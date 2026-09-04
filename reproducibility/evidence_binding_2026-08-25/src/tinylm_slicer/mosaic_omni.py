from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tinylm_slicer.mosaic_resource_profile import process_memory
from tinylm_slicer.mosaic_text_lm import (
    BOS_ID,
    EOS_ID,
    IGNORE_INDEX,
    PAD_ID,
    VOCAB_SIZE,
    MosaicTextConfig,
    MosaicTextLM,
    MosaicTextOutput,
    _sinusoidal_positions,
)
from tinylm_slicer.mosaic_te import (
    MosaicTEConfig,
    MosaicTextEncoderProbe,
)


SLOT_ROLES = (
    *(f"object_{index}" for index in range(8)),
    *(f"relation_{index}" for index in range(8)),
    *(f"action_{index}" for index in range(4)),
    "camera_0",
    "camera_1",
    "lighting_0",
    "lighting_1",
    "environment_0",
    "environment_1",
    "audio_event_0",
    "audio_event_1",
    "narrative",
    "constraints",
    "verification",
    "global",
)
ANSWERABILITY_NGRAM_WIDTHS = (4, 8, 12)


def _compact_body_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    raw = input_ids[:, 1:]
    positions = torch.arange(raw.shape[1], device=raw.device).view(1, -1)
    body_mask = raw.eq(ord("\n")).cumsum(dim=1).gt(0) & raw.ne(PAD_ID)
    order = torch.where(
        body_mask,
        positions,
        positions + raw.shape[1],
    ).argsort(dim=1)
    compact = raw.gather(1, order)
    compact = compact.masked_fill(
        positions >= body_mask.sum(dim=1, keepdim=True),
        PAD_ID,
    )
    return torch.cat((input_ids[:, :1], compact), dim=1)


def _byte_ngram_overlap_features(
    question_input_ids: torch.Tensor,
    evidence_input_ids: torch.Tensor,
    widths: tuple[int, ...] = ANSWERABILITY_NGRAM_WIDTHS,
) -> torch.Tensor:
    question = question_input_ids[:, 1:]
    evidence = evidence_input_ids[:, 1:]
    features = []
    for width in widths:
        if min(question.shape[1], evidence.shape[1]) < width:
            features.append(
                torch.zeros(
                    question.shape[0],
                    device=question.device,
                    dtype=torch.float32,
                )
            )
            continue
        question_windows = question.unfold(1, width, 1)
        evidence_windows = evidence.unfold(1, width, 1)
        question_valid = question_windows.le(255).all(dim=-1)
        evidence_valid = evidence_windows.le(255).all(dim=-1)
        question_hash = torch.zeros_like(question_windows[..., 0])
        evidence_hash = torch.zeros_like(evidence_windows[..., 0])
        for offset in range(width):
            question_hash = question_hash * 257 + question_windows[..., offset] + 1
            evidence_hash = evidence_hash * 257 + evidence_windows[..., offset] + 1
        matches = (
            question_hash.unsqueeze(2).eq(evidence_hash.unsqueeze(1))
            & question_valid.unsqueeze(2)
            & evidence_valid.unsqueeze(1)
        )
        features.append(
            matches.any(dim=2).sum(dim=1).float()
            / question_valid.sum(dim=1).clamp_min(1)
        )
    return torch.stack(features, dim=-1)


def _time_center_object_frame_grid(frame_grid: torch.Tensor) -> torch.Tensor:
    if frame_grid.ndim != 3:
        raise ValueError("object frame grid must be [B*S,F,D]")
    return frame_grid - frame_grid.mean(dim=1, keepdim=True)


def _select_object_temporal_evidence(
    frame_grid: torch.Tensor,
    *,
    time_centered: bool,
    dual_evidence: bool,
    raw_rows: int,
) -> torch.Tensor:
    if not time_centered:
        return frame_grid
    centered = _time_center_object_frame_grid(frame_grid)
    if not dual_evidence:
        return centered
    if not 0 < raw_rows < frame_grid.shape[0]:
        raise ValueError("dual object evidence requires raw and normalized rows")
    return torch.cat((frame_grid[:raw_rows], centered[raw_rows:]), dim=0)


def _normalize_object_frontend_frames(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError("object frontend frames must be [B,C,H,W]")
    mean = frames.mean(dim=(2, 3), keepdim=True)
    scale = frames.var(dim=(2, 3), keepdim=True, unbiased=False).sqrt()
    return (frames - mean) / scale.clamp_min(1e-4)


def _camera_invariant_object_frames(frames: torch.Tensor) -> torch.Tensor:
    """Convert RGB frames to exposure-normalized luminance and edge channels."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("camera-invariant frames must be [B,3,H,W]")
    weights = frames.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    luminance = (frames * weights).sum(dim=1, keepdim=True)
    mean = luminance.mean(dim=(2, 3), keepdim=True)
    scale = luminance.var(dim=(2, 3), keepdim=True, unbiased=False).sqrt()
    normalized = (luminance - mean) / scale.clamp_min(1e-4)
    horizontal = torch.nn.functional.pad(
        normalized[:, :, :, 1:] - normalized[:, :, :, :-1],
        (1, 0, 0, 0),
    )
    vertical = torch.nn.functional.pad(
        normalized[:, :, 1:, :] - normalized[:, :, :-1, :],
        (0, 0, 1, 0),
    )
    return torch.cat((normalized, horizontal, vertical), dim=1)


def _video_camera_statistics(video: torch.Tensor) -> torch.Tensor:
    """Summarize exposure, white balance, contrast, and sharpness per video."""
    if video.ndim != 5 or video.shape[2] != 3:
        raise ValueError("video must be [B,F,3,H,W]")
    means = video.mean(dim=(1, 3, 4))
    deviations = video.std(dim=(1, 3, 4), unbiased=False)
    luma = video.mean(dim=2)
    horizontal = (
        (luma[..., 1:] - luma[..., :-1]).abs().mean(dim=(1, 2, 3))
        if luma.shape[-1] > 1
        else luma.new_zeros(luma.shape[0])
    )
    vertical = (
        (luma[..., 1:, :] - luma[..., :-1, :]).abs().mean(dim=(1, 2, 3))
        if luma.shape[-2] > 1
        else luma.new_zeros(luma.shape[0])
    )
    return torch.cat(
        (means, deviations, horizontal[:, None], vertical[:, None]),
        dim=1,
    )


def _cross_modal_late_summaries(
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    video_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_tokens = torch.nn.functional.normalize(text_tokens, dim=-1)
    video_tokens = torch.nn.functional.normalize(video_tokens, dim=-1)
    similarity = torch.einsum("btd,bvd->btv", text_tokens, video_tokens)
    text_scores = similarity.max(dim=2).values.masked_fill(~text_mask, -torch.inf)
    text_weights = torch.softmax(text_scores, dim=1).masked_fill(~text_mask, 0.0)
    video_scores = similarity.masked_fill(~text_mask.unsqueeze(-1), -torch.inf).max(
        dim=1
    ).values
    video_weights = torch.softmax(video_scores, dim=1)
    return (
        (text_tokens * text_weights.unsqueeze(-1)).sum(dim=1),
        (video_tokens * video_weights.unsqueeze(-1)).sum(dim=1),
    )


def _cross_modal_query_summary(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    score: nn.Linear,
) -> torch.Tensor:
    """Pool locally relevant evidence without conditioning on another modality."""
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("query summary expects [B,T,D] tokens and [B,T] mask")
    if bool((~mask.any(dim=1)).any()):
        raise ValueError("query summary requires at least one active token")
    tokens = torch.nn.functional.normalize(tokens, dim=-1)
    logits = score(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
    weights = torch.softmax(logits, dim=1)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1)


def _cross_modal_sequence_summary(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    encoder: nn.GRU,
) -> torch.Tensor:
    """Preserve evidence order and return the final active recurrent state."""
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("sequence summary expects [B,T,D] tokens and [B,T] mask")
    if bool((~mask.any(dim=1)).any()):
        raise ValueError("sequence summary requires at least one active token")
    encoded, _ = encoder(torch.nn.functional.normalize(tokens, dim=-1))
    last = mask.sum(dim=1) - 1
    summary = encoded[torch.arange(tokens.shape[0], device=tokens.device), last]
    return torch.nn.functional.normalize(summary, dim=-1)


def _cross_modal_last_summary(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the final active contextual token from each sequence."""
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("last summary expects [B,T,D] tokens and [B,T] mask")
    if bool((~mask.any(dim=1)).any()):
        raise ValueError("last summary requires at least one active token")
    last = mask.sum(dim=1) - 1
    summary = tokens[torch.arange(tokens.shape[0], device=tokens.device), last]
    return torch.nn.functional.normalize(summary, dim=-1)


def _spatial_event_features(
    frame_grid: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    """Keep position, event time, and their signed interaction per slot."""
    if frame_grid.ndim != 4 or attention.ndim != 5:
        raise ValueError("spatial event inputs must be [B,S,F,D]/[B,S,F,H,W]")
    if frame_grid.shape[:3] != attention.shape[:3]:
        raise ValueError("spatial event inputs must share batch/slots/frames")
    times = torch.linspace(
        -1.0,
        1.0,
        frame_grid.shape[2],
        device=frame_grid.device,
        dtype=frame_grid.dtype,
    )
    centered = frame_grid - frame_grid.mean(dim=2, keepdim=True)
    activity = centered.square().mean(dim=-1)
    event_time = (activity * times.view(1, 1, -1)).sum(dim=2) / (
        activity.sum(dim=2).clamp_min(1e-6)
    )
    x = torch.linspace(
        -1.0,
        1.0,
        attention.shape[-1],
        device=attention.device,
        dtype=attention.dtype,
    )
    position = (attention[:, :, 0] * x.view(1, 1, 1, -1)).sum(dim=(-2, -1))
    result = torch.zeros_like(frame_grid[:, :, 0])
    result[..., 0] = position
    result[..., 1] = event_time
    result[..., 2] = position * event_time
    return result


def _spatial_temporal_moment(
    feature_map: torch.Tensor,
    *,
    batch: int,
    frames: int,
    axis: str = "x",
) -> torch.Tensor:
    """Summarize whether visual changes move from left-to-right or vice versa."""
    if feature_map.ndim != 4 or feature_map.shape[0] != batch * frames:
        raise ValueError("feature map must be [batch*frames,D,H,W]")
    maps = feature_map.reshape(batch, frames, *feature_map.shape[1:])
    activity = (maps - maps.mean(dim=1, keepdim=True)).square().mean(dim=2)
    time = torch.linspace(
        -1.0,
        1.0,
        frames,
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    if axis not in {"x", "y"}:
        raise ValueError("spatial-temporal moment axis must be x or y")
    spatial_size = feature_map.shape[-1] if axis == "x" else feature_map.shape[-2]
    coordinate = torch.linspace(
        -1.0,
        1.0,
        spatial_size,
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    coordinate = (
        coordinate.view(
            1,
            1,
            1,
            -1,
        )
        if axis == "x"
        else coordinate.view(1, 1, -1, 1)
    )
    signed = activity * time.view(1, frames, 1, 1) * coordinate
    return signed.sum(dim=(1, 2, 3)) / activity.sum(dim=(1, 2, 3)).clamp_min(1e-6)


def _object_attention_trajectory(
    attention: torch.Tensor,
    match_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode each slot's 2D path and visibility confidence without a fixed axis."""
    if attention.ndim != 5:
        raise ValueError("object attention must be [B,S,F,H,W]")
    batch, slots, frames, height, width = attention.shape
    if min(batch, slots, frames, height, width) <= 0:
        raise ValueError("object attention dimensions must be positive")
    probability = attention / attention.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    x = torch.linspace(-1.0, 1.0, width, device=attention.device, dtype=attention.dtype)
    y = torch.linspace(
        -1.0, 1.0, height, device=attention.device, dtype=attention.dtype
    )
    position = torch.stack(
        (
            (probability * x.view(1, 1, 1, 1, -1)).sum(dim=(-2, -1)),
            (probability * y.view(1, 1, 1, -1, 1)).sum(dim=(-2, -1)),
        ),
        dim=-1,
    )
    if match_logits is not None:
        position = _confidence_gated_sequence(position, match_logits)
    time = torch.linspace(
        -1.0, 1.0, frames, device=attention.device, dtype=attention.dtype
    )
    peak = probability.amax(dim=(-2, -1))
    entropy = -(probability * probability.clamp_min(1e-6).log()).sum(
        dim=(-2, -1)
    ) / math.log(max(2, height * width))
    return torch.cat(
        (
            position[:, :, 0],
            position[:, :, -1],
            position[:, :, -1] - position[:, :, 0],
            (position * time.view(1, 1, frames, 1)).mean(dim=2),
            peak.mean(dim=2, keepdim=True),
            (peak * time).mean(dim=2, keepdim=True),
            entropy.mean(dim=2, keepdim=True),
            (entropy * time).mean(dim=2, keepdim=True),
        ),
        dim=-1,
    )


def _confidence_gated_sequence(
    values: torch.Tensor,
    match_logits: torch.Tensor,
) -> torch.Tensor:
    """Carry the last reliable identity state through weak-match frames."""
    if values.ndim < 3 or match_logits.shape != values.shape[:-1]:
        raise ValueError(
            "values/match logits must be [...,frames,features]/[...,frames]"
        )
    frames = values.shape[-2]
    confidence = (torch.softmax(match_logits, dim=-1) * frames).clamp_max(1.0)
    state = values[..., 0, :]
    carried = [state]
    for frame in range(1, frames):
        gate = confidence[..., frame].unsqueeze(-1)
        state = gate * values[..., frame, :] + (1.0 - gate) * state
        carried.append(state)
    return torch.stack(carried, dim=-2)


def _temporal_relative_visibility(visibility_logits: torch.Tensor) -> torch.Tensor:
    """Use within-sequence rank when a role's visibility changes over time."""
    if visibility_logits.ndim != 3 or visibility_logits.shape[1] != 2:
        raise ValueError("temporal visibility logits must be [B,2,F]")
    centered = visibility_logits - visibility_logits.mean(dim=-1, keepdim=True)
    dynamic = visibility_logits.amax(dim=-1, keepdim=True) > visibility_logits.amin(
        dim=-1,
        keepdim=True,
    )
    relative = (centered >= 0).to(visibility_logits.dtype)
    absolute = torch.sigmoid(visibility_logits)
    return torch.where(dynamic, relative, absolute)


def _descriptor_object_memory(
    attention: torch.Tensor,
    visibility_logits: torch.Tensor,
    *,
    temporal_relative_visibility: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write identity positions, hold through occlusion, and read first return."""
    if attention.ndim != 5:
        raise ValueError("descriptor attention must be [B,2,F,H,W]")
    if attention.shape[1] != 2 or visibility_logits.shape != attention.shape[:3]:
        raise ValueError("object memory requires two roles and per-frame visibility")
    height, width = attention.shape[-2:]
    probability = attention / attention.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    x = torch.linspace(-1.0, 1.0, width, device=attention.device, dtype=attention.dtype)
    y = torch.linspace(
        -1.0, 1.0, height, device=attention.device, dtype=attention.dtype
    )
    positions = torch.stack(
        (
            (probability * x.view(1, 1, 1, 1, -1)).sum(dim=(-2, -1)),
            (probability * y.view(1, 1, 1, -1, 1)).sum(dim=(-2, -1)),
        ),
        dim=-1,
    )
    visibility = (
        _temporal_relative_visibility(visibility_logits)
        if temporal_relative_visibility
        else torch.sigmoid(visibility_logits)
    ).to(attention.dtype)
    anchor = torch.zeros_like(positions[:, :, 0])
    anchor_weight = torch.zeros_like(visibility[:, :, 0])
    occluded = torch.zeros_like(anchor_weight)
    returned = torch.zeros_like(anchor_weight)
    read = torch.zeros_like(anchor)
    for frame in range(positions.shape[2]):
        visible = visibility[:, :, frame]
        write = (1.0 - anchor_weight) * visible * (1.0 - occluded)
        anchor = anchor + write.unsqueeze(-1) * positions[:, :, frame]
        anchor_weight = anchor_weight + write
        return_gate = occluded * visible * (1.0 - returned)
        read = read + return_gate.unsqueeze(-1) * positions[:, :, frame]
        returned = returned + return_gate
        occlusion_gate = anchor_weight * (1.0 - visible) * (1.0 - returned)
        occluded = occluded + (1.0 - occluded) * occlusion_gate
    anchor = anchor / anchor_weight.clamp_min(1e-6).unsqueeze(-1)
    read = read / returned.clamp_min(1e-6).unsqueeze(-1)
    delta = read - anchor
    role_features = torch.cat(
        (
            anchor,
            read,
            delta,
            occluded.unsqueeze(-1),
            returned.unsqueeze(-1),
        ),
        dim=-1,
    ).flatten(1)
    same_cost = delta.square().sum(dim=(-2, -1))
    swap_cost = (read[:, 0] - anchor[:, 1]).square().sum(dim=-1) + (
        read[:, 1] - anchor[:, 0]
    ).square().sum(dim=-1)
    margin = same_cost - swap_cost
    features = torch.cat(
        (
            role_features,
            same_cost.unsqueeze(-1),
            swap_cost.unsqueeze(-1),
            margin.unsqueeze(-1),
        ),
        dim=-1,
    )
    return features, margin


def _contrast_memory_summary(
    cosine_peak: torch.Tensor,
    cosine_margin: torch.Tensor,
) -> torch.Tensor:
    """Summarize bounded per-role confidence without collapsing it to one logit."""
    if (
        cosine_peak.ndim != 3
        or cosine_peak.shape[1] != 2
        or cosine_margin.shape != cosine_peak.shape
    ):
        raise ValueError("contrast confidence must be matched [B,2,F] tensors")
    summary = torch.stack(
        (
            cosine_peak.mean(dim=-1),
            0.5 * (cosine_peak.amax(dim=-1) - cosine_peak.amin(dim=-1)),
            0.5 * cosine_margin.mean(dim=-1),
            0.5 * (cosine_margin.amax(dim=-1) - cosine_margin.amin(dim=-1)),
        ),
        dim=-1,
    )
    return summary.flatten(1)


def _normalized_evidence_preference(
    weights: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Enable memory only when normalized evidence outranks raw evidence."""
    if weights.ndim != 2 or weights.shape[1] != 2:
        raise ValueError("evidence weights must be [B,2]")
    if not 0.0 <= margin < 1.0:
        raise ValueError("evidence preference margin must be in [0, 1)")
    return 2.0 * torch.relu(weights[:, 1] - weights[:, 0] - margin) / (
        1.0 - margin
    )


def _sort_object_slots_by_temporal_activity(
    object_slots: torch.Tensor,
    frame_grid: torch.Tensor,
) -> torch.Tensor:
    """Canonicalize spatial slots by their time-varying energy."""
    if object_slots.ndim != 3 or frame_grid.ndim != 4:
        raise ValueError("object slots/frame grid must be [B,S,D]/[B,S,F,D]")
    if object_slots.shape[:2] != frame_grid.shape[:2]:
        raise ValueError("object slots and frame grid must share batch/slots")
    centered = frame_grid - frame_grid.mean(dim=2, keepdim=True)
    activity = centered.square().mean(dim=(2, 3))
    tie_values = torch.cat(
        (centered.flatten(2), object_slots),
        dim=-1,
    )
    feature_index = torch.arange(
        1,
        tie_values.shape[-1] + 1,
        device=tie_values.device,
        dtype=tie_values.dtype,
    )
    fingerprints = torch.stack(
        (
            (tie_values * ((feature_index % 97) + 1)).sum(dim=-1),
            (tie_values * (((feature_index * 17) % 193) + 1)).sum(dim=-1),
            (tie_values * (((feature_index.square()) % 389) + 1)).sum(dim=-1),
        ),
        dim=-1,
    )
    order = torch.arange(
        object_slots.shape[1],
        device=object_slots.device,
    ).expand(object_slots.shape[0], -1)
    for key_index in range(fingerprints.shape[-1] - 1, -1, -1):
        key = torch.gather(fingerprints[:, :, key_index], 1, order)
        local_order = torch.argsort(
            key,
            dim=1,
            descending=True,
            stable=True,
        )
        order = torch.gather(order, 1, local_order)
    activity = torch.gather(activity, 1, order)
    order = torch.gather(
        order,
        1,
        torch.argsort(activity, dim=1, descending=True, stable=True),
    )
    return torch.gather(
        object_slots,
        1,
        order.unsqueeze(-1).expand(-1, -1, object_slots.shape[-1]),
    )


@dataclass(frozen=True)
class MosaicOmniConfig:
    world_slots: int = 32
    world_dim: int = 256
    object_slots: int = 8
    attention_heads: int = 8
    gemma_hidden_dim: int = 3840
    anima_conditioning_tokens: int = 512
    anima_conditioning_dim: int = 1024

    def __post_init__(self) -> None:
        values = (
            self.world_slots,
            self.world_dim,
            self.object_slots,
            self.attention_heads,
            self.gemma_hidden_dim,
            self.anima_conditioning_tokens,
            self.anima_conditioning_dim,
        )
        if min(values) <= 0:
            raise ValueError("MOSAIC-OMNI configuration values must be positive")
        if self.world_slots != len(SLOT_ROLES):
            raise ValueError(
                f"world_slots must match the {len(SLOT_ROLES)} frozen roles"
            )
        if self.object_slots > self.world_slots:
            raise ValueError("object slots cannot exceed world slots")
        if self.world_dim % self.attention_heads:
            raise ValueError("world_dim must be divisible by attention_heads")


@dataclass(frozen=True)
class MosaicUnifiedConfig:
    """One-checkpoint raw-modal input contract for MOSAIC."""

    text: MosaicTextConfig = field(default_factory=MosaicTextConfig)
    omni: MosaicOmniConfig = field(default_factory=MosaicOmniConfig)
    vision_patch_size: int = 16
    visual_semantic_encoder: bool = False
    visual_semantic_split_frontend: bool = False
    visual_semantic_rounds: int = 2
    image_visual_adapter: bool = False
    image_visual_adapter_rank: int = 32
    image_visual_adapter_scale: float = 1.0
    visual_teacher_slot_bridge: bool = False
    visual_teacher_slot_rank: int = 32
    explicit_object_relation_grounder: bool = False
    explicit_object_relation_rank: int = 64
    explicit_relation_classes: int = 6
    audio_patch_samples: int = 320
    world_ffn_dim: int = 1024
    world_rounds: int = 2
    vision_teacher_dim: int = 0
    audio_teacher_dim: int = 0
    audio_temporal_encoder: bool = False
    audio_content_encoder: bool = False
    audio_spectral_content_frontend: bool = False
    audio_spectral_n_fft: int = 400
    audio_spectral_hop_samples: int = 160
    audio_event_slot_injection: bool = False
    audio_ctc_head: bool = False
    audio_grapheme_ctc_vocabulary_size: int = 0
    audio_text_retrieval_head: bool = False
    audio_text_retrieval_text_source: str = "world_global"
    cross_modal_evidence_head: bool = False
    cross_modal_evidence_rank: int = 32
    cross_modal_evidence_direct_features: bool = False
    cross_modal_text_query_pooling: bool = False
    cross_modal_text_sequence_pooling: bool = False
    cross_modal_text_contextual_pooling: bool = False
    narrative_evidence_head: bool = False
    narrative_evidence_hidden_dim: int = 64
    visual_text_retrieval_head: bool = False
    visual_text_retrieval_dim: int = 512
    audio_temporal_binary_head: bool = False
    video_object_temporal_encoder: bool = False
    video_object_frame_normalized_input: bool = False
    video_object_camera_invariant_residual: bool = False
    video_object_frame_normalized_residual_scale: float = 1.0
    video_object_time_centered_input: bool = False
    video_object_activity_sorted_slots: bool = False
    video_object_dual_evidence: bool = False
    video_object_set_decision: bool = False
    video_object_identity_event_binding: bool = False
    video_object_learned_queries: bool = False
    video_object_spatial_coordinates: bool = False
    video_object_spatial_event_binding: bool = False
    video_spatial_temporal_moment: bool = False
    video_query_spatial_temporal_moment: bool = False
    video_spatial_temporal_y_moment: bool = False
    video_spatial_temporal_logit_head: bool = False
    video_spatial_temporal_bilinear_head: bool = False
    video_object_trajectory_binding: bool = False
    video_object_pair_trajectory_binding: bool = False
    video_descriptor_trajectory_binding: bool = False
    video_descriptor_pair_centered_queries: bool = False
    video_descriptor_persistent_identity_state: bool = False
    video_descriptor_object_memory: bool = False
    video_descriptor_object_memory_scale: float = 1.0
    video_descriptor_object_memory_query_gate: bool = False
    video_descriptor_object_memory_reliability_gate: bool = False
    video_descriptor_object_memory_contrast_visibility: bool = False
    video_descriptor_object_memory_evidence_routing: bool = False
    video_descriptor_object_memory_evidence_routing_margin: float = 0.0
    video_descriptor_object_memory_contrast_readout: bool = False
    video_descriptor_object_memory_temporal_relative_visibility: bool = False
    video_descriptor_object_memory_temporal_relative_readout: bool = False
    video_isolated_identity_descriptors: bool = False
    video_query_conditioned_head: bool = False
    video_camera_robustness_adapter: bool = False
    video_camera_robustness_nonlinear_gate: bool = False
    video_camera_pose_dim: int = 0
    video_spatial_relation_classes: int = 0
    video_action_dim: int = 0
    video_egomotion_classes: int = 0
    video_egomotion_validity_head: bool = False
    video_egomotion_evidence_gate: bool = False
    video_egomotion_minimum_motion_evidence: float = 1e-6
    video_uses_visual_semantic_encoder: bool = False
    video_visual_semantic_scale: float = 1.0
    video_explicit_temporal_delta: bool = False
    video_explicit_temporal_delta_scale: float = 1.0
    video_separate_temporal_delta_projection: bool = False
    long_video_world_accumulator: bool = False
    long_video_transition_features: bool = False
    text_only_bridge_adapter: bool = False
    text_only_output_adapter: bool = False
    text_only_cross_memory_adapter: bool = False
    text_only_hidden_cross_memory_adapter: bool = False
    text_answerability_head: bool = False
    text_answerability_mode: str = "pooled"
    text_answerability_classes: int = 2
    text_epistemic_memory_adapter: bool = False
    text_epistemic_memory_slots: int = 1
    text_epistemic_output_rank: int = 0
    text_epistemic_supported_class: int = 0
    text_epistemic_output_threshold: float = 0.5
    text_answerability_fallback_bytes: tuple[int, ...] = ()
    text_answerability_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (
            min(
                self.vision_patch_size,
                self.visual_semantic_rounds,
                self.image_visual_adapter_rank,
                self.visual_teacher_slot_rank,
                self.explicit_object_relation_rank,
                self.explicit_relation_classes,
                self.audio_patch_samples,
                self.audio_spectral_n_fft,
                self.audio_spectral_hop_samples,
                self.cross_modal_evidence_rank,
                self.narrative_evidence_hidden_dim,
                self.world_ffn_dim,
                self.world_rounds,
            )
            <= 0
        ):
            raise ValueError(
                "unified model patch and world dimensions must be positive"
            )
        if min(self.vision_teacher_dim, self.audio_teacher_dim) < 0:
            raise ValueError("teacher dimensions must not be negative")
        if not 0 <= self.video_camera_pose_dim <= 32:
            raise ValueError("video camera pose dimension must be in [0, 32]")
        if not 0 <= self.video_spatial_relation_classes <= 16:
            raise ValueError("video spatial relation classes must be in [0, 16]")
        if not 0 <= self.video_action_dim <= 64:
            raise ValueError("video action dimension must be in [0, 64]")
        if not 0 <= self.video_egomotion_classes <= 16:
            raise ValueError("video egomotion classes must be in [0, 16]")
        if self.video_egomotion_classes and not self.video_action_dim:
            raise ValueError("video egomotion requires action input")
        if self.video_egomotion_validity_head and not self.video_egomotion_classes:
            raise ValueError("egomotion validity requires egomotion classes")
        if self.video_egomotion_evidence_gate and not self.video_egomotion_classes:
            raise ValueError("egomotion evidence gate requires egomotion classes")
        if self.video_egomotion_minimum_motion_evidence < 0:
            raise ValueError("minimum motion evidence must be non-negative")
        if sum(
            (
                self.cross_modal_text_query_pooling,
                self.cross_modal_text_sequence_pooling,
                self.cross_modal_text_contextual_pooling,
            )
        ) > 1:
            raise ValueError("cross-modal text pooling modes are mutually exclusive")
        if self.audio_grapheme_ctc_vocabulary_size < 0:
            raise ValueError("grapheme CTC vocabulary size must not be negative")
        if self.visual_text_retrieval_dim <= 0:
            raise ValueError("visual/text retrieval dimension must be positive")
        if (
            self.visual_teacher_slot_bridge
            and self.visual_teacher_slot_rank > self.omni.world_dim
        ):
            raise ValueError("visual teacher slot rank must fit the world dimension")
        if (
            self.explicit_object_relation_grounder
            and self.explicit_object_relation_rank > self.omni.world_dim
        ):
            raise ValueError("object relation rank must fit the world dimension")
        if self.explicit_object_relation_grounder and (
            self.omni.object_slots < 2
            or self.omni.object_slots >= self.omni.world_slots
        ):
            raise ValueError(
                "object relation grounding requires two object slots and one relation slot"
            )
        if self.audio_grapheme_ctc_vocabulary_size and not self.audio_content_encoder:
            raise ValueError("grapheme CTC head requires the content encoder")
        if self.text_answerability_mode not in {
            "pooled",
            "token-cross",
            "contextual-cross",
            "consistency-cross",
            "body-cross",
            "core-body-cross",
            "core-compact-body-cross",
            "core-projected-compact-body-cross",
            "core-lexical-compact-body-cross",
            "core-lexical-consistency-cross",
        }:
            raise ValueError("unsupported text answerability mode")
        if any(
            value < 0 or value > 255 for value in self.text_answerability_fallback_bytes
        ):
            raise ValueError("answerability fallback bytes must be raw bytes")
        if not 2 <= self.text_answerability_classes <= 8:
            raise ValueError("answerability classes must be in [2, 8]")
        if self.text_epistemic_memory_adapter and not self.text_answerability_head:
            raise ValueError("epistemic memory adapter requires answerability head")
        if not 1 <= self.text_epistemic_memory_slots <= self.omni.world_slots:
            raise ValueError("epistemic memory slots must fit the world workspace")
        if not 0 <= self.text_epistemic_output_rank <= self.text.model_dim:
            raise ValueError("epistemic output rank must fit the text model")
        if self.text_epistemic_output_rank and not self.text_answerability_head:
            raise ValueError("epistemic output adapter requires answerability head")
        if self.audio_temporal_binary_head and not self.audio_temporal_encoder:
            raise ValueError(
                "audio temporal binary head requires temporal audio encoder"
            )
        if self.video_uses_visual_semantic_encoder and not self.visual_semantic_encoder:
            raise ValueError(
                "video semantic frames require the visual semantic encoder"
            )
        if self.image_visual_adapter and not (
            self.visual_semantic_encoder and self.visual_semantic_split_frontend
        ):
            raise ValueError(
                "image visual adapter requires the split visual semantic encoder"
            )
        if not 0.0 < self.image_visual_adapter_scale <= 1.0:
            raise ValueError("image visual adapter scale must be in (0, 1]")
        if not 0.0 < self.video_visual_semantic_scale <= 1.0:
            raise ValueError("video visual semantic scale must be in (0, 1]")
        if self.video_explicit_temporal_delta_scale <= 0.0:
            raise ValueError("video explicit temporal delta scale must be positive")
        if (
            self.video_separate_temporal_delta_projection
            and not self.video_explicit_temporal_delta
        ):
            raise ValueError(
                "separate video delta projection requires explicit temporal delta"
            )
        if (
            self.long_video_transition_features
            and not self.long_video_world_accumulator
        ):
            raise ValueError("long-video transition features require the accumulator")
        if self.audio_spectral_content_frontend and not self.audio_content_encoder:
            raise ValueError("spectral content frontend requires the content encoder")
        if self.audio_spectral_hop_samples > self.audio_spectral_n_fft:
            raise ValueError("spectral hop must not exceed the FFT window")
        if self.audio_text_retrieval_text_source not in {
            "world_global",
            "text_token_mean",
        }:
            raise ValueError("unsupported audio/text retrieval text source")
        if (
            not 0
            <= self.text_epistemic_supported_class
            < (self.text_answerability_classes)
        ):
            raise ValueError("epistemic supported class is out of range")
        if not 0.0 < self.text_epistemic_output_threshold <= 1.0:
            raise ValueError("epistemic output threshold must be in (0, 1]")
        if not 0.0 < self.text_answerability_threshold < 1.0:
            raise ValueError("answerability threshold must be in (0, 1)")
        if (
            self.video_object_frame_normalized_input
            or self.video_object_camera_invariant_residual
            or self.video_object_time_centered_input
            or self.video_object_activity_sorted_slots
            or self.video_object_dual_evidence
            or self.video_object_learned_queries
            or self.video_object_spatial_coordinates
            or self.video_object_spatial_event_binding
            or self.video_spatial_temporal_moment
            or self.video_query_spatial_temporal_moment
            or self.video_spatial_temporal_y_moment
            or self.video_spatial_temporal_logit_head
            or self.video_spatial_temporal_bilinear_head
            or self.video_object_trajectory_binding
            or self.video_object_pair_trajectory_binding
            or self.video_descriptor_trajectory_binding
        ) and not self.video_object_temporal_encoder:
            raise ValueError("object-temporal invariance requires the object encoder")
        if (
            self.video_object_spatial_coordinates
            and not self.video_object_learned_queries
        ):
            raise ValueError(
                "object spatial coordinates require learned object queries"
            )
        if self.video_object_spatial_event_binding and not (
            self.video_object_spatial_coordinates
            and self.video_object_identity_event_binding
        ):
            raise ValueError(
                "spatial event binding requires spatial queries and identity-event binding"
            )
        if self.video_spatial_temporal_moment and not self.video_object_dual_evidence:
            raise ValueError("spatial-temporal moment requires dual evidence")
        if self.video_query_spatial_temporal_moment and not (
            self.video_spatial_temporal_moment and self.video_query_conditioned_head
        ):
            raise ValueError(
                "query spatial-temporal moment requires query-conditioned video and moment"
            )
        if self.video_spatial_temporal_y_moment and not (
            self.video_spatial_temporal_moment
            and self.video_query_spatial_temporal_moment
        ):
            raise ValueError("y spatial-temporal moment requires query-gated x moment")
        if self.video_spatial_temporal_logit_head and not (
            self.video_spatial_temporal_y_moment and self.video_query_conditioned_head
        ):
            raise ValueError(
                "spatial-temporal logit head requires xy moments and query conditioning"
            )
        if self.video_spatial_temporal_bilinear_head and not (
            self.video_spatial_temporal_y_moment and self.video_query_conditioned_head
        ):
            raise ValueError(
                "spatial-temporal bilinear head requires xy moments and query conditioning"
            )
        if self.video_object_trajectory_binding and not (
            self.video_object_dual_evidence
            and self.video_object_learned_queries
            and self.video_object_spatial_coordinates
            and self.video_query_conditioned_head
        ):
            raise ValueError(
                "object trajectory binding requires dual evidence, spatial learned queries, and query conditioning"
            )
        if self.video_object_pair_trajectory_binding and not (
            self.video_object_dual_evidence
            and self.video_object_learned_queries
            and self.video_object_spatial_coordinates
            and self.video_query_conditioned_head
        ):
            raise ValueError(
                "object pair trajectory binding requires dual evidence, spatial learned queries, and query conditioning"
            )
        if (
            self.video_descriptor_trajectory_binding
            and not self.video_query_conditioned_head
        ):
            raise ValueError(
                "descriptor trajectory binding requires query-conditioned video"
            )
        if (
            self.video_spatial_relation_classes
            and not self.video_descriptor_trajectory_binding
        ):
            raise ValueError(
                "video spatial relations require descriptor trajectory binding"
            )
        if (
            self.video_descriptor_pair_centered_queries
            and not self.video_descriptor_trajectory_binding
        ):
            raise ValueError(
                "descriptor pair centering requires descriptor trajectory binding"
            )
        if (
            self.video_descriptor_persistent_identity_state
            and not self.video_descriptor_trajectory_binding
        ):
            raise ValueError(
                "persistent descriptor identity requires descriptor trajectory binding"
            )
        if (
            self.video_descriptor_object_memory
            and not self.video_descriptor_trajectory_binding
        ):
            raise ValueError(
                "descriptor object memory requires descriptor trajectory binding"
            )
        if not 0.0 <= self.video_descriptor_object_memory_scale <= 1.0:
            raise ValueError("descriptor object memory scale must be in [0, 1]")
        if (
            self.video_descriptor_object_memory_query_gate
            and not self.video_descriptor_object_memory
        ):
            raise ValueError("descriptor memory query gate requires object memory")
        if (
            self.video_descriptor_object_memory_reliability_gate
            and not self.video_descriptor_object_memory
        ):
            raise ValueError(
                "descriptor memory reliability gate requires object memory"
            )
        if (
            self.video_descriptor_object_memory_contrast_visibility
            and not self.video_descriptor_object_memory
        ):
            raise ValueError(
                "descriptor memory contrast visibility requires object memory"
            )
        if self.video_descriptor_object_memory_evidence_routing and not (
            self.video_descriptor_object_memory and self.video_object_dual_evidence
        ):
            raise ValueError(
                "descriptor memory evidence routing requires object memory and dual evidence"
            )
        if not 0.0 <= self.video_descriptor_object_memory_evidence_routing_margin < 1.0:
            raise ValueError("descriptor memory evidence routing margin must be in [0, 1)")
        if (
            self.video_descriptor_object_memory_evidence_routing_margin
            and not self.video_descriptor_object_memory_evidence_routing
        ):
            raise ValueError(
                "descriptor memory evidence routing margin requires evidence routing"
            )
        if self.video_descriptor_object_memory_contrast_readout and not (
            self.video_descriptor_object_memory
            and self.video_descriptor_object_memory_contrast_visibility
        ):
            raise ValueError(
                "descriptor memory contrast readout requires contrast visibility"
            )
        if (
            self.video_descriptor_object_memory_temporal_relative_visibility
            and not self.video_descriptor_object_memory
        ):
            raise ValueError(
                "descriptor memory temporal relative visibility requires object memory"
            )
        if self.video_descriptor_object_memory_temporal_relative_readout and not (
            self.video_descriptor_object_memory
            and self.video_descriptor_object_memory_evidence_routing
        ):
            raise ValueError(
                "descriptor memory temporal relative readout requires object memory "
                "and evidence routing"
            )
        if (
            self.video_descriptor_object_memory_temporal_relative_readout
            and self.video_descriptor_object_memory_temporal_relative_visibility
        ):
            raise ValueError(
                "descriptor memory temporal relative readout preserves the parent "
                "visibility path and cannot replace it"
            )
        if (
            self.video_descriptor_object_memory_query_gate
            and self.video_descriptor_object_memory_reliability_gate
        ):
            raise ValueError("descriptor memory gates are mutually exclusive")
        if self.video_isolated_identity_descriptors and not (
            self.video_object_pair_trajectory_binding
            or self.video_descriptor_trajectory_binding
        ):
            raise ValueError(
                "isolated identity descriptors require pair or dense binding"
            )
        if self.video_object_dual_evidence and not self.video_query_conditioned_head:
            raise ValueError("dual object evidence requires query-conditioned video")
        if self.video_object_set_decision and not self.video_object_dual_evidence:
            raise ValueError("object set decision requires dual evidence")
        if (
            self.video_camera_robustness_nonlinear_gate
            and not self.video_camera_robustness_adapter
        ):
            raise ValueError("nonlinear camera gate requires camera robustness adapter")
        if (
            self.video_object_identity_event_binding
            and not self.video_object_dual_evidence
        ):
            raise ValueError("identity-event binding requires dual evidence")
        if not 0.0 <= self.video_object_frame_normalized_residual_scale <= 1.0:
            raise ValueError("object frame-normalized residual scale must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "mosaic-unified-config-v0",
            "text": self.text.to_dict(),
            "omni": asdict(self.omni),
            "vision_patch_size": self.vision_patch_size,
            "visual_semantic_encoder": self.visual_semantic_encoder,
            "visual_semantic_split_frontend": (self.visual_semantic_split_frontend),
            "visual_semantic_rounds": self.visual_semantic_rounds,
            "image_visual_adapter": self.image_visual_adapter,
            "image_visual_adapter_rank": self.image_visual_adapter_rank,
            "image_visual_adapter_scale": self.image_visual_adapter_scale,
            "visual_teacher_slot_bridge": self.visual_teacher_slot_bridge,
            "visual_teacher_slot_rank": self.visual_teacher_slot_rank,
            "explicit_object_relation_grounder": (
                self.explicit_object_relation_grounder
            ),
            "explicit_object_relation_rank": self.explicit_object_relation_rank,
            "explicit_relation_classes": self.explicit_relation_classes,
            "audio_patch_samples": self.audio_patch_samples,
            "world_ffn_dim": self.world_ffn_dim,
            "world_rounds": self.world_rounds,
            "vision_teacher_dim": self.vision_teacher_dim,
            "audio_teacher_dim": self.audio_teacher_dim,
            "audio_temporal_encoder": self.audio_temporal_encoder,
            "audio_content_encoder": self.audio_content_encoder,
            "audio_spectral_content_frontend": (self.audio_spectral_content_frontend),
            "audio_spectral_n_fft": self.audio_spectral_n_fft,
            "audio_spectral_hop_samples": self.audio_spectral_hop_samples,
            "audio_event_slot_injection": self.audio_event_slot_injection,
            "audio_ctc_head": self.audio_ctc_head,
            "audio_grapheme_ctc_vocabulary_size": (
                self.audio_grapheme_ctc_vocabulary_size
            ),
            "audio_text_retrieval_head": self.audio_text_retrieval_head,
            "audio_text_retrieval_text_source": (self.audio_text_retrieval_text_source),
            "cross_modal_evidence_head": self.cross_modal_evidence_head,
            "cross_modal_evidence_rank": self.cross_modal_evidence_rank,
            "cross_modal_evidence_direct_features": (
                self.cross_modal_evidence_direct_features
            ),
            "cross_modal_text_query_pooling": self.cross_modal_text_query_pooling,
            "cross_modal_text_sequence_pooling": (
                self.cross_modal_text_sequence_pooling
            ),
            "cross_modal_text_contextual_pooling": (
                self.cross_modal_text_contextual_pooling
            ),
            "narrative_evidence_head": self.narrative_evidence_head,
            "narrative_evidence_hidden_dim": self.narrative_evidence_hidden_dim,
            "visual_text_retrieval_head": self.visual_text_retrieval_head,
            "visual_text_retrieval_dim": self.visual_text_retrieval_dim,
            "audio_temporal_binary_head": self.audio_temporal_binary_head,
            "video_object_temporal_encoder": (self.video_object_temporal_encoder),
            "video_object_frame_normalized_input": (
                self.video_object_frame_normalized_input
            ),
            "video_object_camera_invariant_residual": (
                self.video_object_camera_invariant_residual
            ),
            "video_object_frame_normalized_residual_scale": (
                self.video_object_frame_normalized_residual_scale
            ),
            "video_object_time_centered_input": (self.video_object_time_centered_input),
            "video_object_activity_sorted_slots": (
                self.video_object_activity_sorted_slots
            ),
            "video_object_dual_evidence": self.video_object_dual_evidence,
            "video_object_set_decision": self.video_object_set_decision,
            "video_object_identity_event_binding": (
                self.video_object_identity_event_binding
            ),
            "video_object_learned_queries": (self.video_object_learned_queries),
            "video_object_spatial_coordinates": (self.video_object_spatial_coordinates),
            "video_object_spatial_event_binding": (
                self.video_object_spatial_event_binding
            ),
            "video_spatial_temporal_moment": (self.video_spatial_temporal_moment),
            "video_query_spatial_temporal_moment": (
                self.video_query_spatial_temporal_moment
            ),
            "video_spatial_temporal_y_moment": (self.video_spatial_temporal_y_moment),
            "video_spatial_temporal_logit_head": (
                self.video_spatial_temporal_logit_head
            ),
            "video_spatial_temporal_bilinear_head": (
                self.video_spatial_temporal_bilinear_head
            ),
            "video_object_trajectory_binding": (self.video_object_trajectory_binding),
            "video_object_pair_trajectory_binding": (
                self.video_object_pair_trajectory_binding
            ),
            "video_descriptor_trajectory_binding": (
                self.video_descriptor_trajectory_binding
            ),
            "video_descriptor_pair_centered_queries": (
                self.video_descriptor_pair_centered_queries
            ),
            "video_descriptor_persistent_identity_state": (
                self.video_descriptor_persistent_identity_state
            ),
            "video_descriptor_object_memory": (self.video_descriptor_object_memory),
            "video_descriptor_object_memory_scale": (
                self.video_descriptor_object_memory_scale
            ),
            "video_descriptor_object_memory_query_gate": (
                self.video_descriptor_object_memory_query_gate
            ),
            "video_descriptor_object_memory_reliability_gate": (
                self.video_descriptor_object_memory_reliability_gate
            ),
            "video_descriptor_object_memory_contrast_visibility": (
                self.video_descriptor_object_memory_contrast_visibility
            ),
            "video_descriptor_object_memory_evidence_routing": (
                self.video_descriptor_object_memory_evidence_routing
            ),
            "video_descriptor_object_memory_evidence_routing_margin": (
                self.video_descriptor_object_memory_evidence_routing_margin
            ),
            "video_descriptor_object_memory_contrast_readout": (
                self.video_descriptor_object_memory_contrast_readout
            ),
            "video_descriptor_object_memory_temporal_relative_visibility": (
                self.video_descriptor_object_memory_temporal_relative_visibility
            ),
            "video_descriptor_object_memory_temporal_relative_readout": (
                self.video_descriptor_object_memory_temporal_relative_readout
            ),
            "video_isolated_identity_descriptors": (
                self.video_isolated_identity_descriptors
            ),
            "video_query_conditioned_head": (self.video_query_conditioned_head),
            "video_camera_robustness_adapter": (
                self.video_camera_robustness_adapter
            ),
            "video_camera_robustness_nonlinear_gate": (
                self.video_camera_robustness_nonlinear_gate
            ),
            "video_camera_pose_dim": self.video_camera_pose_dim,
            "video_spatial_relation_classes": self.video_spatial_relation_classes,
            "video_action_dim": self.video_action_dim,
            "video_egomotion_classes": self.video_egomotion_classes,
            "video_egomotion_validity_head": self.video_egomotion_validity_head,
            "video_egomotion_evidence_gate": self.video_egomotion_evidence_gate,
            "video_egomotion_minimum_motion_evidence": (
                self.video_egomotion_minimum_motion_evidence
            ),
            "video_uses_visual_semantic_encoder": (
                self.video_uses_visual_semantic_encoder
            ),
            "video_visual_semantic_scale": self.video_visual_semantic_scale,
            "video_explicit_temporal_delta": (self.video_explicit_temporal_delta),
            "video_explicit_temporal_delta_scale": (
                self.video_explicit_temporal_delta_scale
            ),
            "video_separate_temporal_delta_projection": (
                self.video_separate_temporal_delta_projection
            ),
            "long_video_world_accumulator": (self.long_video_world_accumulator),
            "long_video_transition_features": (self.long_video_transition_features),
            "text_only_bridge_adapter": self.text_only_bridge_adapter,
            "text_only_output_adapter": self.text_only_output_adapter,
            "text_only_cross_memory_adapter": (self.text_only_cross_memory_adapter),
            "text_only_hidden_cross_memory_adapter": (
                self.text_only_hidden_cross_memory_adapter
            ),
            "text_answerability_head": self.text_answerability_head,
            "text_answerability_mode": self.text_answerability_mode,
            "text_answerability_classes": self.text_answerability_classes,
            "text_epistemic_memory_adapter": self.text_epistemic_memory_adapter,
            "text_epistemic_memory_slots": self.text_epistemic_memory_slots,
            "text_epistemic_output_rank": self.text_epistemic_output_rank,
            "text_epistemic_supported_class": self.text_epistemic_supported_class,
            "text_epistemic_output_threshold": (self.text_epistemic_output_threshold),
            "text_answerability_fallback_bytes": list(
                self.text_answerability_fallback_bytes
            ),
            "text_answerability_threshold": self.text_answerability_threshold,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "MosaicUnifiedConfig":
        values = dict(values)
        schema = values.pop("schema_version", None)
        if schema != "mosaic-unified-config-v0":
            raise ValueError(f"unsupported unified config schema: {schema}")
        if "text_answerability_fallback_bytes" in values:
            values["text_answerability_fallback_bytes"] = tuple(
                values["text_answerability_fallback_bytes"]
            )
        return cls(
            text=MosaicTextConfig.from_dict(dict(values.pop("text"))),
            omni=MosaicOmniConfig(**dict(values.pop("omni"))),
            **values,
        )


@dataclass(frozen=True)
class SurfaceResidualRef:
    modality: str
    storage_key: str
    shape: tuple[int, ...]
    dirty_regions: tuple[str, ...] = ()

    def mark_dirty(self, *regions: str) -> "SurfaceResidualRef":
        combined = tuple(dict.fromkeys((*self.dirty_regions, *regions)))
        return replace(self, dirty_regions=combined)


@dataclass(frozen=True)
class WorldState:
    semantic_slots: torch.Tensor
    active_mask: torch.Tensor
    dirty_mask: torch.Tensor
    source: str
    surface_refs: tuple[SurfaceResidualRef, ...] = ()

    def validate(self, config: MosaicOmniConfig) -> None:
        if self.semantic_slots.ndim != 3:
            raise ValueError("semantic_slots must have shape [batch, slots, dim]")
        expected = (
            self.semantic_slots.shape[0],
            config.world_slots,
            config.world_dim,
        )
        if tuple(self.semantic_slots.shape) != expected:
            raise ValueError(
                f"expected semantic slot shape {expected}, "
                f"got {tuple(self.semantic_slots.shape)}"
            )
        mask_shape = (self.semantic_slots.shape[0], config.world_slots)
        if tuple(self.active_mask.shape) != mask_shape:
            raise ValueError("active_mask must have shape [batch, slots]")
        if tuple(self.dirty_mask.shape) != mask_shape:
            raise ValueError("dirty_mask must have shape [batch, slots]")
        if self.active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be boolean")
        if self.dirty_mask.dtype != torch.bool:
            raise ValueError("dirty_mask must be boolean")


class ModalToWorldAdapter(nn.Module):
    def __init__(self, source_dim: int, config: MosaicOmniConfig) -> None:
        super().__init__()
        if source_dim <= 0:
            raise ValueError("source_dim must be positive")
        self.config = config
        self.source_projection = nn.Sequential(
            nn.LayerNorm(source_dim),
            nn.Linear(source_dim, config.world_dim),
        )
        self.world_queries = nn.Parameter(
            torch.empty(config.world_slots, config.world_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            config.world_dim,
            config.attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(config.world_dim)
        nn.init.normal_(self.world_queries, mean=0.0, std=0.02)

    def forward(
        self,
        source_states: torch.Tensor,
        *,
        source_mask: torch.Tensor | None = None,
        source: str,
    ) -> WorldState:
        if source_states.ndim != 3:
            raise ValueError("source_states must have shape [batch, tokens, dim]")
        if source_mask is None:
            source_mask = torch.ones(
                source_states.shape[:2],
                dtype=torch.bool,
                device=source_states.device,
            )
        if tuple(source_mask.shape) != tuple(source_states.shape[:2]):
            raise ValueError("source_mask must have shape [batch, tokens]")
        memory = self.source_projection(source_states)
        queries = self.world_queries.unsqueeze(0).expand(
            source_states.shape[0],
            -1,
            -1,
        )
        slots, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        slots = self.output_norm(slots + queries)
        mask = torch.ones(
            slots.shape[:2],
            dtype=torch.bool,
            device=slots.device,
        )
        state = WorldState(
            semantic_slots=slots,
            active_mask=mask,
            dirty_mask=torch.zeros_like(mask),
            source=source,
        )
        state.validate(self.config)
        return state


@dataclass(frozen=True)
class MosaicUnifiedOutput:
    text: MosaicTextOutput
    world_state: WorldState
    modalities: tuple[str, ...]
    video_order_logits: torch.Tensor | None = None
    video_object_evidence_weights: torch.Tensor | None = None
    video_object_attention: torch.Tensor | None = None
    video_object_trajectory_weights: torch.Tensor | None = None
    video_object_pair_trajectory_weights: torch.Tensor | None = None
    video_descriptor_trajectory_attention: torch.Tensor | None = None
    video_descriptor_visibility_logits: torch.Tensor | None = None
    video_descriptor_memory_margin: torch.Tensor | None = None
    video_descriptor_memory_reliability_logits: torch.Tensor | None = None
    video_camera_robustness_gate: torch.Tensor | None = None
    video_spatial_relation_logits: torch.Tensor | None = None
    video_egomotion_logits: torch.Tensor | None = None
    video_egomotion_validity_logits: torch.Tensor | None = None
    video_egomotion_motion_evidence: torch.Tensor | None = None
    video_egomotion_sufficient_mask: torch.Tensor | None = None
    audio_temporal_logits: torch.Tensor | None = None
    video_embedding: torch.Tensor | None = None
    audio_embedding: torch.Tensor | None = None
    visual_embedding: torch.Tensor | None = None
    text_retrieval_embedding: torch.Tensor | None = None
    video_teacher_embedding: torch.Tensor | None = None
    audio_teacher_embedding: torch.Tensor | None = None
    audio_teacher_temporal_states: torch.Tensor | None = None
    audio_world_teacher_embedding: torch.Tensor | None = None
    audio_ctc_logits: torch.Tensor | None = None
    audio_grapheme_ctc_logits: torch.Tensor | None = None
    audio_text_retrieval_embedding: torch.Tensor | None = None
    text_audio_retrieval_embedding: torch.Tensor | None = None
    cross_modal_evidence_logits: torch.Tensor | None = None
    cross_modal_evidence_delta: torch.Tensor | None = None
    visual_text_retrieval_embedding: torch.Tensor | None = None
    text_visual_retrieval_embedding: torch.Tensor | None = None
    answerability_logits: torch.Tensor | None = None
    answerability_loss: torch.Tensor | None = None
    explicit_relation_logits: torch.Tensor | None = None
    explicit_object_attention: torch.Tensor | None = None

    @property
    def logits(self) -> torch.Tensor:
        return self.text.logits

    @property
    def loss(self) -> torch.Tensor | None:
        return self.text.loss


@dataclass(frozen=True)
class NarrativeContinuityOutput:
    score: torch.Tensor
    world_delta: torch.Tensor


@dataclass(frozen=True)
class LongVideoWorldOutput:
    world_state: WorldState
    order_logits: torch.Tensor
    checkpoint_states: torch.Tensor


class LongVideoWorldAccumulator(nn.Module):
    """Reuse one slot-wise cell while a bounded World State watches clips."""

    def __init__(
        self,
        config: MosaicOmniConfig,
        *,
        transition_features: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.position = nn.Linear(2, config.world_dim, bias=False)
        self.transition = (
            nn.Linear(config.world_dim * 2, config.world_dim)
            if transition_features
            else None
        )
        self.cell = nn.GRUCell(config.world_dim, config.world_dim)
        self.norm = nn.LayerNorm(config.world_dim)
        self.order_head = nn.Sequential(
            nn.LayerNorm(config.world_dim),
            nn.Linear(config.world_dim, 2),
        )

    def forward(
        self,
        clip_world_states: torch.Tensor,
        clip_mask: torch.Tensor,
        *,
        initial_state: WorldState | None = None,
    ) -> LongVideoWorldOutput:
        if clip_world_states.ndim != 4:
            raise ValueError("clip_world_states must be [batch, clips, slots, dim]")
        batch, clips, slots, dimension = clip_world_states.shape
        expected = (batch, clips)
        if tuple(clip_mask.shape) != expected or clip_mask.dtype != torch.bool:
            raise ValueError("clip_mask must be boolean [batch, clips]")
        if slots != self.config.world_slots or dimension != self.config.world_dim:
            raise ValueError("clip World State shape does not match the core")
        if not bool(clip_mask.any(dim=1).all()):
            raise ValueError("every sequence needs at least one observed clip")

        if initial_state is None:
            hidden = torch.zeros(
                (batch, slots, dimension),
                dtype=clip_world_states.dtype,
                device=clip_world_states.device,
            )
            active_mask = torch.zeros(
                (batch, slots),
                dtype=torch.bool,
                device=clip_world_states.device,
            )
        else:
            initial_state.validate(self.config)
            if initial_state.semantic_slots.shape[0] != batch:
                raise ValueError("initial World State batch does not match")
            hidden = initial_state.semantic_slots
            active_mask = initial_state.active_mask.clone()

        denominators = clip_mask.sum(dim=1).sub(1).clamp_min(1)
        observed = torch.zeros(
            (batch,),
            dtype=torch.long,
            device=clip_world_states.device,
        )
        previous_observation = torch.zeros_like(hidden)
        checkpoints = []
        for index in range(clips):
            present = clip_mask[:, index]
            normalized = observed.to(clip_world_states.dtype) / denominators
            delta = present.to(clip_world_states.dtype) / denominators
            position = self.position(
                torch.stack((normalized, delta), dim=-1)
            ).unsqueeze(1)
            observation = clip_world_states[:, index]
            cell_input = observation
            if self.transition is not None:
                cell_input = self.transition(
                    torch.cat(
                        (
                            observation,
                            observation - previous_observation,
                        ),
                        dim=-1,
                    )
                )
            candidate = self.cell(
                (cell_input + position).reshape(
                    batch * slots,
                    dimension,
                ),
                hidden.reshape(batch * slots, dimension),
            ).reshape(batch, slots, dimension)
            hidden = torch.where(
                present.view(batch, 1, 1),
                candidate,
                hidden,
            )
            previous_observation = torch.where(
                present.view(batch, 1, 1),
                observation,
                previous_observation,
            )
            active_mask |= present.view(batch, 1)
            observed += present
            checkpoints.append(self.norm(hidden))

        semantic_slots = self.norm(hidden)
        dirty_mask = torch.zeros_like(active_mask)
        world_state = WorldState(
            semantic_slots=semantic_slots,
            active_mask=active_mask,
            dirty_mask=dirty_mask,
            source="unified:long-video",
        )
        world_state.validate(self.config)
        return LongVideoWorldOutput(
            world_state=world_state,
            order_logits=self.order_head(semantic_slots[:, -1]),
            checkpoint_states=torch.stack(checkpoints, dim=1),
        )

    @property
    def loss(self) -> torch.Tensor | None:
        return self.text.loss


class VisualTeacherSlotBridge(nn.Module):
    """Add a bounded low-rank image residual to the existing World slots."""

    def __init__(self, world_dim: int, rank: int) -> None:
        super().__init__()
        if world_dim <= 0 or rank <= 0 or rank > world_dim:
            raise ValueError("visual teacher bridge dimensions are invalid")
        self.world_dim = world_dim
        self.rank = rank
        self.slot_norm = nn.LayerNorm(world_dim)
        self.patch_norm = nn.LayerNorm(world_dim)
        self.query = nn.Linear(world_dim, rank, bias=False)
        self.key = nn.Linear(world_dim, rank, bias=False)
        self.value_down = nn.Linear(world_dim, rank)
        self.value_up = nn.Linear(rank, world_dim)
        nn.init.zeros_(self.value_up.weight)
        nn.init.zeros_(self.value_up.bias)

    def _position(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.world_dim % 4:
            raise ValueError("world dimension must be divisible by four")
        quarter = self.world_dim // 4
        denominator = max(1, quarter - 1)
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(quarter, device=device, dtype=torch.float32)
            / denominator
        )
        vertical = torch.linspace(
            -1.0,
            1.0,
            height,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        horizontal = torch.linspace(
            -1.0,
            1.0,
            width,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        vertical = vertical * frequencies.unsqueeze(0)
        horizontal = horizontal * frequencies.unsqueeze(0)
        y = torch.cat((vertical.sin(), vertical.cos()), dim=-1)
        x = torch.cat((horizontal.sin(), horizontal.cos()), dim=-1)
        return (
            torch.cat(
                (
                    y[:, None, :].expand(height, width, -1),
                    x[None, :, :].expand(height, width, -1),
                ),
                dim=-1,
            )
            .reshape(height * width, self.world_dim)
            .to(dtype=dtype)
        )

    def forward(
        self,
        world_slots: torch.Tensor,
        image_patches: torch.Tensor,
        *,
        patch_height: int,
        patch_width: int,
    ) -> torch.Tensor:
        if world_slots.ndim != 3 or image_patches.ndim != 3:
            raise ValueError("visual teacher bridge inputs must be sequences")
        if image_patches.shape[1] != patch_height * patch_width:
            raise ValueError("visual patch grid does not match its sequence")
        position = self._position(
            patch_height,
            patch_width,
            device=image_patches.device,
            dtype=image_patches.dtype,
        )
        patches = self.patch_norm(image_patches + position.unsqueeze(0))
        scores = self.query(self.slot_norm(world_slots)) @ self.key(patches).transpose(
            1, 2
        )
        weights = torch.softmax(scores / math.sqrt(self.rank), dim=-1)
        attended = weights @ self.value_down(patches)
        return 0.25 * torch.tanh(self.value_up(attended))


class ExplicitObjectRelationGrounder(nn.Module):
    """Bind two query descriptors to image regions and derive one relation."""

    def __init__(self, world_dim: int, rank: int) -> None:
        super().__init__()
        if world_dim <= 0 or rank <= 0 or rank > world_dim:
            raise ValueError("object relation dimensions are invalid")
        self.world_dim = world_dim
        self.rank = rank
        self.descriptor_norm = nn.LayerNorm(world_dim)
        self.patch_norm = nn.LayerNorm(world_dim)
        self.query = nn.Linear(world_dim, rank, bias=False)
        self.key = nn.Linear(world_dim, rank, bias=False)
        self.value = nn.Linear(world_dim, rank)
        self.object_up = nn.Linear(rank + 4, world_dim)
        relation_rank = min(world_dim, rank * 2)
        self.relation_norm = nn.LayerNorm(world_dim * 4)
        self.relation_down = nn.Linear(
            world_dim * 4,
            relation_rank,
        )
        self.relation_up = nn.Linear(relation_rank, world_dim)

    @staticmethod
    def _coordinates(
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        vertical = torch.linspace(
            -1.0,
            1.0,
            height,
            device=device,
            dtype=dtype,
        )
        horizontal = torch.linspace(
            -1.0,
            1.0,
            width,
            device=device,
            dtype=dtype,
        )
        y, x = torch.meshgrid(vertical, horizontal, indexing="ij")
        return torch.stack((x, y, x.square(), y.square()), dim=-1).reshape(
            height * width,
            4,
        )

    def _bind(
        self,
        descriptor: torch.Tensor,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(self.descriptor_norm(descriptor)).unsqueeze(1)
        scores = query @ self.key(patches).transpose(1, 2)
        weights = torch.softmax(scores / math.sqrt(self.rank), dim=-1)
        appearance = (weights @ self.value(patches)).squeeze(1)
        moments = (weights @ coordinates.unsqueeze(0)).squeeze(1)
        slot = self.object_up(torch.cat((appearance, moments), dim=-1))
        return slot, weights.squeeze(1)

    def forward(
        self,
        image_patches: torch.Tensor,
        subject_descriptor: torch.Tensor,
        object_descriptor: torch.Tensor,
        *,
        patch_height: int,
        patch_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_patches.ndim != 3:
            raise ValueError("image patches must have shape [batch, patches, dim]")
        expected_descriptor = (image_patches.shape[0], self.world_dim)
        if (
            subject_descriptor.shape != expected_descriptor
            or object_descriptor.shape != expected_descriptor
        ):
            raise ValueError("object descriptors must match batch and world dim")
        if (
            image_patches.shape[1] != patch_height * patch_width
            or image_patches.shape[2] != self.world_dim
        ):
            raise ValueError("image patch grid does not match the grounder")
        patches = self.patch_norm(image_patches)
        coordinates = self._coordinates(
            patch_height,
            patch_width,
            device=patches.device,
            dtype=patches.dtype,
        )
        subject, subject_attention = self._bind(
            subject_descriptor,
            patches,
            coordinates,
        )
        object_, object_attention = self._bind(
            object_descriptor,
            patches,
            coordinates,
        )
        relation_input = torch.cat(
            (subject, object_, subject - object_, subject * object_),
            dim=-1,
        )
        relation = self.relation_up(
            torch.nn.functional.gelu(
                self.relation_down(self.relation_norm(relation_input))
            )
        )
        slots = torch.stack((subject, object_, relation), dim=1)
        attention = torch.stack(
            (subject_attention, object_attention),
            dim=1,
        )
        return slots, attention


class ExplicitRelationHead(nn.Module):
    """Decode one ordered Relation Slot without a separate sidecar."""

    def __init__(self, dimension: int, classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, classes)

    def forward(self, relation_slot: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(relation_slot))


class LearnedVideoObjectTracker(nn.Module):
    """Track learned object queries through a sequence of spatial token sets."""

    def __init__(
        self,
        slots: int,
        dim: int,
        *,
        spatial_coordinates: bool = False,
    ) -> None:
        super().__init__()
        if slots <= 0 or dim <= 0:
            raise ValueError("learned object tracker dimensions must be positive")
        self.slots = slots
        self.dim = dim
        self.queries = nn.Parameter(torch.empty(slots, dim))
        nn.init.normal_(self.queries, std=dim**-0.5)
        self.query_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.position_projection = (
            nn.Linear(2, dim, bias=False) if spatial_coordinates else None
        )
        if self.position_projection is not None:
            nn.init.zeros_(self.position_projection.weight)
            self.position_projection.weight.data[0, 0] = 1.0
            self.position_projection.weight.data[1, 1] = 1.0

    def track_with_attention(
        self,
        feature_map: torch.Tensor,
        *,
        batch: int,
        frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            feature_map.ndim != 4
            or feature_map.shape[0] != batch * frames
            or feature_map.shape[1] != self.dim
        ):
            raise ValueError(
                "object feature map must be [batch*frames,dim,height,width]"
            )
        tokens = (
            feature_map.flatten(2)
            .transpose(1, 2)
            .reshape(
                batch,
                frames,
                -1,
                self.dim,
            )
        )
        if self.position_projection is not None:
            height, width = feature_map.shape[2:]
            y, x = torch.meshgrid(
                torch.linspace(
                    -1.0,
                    1.0,
                    height,
                    device=feature_map.device,
                    dtype=feature_map.dtype,
                ),
                torch.linspace(
                    -1.0,
                    1.0,
                    width,
                    device=feature_map.device,
                    dtype=feature_map.dtype,
                ),
                indexing="ij",
            )
            position = self.position_projection(
                torch.stack((x, y), dim=-1).reshape(-1, 2)
            )
            tokens = tokens + position.view(1, 1, -1, self.dim)
        learned = self.queries.unsqueeze(0).expand(batch, -1, -1)
        state = torch.zeros_like(learned)
        tracks = []
        attentions = []
        for frame_index in range(frames):
            query = self.query_norm(state + learned)
            token = tokens[:, frame_index]
            logits = torch.einsum(
                "bsd,bpd->bsp",
                query,
                self.token_norm(token),
            ) / math.sqrt(self.dim)
            attention = torch.softmax(logits, dim=1).clamp_min(1e-6)
            attention = attention / attention.sum(dim=-1, keepdim=True)
            state = torch.bmm(attention, token)
            tracks.append(state)
            attentions.append(attention)
        return (
            torch.stack(tracks, dim=2).reshape(
                batch * self.slots,
                frames,
                self.dim,
            ),
            torch.stack(attentions, dim=2).reshape(
                batch,
                self.slots,
                frames,
                *feature_map.shape[2:],
            ),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        *,
        batch: int,
        frames: int,
    ) -> torch.Tensor:
        tracks, _ = self.track_with_attention(
            feature_map,
            batch=batch,
            frames=frames,
        )
        return tracks


class QueryConditionedObjectTrajectoryBinding(nn.Module):
    """Select an identity slot with text, then expose that slot's trajectory."""

    trajectory_width = 12

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.identity_norm = nn.LayerNorm(dim)
        self.trajectory_norm = nn.LayerNorm(dim * 3 + self.trajectory_width)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim * 3 + self.trajectory_width, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)
        nn.init.zeros_(self.value.weight)
        self.value.weight.data[:, dim : dim * 2] = torch.eye(dim)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        identity: torch.Tensor,
        event: torch.Tensor,
        delta: torch.Tensor,
        trajectory: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if identity.shape != event.shape or identity.shape != delta.shape:
            raise ValueError("identity, event, and delta slots must match")
        if identity.ndim != 3 or identity.shape[-1] != self.dim:
            raise ValueError("object trajectory slots must be [B,S,D]")
        if trajectory.shape != (*identity.shape[:2], self.trajectory_width):
            raise ValueError("object trajectory geometry must be [B,S,12]")
        if query.shape != (identity.shape[0], self.dim):
            raise ValueError("object trajectory query must be [B,D]")
        weights = torch.softmax(
            torch.einsum(
                "bd,bsd->bs",
                self.query(query),
                self.key(self.identity_norm(identity)),
            )
            / math.sqrt(self.dim),
            dim=-1,
        )
        values = torch.nn.functional.gelu(
            self.value(
                self.trajectory_norm(
                    torch.cat((identity, event, delta, trajectory), dim=-1)
                )
            )
        )
        return self.output((values * weights.unsqueeze(-1)).sum(dim=1)), weights


class QueryConditionedObjectPairTrajectoryBinding(nn.Module):
    """Select A/B object trajectories from role-specific text descriptors."""

    trajectory_width = 12

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.identity_norm = nn.LayerNorm(dim)
        self.trajectory_norm = nn.LayerNorm(dim * 3 + self.trajectory_width)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim * 3 + self.trajectory_width, dim, bias=False)
        self.output = nn.Linear(dim * 4, dim, bias=False)
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)
        nn.init.zeros_(self.value.weight)
        self.value.weight.data[:, dim : dim * 2] = torch.eye(dim)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        identity: torch.Tensor,
        event: torch.Tensor,
        delta: torch.Tensor,
        trajectory: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if identity.shape != event.shape or identity.shape != delta.shape:
            raise ValueError("identity, event, and delta slots must match")
        if identity.ndim != 3 or identity.shape[-1] != self.dim:
            raise ValueError("object trajectory slots must be [B,S,D]")
        if trajectory.shape != (*identity.shape[:2], self.trajectory_width):
            raise ValueError("object trajectory geometry must be [B,S,12]")
        if queries.shape != (identity.shape[0], 2, self.dim):
            raise ValueError("object pair queries must be [B,2,D]")
        weights = torch.softmax(
            torch.einsum(
                "bqd,bsd->bqs",
                self.query(queries),
                self.key(self.identity_norm(identity)),
            )
            / math.sqrt(self.dim),
            dim=-1,
        )
        values = torch.nn.functional.gelu(
            self.value(
                self.trajectory_norm(
                    torch.cat((identity, event, delta, trajectory), dim=-1)
                )
            )
        )
        selected = torch.einsum("bqs,bsd->bqd", weights, values)
        first, second = selected.unbind(dim=1)
        decision = self.output(
            torch.cat(
                (first, second, first - second, first * second),
                dim=-1,
            )
        )
        return decision, weights


class DescriptorConditionedDenseTrajectoryBinding(nn.Module):
    """Track named A/B roles directly over dense per-frame visual features."""

    trajectory_width = 12
    object_memory_width = 19
    object_memory_visibility_width = 2
    object_memory_contrast_visibility_width = 4
    object_memory_contrast_summary_width = 8

    def __init__(
        self,
        dim: int,
        *,
        pair_centered_queries: bool = False,
        persistent_identity_state: bool = False,
        object_memory: bool = False,
        object_memory_scale: float = 1.0,
        object_memory_query_gate: bool = False,
        object_memory_reliability_gate: bool = False,
        object_memory_contrast_visibility: bool = False,
        object_memory_contrast_readout: bool = False,
        object_memory_temporal_relative_visibility: bool = False,
        object_memory_temporal_relative_readout: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pair_centered_queries = pair_centered_queries
        self.persistent_identity_state = persistent_identity_state
        self.object_memory = object_memory
        self.object_memory_scale = object_memory_scale
        self.object_memory_query_gate = object_memory_query_gate
        self.object_memory_reliability_gate = object_memory_reliability_gate
        self.object_memory_contrast_visibility = object_memory_contrast_visibility
        self.object_memory_contrast_readout = object_memory_contrast_readout
        self.object_memory_temporal_relative_visibility = (
            object_memory_temporal_relative_visibility
        )
        self.object_memory_temporal_relative_readout = (
            object_memory_temporal_relative_readout
        )
        if object_memory_contrast_readout and not (
            object_memory and object_memory_contrast_visibility
        ):
            raise ValueError("contrast readout requires contrast object memory")
        if object_memory_temporal_relative_readout and not object_memory:
            raise ValueError("temporal relative readout requires object memory")
        if (
            object_memory_temporal_relative_readout
            and object_memory_temporal_relative_visibility
        ):
            raise ValueError(
                "temporal relative readout cannot replace the parent visibility path"
            )
        self.feature_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.trajectory_norm = nn.LayerNorm(dim * 3 + self.trajectory_width)
        self.value = nn.Linear(dim * 3 + self.trajectory_width, dim, bias=False)
        self.output = nn.Linear(dim * 4, dim, bias=False)
        visibility_width = (
            self.object_memory_contrast_visibility_width
            if object_memory_contrast_visibility
            else self.object_memory_visibility_width
        )
        self.memory_visibility = (
            nn.Linear(visibility_width, 1) if object_memory else None
        )
        self.memory_output = (
            nn.Linear(self.object_memory_width, dim, bias=False)
            if object_memory
            else None
        )
        self.memory_contrast_output = (
            nn.Linear(self.object_memory_contrast_summary_width, dim, bias=False)
            if object_memory_contrast_readout
            else None
        )
        self.memory_temporal_relative_output = (
            nn.Linear(self.object_memory_width, dim, bias=False)
            if object_memory_temporal_relative_readout
            else None
        )
        self.memory_gate = (
            nn.Linear(dim, 1) if object_memory and object_memory_query_gate else None
        )
        self.memory_reliability_gate = (
            nn.Linear(self.object_memory_width, 1)
            if object_memory and object_memory_reliability_gate
            else None
        )
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)
        nn.init.zeros_(self.value.weight)
        self.value.weight.data[:, dim : dim * 2] = torch.eye(dim)
        nn.init.zeros_(self.output.weight)
        if self.memory_visibility is not None and self.memory_output is not None:
            nn.init.zeros_(self.memory_visibility.weight)
            nn.init.zeros_(self.memory_visibility.bias)
            nn.init.zeros_(self.memory_output.weight)
        if self.memory_contrast_output is not None:
            nn.init.zeros_(self.memory_contrast_output.weight)
        if self.memory_temporal_relative_output is not None:
            nn.init.zeros_(self.memory_temporal_relative_output.weight)
        if self.memory_gate is not None:
            nn.init.zeros_(self.memory_gate.weight)
            nn.init.zeros_(self.memory_gate.bias)
        if self.memory_reliability_gate is not None:
            nn.init.zeros_(self.memory_reliability_gate.weight)
            nn.init.zeros_(self.memory_reliability_gate.bias)

    def forward(
        self,
        features: torch.Tensor,
        queries: torch.Tensor,
        condition: torch.Tensor | None = None,
        memory_scale: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if features.ndim != 5 or features.shape[2] != self.dim:
            raise ValueError("dense video features must be [B,F,D,H,W]")
        if queries.shape != (features.shape[0], 2, self.dim):
            raise ValueError("descriptor queries must be [B,2,D]")
        if memory_scale is not None and memory_scale.shape != (features.shape[0],):
            raise ValueError("descriptor memory scale must be [B]")
        tokens = features.permute(0, 1, 3, 4, 2)
        if self.pair_centered_queries:
            queries = queries - queries.mean(dim=1, keepdim=True)
            queries = torch.nn.functional.layer_norm(queries, (self.dim,))
        projected_queries = self.query(queries)
        projected_keys = self.key(self.feature_norm(tokens))
        logits = torch.einsum(
            "bqd,bfhwd->bqfhw",
            projected_queries,
            projected_keys,
        ) / math.sqrt(self.dim)
        attention = torch.softmax(logits.flatten(-2), dim=-1).reshape_as(logits)
        visibility_logits = None
        memory_margin = None
        memory_decision = None
        contrast_memory_summary = None
        memory_reliability_logits = None
        if self.memory_visibility is not None and self.memory_output is not None:
            peak = attention.amax(dim=(-2, -1))
            entropy = -(attention * attention.clamp_min(1e-6).log()).sum(
                dim=(-2, -1)
            ) / math.log(max(2, attention.shape[-2] * attention.shape[-1]))
            visibility_features = [peak, 1.0 - entropy]
            if self.object_memory_contrast_visibility:
                cosine_logits = torch.einsum(
                    "bqd,bfhwd->bqfhw",
                    torch.nn.functional.normalize(projected_queries, dim=-1),
                    torch.nn.functional.normalize(projected_keys, dim=-1),
                )
                cosine_scores = cosine_logits.flatten(-2)
                cosine_peak = cosine_scores.amax(dim=-1)
                if cosine_scores.shape[-1] > 1:
                    top_two = cosine_scores.topk(2, dim=-1).values
                    cosine_margin = top_two[..., 0] - top_two[..., 1]
                else:
                    cosine_margin = torch.zeros_like(cosine_peak)
                visibility_features.extend((cosine_peak, cosine_margin))
                if self.memory_contrast_output is not None:
                    contrast_memory_summary = _contrast_memory_summary(
                        cosine_peak,
                        cosine_margin,
                    )
            visibility_logits = self.memory_visibility(
                torch.stack(visibility_features, dim=-1)
            ).squeeze(-1)
            memory_features, memory_margin = _descriptor_object_memory(
                attention,
                visibility_logits,
                temporal_relative_visibility=(
                    self.object_memory_temporal_relative_visibility
                ),
            )
            memory_decision = self.memory_output(memory_features)
            if contrast_memory_summary is not None:
                memory_decision = memory_decision + self.memory_contrast_output(
                    contrast_memory_summary
                )
            if self.memory_reliability_gate is not None:
                memory_reliability_logits = self.memory_reliability_gate(
                    memory_features
                ).squeeze(-1)
                memory_decision = memory_decision * (
                    2.0 * torch.sigmoid(memory_reliability_logits).unsqueeze(-1)
                )
            if self.memory_gate is not None:
                if condition is None or condition.shape != (
                    features.shape[0],
                    self.dim,
                ):
                    raise ValueError(
                        "descriptor memory query gate requires [B,D] condition"
                    )
                memory_decision = memory_decision * (
                    2.0 * torch.sigmoid(self.memory_gate(condition))
                )
            if self.memory_temporal_relative_output is not None:
                temporal_relative_features, memory_margin = _descriptor_object_memory(
                    attention,
                    visibility_logits,
                    temporal_relative_visibility=True,
                )
                temporal_relative_decision = self.memory_temporal_relative_output(
                    temporal_relative_features
                )
                if memory_scale is not None:
                    temporal_relative_decision = temporal_relative_decision * (
                        memory_scale.unsqueeze(-1)
                    )
                memory_decision = memory_decision + temporal_relative_decision
                memory_scale = None
        selected = torch.einsum("bqfhw,bfhwd->bqfd", attention, tokens)
        match_logits = logits.amax(dim=(-2, -1))
        if self.persistent_identity_state:
            selected = _confidence_gated_sequence(
                selected,
                match_logits,
            )
        times = torch.linspace(
            -1.0,
            1.0,
            selected.shape[2],
            device=selected.device,
            dtype=selected.dtype,
        ).view(1, 1, -1, 1)
        identity = (selected[:, :, 0] + selected[:, :, -1]) * 0.5
        event = ((selected - selected.mean(dim=2, keepdim=True)) * times).mean(dim=2)
        delta = selected[:, :, -1] - selected[:, :, 0]
        geometry = _object_attention_trajectory(
            attention,
            match_logits if self.persistent_identity_state else None,
        )
        values = torch.nn.functional.gelu(
            self.value(
                self.trajectory_norm(
                    torch.cat((identity, event, delta, geometry), dim=-1)
                )
            )
        )
        first, second = values.unbind(dim=1)
        decision = self.output(
            torch.cat(
                (first, second, first - second, first * second),
                dim=-1,
            )
        )
        if memory_decision is not None:
            scaled_memory = self.object_memory_scale * memory_decision
            if memory_scale is not None:
                scaled_memory = scaled_memory * memory_scale.unsqueeze(-1)
            decision = decision + scaled_memory
        return (
            decision,
            attention,
            visibility_logits,
            memory_margin,
            memory_reliability_logits,
        )


class VideoEgomotionReasoner(nn.Module):
    """Read stereo endpoint changes directly, without oracle camera poses."""

    def __init__(self, world_dim: int) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(64, world_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] < 4 or video.shape[1] % 2:
            raise ValueError("egomotion video must contain paired stereo endpoints")
        left_before, right_before = video[:, 0], video[:, 1]
        left_after, right_after = video[:, -2], video[:, -1]
        features = torch.cat(
            (
                left_before,
                right_before,
                left_after,
                right_after,
                left_after - left_before,
                right_after - right_before,
                right_before - left_before,
                right_after - left_after,
            ),
            dim=1,
        )
        encoded = self.frontend(features.to(dtype=self.output.weight.dtype)).flatten(1)
        return self.output(encoded)


def _video_egomotion_validity_statistics(video: torch.Tensor) -> torch.Tensor:
    """Expose exact endpoint change magnitudes without creating persistent state."""
    if video.ndim != 5 or video.shape[1] < 4 or video.shape[1] % 2:
        raise ValueError("egomotion video must contain paired stereo endpoints")
    delta = torch.stack(
        (
            video[:, -2] - video[:, 0],
            video[:, -1] - video[:, 1],
        ),
        dim=1,
    ).abs()
    return torch.stack(
        (
            delta.mean(dim=(1, 2, 3, 4)),
            delta.amax(dim=(1, 2, 3, 4)),
        ),
        dim=-1,
    )


class VideoSpatialGeometryReasoner(nn.Module):
    """Fuse per-view named-object geometry with camera pose, order-invariantly."""

    output_dim = 192

    def __init__(self, camera_pose_dim: int) -> None:
        super().__init__()
        if camera_pose_dim <= 0:
            raise ValueError("camera pose dimension must be positive")
        self.camera_pose_dim = camera_pose_dim
        self.frame = nn.Sequential(
            nn.LayerNorm(camera_pose_dim + 16),
            nn.Linear(camera_pose_dim + 16, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.stereo_pair = nn.Sequential(
            nn.LayerNorm(camera_pose_dim * 3 + 28),
            nn.Linear(camera_pose_dim * 3 + 28, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )

    def _stereo_summary(
        self,
        role: torch.Tensor,
        camera_pose: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, frames, _ = role.shape
        if self.camera_pose_dim < 14 or frames % 2:
            return role.new_zeros(batch, 64)
        pairs = frames // 2
        paired_role = role.reshape(batch, 2, pairs, 2, 4).permute(0, 2, 3, 1, 4)
        paired_pose = camera_pose.reshape(batch, pairs, 2, self.camera_pose_dim)
        eye_sign = paired_pose[..., 12]
        left_index = eye_sign.argmin(dim=2)
        right_index = eye_sign.argmax(dim=2)
        role_index_shape = (-1, -1, 1, 2, 4)
        pose_index_shape = (-1, -1, 1, self.camera_pose_dim)
        left_role = paired_role.gather(
            2,
            left_index[..., None, None, None].expand(*role_index_shape),
        ).squeeze(2)
        right_role = paired_role.gather(
            2,
            right_index[..., None, None, None].expand(*role_index_shape),
        ).squeeze(2)
        left_pose = paired_pose.gather(
            2,
            left_index[..., None, None].expand(*pose_index_shape),
        ).squeeze(2)
        right_pose = paired_pose.gather(
            2,
            right_index[..., None, None].expand(*pose_index_shape),
        ).squeeze(2)
        disparity = left_role - right_role
        pair = self.stereo_pair(
            torch.cat(
                (
                    left_role.flatten(-2),
                    right_role.flatten(-2),
                    disparity.flatten(-2),
                    disparity[:, :, 0] - disparity[:, :, 1],
                    left_pose,
                    right_pose,
                    right_pose - left_pose,
                ),
                dim=-1,
            )
        )
        return torch.cat((pair.mean(dim=1), pair.amax(dim=1)), dim=-1)

    def forward(
        self,
        attention: torch.Tensor,
        camera_pose: torch.Tensor,
    ) -> torch.Tensor:
        if attention.ndim != 5 or attention.shape[1] != 2:
            raise ValueError("spatial reasoning attention must be [B,2,F,H,W]")
        if camera_pose.shape != (
            attention.shape[0],
            attention.shape[2],
            self.camera_pose_dim,
        ):
            raise ValueError("camera pose must align with spatial attention frames")
        probability = attention / attention.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-6)
        height, width = attention.shape[-2:]
        x = torch.linspace(
            -1.0, 1.0, width, device=attention.device, dtype=attention.dtype
        )
        y = torch.linspace(
            -1.0, 1.0, height, device=attention.device, dtype=attention.dtype
        )
        position = torch.stack(
            (
                (probability * x.view(1, 1, 1, 1, -1)).sum(dim=(-2, -1)),
                (probability * y.view(1, 1, 1, -1, 1)).sum(dim=(-2, -1)),
            ),
            dim=-1,
        )
        peak = probability.amax(dim=(-2, -1), keepdim=False).unsqueeze(-1)
        entropy = (
            -(probability * probability.clamp_min(1e-6).log()).sum(dim=(-2, -1))
            / math.log(max(2, height * width))
        ).unsqueeze(-1)
        role = torch.cat((position, peak, entropy), dim=-1)
        first, second = role.unbind(dim=1)
        frame = self.frame(
            torch.cat(
                (
                    first,
                    second,
                    first - second,
                    first * second,
                    camera_pose.to(attention.dtype),
                ),
                dim=-1,
            )
        )
        return torch.cat(
            (
                frame.mean(dim=1),
                frame.amax(dim=1),
                self._stereo_summary(role, camera_pose.to(attention.dtype)),
            ),
            dim=-1,
        )


class TextOnlyCrossMemoryAdapter(nn.Module):
    """Small evidence reader that leaves non-text branches untouched."""

    def __init__(self, model_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.query = nn.Linear(VOCAB_SIZE, model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output = nn.Linear(model_dim, VOCAB_SIZE)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        logits: torch.Tensor,
        evidence_states: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, patches, patch_size, _ = logits.shape
        query = self.query(logits.flatten(1, 2))
        context, _ = self.cross_attention(
            query,
            evidence_states,
            evidence_states,
            key_padding_mask=~evidence_mask,
            need_weights=False,
        )
        return self.output(torch.nn.functional.gelu(context)).view(
            batch,
            patches,
            patch_size,
            VOCAB_SIZE,
        )


class TextOnlyHiddenCrossMemoryAdapter(nn.Module):
    """Read exact evidence from the decoder's contextual hidden state."""

    def __init__(self, model_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output = nn.Linear(model_dim, VOCAB_SIZE)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        decoder_states: torch.Tensor,
        evidence_states: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, patches, patch_size, _ = decoder_states.shape
        query = self.query(decoder_states.flatten(1, 2))
        context, _ = self.cross_attention(
            query,
            evidence_states,
            evidence_states,
            key_padding_mask=~evidence_mask,
            need_weights=False,
        )
        return self.output(torch.nn.functional.gelu(context)).view(
            batch,
            patches,
            patch_size,
            VOCAB_SIZE,
        )


class TextEpistemicOutputAdapter(nn.Module):
    """Low-rank logit residual that is exactly zero for supported inputs."""

    def __init__(self, model_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(model_dim, rank, bias=False)
        self.output = nn.Linear(rank, VOCAB_SIZE)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        decoder_states: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.output(torch.nn.functional.gelu(self.down(decoder_states)))
        return residual * active.to(residual.dtype)[:, None, None, None]


class TextAnswerabilityVerifier(nn.Module):
    """Compare question patches with evidence patches before generation."""

    def __init__(
        self,
        world_dim: int,
        attention_heads: int,
        *,
        contextual: bool = False,
        evidence_consistency: bool = False,
        question_body_only: bool = False,
        source_dim: int | None = None,
        extra_feature_dim: int = 0,
        output_classes: int = 2,
    ) -> None:
        super().__init__()
        self.evidence_consistency = evidence_consistency
        self.question_body_only = question_body_only
        self.input_projection = (
            nn.Sequential(
                nn.LayerNorm(source_dim),
                nn.Linear(source_dim, world_dim),
            )
            if source_dim is not None and source_dim != world_dim
            else nn.Identity()
        )
        self.extra_feature_dim = extra_feature_dim
        self.context_encoder = (
            nn.TransformerEncoderLayer(
                d_model=world_dim,
                nhead=attention_heads,
                dim_feedforward=world_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            if contextual
            else None
        )
        self.question_norm = nn.LayerNorm(world_dim)
        self.evidence_norm = nn.LayerNorm(world_dim)
        self.cross_attention = nn.MultiheadAttention(
            world_dim,
            attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(world_dim * 4),
            nn.Linear(world_dim * 4, world_dim),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(
                world_dim * (4 if evidence_consistency else 2)
                + (3 if evidence_consistency else 2)
                + extra_feature_dim
            ),
            nn.Linear(
                world_dim * (4 if evidence_consistency else 2)
                + (3 if evidence_consistency else 2)
                + extra_feature_dim,
                world_dim,
            ),
            nn.GELU(),
            nn.Linear(world_dim, output_classes),
        )

    def forward(
        self,
        question_states: torch.Tensor,
        question_mask: torch.Tensor,
        evidence_states: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_title_mask: torch.Tensor | None = None,
        evidence_body_mask: torch.Tensor | None = None,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        question_states = self.input_projection(question_states)
        evidence_states = self.input_projection(evidence_states)
        if self.context_encoder is not None:
            question_states = self.context_encoder(
                question_states
                + _sinusoidal_positions(
                    question_states.shape[1],
                    question_states.shape[2],
                    device=question_states.device,
                    dtype=question_states.dtype,
                ),
                src_key_padding_mask=~question_mask,
            )
            evidence_context_mask = evidence_mask
            if self.question_body_only:
                if evidence_body_mask is None:
                    raise ValueError("body mask is required for body-only verification")
                evidence_context_mask = evidence_body_mask.clone()
                evidence_context_mask[:, 0] |= ~evidence_context_mask.any(dim=1)
            evidence_states = self.context_encoder(
                evidence_states
                + _sinusoidal_positions(
                    evidence_states.shape[1],
                    evidence_states.shape[2],
                    device=evidence_states.device,
                    dtype=evidence_states.dtype,
                ),
                src_key_padding_mask=~evidence_context_mask,
            )
        question = self.question_norm(question_states)
        evidence = self.evidence_norm(evidence_states)
        question_evidence_mask = evidence_mask
        if self.question_body_only:
            question_evidence_mask = evidence_body_mask.clone()
            question_evidence_mask[:, 0] |= ~question_evidence_mask.any(dim=1)
        context, _ = self.cross_attention(
            question,
            evidence,
            evidence,
            key_padding_mask=~question_evidence_mask,
            need_weights=False,
        )
        question_interactions = self.token_projection(
            torch.cat(
                (
                    question,
                    context,
                    (question - context).abs(),
                    question * context,
                ),
                dim=-1,
            )
        )
        features = list(self._pool(question_interactions, question_mask))
        densities = [
            question_mask.float().mean(dim=1),
            (evidence_body_mask if self.question_body_only else evidence_mask)
            .float()
            .mean(dim=1),
        ]
        if self.evidence_consistency:
            if evidence_title_mask is None or evidence_body_mask is None:
                raise ValueError("evidence consistency masks are required")
            safe_body_mask = evidence_body_mask.clone()
            safe_body_mask[:, 0] |= ~safe_body_mask.any(dim=1)
            title_context, _ = self.cross_attention(
                evidence,
                evidence,
                evidence,
                key_padding_mask=~safe_body_mask,
                need_weights=False,
            )
            title_interactions = self.token_projection(
                torch.cat(
                    (
                        evidence,
                        title_context,
                        (evidence - title_context).abs(),
                        evidence * title_context,
                    ),
                    dim=-1,
                )
            )
            features.extend(self._pool(title_interactions, evidence_title_mask))
            densities = [
                question_mask.float().mean(dim=1),
                evidence_title_mask.float().mean(dim=1),
                evidence_body_mask.float().mean(dim=1),
            ]
        density = torch.stack(densities, dim=-1)
        if self.extra_feature_dim:
            if extra_features is None or extra_features.shape != (
                question_states.shape[0],
                self.extra_feature_dim,
            ):
                raise ValueError("answerability extra features have wrong shape")
            features.append(extra_features.to(density.dtype))
        return self.output(torch.cat((*features, density), dim=-1))

    @staticmethod
    def _pool(
        states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_mask = mask.clone()
        safe_mask[:, 0] |= ~safe_mask.any(dim=1)
        active = safe_mask.unsqueeze(-1)
        mean = (states * active.to(states.dtype)).sum(dim=1)
        mean = mean / active.sum(dim=1).clamp_min(1)
        maximum = (
            states.masked_fill(
                ~active,
                float("-inf"),
            )
            .max(dim=1)
            .values
        )
        return mean, maximum


class MosaicUnifiedForConditionalGeneration(nn.Module):
    """Native raw-modal frontends and one shared World Latent in one model."""

    def __init__(self, config: MosaicUnifiedConfig) -> None:
        super().__init__()
        self.config = config
        world_dim = config.omni.world_dim
        self.text_core = MosaicTextLM(config.text)
        self.text_to_world = nn.Linear(config.text.model_dim, world_dim)
        self.text_only_to_world = (
            nn.Linear(config.text.model_dim, world_dim)
            if config.text_only_bridge_adapter
            else None
        )
        self.vision_frontend = nn.Conv2d(
            3,
            world_dim,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
        )
        self.visual_semantic_frontend = (
            nn.Conv2d(
                3,
                world_dim,
                kernel_size=config.vision_patch_size,
                stride=config.vision_patch_size,
            )
            if (
                config.visual_semantic_encoder and config.visual_semantic_split_frontend
            )
            else None
        )
        self.visual_position = (
            nn.Linear(2, world_dim, bias=False)
            if config.visual_semantic_encoder
            else None
        )
        self.visual_semantic_cell = (
            nn.TransformerEncoderLayer(
                d_model=world_dim,
                nhead=config.omni.attention_heads,
                dim_feedforward=config.world_ffn_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            if config.visual_semantic_encoder
            else None
        )
        self.visual_semantic_norm = (
            nn.LayerNorm(world_dim) if config.visual_semantic_encoder else None
        )
        self.image_visual_adapter_down = (
            nn.Conv2d(
                3,
                config.image_visual_adapter_rank,
                kernel_size=config.vision_patch_size,
                stride=config.vision_patch_size,
                bias=False,
            )
            if config.image_visual_adapter
            else None
        )
        self.image_visual_adapter_up = (
            nn.Conv2d(
                config.image_visual_adapter_rank,
                world_dim,
                kernel_size=1,
                bias=False,
            )
            if config.image_visual_adapter
            else None
        )
        if self.image_visual_adapter_up is not None:
            nn.init.zeros_(self.image_visual_adapter_up.weight)
        self.visual_teacher_slot_bridge = (
            VisualTeacherSlotBridge(
                world_dim,
                config.visual_teacher_slot_rank,
            )
            if config.visual_teacher_slot_bridge
            else None
        )
        self.explicit_object_relation_grounder = (
            ExplicitObjectRelationGrounder(
                world_dim,
                config.explicit_object_relation_rank,
            )
            if config.explicit_object_relation_grounder
            else None
        )
        self.explicit_relation_head = (
            ExplicitRelationHead(
                world_dim,
                config.explicit_relation_classes,
            )
            if config.explicit_object_relation_grounder
            else None
        )
        self.audio_frontend = nn.Conv1d(
            1,
            world_dim,
            kernel_size=config.audio_patch_samples,
            stride=config.audio_patch_samples,
        )
        self.audio_temporal_cell = (
            nn.GRU(world_dim, world_dim, batch_first=True)
            if config.audio_temporal_encoder
            else None
        )
        self.audio_temporal_to_world = (
            nn.Linear(world_dim, world_dim) if config.audio_temporal_encoder else None
        )
        self.audio_content_temporal_cell = (
            nn.GRU(world_dim, world_dim, batch_first=True)
            if config.audio_content_encoder
            else None
        )
        self.audio_content_to_world = (
            nn.Linear(world_dim, world_dim) if config.audio_content_encoder else None
        )
        self.audio_spectral_projection = (
            nn.Linear(config.audio_spectral_n_fft // 2 + 1, world_dim)
            if config.audio_spectral_content_frontend
            else None
        )
        if config.audio_spectral_content_frontend:
            self.register_buffer(
                "audio_spectral_window",
                torch.hann_window(config.audio_spectral_n_fft),
                persistent=False,
            )
        else:
            self.audio_spectral_window = None
        self.audio_event_slot_projection = (
            nn.Linear(world_dim, world_dim)
            if config.audio_event_slot_injection
            else None
        )
        self.audio_ctc_projection = (
            nn.Linear(world_dim, VOCAB_SIZE) if config.audio_ctc_head else None
        )
        self.audio_grapheme_ctc_projection = (
            nn.Linear(
                world_dim,
                config.audio_grapheme_ctc_vocabulary_size + 1,
            )
            if config.audio_grapheme_ctc_vocabulary_size
            else None
        )
        self.audio_text_retrieval_projection = (
            nn.Linear(world_dim, world_dim, bias=False)
            if config.audio_text_retrieval_head
            else None
        )
        self.text_audio_retrieval_projection = (
            nn.Linear(world_dim, world_dim, bias=False)
            if config.audio_text_retrieval_head
            else None
        )
        self.cross_modal_text_projection = (
            nn.Linear(world_dim, config.cross_modal_evidence_rank, bias=False)
            if config.cross_modal_evidence_head
            else None
        )
        self.cross_modal_audio_projection = (
            nn.Linear(world_dim, config.cross_modal_evidence_rank, bias=False)
            if config.cross_modal_evidence_head
            else None
        )
        self.cross_modal_video_projection = (
            nn.Linear(world_dim, config.cross_modal_evidence_rank, bias=False)
            if config.cross_modal_evidence_head
            else None
        )
        self.cross_modal_text_evidence_score = (
            nn.Linear(config.cross_modal_evidence_rank, 1, bias=False)
            if config.cross_modal_evidence_head
            and config.cross_modal_text_query_pooling
            else None
        )
        self.cross_modal_text_sequence_encoder = (
            nn.GRU(
                config.cross_modal_evidence_rank,
                config.cross_modal_evidence_rank,
                batch_first=True,
            )
            if config.cross_modal_evidence_head
            and config.cross_modal_text_sequence_pooling
            else None
        )
        self.cross_modal_evidence_to_world = (
            nn.Linear(
                config.cross_modal_evidence_rank
                * (6 if config.cross_modal_evidence_direct_features else 3),
                world_dim,
            )
            if config.cross_modal_evidence_head
            else None
        )
        self.cross_modal_evidence_norm = (
            nn.BatchNorm1d(
                config.cross_modal_evidence_rank
                * (6 if config.cross_modal_evidence_direct_features else 3),
                affine=False,
                momentum=0.1,
            )
            if config.cross_modal_evidence_head
            else None
        )
        self.cross_modal_evidence_head = (
            nn.Sequential(nn.LayerNorm(world_dim), nn.Linear(world_dim, 2))
            if config.cross_modal_evidence_head
            else None
        )
        self.cross_modal_evidence_activation = (
            nn.GELU() if config.cross_modal_evidence_head else None
        )
        self.narrative_continuity_head = (
            nn.Sequential(
                nn.Linear(config.text.model_dim * 8, config.narrative_evidence_hidden_dim),
                nn.GELU(),
                nn.Linear(config.narrative_evidence_hidden_dim, 1),
            )
            if config.narrative_evidence_head
            else None
        )
        self.narrative_evidence_to_world = (
            nn.Linear(1, world_dim, bias=False)
            if config.narrative_evidence_head
            else None
        )
        if self.narrative_evidence_to_world is not None:
            nn.init.zeros_(self.narrative_evidence_to_world.weight)
        self.visual_text_retrieval_projection = (
            nn.Linear(
                world_dim,
                config.visual_text_retrieval_dim,
                bias=False,
            )
            if config.visual_text_retrieval_head
            else None
        )
        self.text_visual_retrieval_projection = (
            nn.Linear(
                world_dim,
                config.visual_text_retrieval_dim,
                bias=False,
            )
            if config.visual_text_retrieval_head
            else None
        )
        self.audio_temporal_head = (
            nn.Sequential(
                nn.LayerNorm(world_dim),
                nn.Linear(world_dim, 2),
            )
            if config.audio_temporal_binary_head
            else None
        )
        self.video_time = nn.Linear(1, world_dim, bias=False)
        self.video_temporal_mixer = nn.Conv3d(
            world_dim,
            world_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=world_dim,
            bias=False,
        )
        self.video_temporal_cell = nn.GRU(
            world_dim,
            world_dim,
            batch_first=True,
        )
        self.video_temporal_to_world = nn.Linear(world_dim, world_dim)
        self.video_temporal_delta_to_world = (
            nn.Linear(world_dim, world_dim, bias=False)
            if config.video_separate_temporal_delta_projection
            else None
        )
        if self.video_temporal_delta_to_world is not None:
            nn.init.eye_(self.video_temporal_delta_to_world.weight)
        self.video_object_temporal_cell = (
            nn.GRU(world_dim, world_dim, batch_first=True)
            if config.video_object_temporal_encoder
            else None
        )
        object_patch_size = max(2, config.vision_patch_size // 2)
        self.video_object_frontend = (
            nn.Conv2d(
                3,
                world_dim,
                kernel_size=object_patch_size,
                stride=object_patch_size,
            )
            if config.video_object_temporal_encoder
            else None
        )
        self.video_object_camera_invariant_frontend = (
            nn.Conv2d(
                3,
                world_dim,
                kernel_size=object_patch_size,
                stride=object_patch_size,
            )
            if config.video_object_camera_invariant_residual
            else None
        )
        if self.video_object_camera_invariant_frontend is not None:
            nn.init.zeros_(self.video_object_camera_invariant_frontend.weight)
            nn.init.zeros_(self.video_object_camera_invariant_frontend.bias)
        self.video_object_tracker = (
            LearnedVideoObjectTracker(
                config.omni.object_slots,
                world_dim,
                spatial_coordinates=(config.video_object_spatial_coordinates),
            )
            if config.video_object_learned_queries
            else None
        )
        self.video_object_to_world = (
            nn.Linear(world_dim, world_dim)
            if config.video_object_temporal_encoder
            else None
        )
        self.video_object_statistics = (
            nn.Linear(world_dim * 3, world_dim)
            if config.video_object_temporal_encoder
            else None
        )
        self.video_object_decision = (
            nn.Sequential(
                nn.LayerNorm(world_dim * (config.omni.object_slots + 1)),
                nn.Linear(
                    world_dim * (config.omni.object_slots + 1),
                    world_dim,
                ),
                nn.GELU(),
            )
            if config.video_object_temporal_encoder
            else None
        )
        self.video_object_set_decision = (
            nn.Sequential(
                nn.LayerNorm(world_dim * 2),
                nn.Linear(world_dim * 2, world_dim),
                nn.GELU(),
            )
            if config.video_object_set_decision
            else None
        )
        if self.video_object_set_decision is not None:
            nn.init.zeros_(self.video_object_set_decision[1].weight)
            nn.init.zeros_(self.video_object_set_decision[1].bias)
        self.video_object_binding_decision = (
            nn.Sequential(
                nn.LayerNorm(world_dim * 4),
                nn.Linear(world_dim * 4, world_dim),
                nn.GELU(),
            )
            if config.video_object_identity_event_binding
            else None
        )
        if self.video_object_binding_decision is not None:
            nn.init.zeros_(self.video_object_binding_decision[1].weight)
            nn.init.zeros_(self.video_object_binding_decision[1].bias)
        self.video_spatial_temporal_moment_to_world = (
            nn.Linear(2, world_dim, bias=False)
            if config.video_spatial_temporal_moment
            else None
        )
        if self.video_spatial_temporal_moment_to_world is not None:
            nn.init.zeros_(self.video_spatial_temporal_moment_to_world.weight)
        self.video_spatial_temporal_moment_gate = (
            nn.Linear(world_dim, 2)
            if config.video_query_spatial_temporal_moment
            else None
        )
        if self.video_spatial_temporal_moment_gate is not None:
            nn.init.zeros_(self.video_spatial_temporal_moment_gate.weight)
            nn.init.zeros_(self.video_spatial_temporal_moment_gate.bias)
        self.video_spatial_temporal_y_moment_to_world = (
            nn.Linear(2, world_dim, bias=False)
            if config.video_spatial_temporal_y_moment
            else None
        )
        self.video_spatial_temporal_y_moment_gate = (
            nn.Linear(world_dim, 2) if config.video_spatial_temporal_y_moment else None
        )
        if self.video_spatial_temporal_y_moment_to_world is not None:
            nn.init.zeros_(self.video_spatial_temporal_y_moment_to_world.weight)
            nn.init.zeros_(self.video_spatial_temporal_y_moment_gate.weight)
            nn.init.zeros_(self.video_spatial_temporal_y_moment_gate.bias)
        self.video_spatial_temporal_logit_head = (
            nn.Linear(world_dim + 4, 2)
            if config.video_spatial_temporal_logit_head
            else None
        )
        if self.video_spatial_temporal_logit_head is not None:
            nn.init.zeros_(self.video_spatial_temporal_logit_head.weight)
            nn.init.zeros_(self.video_spatial_temporal_logit_head.bias)
        self.video_spatial_temporal_bilinear_head = (
            nn.Linear(world_dim, 4)
            if config.video_spatial_temporal_bilinear_head
            else None
        )
        if self.video_spatial_temporal_bilinear_head is not None:
            nn.init.zeros_(self.video_spatial_temporal_bilinear_head.weight)
            nn.init.zeros_(self.video_spatial_temporal_bilinear_head.bias)
        self.video_object_trajectory_binding = (
            QueryConditionedObjectTrajectoryBinding(world_dim)
            if config.video_object_trajectory_binding
            else None
        )
        self.video_object_pair_trajectory_binding = (
            QueryConditionedObjectPairTrajectoryBinding(world_dim)
            if config.video_object_pair_trajectory_binding
            else None
        )
        self.video_descriptor_trajectory_binding = (
            DescriptorConditionedDenseTrajectoryBinding(
                world_dim,
                pair_centered_queries=(config.video_descriptor_pair_centered_queries),
                persistent_identity_state=(
                    config.video_descriptor_persistent_identity_state
                ),
                object_memory=config.video_descriptor_object_memory,
                object_memory_scale=(config.video_descriptor_object_memory_scale),
                object_memory_query_gate=(
                    config.video_descriptor_object_memory_query_gate
                ),
                object_memory_reliability_gate=(
                    config.video_descriptor_object_memory_reliability_gate
                ),
                object_memory_contrast_visibility=(
                    config.video_descriptor_object_memory_contrast_visibility
                ),
                object_memory_contrast_readout=(
                    config.video_descriptor_object_memory_contrast_readout
                ),
                object_memory_temporal_relative_visibility=(
                    config.video_descriptor_object_memory_temporal_relative_visibility
                ),
                object_memory_temporal_relative_readout=(
                    config.video_descriptor_object_memory_temporal_relative_readout
                ),
            )
            if config.video_descriptor_trajectory_binding
            else None
        )
        self.video_query_conditioning = (
            nn.Linear(world_dim, world_dim * 2)
            if config.video_query_conditioned_head
            else None
        )
        if self.video_query_conditioning is not None:
            nn.init.zeros_(self.video_query_conditioning.weight)
            nn.init.zeros_(self.video_query_conditioning.bias)
        self.video_object_evidence_gate = (
            nn.Linear(world_dim, 2) if config.video_object_dual_evidence else None
        )
        if self.video_object_evidence_gate is not None:
            nn.init.zeros_(self.video_object_evidence_gate.weight)
            nn.init.zeros_(self.video_object_evidence_gate.bias)
        self.long_video_accumulator = (
            LongVideoWorldAccumulator(
                config.omni,
                transition_features=config.long_video_transition_features,
            )
            if config.long_video_world_accumulator
            else None
        )
        self.modality_embedding = nn.Embedding(4, world_dim)
        self.to_world = ModalToWorldAdapter(world_dim, config.omni)
        self.world_cell = nn.TransformerEncoderLayer(
            d_model=world_dim,
            nhead=config.omni.attention_heads,
            dim_feedforward=config.world_ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.world_norm = nn.LayerNorm(world_dim)
        self.world_to_text_memory = nn.Linear(
            world_dim,
            config.text.retriever_dim,
        )
        self.text_only_world_to_text_memory = (
            nn.Linear(world_dim, config.text.retriever_dim)
            if config.text_only_bridge_adapter
            else None
        )
        self.text_only_logit_adapter = (
            nn.Sequential(
                nn.Linear(VOCAB_SIZE, VOCAB_SIZE * 2),
                nn.GELU(),
                nn.Linear(VOCAB_SIZE * 2, VOCAB_SIZE),
            )
            if config.text_only_output_adapter
            else None
        )
        self.text_only_cross_memory = (
            TextOnlyCrossMemoryAdapter(
                config.text.model_dim,
                config.text.attention_heads,
            )
            if config.text_only_cross_memory_adapter
            else None
        )
        self.text_only_hidden_cross_memory = (
            TextOnlyHiddenCrossMemoryAdapter(
                config.text.model_dim,
                config.text.attention_heads,
            )
            if config.text_only_hidden_cross_memory_adapter
            else None
        )
        self.text_answerability_head = None
        if config.text_answerability_head:
            if config.text_answerability_mode in {
                "token-cross",
                "contextual-cross",
                "consistency-cross",
                "body-cross",
                "core-body-cross",
                "core-compact-body-cross",
                "core-projected-compact-body-cross",
                "core-lexical-compact-body-cross",
                "core-lexical-consistency-cross",
            }:
                self.text_answerability_head = TextAnswerabilityVerifier(
                    world_dim,
                    config.omni.attention_heads,
                    contextual=(
                        config.text_answerability_mode
                        in {
                            "contextual-cross",
                            "consistency-cross",
                            "body-cross",
                        }
                    ),
                    evidence_consistency=(
                        config.text_answerability_mode
                        in {
                            "consistency-cross",
                            "core-lexical-consistency-cross",
                        }
                    ),
                    question_body_only=(
                        config.text_answerability_mode
                        in {
                            "body-cross",
                            "core-body-cross",
                            "core-compact-body-cross",
                            "core-projected-compact-body-cross",
                            "core-lexical-compact-body-cross",
                            "core-lexical-consistency-cross",
                        }
                    ),
                    source_dim=(
                        config.text.model_dim
                        if config.text_answerability_mode
                        in {
                            "core-projected-compact-body-cross",
                            "core-lexical-compact-body-cross",
                            "core-lexical-consistency-cross",
                        }
                        else None
                    ),
                    extra_feature_dim=(
                        len(ANSWERABILITY_NGRAM_WIDTHS)
                        if config.text_answerability_mode
                        in {
                            "core-lexical-compact-body-cross",
                            "core-lexical-consistency-cross",
                        }
                        else 0
                    ),
                    output_classes=config.text_answerability_classes,
                )
            else:
                self.text_answerability_head = nn.Sequential(
                    nn.LayerNorm(world_dim * 4 + 2),
                    nn.Linear(world_dim * 4 + 2, world_dim),
                    nn.GELU(),
                    nn.Linear(world_dim, config.text_answerability_classes),
                )
        self.text_epistemic_memory = (
            nn.Linear(
                config.text_answerability_classes,
                config.text.retriever_dim,
                bias=False,
            )
            if config.text_epistemic_memory_adapter
            else None
        )
        self.text_epistemic_output = (
            TextEpistemicOutputAdapter(
                config.text.model_dim,
                config.text_epistemic_output_rank,
            )
            if config.text_epistemic_output_rank
            else None
        )
        for adapter in (
            self.text_only_to_world,
            self.text_only_world_to_text_memory,
        ):
            if adapter is not None:
                nn.init.zeros_(adapter.weight)
                nn.init.zeros_(adapter.bias)
        if self.text_only_logit_adapter is not None:
            nn.init.zeros_(self.text_only_logit_adapter[-1].weight)
            nn.init.zeros_(self.text_only_logit_adapter[-1].bias)
        if self.text_epistemic_memory is not None:
            nn.init.zeros_(self.text_epistemic_memory.weight)
        self.video_order_head = nn.Sequential(
            nn.LayerNorm(world_dim),
            nn.Linear(world_dim, 2),
        )
        self.video_camera_robustness_gate = (
            nn.Sequential(
                nn.Linear(8, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            if config.video_camera_robustness_nonlinear_gate
            else nn.Linear(8, 1)
            if config.video_camera_robustness_adapter
            else None
        )
        self.video_camera_robustness_head = (
            nn.Linear(world_dim, 2, bias=False)
            if config.video_camera_robustness_adapter
            else None
        )
        if self.video_camera_robustness_gate is not None:
            if config.video_camera_robustness_nonlinear_gate:
                nn.init.zeros_(self.video_camera_robustness_gate[0].weight)
                with torch.no_grad():
                    identity = torch.eye(8)
                    self.video_camera_robustness_gate[0].weight[:8].copy_(identity)
                    self.video_camera_robustness_gate[0].weight[8:].copy_(-identity)
                nn.init.zeros_(self.video_camera_robustness_gate[0].bias)
                nn.init.zeros_(self.video_camera_robustness_gate[2].weight)
                nn.init.constant_(self.video_camera_robustness_gate[2].bias, -4.0)
            else:
                nn.init.zeros_(self.video_camera_robustness_gate.weight)
                nn.init.constant_(self.video_camera_robustness_gate.bias, -4.0)
            nn.init.zeros_(self.video_camera_robustness_head.weight)
        self.video_camera_pose_encoder = (
            nn.Linear(config.video_camera_pose_dim, world_dim)
            if config.video_camera_pose_dim
            else None
        )
        if self.video_camera_pose_encoder is not None:
            nn.init.zeros_(self.video_camera_pose_encoder.weight)
            nn.init.zeros_(self.video_camera_pose_encoder.bias)
        self.video_spatial_geometry_reasoner = (
            VideoSpatialGeometryReasoner(config.video_camera_pose_dim)
            if config.video_camera_pose_dim
            and config.video_spatial_relation_classes
            else None
        )
        spatial_relation_input_dim = (
            world_dim * 2
            + 48
            + (
                VideoSpatialGeometryReasoner.output_dim
                if self.video_spatial_geometry_reasoner is not None
                else 0
            )
        )
        self.video_spatial_relation_head = (
            nn.Sequential(
                nn.LayerNorm(spatial_relation_input_dim),
                nn.Linear(spatial_relation_input_dim, world_dim),
                nn.GELU(),
                nn.Linear(world_dim, config.video_spatial_relation_classes),
            )
            if config.video_spatial_relation_classes
            else None
        )
        if self.video_spatial_relation_head is not None:
            nn.init.zeros_(self.video_spatial_relation_head[3].weight)
            nn.init.zeros_(self.video_spatial_relation_head[3].bias)
        self.video_action_encoder = (
            nn.Linear(config.video_action_dim, world_dim)
            if config.video_action_dim
            else None
        )
        self.video_egomotion_reasoner = (
            VideoEgomotionReasoner(world_dim)
            if config.video_egomotion_classes
            else None
        )
        self.video_egomotion_head = (
            nn.Sequential(
                nn.LayerNorm(world_dim * 3),
                nn.Linear(world_dim * 3, world_dim),
                nn.GELU(),
                nn.Linear(world_dim, config.video_egomotion_classes),
            )
            if config.video_egomotion_classes
            else None
        )
        if self.video_action_encoder is not None:
            nn.init.zeros_(self.video_action_encoder.weight)
            nn.init.zeros_(self.video_action_encoder.bias)
        if self.video_egomotion_head is not None:
            nn.init.zeros_(self.video_egomotion_head[3].weight)
            nn.init.zeros_(self.video_egomotion_head[3].bias)
        self.video_egomotion_validity_head = (
            nn.Sequential(
                nn.LayerNorm(world_dim + 2),
                nn.Linear(world_dim + 2, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
            if config.video_egomotion_validity_head
            else None
        )
        if self.video_egomotion_validity_head is not None:
            nn.init.zeros_(self.video_egomotion_validity_head[3].weight)
            nn.init.zeros_(self.video_egomotion_validity_head[3].bias)
        self.video_teacher_projection = (
            nn.Linear(world_dim, config.vision_teacher_dim)
            if config.vision_teacher_dim
            else None
        )
        self.audio_teacher_projection = (
            nn.Linear(world_dim, config.audio_teacher_dim)
            if config.audio_teacher_dim
            else None
        )

    def _encode_visual_map(
        self,
        normalized_pixels: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.vision_frontend(normalized_pixels)
        if self.visual_semantic_frontend is not None:
            return visual
        return self._apply_visual_semantic_encoder(visual)

    def _apply_visual_semantic_encoder(
        self,
        visual: torch.Tensor,
    ) -> torch.Tensor:
        if self.visual_semantic_cell is None:
            return visual
        batch, channels, height, width = visual.shape
        vertical = torch.linspace(
            -1.0,
            1.0,
            height,
            dtype=visual.dtype,
            device=visual.device,
        )
        horizontal = torch.linspace(
            -1.0,
            1.0,
            width,
            dtype=visual.dtype,
            device=visual.device,
        )
        grid_y, grid_x = torch.meshgrid(
            vertical,
            horizontal,
            indexing="ij",
        )
        coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(
            1,
            height * width,
            2,
        )
        tokens = visual.flatten(2).transpose(1, 2)
        tokens = tokens + self.visual_position(coordinates)
        for _ in range(self.config.visual_semantic_rounds):
            tokens = self.visual_semantic_cell(tokens)
        tokens = self.visual_semantic_norm(tokens)
        return tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )

    def _encode_visual_semantic_map(
        self,
        normalized_pixels: torch.Tensor,
    ) -> torch.Tensor:
        if self.visual_semantic_frontend is None:
            return self._encode_visual_map(normalized_pixels)
        return self._apply_visual_semantic_encoder(
            self.visual_semantic_frontend(normalized_pixels)
        )

    def _encode_image_visual_semantic_map(
        self,
        normalized_pixels: torch.Tensor,
    ) -> torch.Tensor:
        if self.image_visual_adapter_down is None:
            return self._encode_visual_semantic_map(normalized_pixels)
        visual = self.visual_semantic_frontend(normalized_pixels)
        residual = self.image_visual_adapter_up(
            self.image_visual_adapter_down(normalized_pixels)
        )
        return self._apply_visual_semantic_encoder(
            visual + self.config.image_visual_adapter_scale * residual
        )

    def encode_visual_summary(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return the shared per-image semantic summary without World decoding."""
        self._validate_image(
            pixel_values,
            pixel_values.shape[0],
            "pixel_values",
        )
        visual = self._encode_image_visual_semantic_map(
            self._frontend_input(pixel_values)
        )
        return visual.mean(dim=(2, 3))

    def encode_visual_regions(
        self,
        pixel_values: torch.Tensor,
        boxes_xyxy_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Average a 2x2 feature grid inside normalized regions."""
        return self.encode_visual_region_grids(
            pixel_values,
            boxes_xyxy_normalized,
            grid_size=2,
        ).mean(dim=(-1, -2))

    def encode_visual_region_grids(
        self,
        pixel_values: torch.Tensor,
        boxes_xyxy_normalized: torch.Tensor,
        *,
        grid_size: int = 2,
    ) -> torch.Tensor:
        """Sample fixed-size feature grids inside normalized regions."""
        self._validate_image(
            pixel_values,
            pixel_values.shape[0],
            "pixel_values",
        )
        if (
            boxes_xyxy_normalized.ndim != 3
            or boxes_xyxy_normalized.shape[0] != pixel_values.shape[0]
            or boxes_xyxy_normalized.shape[2] != 4
        ):
            raise ValueError(
                "boxes_xyxy_normalized must have shape [batch, regions, 4]"
            )
        boxes = boxes_xyxy_normalized.to(
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        if bool(((boxes < 0.0) | (boxes > 1.0)).any()):
            raise ValueError("normalized boxes must stay inside [0, 1]")
        if bool(
            ((boxes[..., 0] >= boxes[..., 2]) | (boxes[..., 1] >= boxes[..., 3])).any()
        ):
            raise ValueError("normalized boxes must have positive area")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        visual = self._encode_image_visual_semantic_map(
            self._frontend_input(pixel_values)
        )
        fractions = (
            torch.arange(
                grid_size,
                device=visual.device,
                dtype=visual.dtype,
            )
            + 0.5
        ) / grid_size
        fraction_y, fraction_x = torch.meshgrid(
            fractions,
            fractions,
            indexing="ij",
        )
        grid_x = (
            boxes[..., 0, None, None]
            + (boxes[..., 2, None, None] - boxes[..., 0, None, None]) * fraction_x
        )
        grid_y = (
            boxes[..., 1, None, None]
            + (boxes[..., 3, None, None] - boxes[..., 1, None, None]) * fraction_y
        )
        regions = boxes.shape[1]
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = (
            grid.mul(2.0)
            .sub(1.0)
            .reshape(
                pixel_values.shape[0],
                regions * grid_size,
                grid_size,
                2,
            )
        )
        sampled = torch.nn.functional.grid_sample(
            visual,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.reshape(
            visual.shape[0],
            visual.shape[1],
            regions,
            grid_size,
            grid_size,
        ).permute(0, 2, 1, 3, 4)

    def _encode_text_descriptor(self, input_ids: torch.Tensor) -> torch.Tensor:
        if (
            input_ids.ndim != 2
            or input_ids.shape[1] < 2
            or bool((input_ids[:, 0] != BOS_ID).any())
        ):
            raise ValueError("descriptor inputs must be BOS-prefixed text batches")
        patches = self.text_core._pad_tokens(input_ids[:, 1:], PAD_ID)
        patch_mask = patches.ne(PAD_ID).any(dim=-1)
        if bool((~patch_mask.any(dim=1)).any()):
            raise ValueError("every descriptor input must contain text")
        states = self.text_core._encode_patches(patches)
        world = self.text_to_world(states)
        active = patch_mask.unsqueeze(-1).to(world.dtype)
        return (world * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)

    def _narrative_text_summary(self, input_ids: torch.Tensor) -> torch.Tensor:
        if (
            input_ids.ndim != 2
            or input_ids.shape[1] < 2
            or bool((input_ids[:, 0] != BOS_ID).any())
        ):
            raise ValueError("narrative inputs must be BOS-prefixed text batches")
        patches = self.text_core._pad_tokens(input_ids[:, 1:], PAD_ID)
        patch_mask = patches.ne(PAD_ID).any(dim=-1)
        if bool((~patch_mask.any(dim=1)).any()):
            raise ValueError("every narrative input must contain text")
        output = self.text_core(input_ids, rounds=1)
        if output.context_states is None:
            raise RuntimeError("text core did not return contextual states")
        states = output.context_states
        active = patch_mask.unsqueeze(-1).to(states.dtype)
        mean = (states * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
        last_index = patch_mask.sum(dim=1) - 1
        last = states[torch.arange(states.shape[0], device=states.device), last_index]
        return torch.cat((F.normalize(mean.float()), F.normalize(last.float())), dim=-1)

    def score_narrative_continuity(
        self,
        anchor_input_ids: torch.Tensor,
        candidate_input_ids: torch.Tensor,
    ) -> NarrativeContinuityOutput:
        if self.narrative_continuity_head is None or self.narrative_evidence_to_world is None:
            raise RuntimeError("narrative evidence head is disabled")
        if anchor_input_ids.shape[0] != candidate_input_ids.shape[0]:
            raise ValueError("anchor and candidate batches must match")
        anchor = self._narrative_text_summary(anchor_input_ids)
        candidate = self._narrative_text_summary(candidate_input_ids)
        features = torch.cat(
            (anchor, candidate, (anchor - candidate).abs(), anchor * candidate),
            dim=-1,
        )
        score = self.narrative_continuity_head(features).squeeze(-1)
        return NarrativeContinuityOutput(
            score=score,
            world_delta=self.narrative_evidence_to_world(score.unsqueeze(-1)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        world_input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        subject_descriptor_mask: torch.Tensor | None = None,
        object_descriptor_mask: torch.Tensor | None = None,
        subject_descriptor_input_ids: torch.Tensor | None = None,
        object_descriptor_input_ids: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
        video_values: torch.Tensor | None = None,
        camera_pose_values: torch.Tensor | None = None,
        action_values: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        question_input_ids: torch.Tensor | None = None,
        answerability_labels: torch.Tensor | None = None,
        text_rounds: int | None = None,
    ) -> MosaicUnifiedOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if bool((input_ids[:, 0] != BOS_ID).any()):
            raise ValueError("every unified input sequence must begin with BOS_ID")
        batch = input_ids.shape[0]
        text_only = all(
            value is None
            for value in (
                pixel_values,
                audio_values,
                video_values,
            )
        )
        if world_input_ids is None:
            world_input_ids = input_ids
        if (
            world_input_ids.ndim != 2
            or world_input_ids.shape[0] != batch
            or world_input_ids.shape[1] < 1
            or bool((world_input_ids[:, 0] != BOS_ID).any())
        ):
            raise ValueError(
                "world_input_ids must be a BOS-prefixed batch matching input_ids"
            )
        if question_input_ids is not None and (
            question_input_ids.ndim != 2
            or question_input_ids.shape[0] != batch
            or question_input_ids.shape[1] < 1
            or bool((question_input_ids[:, 0] != BOS_ID).any())
        ):
            raise ValueError(
                "question_input_ids must be a BOS-prefixed batch matching input_ids"
            )
        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        modalities: list[str] = []
        video_temporal_summary: torch.Tensor | None = None
        video_evidence_tokens: torch.Tensor | None = None
        video_temporal_delta: torch.Tensor | None = None
        video_temporal_delta_world: torch.Tensor | None = None
        video_object_slots: torch.Tensor | None = None
        video_object_normalized_slots: torch.Tensor | None = None
        video_object_binding_features: torch.Tensor | None = None
        video_object_trajectory_features: tuple[torch.Tensor, ...] | None = None
        video_object_evidence_weights: torch.Tensor | None = None
        video_object_attention_output: torch.Tensor | None = None
        video_object_trajectory_weights: torch.Tensor | None = None
        video_object_pair_trajectory_weights: torch.Tensor | None = None
        video_descriptor_trajectory_attention: torch.Tensor | None = None
        video_descriptor_visibility_logits: torch.Tensor | None = None
        video_descriptor_memory_margin: torch.Tensor | None = None
        video_descriptor_memory_reliability_logits: torch.Tensor | None = None
        video_camera_statistics: torch.Tensor | None = None
        video_camera_robustness_gate: torch.Tensor | None = None
        video_camera_pose_summary: torch.Tensor | None = None
        video_action_summary: torch.Tensor | None = None
        video_egomotion_summary: torch.Tensor | None = None
        video_dense_object_features: tuple[torch.Tensor, ...] | None = None
        video_spatial_temporal_moment: torch.Tensor | None = None
        video_spatial_temporal_y_moment: torch.Tensor | None = None
        video_spatial_temporal_features: torch.Tensor | None = None
        video_spatial_relation_geometry: torch.Tensor | None = None
        video_spatial_reasoning: torch.Tensor | None = None
        video_world_summary: torch.Tensor | None = None
        image_summary: torch.Tensor | None = None
        audio_summary: torch.Tensor | None = None
        audio_world_summary: torch.Tensor | None = None
        audio_temporal_states: torch.Tensor | None = None
        audio_temporal_features: torch.Tensor | None = None
        audio_event_slots: torch.Tensor | None = None
        audio_content_summary: torch.Tensor | None = None
        audio_content_features: torch.Tensor | None = None
        audio_content_event_slots: torch.Tensor | None = None
        image_patch_tokens: torch.Tensor | None = None
        image_patch_grid: tuple[int, int] | None = None
        explicit_relation_logits: torch.Tensor | None = None
        explicit_object_attention: torch.Tensor | None = None
        descriptor_queries: tuple[torch.Tensor, torch.Tensor] | None = None

        if camera_pose_values is not None:
            if self.video_camera_pose_encoder is None:
                raise ValueError("camera pose values require the camera pose encoder")
            if video_values is None:
                raise ValueError("camera pose values require video input")
            expected_pose = (
                batch,
                video_values.shape[1],
                self.config.video_camera_pose_dim,
            )
            if tuple(camera_pose_values.shape) != expected_pose:
                raise ValueError(
                    f"camera pose values must have shape {expected_pose}"
                )
            video_camera_pose_states = self.video_camera_pose_encoder(
                camera_pose_values.to(dtype=self.video_camera_pose_encoder.weight.dtype)
            )
            video_camera_pose_summary = video_camera_pose_states.mean(dim=1)

        if action_values is not None:
            if self.video_action_encoder is None:
                raise ValueError("action values require the video action encoder")
            if video_values is None:
                raise ValueError("action values require video input")
            expected_action = (batch, self.config.video_action_dim)
            if tuple(action_values.shape) != expected_action:
                raise ValueError(f"action values must have shape {expected_action}")
            video_action_summary = self.video_action_encoder(
                action_values.to(dtype=self.video_action_encoder.weight.dtype)
            )

        if (
            video_values is not None
            and action_values is not None
            and self.video_egomotion_reasoner is not None
        ):
            video_egomotion_summary = self.video_egomotion_reasoner(video_values)

        if video_values is not None and self.video_camera_robustness_gate is not None:
            video_camera_statistics = _video_camera_statistics(video_values)

        raw_text = world_input_ids[:, 1:]
        text_patches = self.text_core._pad_tokens(raw_text, PAD_ID)
        text_mask = text_patches.ne(PAD_ID).any(dim=-1)
        bos = self.text_core.bos_patch.view(1, 1, -1).expand(batch, -1, -1)
        text_states = torch.cat(
            (bos, self.text_core._encode_patches(text_patches)),
            dim=1,
        )
        text_world = self.text_to_world(text_states)
        if text_only and self.text_only_to_world is not None:
            text_world = text_world + self.text_only_to_world(text_states)
        text_source_mask = torch.cat(
            (
                torch.ones(
                    (batch, 1),
                    dtype=torch.bool,
                    device=world_input_ids.device,
                ),
                text_mask,
            ),
            dim=1,
        )
        text_retrieval_summary = (text_world * text_source_mask.unsqueeze(-1)).sum(
            dim=1
        ) / text_source_mask.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1)
        tokens.append(text_world + self.modality_embedding.weight[0].view(1, 1, -1))
        masks.append(text_source_mask)
        modalities.append("text")

        if pixel_values is not None:
            self._validate_image(pixel_values, batch, "pixel_values")
            frontend_image = self._frontend_input(pixel_values)
            image_geometry = self._encode_visual_map(frontend_image)
            image_patch_grid = (
                image_geometry.shape[2],
                image_geometry.shape[3],
            )
            image_patch_tokens = image_geometry.flatten(2).transpose(1, 2)
            image = (
                (
                    self._encode_image_visual_semantic_map(frontend_image)
                    if self.visual_semantic_frontend is not None
                    else image_geometry
                )
                .flatten(2)
                .transpose(1, 2)
            )
            image_summary = image.mean(dim=1)
            tokens.append(image + self.modality_embedding.weight[1].view(1, 1, -1))
            masks.append(self._full_mask(image))
            modalities.append("image")

        if audio_values is not None:
            if audio_values.ndim == 2:
                audio_values = audio_values.unsqueeze(1)
            if (
                audio_values.ndim != 3
                or audio_values.shape[0] != batch
                or audio_values.shape[1] != 1
                or audio_values.shape[2] < self.config.audio_patch_samples
            ):
                raise ValueError(
                    "audio_values must be [batch, samples] or [batch, 1, samples]"
                )
            raw_audio = self._frontend_input(audio_values)
            audio = self.audio_frontend(raw_audio)
            audio = audio.transpose(1, 2)
            audio_frontend_features = audio
            if self.audio_temporal_cell is None:
                audio_summary = audio.mean(dim=1)
            else:
                audio_temporal_states, audio_temporal_hidden = self.audio_temporal_cell(
                    audio
                )
                audio = audio + audio_temporal_states
                audio_temporal_features = audio
                audio_summary = audio_temporal_hidden[-1]
                audio_world_summary = self.audio_temporal_to_world(audio_summary)
            if self.audio_content_temporal_cell is not None:
                audio_content_input = audio_frontend_features
                if self.audio_spectral_projection is not None:
                    waveform = raw_audio[:, 0].float()
                    if waveform.shape[1] < self.config.audio_spectral_n_fft:
                        waveform = torch.nn.functional.pad(
                            waveform,
                            (
                                0,
                                self.config.audio_spectral_n_fft - waveform.shape[1],
                            ),
                        )
                    spectrum = torch.stft(
                        waveform,
                        n_fft=self.config.audio_spectral_n_fft,
                        hop_length=self.config.audio_spectral_hop_samples,
                        win_length=self.config.audio_spectral_n_fft,
                        window=self.audio_spectral_window.float(),
                        center=False,
                        return_complex=True,
                    )
                    log_power = torch.log1p(spectrum.abs().square())
                    spectral = self.audio_spectral_projection(
                        log_power.transpose(1, 2).to(
                            self.audio_spectral_projection.weight.dtype
                        )
                    )
                    audio_content_input = torch.nn.functional.adaptive_avg_pool1d(
                        spectral.transpose(1, 2),
                        audio_frontend_features.shape[1],
                    ).transpose(1, 2)
                (
                    audio_content_states,
                    audio_content_hidden,
                ) = self.audio_content_temporal_cell(audio_content_input)
                audio_content_features = audio_content_input + audio_content_states
                audio_content_summary = audio_content_hidden[-1]
                if audio_content_states.shape[1] == audio.shape[1]:
                    audio = audio + audio_content_states
            tokens.append(audio + self.modality_embedding.weight[2].view(1, 1, -1))
            masks.append(self._full_mask(audio))
            modalities.append("audio")

        if video_values is not None:
            if video_values.ndim != 5 or video_values.shape[0] != batch:
                raise ValueError(
                    "video_values must have shape [batch, frames, 3, height, width]"
                )
            frames = video_values.shape[1]
            flattened = video_values.reshape(
                batch * frames,
                *video_values.shape[2:],
            )
            self._validate_image(flattened, batch * frames, "video_values")
            frontend_video = self._frontend_input(flattened)
            video = self._encode_visual_map(frontend_video)
            if self.config.video_uses_visual_semantic_encoder:
                semantic_video = self._encode_visual_semantic_map(frontend_video)
                video = video + self.config.video_visual_semantic_scale * (
                    semantic_video - video
                )
            object_video = None
            object_evidence_count = 1
            if self.video_object_frontend is not None:
                camera_invariant_residual = (
                    self.video_object_camera_invariant_frontend(
                        _camera_invariant_object_frames(frontend_video)
                    )
                    if self.video_object_camera_invariant_frontend is not None
                    else 0.0
                )
                if self.config.video_object_dual_evidence:
                    raw_object_video = (
                        self.video_object_frontend(frontend_video)
                        + camera_invariant_residual
                    )
                    normalized_object_video = self.video_object_frontend(
                        _normalize_object_frontend_frames(frontend_video)
                    ) + camera_invariant_residual
                    object_video = torch.cat(
                        (raw_object_video, normalized_object_video),
                        dim=0,
                    )
                    object_evidence_count = 2
                elif not self.config.video_object_frame_normalized_input:
                    object_video = (
                        self.video_object_frontend(frontend_video)
                        + camera_invariant_residual
                    )
                else:
                    normalized_object_video = self.video_object_frontend(
                        _normalize_object_frontend_frames(frontend_video)
                    ) + camera_invariant_residual
                    scale = self.config.video_object_frame_normalized_residual_scale
                    if scale == 1.0:
                        object_video = normalized_object_video
                    else:
                        raw_object_video = (
                            self.video_object_frontend(frontend_video)
                            + camera_invariant_residual
                        )
                        object_video = raw_object_video + scale * (
                            normalized_object_video - raw_object_video
                        )
                if self.video_descriptor_trajectory_binding is not None:
                    video_dense_object_features = tuple(
                        evidence.reshape(
                            batch,
                            frames,
                            -1,
                            *object_video.shape[-2:],
                        )
                        for evidence in object_video.chunk(
                            object_evidence_count,
                            dim=0,
                        )
                    )
            if self.config.video_spatial_temporal_moment:
                video_spatial_temporal_moment = torch.stack(
                    (
                        _spatial_temporal_moment(
                            raw_object_video,
                            batch=batch,
                            frames=frames,
                        ),
                        _spatial_temporal_moment(
                            normalized_object_video,
                            batch=batch,
                            frames=frames,
                        ),
                    ),
                    dim=-1,
                )
            if self.config.video_spatial_temporal_y_moment:
                video_spatial_temporal_y_moment = torch.stack(
                    (
                        _spatial_temporal_moment(
                            raw_object_video,
                            batch=batch,
                            frames=frames,
                            axis="y",
                        ),
                        _spatial_temporal_moment(
                            normalized_object_video,
                            batch=batch,
                            frames=frames,
                            axis="y",
                        ),
                    ),
                    dim=-1,
                )
                video_spatial_temporal_features = torch.cat(
                    (
                        video_spatial_temporal_moment,
                        video_spatial_temporal_y_moment,
                    ),
                    dim=-1,
                )
            height, width = video.shape[2:]
            video = video.reshape(
                batch,
                frames,
                video.shape[1],
                height,
                width,
            ).transpose(1, 2)
            if self.config.video_explicit_temporal_delta:
                unmixed_frame_states = video.mean(dim=(3, 4)).transpose(1, 2)
                video_temporal_delta = (
                    unmixed_frame_states[:, -1] - unmixed_frame_states[:, 0]
                )
            video = video + self.video_temporal_mixer(video)
            frame_times = torch.linspace(
                0.0,
                1.0,
                frames,
                device=video.device,
                dtype=video.dtype,
            ).view(1, frames, 1)
            frame_states = video.mean(dim=(3, 4)).transpose(1, 2)
            temporal_states, temporal_hidden = self.video_temporal_cell(
                frame_states + self.video_time(frame_times)
            )
            video_temporal_summary = temporal_hidden[-1]
            video_evidence_tokens = self.video_temporal_to_world(temporal_states)
            if self.video_object_temporal_cell is not None:
                object_batch = batch * object_evidence_count
                object_slots = self.config.omni.object_slots
                video_object_attention = None
                if self.video_object_tracker is None:
                    grid_rows = math.isqrt(object_slots)
                    while object_slots % grid_rows:
                        grid_rows -= 1
                    grid_columns = object_slots // grid_rows
                    frame_grid = torch.nn.functional.adaptive_avg_pool2d(
                        object_video,
                        (grid_rows, grid_columns),
                    )
                    frame_grid = (
                        frame_grid.reshape(
                            object_batch,
                            frames,
                            -1,
                            object_slots,
                        )
                        .permute(0, 3, 1, 2)
                        .reshape(object_batch * object_slots, frames, -1)
                    )
                else:
                    if (
                        self.config.video_object_spatial_event_binding
                        or self.config.video_object_trajectory_binding
                        or self.config.video_object_pair_trajectory_binding
                    ):
                        (
                            frame_grid,
                            video_object_attention,
                        ) = self.video_object_tracker.track_with_attention(
                            object_video,
                            batch=object_batch,
                            frames=frames,
                        )
                    else:
                        frame_grid = self.video_object_tracker(
                            object_video,
                            batch=object_batch,
                            frames=frames,
                        )
                raw_object_frame_grid = frame_grid
                if (
                    self.video_object_trajectory_binding is not None
                    or self.video_object_pair_trajectory_binding is not None
                ):
                    if video_object_attention is None:
                        raise RuntimeError(
                            "object trajectory binding requires object attention"
                        )
                    raw_grid = raw_object_frame_grid[: batch * object_slots].reshape(
                        batch, object_slots, frames, -1
                    )
                    identity = (raw_grid[:, :, 0] + raw_grid[:, :, -1]) * 0.5
                    trajectory_times = torch.linspace(
                        -1.0,
                        1.0,
                        frames,
                        device=raw_grid.device,
                        dtype=raw_grid.dtype,
                    ).view(1, 1, frames, 1)
                    event = (
                        (raw_grid - raw_grid.mean(dim=2, keepdim=True))
                        * trajectory_times
                    ).mean(dim=2)
                    delta = raw_grid[:, :, -1] - raw_grid[:, :, 0]
                    geometry = _object_attention_trajectory(
                        video_object_attention[:batch]
                    )
                    video_object_trajectory_features = (
                        identity,
                        event,
                        delta,
                        geometry,
                    )
                    video_object_attention_output = video_object_attention[:batch]
                if self.video_object_binding_decision is not None:
                    raw_grid = raw_object_frame_grid[: batch * object_slots].reshape(
                        batch, object_slots, frames, -1
                    )
                    identity_features = raw_grid.mean(dim=2)
                    centered_raw_grid = raw_grid - identity_features.unsqueeze(2)
                    binding_times = torch.linspace(
                        -1.0,
                        1.0,
                        frames,
                        device=raw_grid.device,
                        dtype=raw_grid.dtype,
                    ).view(1, 1, frames, 1)
                    event_features = (centered_raw_grid * binding_times).mean(dim=2)
                    if self.config.video_object_spatial_event_binding:
                        if video_object_attention is None:
                            raise RuntimeError(
                                "spatial event binding requires object attention"
                            )
                        event_features = event_features + _spatial_event_features(
                            raw_grid,
                            video_object_attention[:batch],
                        )
                    video_object_binding_features = torch.cat(
                        (
                            identity_features,
                            event_features,
                            identity_features * event_features,
                        ),
                        dim=-1,
                    )
                frame_grid = _select_object_temporal_evidence(
                    raw_object_frame_grid,
                    time_centered=(self.config.video_object_time_centered_input),
                    dual_evidence=self.config.video_object_dual_evidence,
                    raw_rows=batch * object_slots,
                )
                centered_times = torch.linspace(
                    -1.0,
                    1.0,
                    frames,
                    device=frame_grid.device,
                    dtype=frame_grid.dtype,
                ).view(1, frames, 1)
                endpoint_delta = frame_grid[:, -1] - frame_grid[:, 0]
                temporal_moment = (frame_grid * centered_times).mean(dim=1)
                first_reveal = min(frames - 1, max(1, frames // 2 - 1))
                reveal_delta = frame_grid[:, first_reveal] - frame_grid[:, 0]
                object_statistics = self.video_object_statistics(
                    torch.cat(
                        (
                            endpoint_delta,
                            temporal_moment,
                            reveal_delta,
                        ),
                        dim=-1,
                    )
                )
                object_times = self.video_time(frame_times).unsqueeze(1)
                object_times = object_times.expand(
                    object_batch,
                    object_slots,
                    frames,
                    -1,
                ).reshape(object_batch * object_slots, frames, -1)
                _, object_hidden = self.video_object_temporal_cell(
                    frame_grid + object_times
                )
                encoded_object_slots = (object_hidden[-1] + object_statistics).reshape(
                    object_batch, object_slots, -1
                )
                if self.config.video_object_activity_sorted_slots:
                    encoded_object_slots = _sort_object_slots_by_temporal_activity(
                        encoded_object_slots,
                        raw_object_frame_grid.reshape(
                            object_batch,
                            object_slots,
                            frames,
                            -1,
                        ),
                    )
                if self.config.video_object_dual_evidence:
                    (
                        video_object_slots,
                        video_object_normalized_slots,
                    ) = encoded_object_slots.split(batch, dim=0)
                else:
                    video_object_slots = encoded_object_slots
            video = video.transpose(1, 2).reshape(
                batch * frames,
                -1,
                height,
                width,
            )
            spatial_tokens = video.shape[2] * video.shape[3]
            video = (
                video.flatten(2)
                .transpose(1, 2)
                .reshape(batch, frames, spatial_tokens, -1)
            )
            timeline = torch.linspace(
                0.0,
                1.0,
                frames,
                device=video.device,
                dtype=video.dtype,
            ).view(1, frames, 1, 1)
            video = video + self.video_time(timeline).expand(
                batch,
                -1,
                spatial_tokens,
                -1,
            )
            video = video.flatten(1, 2)
            video = torch.cat((video, temporal_states), dim=1)
            tokens.append(video + self.modality_embedding.weight[3].view(1, 1, -1))
            masks.append(self._full_mask(video))
            modalities.append("video")

        source_tokens = torch.cat(tokens, dim=1)
        source_mask = torch.cat(masks, dim=1)
        world_state = self.to_world(
            source_tokens,
            source_mask=source_mask,
            source="unified:" + "+".join(modalities),
        )
        world = world_state.semantic_slots
        if video_camera_pose_summary is not None:
            world = world.clone()
            first_camera = SLOT_ROLES.index("camera_0")
            world[:, first_camera] = (
                world[:, first_camera] + video_camera_pose_states[:, 0]
            )
            world[:, first_camera + 1] = (
                world[:, first_camera + 1] + video_camera_pose_states[:, -1]
            )
        if video_action_summary is not None:
            if world is world_state.semantic_slots:
                world = world.clone()
            world[:, SLOT_ROLES.index("action_0")] = (
                world[:, SLOT_ROLES.index("action_0")] + video_action_summary
            )
        descriptor_masks = (
            subject_descriptor_mask,
            object_descriptor_mask,
        )
        descriptor_inputs = (
            subject_descriptor_input_ids,
            object_descriptor_input_ids,
        )
        if any(mask is not None for mask in descriptor_masks) and any(
            value is not None for value in descriptor_inputs
        ):
            raise ValueError("descriptor masks and descriptor inputs are exclusive")
        descriptors: list[torch.Tensor] | None = None
        if any(value is not None for value in descriptor_inputs):
            if any(value is None for value in descriptor_inputs):
                raise ValueError("object binding requires both descriptor inputs")
            descriptors = [
                self._encode_text_descriptor(value) for value in descriptor_inputs
            ]
        elif any(mask is not None for mask in descriptor_masks):
            if any(mask is None for mask in descriptor_masks):
                raise ValueError("object binding requires both descriptor masks")
            expected_mask = text_world.shape[:2]
            normalized_masks = []
            for mask in descriptor_masks:
                if mask.shape != expected_mask:
                    raise ValueError(
                        "descriptor masks must match the text World sequence"
                    )
                normalized = mask.to(
                    device=text_world.device,
                    dtype=torch.bool,
                )
                if bool((~normalized.any(dim=1)).any()):
                    raise ValueError("every descriptor mask must select text")
                normalized_masks.append(normalized)
            descriptors = []
            for mask in normalized_masks:
                active = mask.unsqueeze(-1).to(text_world.dtype)
                descriptors.append(
                    (text_world * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
                )
        if descriptors is not None:
            descriptor_queries = (descriptors[0], descriptors[1])
            if (
                image_patch_tokens is not None
                and image_patch_grid is not None
                and self.explicit_object_relation_grounder is not None
            ):
                explicit_slots, explicit_object_attention = (
                    self.explicit_object_relation_grounder(
                        image_patch_tokens,
                        descriptors[0],
                        descriptors[1],
                        patch_height=image_patch_grid[0],
                        patch_width=image_patch_grid[1],
                    )
                )
                explicit_relation_logits = self.explicit_relation_head(
                    explicit_slots[:, 2]
                )
                world = world.clone()
                world[:, 0] = world[:, 0] + explicit_slots[:, 0]
                world[:, 1] = world[:, 1] + explicit_slots[:, 1]
                relation_slot = self.config.omni.object_slots
                world[:, relation_slot] = world[:, relation_slot] + explicit_slots[:, 2]
            elif (
                self.video_object_pair_trajectory_binding is None
                and self.video_descriptor_trajectory_binding is None
            ):
                raise ValueError(
                    "descriptor masks require image grounding or video pair binding"
                )
        if (
            self.visual_teacher_slot_bridge is not None
            and image_patch_tokens is not None
            and image_patch_grid is not None
        ):
            world = world + self.visual_teacher_slot_bridge(
                world,
                image_patch_tokens,
                patch_height=image_patch_grid[0],
                patch_width=image_patch_grid[1],
            )
        if video_object_slots is not None:
            world = world.clone()
            world[:, : self.config.omni.object_slots] = world[
                :, : self.config.omni.object_slots
            ] + self.video_object_to_world(video_object_slots)
        if video_temporal_summary is not None:
            video_world_summary = self.video_temporal_to_world(video_temporal_summary)
            world = world.clone()
            world[:, -1] = world[:, -1] + video_world_summary
        if video_temporal_delta is not None:
            scaled_temporal_delta = (
                self.config.video_explicit_temporal_delta_scale * video_temporal_delta
            )
            video_temporal_delta_world = (
                self.video_temporal_delta_to_world(scaled_temporal_delta)
                if self.video_temporal_delta_to_world is not None
                else torch.nn.functional.linear(
                    scaled_temporal_delta,
                    self.video_temporal_to_world.weight,
                )
            )
            world = world.clone()
            action_slot = SLOT_ROLES.index("action_0")
            world[:, action_slot] = world[:, action_slot] + video_temporal_delta_world
        if audio_world_summary is not None:
            world = world.clone()
            world[:, -1] = world[:, -1] + audio_world_summary
        if audio_temporal_features is not None and (
            self.audio_event_slot_projection is not None
            or self.audio_temporal_head is not None
        ):
            audio_event_slots = torch.nn.functional.adaptive_avg_pool1d(
                audio_temporal_features.transpose(1, 2),
                2,
            ).transpose(1, 2)
        if audio_content_features is not None:
            audio_content_event_slots = torch.nn.functional.adaptive_avg_pool1d(
                audio_content_features.transpose(1, 2),
                2,
            ).transpose(1, 2)
        if (
            audio_event_slots is not None
            and self.audio_event_slot_projection is not None
        ):
            world = world.clone()
            world[:, 26:28] = world[:, 26:28] + self.audio_event_slot_projection(
                audio_event_slots
            )
        if audio_content_event_slots is not None:
            world = world.clone()
            world[:, 26:28] = world[:, 26:28] + self.audio_content_to_world(
                audio_content_event_slots
            )
        for _ in range(self.config.world_rounds):
            world = self.world_cell(
                world,
                src_key_padding_mask=~world_state.active_mask,
            )
        if (
            audio_event_slots is not None
            and self.audio_event_slot_projection is not None
        ):
            world = world.clone()
            world[:, 26:28] = self.audio_event_slot_projection(audio_event_slots)
        if audio_content_event_slots is not None:
            world = world.clone()
            world[:, 26:28] = world[:, 26:28] + self.audio_content_to_world(
                audio_content_event_slots
            )
        cross_modal_evidence: torch.Tensor | None = None
        cross_modal_evidence_active = (
            self.cross_modal_evidence_to_world is not None
            and (audio_summary is not None or video_temporal_summary is not None)
        )
        if cross_modal_evidence_active:
            text_evidence = self.cross_modal_text_projection(text_retrieval_summary)
            audio_evidence = (
                self.cross_modal_audio_projection(
                    audio_content_summary
                    if audio_content_summary is not None
                    else audio_summary
                )
                if audio_summary is not None
                else torch.zeros_like(text_evidence)
            )
            video_evidence = (
                self.cross_modal_video_projection(video_temporal_summary)
                if video_temporal_summary is not None
                else torch.zeros_like(text_evidence)
            )
            if self.config.cross_modal_evidence_direct_features:
                evidence_text_mask = text_source_mask.clone()
                evidence_text_mask[:, 0] = ~evidence_text_mask[:, 1:].any(dim=1)
                evidence_text_tokens = text_world
                if self.config.cross_modal_text_contextual_pooling:
                    context_output = self.text_core(world_input_ids, rounds=1)
                    evidence_text_tokens = self.text_to_world(
                        context_output.context_states
                    )
                    evidence_text_mask = text_mask
                projected_text_tokens = self.cross_modal_text_projection(
                    evidence_text_tokens
                )
                if audio_summary is None and video_temporal_summary is not None:
                    if video_object_slots is not None:
                        video_evidence_tokens = self.video_object_to_world(
                            video_object_slots
                        )
                    if video_evidence_tokens is None:
                        raise RuntimeError("video evidence tokens are unavailable")
                    late_text, late_video = _cross_modal_late_summaries(
                        projected_text_tokens,
                        evidence_text_mask,
                        self.cross_modal_video_projection(video_evidence_tokens),
                    )
                    if self.config.cross_modal_text_contextual_pooling:
                        late_text = _cross_modal_last_summary(
                            projected_text_tokens,
                            evidence_text_mask,
                        )
                    elif self.cross_modal_text_sequence_encoder is not None:
                        late_text = _cross_modal_sequence_summary(
                            projected_text_tokens,
                            evidence_text_mask,
                            self.cross_modal_text_sequence_encoder,
                        )
                    elif self.cross_modal_text_evidence_score is not None:
                        late_text = _cross_modal_query_summary(
                            projected_text_tokens,
                            evidence_text_mask,
                            self.cross_modal_text_evidence_score,
                        )
                    interaction = torch.cat(
                        (
                            text_evidence,
                            late_text,
                            video_evidence,
                            late_video,
                            text_evidence * video_evidence,
                            late_text * late_video,
                        ),
                        dim=-1,
                    )
                elif video_temporal_summary is None and audio_summary is not None:
                    audio_evidence_tokens = (
                        audio_content_features
                        if audio_content_features is not None
                        else audio_temporal_features
                    )
                    if audio_evidence_tokens is None:
                        audio_evidence_tokens = audio
                    late_text, late_audio = _cross_modal_late_summaries(
                        projected_text_tokens,
                        evidence_text_mask,
                        self.cross_modal_audio_projection(audio_evidence_tokens),
                    )
                    if self.config.cross_modal_text_contextual_pooling:
                        late_text = _cross_modal_last_summary(
                            projected_text_tokens,
                            evidence_text_mask,
                        )
                    elif self.cross_modal_text_sequence_encoder is not None:
                        late_text = _cross_modal_sequence_summary(
                            projected_text_tokens,
                            evidence_text_mask,
                            self.cross_modal_text_sequence_encoder,
                        )
                    elif self.cross_modal_text_evidence_score is not None:
                        late_text = _cross_modal_query_summary(
                            projected_text_tokens,
                            evidence_text_mask,
                            self.cross_modal_text_evidence_score,
                        )
                    interaction = torch.cat(
                        (
                            text_evidence,
                            late_text,
                            audio_evidence,
                            late_audio,
                            text_evidence * audio_evidence,
                            late_text * late_audio,
                        ),
                        dim=-1,
                    )
                else:
                    interaction = torch.cat(
                        (
                            text_evidence,
                            audio_evidence,
                            video_evidence,
                            text_evidence * audio_evidence,
                            text_evidence * video_evidence,
                            audio_evidence * video_evidence,
                        ),
                        dim=-1,
                    )
            else:
                text_evidence = torch.nn.functional.normalize(text_evidence, dim=-1)
                audio_evidence = torch.nn.functional.normalize(audio_evidence, dim=-1)
                video_evidence = torch.nn.functional.normalize(video_evidence, dim=-1)
                interaction = torch.cat(
                    (
                        text_evidence * audio_evidence,
                        text_evidence * video_evidence,
                        audio_evidence * video_evidence,
                    ),
                    dim=-1,
                )
            cross_modal_evidence = self.cross_modal_evidence_activation(
                self.cross_modal_evidence_to_world(
                    self.cross_modal_evidence_norm(interaction)
                )
            )
        world = self.world_norm(world)
        cross_modal_evidence_logits = (
            self.cross_modal_evidence_head(cross_modal_evidence)
            if cross_modal_evidence is not None
            else None
        )
        video_decision_state = world[:, -1]
        text_query: torch.Tensor | None = None
        if video_values is not None and self.video_query_conditioning is not None:
            text_active = torch.cat(
                (
                    torch.ones(
                        (batch, 1),
                        dtype=torch.bool,
                        device=text_mask.device,
                    ),
                    text_mask,
                ),
                dim=1,
            )
            text_query = (
                text_world * text_active.unsqueeze(-1).to(text_world.dtype)
            ).sum(dim=1)
            text_query = text_query / text_active.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1)
        if video_temporal_delta_world is not None:
            video_decision_state = video_decision_state + video_temporal_delta_world
        if video_object_slots is not None:
            if video_object_normalized_slots is None:
                decision_input = torch.cat(
                    (
                        world[:, : self.config.omni.object_slots].flatten(1),
                        world[:, -1],
                    ),
                    dim=-1,
                )
                object_decision = self.video_object_decision(decision_input)
            else:
                if text_query is None:
                    raise RuntimeError("dual object evidence requires a text query")
                raw_world_slots = self.video_object_to_world(video_object_slots)
                normalized_world_slots = self.video_object_to_world(
                    video_object_normalized_slots
                )
                if self.video_object_set_decision is None:
                    raw_decision_input = torch.cat(
                        (raw_world_slots.flatten(1), text_query),
                        dim=-1,
                    )
                    normalized_decision_input = torch.cat(
                        (normalized_world_slots.flatten(1), text_query),
                        dim=-1,
                    )
                    raw_decision = self.video_object_decision(raw_decision_input)
                    normalized_decision = self.video_object_decision(
                        normalized_decision_input
                    )
                else:
                    query_slots = text_query.unsqueeze(1).expand(
                        -1,
                        self.config.omni.object_slots,
                        -1,
                    )
                    raw_decision = self.video_object_set_decision(
                        torch.cat((raw_world_slots, query_slots), dim=-1)
                    ).mean(dim=1)
                    normalized_decision = self.video_object_set_decision(
                        torch.cat(
                            (normalized_world_slots, query_slots),
                            dim=-1,
                        )
                    ).mean(dim=1)
                video_object_evidence_weights = torch.softmax(
                    self.video_object_evidence_gate(text_query),
                    dim=-1,
                )
                object_decision = (
                    raw_decision * video_object_evidence_weights[:, :1]
                    + normalized_decision * video_object_evidence_weights[:, 1:]
                )
            video_decision_state = video_decision_state + object_decision
        if video_object_binding_features is not None:
            if text_query is None:
                raise RuntimeError("identity-event binding requires a text query")
            binding_query = text_query.unsqueeze(1).expand(
                -1,
                self.config.omni.object_slots,
                -1,
            )
            binding_decision = self.video_object_binding_decision(
                torch.cat(
                    (video_object_binding_features, binding_query),
                    dim=-1,
                )
            ).mean(dim=1)
            video_decision_state = video_decision_state + binding_decision
        if (
            video_object_trajectory_features is not None
            and self.video_object_trajectory_binding is not None
        ):
            if text_query is None:
                raise RuntimeError("object trajectory binding requires a text query")
            (
                trajectory_decision,
                video_object_trajectory_weights,
            ) = self.video_object_trajectory_binding(
                *video_object_trajectory_features,
                text_query,
            )
            video_decision_state = video_decision_state + trajectory_decision
        if (
            video_object_trajectory_features is not None
            and self.video_object_pair_trajectory_binding is not None
            and descriptor_queries is not None
        ):
            (
                pair_trajectory_decision,
                video_object_pair_trajectory_weights,
            ) = self.video_object_pair_trajectory_binding(
                *video_object_trajectory_features,
                torch.stack(descriptor_queries, dim=1),
            )
            video_decision_state = video_decision_state + pair_trajectory_decision
        if (
            video_dense_object_features is not None
            and self.video_descriptor_trajectory_binding is not None
            and descriptor_queries is not None
        ):
            memory_scales: tuple[torch.Tensor | None, ...] = (None,) * len(
                video_dense_object_features
            )
            if self.config.video_descriptor_object_memory_evidence_routing:
                if (
                    video_object_evidence_weights is None
                    or video_object_evidence_weights.shape[1]
                    != len(video_dense_object_features)
                ):
                    raise RuntimeError(
                        "descriptor memory evidence routing requires matched evidence weights"
                    )
                memory_scale = _normalized_evidence_preference(
                    video_object_evidence_weights,
                    self.config.video_descriptor_object_memory_evidence_routing_margin,
                )
                memory_scales = (memory_scale,) * len(video_dense_object_features)
            descriptor_trajectory_outputs = [
                self.video_descriptor_trajectory_binding(
                    evidence,
                    torch.stack(descriptor_queries, dim=1),
                    text_query,
                    memory_scale,
                )
                for evidence, memory_scale in zip(
                    video_dense_object_features,
                    memory_scales,
                    strict=True,
                )
            ]
            if len(descriptor_trajectory_outputs) == 1:
                (
                    descriptor_trajectory_decision,
                    video_descriptor_trajectory_attention,
                    video_descriptor_visibility_logits,
                    video_descriptor_memory_margin,
                    video_descriptor_memory_reliability_logits,
                ) = descriptor_trajectory_outputs[0]
            else:
                descriptor_trajectory_decision = sum(
                    output[0] for output in descriptor_trajectory_outputs
                ) / len(descriptor_trajectory_outputs)
                video_descriptor_trajectory_attention = sum(
                    output[1] for output in descriptor_trajectory_outputs
                ) / len(descriptor_trajectory_outputs)
                visibility = [
                    output[2]
                    for output in descriptor_trajectory_outputs
                    if output[2] is not None
                ]
                margins = [
                    output[3]
                    for output in descriptor_trajectory_outputs
                    if output[3] is not None
                ]
                reliability_logits = [
                    output[4]
                    for output in descriptor_trajectory_outputs
                    if output[4] is not None
                ]
                if visibility:
                    video_descriptor_visibility_logits = sum(visibility) / len(
                        visibility
                    )
                if margins:
                    video_descriptor_memory_margin = sum(margins) / len(margins)
                if reliability_logits:
                    video_descriptor_memory_reliability_logits = sum(
                        reliability_logits
                    ) / len(reliability_logits)
            video_decision_state = video_decision_state + descriptor_trajectory_decision
        if video_spatial_temporal_moment is not None:
            if self.video_spatial_temporal_moment_gate is not None:
                if text_query is None:
                    raise RuntimeError(
                        "query spatial-temporal moment requires a text query"
                    )
                video_spatial_temporal_moment = video_spatial_temporal_moment * (
                    1.0
                    + torch.tanh(self.video_spatial_temporal_moment_gate(text_query))
                )
            video_decision_state = (
                video_decision_state
                + self.video_spatial_temporal_moment_to_world(
                    video_spatial_temporal_moment
                )
            )
        if video_spatial_temporal_y_moment is not None:
            if text_query is None:
                raise RuntimeError("y spatial-temporal moment requires a text query")
            video_spatial_temporal_y_moment = video_spatial_temporal_y_moment * (
                1.0 + torch.tanh(self.video_spatial_temporal_y_moment_gate(text_query))
            )
            video_decision_state = (
                video_decision_state
                + self.video_spatial_temporal_y_moment_to_world(
                    video_spatial_temporal_y_moment
                )
            )
        if video_values is not None and self.video_query_conditioning is not None:
            query_scale, query_shift = self.video_query_conditioning(text_query).chunk(
                2, dim=-1
            )
            video_decision_state = (
                video_decision_state * (1.0 + torch.tanh(query_scale)) + query_shift
            )
        world_state = replace(world_state, semantic_slots=world)
        world_state.validate(self.config.omni)
        answerability_logits: torch.Tensor | None = None
        answerability_loss: torch.Tensor | None = None
        epistemic_output_active: torch.Tensor | None = None
        if self.text_answerability_head is not None and text_only:
            question_ids = (
                input_ids if question_input_ids is None else question_input_ids
            )
            question_patches = self.text_core._pad_tokens(
                question_ids[:, 1:],
                PAD_ID,
            )
            question_mask = question_patches.ne(PAD_ID).any(dim=-1)
            core_body_cross = self.config.text_answerability_mode in {
                "core-body-cross",
                "core-compact-body-cross",
                "core-projected-compact-body-cross",
                "core-lexical-compact-body-cross",
                "core-lexical-consistency-cross",
            }
            if core_body_cross:
                question_core = self.text_core(
                    question_ids,
                    rounds=text_rounds,
                )
                verifier_world_input_ids = (
                    _compact_body_input_ids(world_input_ids)
                    if self.config.text_answerability_mode
                    in {
                        "core-compact-body-cross",
                        "core-projected-compact-body-cross",
                        "core-lexical-compact-body-cross",
                    }
                    else world_input_ids
                )
                evidence_core = self.text_core(
                    verifier_world_input_ids,
                    rounds=text_rounds,
                )
                if (
                    question_core.context_states is None
                    or evidence_core.context_states is None
                ):
                    raise RuntimeError("text contextual states are unavailable")
                question_states = question_core.context_states
                verifier_text_states = evidence_core.context_states
                verifier_text_patches = self.text_core._pad_tokens(
                    verifier_world_input_ids[:, 1:],
                    PAD_ID,
                )
                verifier_text_mask = verifier_text_patches.ne(PAD_ID).any(dim=-1)
            else:
                question_states = torch.cat(
                    (
                        self.text_core.bos_patch.view(1, 1, -1).expand(
                            batch,
                            -1,
                            -1,
                        ),
                        self.text_core._encode_patches(question_patches),
                    ),
                    dim=1,
                )
                verifier_text_states = text_states
                verifier_text_mask = text_mask
            question_world = self.text_to_world(question_states)
            if self.text_only_to_world is not None:
                question_world = question_world + self.text_only_to_world(
                    question_states
                )
            verifier_text_world = self.text_to_world(verifier_text_states)
            if self.text_only_to_world is not None:
                verifier_text_world = verifier_text_world + self.text_only_to_world(
                    verifier_text_states
                )
            verifier_question_states = (
                question_states
                if self.config.text_answerability_mode
                in {
                    "core-projected-compact-body-cross",
                    "core-lexical-compact-body-cross",
                    "core-lexical-consistency-cross",
                }
                else question_world
            )
            verifier_evidence_states = (
                verifier_text_states
                if self.config.text_answerability_mode
                in {
                    "core-projected-compact-body-cross",
                    "core-lexical-compact-body-cross",
                    "core-lexical-consistency-cross",
                }
                else verifier_text_world
            )
            verifier_extra_features = (
                _byte_ngram_overlap_features(
                    question_ids,
                    (
                        _compact_body_input_ids(world_input_ids)
                        if self.config.text_answerability_mode
                        == "core-lexical-consistency-cross"
                        else verifier_world_input_ids
                    ),
                )
                if self.config.text_answerability_mode
                in {
                    "core-lexical-compact-body-cross",
                    "core-lexical-consistency-cross",
                }
                else None
            )
            question_active = (
                question_mask
                if core_body_cross
                else torch.cat(
                    (
                        torch.ones(
                            (batch, 1),
                            dtype=torch.bool,
                            device=question_ids.device,
                        ),
                        question_mask,
                    ),
                    dim=1,
                )
            )
            evidence_active = (
                verifier_text_mask
                if core_body_cross
                else torch.cat(
                    (
                        torch.ones(
                            (batch, 1),
                            dtype=torch.bool,
                            device=world_input_ids.device,
                        ),
                        text_mask,
                    ),
                    dim=1,
                )
            )
            question_summary = (
                question_world * question_active.unsqueeze(-1).to(question_world.dtype)
            ).sum(dim=1) / question_active.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1)
            evidence_summary = (
                verifier_text_world
                * evidence_active.unsqueeze(-1).to(verifier_text_world.dtype)
            ).sum(dim=1) / evidence_active.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1)
            if self.config.text_answerability_mode in {
                "token-cross",
                "contextual-cross",
                "consistency-cross",
                "body-cross",
                "core-body-cross",
                "core-compact-body-cross",
                "core-projected-compact-body-cross",
                "core-lexical-compact-body-cross",
                "core-lexical-consistency-cross",
            }:
                evidence_title_mask = None
                evidence_body_mask = None
                if self.config.text_answerability_mode in {
                    "consistency-cross",
                    "body-cross",
                    "core-body-cross",
                    "core-compact-body-cross",
                    "core-projected-compact-body-cross",
                    "core-lexical-compact-body-cross",
                    "core-lexical-consistency-cross",
                }:
                    body_patches = (
                        text_patches.eq(ord("\n")).any(dim=-1).cumsum(dim=-1) > 0
                    )
                    evidence_body_mask = (
                        verifier_text_mask
                        if self.config.text_answerability_mode
                        in {
                            "core-compact-body-cross",
                            "core-projected-compact-body-cross",
                            "core-lexical-compact-body-cross",
                        }
                        else body_patches & text_mask
                        if core_body_cross
                        else torch.cat(
                            (
                                torch.zeros(
                                    (batch, 1),
                                    dtype=torch.bool,
                                    device=text_patches.device,
                                ),
                                body_patches & text_mask,
                            ),
                            dim=1,
                        )
                    )
                    evidence_title_mask = evidence_active & ~evidence_body_mask
                    evidence_title_mask[:, 0] = False
                answerability_logits = self.text_answerability_head(
                    verifier_question_states,
                    question_active,
                    verifier_evidence_states,
                    evidence_active,
                    evidence_title_mask,
                    evidence_body_mask,
                    verifier_extra_features,
                )
            else:
                answerability_features = torch.cat(
                    (
                        question_summary,
                        evidence_summary,
                        (question_summary - evidence_summary).abs(),
                        question_summary * evidence_summary,
                        question_active.float().mean(dim=1, keepdim=True),
                        evidence_active.float().mean(dim=1, keepdim=True),
                    ),
                    dim=-1,
                )
                answerability_logits = self.text_answerability_head(
                    answerability_features
                )
            if answerability_labels is not None:
                if answerability_labels.shape != (batch,) or bool(
                    (
                        (answerability_labels < 0)
                        | (
                            answerability_labels
                            >= self.config.text_answerability_classes
                        )
                    ).any()
                ):
                    raise ValueError(
                        "answerability_labels must be [batch] class indices"
                    )
                answerability_loss = torch.nn.functional.cross_entropy(
                    answerability_logits,
                    answerability_labels.to(
                        device=answerability_logits.device,
                        dtype=torch.long,
                    ),
                )
            if answerability_labels is not None:
                epistemic_output_active = answerability_labels.to(
                    device=answerability_logits.device,
                    dtype=torch.long,
                ).ne(self.config.text_epistemic_supported_class)
            else:
                answerability_distribution = (
                    answerability_logits.detach().float().softmax(dim=-1)
                )
                unsupported_probability = (
                    1.0
                    - answerability_distribution[
                        :, self.config.text_epistemic_supported_class
                    ]
                )
                epistemic_output_active = unsupported_probability.ge(
                    self.config.text_epistemic_output_threshold
                )
        text_memory = self.world_to_text_memory(world)
        if text_only and self.text_only_world_to_text_memory is not None:
            text_memory = text_memory + self.text_only_world_to_text_memory(world)
        if text_only and self.text_epistemic_memory is not None:
            if answerability_logits is None:
                raise RuntimeError("epistemic logits are unavailable")
            epistemic_distribution = (
                answerability_logits.detach()
                .float()
                .softmax(dim=-1)
                .to(text_memory.dtype)
            )
            epistemic_memory = self.text_epistemic_memory(epistemic_distribution)
            text_memory = text_memory.clone()
            active_slots = self.config.text_epistemic_memory_slots
            text_memory[:, -active_slots:] = text_memory[
                :, -active_slots:
            ] + epistemic_memory.unsqueeze(1)
        text = self.text_core(
            input_ids,
            targets=targets,
            rounds=text_rounds,
            memory_summary=text_memory,
        )
        if text_only and (
            self.text_only_logit_adapter is not None
            or self.text_only_cross_memory is not None
            or self.text_only_hidden_cross_memory is not None
            or self.text_epistemic_output is not None
        ):
            logits = text.logits
            if self.text_only_logit_adapter is not None:
                logits = logits + self.text_only_logit_adapter(text.logits)
            if self.text_only_cross_memory is not None:
                logits = logits + self.text_only_cross_memory(
                    text.logits,
                    text_states,
                    torch.cat(
                        (
                            torch.ones(
                                (batch, 1),
                                dtype=torch.bool,
                                device=world_input_ids.device,
                            ),
                            text_mask,
                        ),
                        dim=1,
                    ),
                )
            if self.text_only_hidden_cross_memory is not None:
                if text.decoder_states is None:
                    raise RuntimeError("text decoder states are unavailable")
                logits = logits + self.text_only_hidden_cross_memory(
                    text.decoder_states,
                    text_states,
                    torch.cat(
                        (
                            torch.ones(
                                (batch, 1),
                                dtype=torch.bool,
                                device=world_input_ids.device,
                            ),
                            text_mask,
                        ),
                        dim=1,
                    ),
                )
            if self.text_epistemic_output is not None:
                if text.decoder_states is None or epistemic_output_active is None:
                    raise RuntimeError("epistemic decoder state is unavailable")
                logits = logits + self.text_epistemic_output(
                    text.decoder_states,
                    epistemic_output_active,
                )
            labels = self.text_core._labels(
                input_ids,
                targets,
                logits.shape[1],
            )
            target_mask = labels.ne(IGNORE_INDEX)
            text = MosaicTextOutput(
                logits=logits,
                loss=(
                    torch.nn.functional.cross_entropy(
                        logits.reshape(-1, VOCAB_SIZE),
                        labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                    if bool(target_mask.any())
                    else None
                ),
                rounds=text.rounds,
                target_mask=target_mask,
                decoder_states=text.decoder_states,
                context_states=text.context_states,
            )
        video_teacher_embedding = None
        if (
            video_world_summary is not None
            and self.video_teacher_projection is not None
        ):
            video_teacher_embedding = self.video_teacher_projection(video_world_summary)
            if video_temporal_delta_world is not None:
                midpoint = video_teacher_embedding.shape[-1] // 2
                video_teacher_embedding = torch.cat(
                    (
                        video_teacher_embedding[:, :midpoint],
                        torch.nn.functional.linear(
                            video_temporal_delta_world,
                            self.video_teacher_projection.weight[midpoint:],
                        ),
                    ),
                    dim=-1,
                )
        video_order_logits = (
            self.video_order_head(video_decision_state)
            if video_values is not None
            else None
        )
        if video_descriptor_trajectory_attention is not None:
            relation_paths = _object_attention_trajectory(
                video_descriptor_trajectory_attention
            )
            first_path, second_path = relation_paths.unbind(dim=1)
            video_spatial_relation_geometry = torch.cat(
                (
                    first_path,
                    second_path,
                    first_path - second_path,
                    first_path * second_path,
                ),
                dim=-1,
            )
            if (
                self.video_spatial_geometry_reasoner is not None
                and camera_pose_values is not None
            ):
                video_spatial_reasoning = self.video_spatial_geometry_reasoner(
                    video_descriptor_trajectory_attention,
                    camera_pose_values,
                )
        video_spatial_relation_logits = (
            self.video_spatial_relation_head(
                torch.cat(
                    (
                        video_decision_state,
                        video_camera_pose_summary
                        if video_camera_pose_summary is not None
                        else torch.zeros_like(video_decision_state),
                        video_spatial_relation_geometry
                        if video_spatial_relation_geometry is not None
                        else video_decision_state.new_zeros(
                            video_decision_state.shape[0], 48
                        ),
                        video_spatial_reasoning
                        if video_spatial_reasoning is not None
                        else video_decision_state.new_zeros(
                            video_decision_state.shape[0],
                            VideoSpatialGeometryReasoner.output_dim,
                        )
                        if self.video_spatial_geometry_reasoner is not None
                        else video_decision_state.new_zeros(
                            video_decision_state.shape[0], 0
                        ),
                    ),
                    dim=-1,
                )
            )
            if video_values is not None
            and self.video_spatial_relation_head is not None
            else None
        )
        video_egomotion_logits = (
            self.video_egomotion_head(
                torch.cat(
                    (
                        video_decision_state,
                        video_action_summary,
                        video_egomotion_summary,
                    ),
                    dim=-1,
                )
            )
            if video_values is not None
            and video_action_summary is not None
            and video_egomotion_summary is not None
            and self.video_egomotion_head is not None
            else None
        )
        video_egomotion_validity_logits = (
            self.video_egomotion_validity_head(
                torch.cat(
                    (
                        video_egomotion_summary,
                        _video_egomotion_validity_statistics(video_values).to(
                            dtype=video_egomotion_summary.dtype
                        ),
                    ),
                    dim=-1,
                )
            )
            if video_egomotion_summary is not None
            and self.video_egomotion_validity_head is not None
            else None
        )
        video_egomotion_motion_evidence = (
            _video_egomotion_validity_statistics(video_values)[:, 1]
            if video_values is not None and self.config.video_egomotion_evidence_gate
            else None
        )
        video_egomotion_sufficient_mask = (
            video_egomotion_motion_evidence
            > self.config.video_egomotion_minimum_motion_evidence
            if video_egomotion_motion_evidence is not None
            else None
        )
        if video_values is not None and self.video_camera_robustness_gate is not None:
            if video_camera_statistics is None:
                raise RuntimeError("camera robustness statistics are unavailable")
            video_camera_robustness_gate = torch.sigmoid(
                self.video_camera_robustness_gate(video_camera_statistics)
            )
            camera_residual = self.video_camera_robustness_head(
                self.video_order_head[0](video_decision_state)
            )
            video_order_logits = (
                video_order_logits
                + video_camera_robustness_gate * camera_residual
            )
        if self.video_spatial_temporal_logit_head is not None:
            if text_query is None or video_spatial_temporal_features is None:
                raise RuntimeError(
                    "spatial-temporal logit head requires query and xy moments"
                )
            video_order_logits = (
                video_order_logits
                + self.video_spatial_temporal_logit_head(
                    torch.cat(
                        (text_query, video_spatial_temporal_features),
                        dim=-1,
                    )
                )
            )
        if self.video_spatial_temporal_bilinear_head is not None:
            if text_query is None or video_spatial_temporal_features is None:
                raise RuntimeError(
                    "spatial-temporal bilinear head requires query and xy moments"
                )
            spatial_score = (
                self.video_spatial_temporal_bilinear_head(text_query)
                * video_spatial_temporal_features
            ).sum(dim=-1)
            video_order_logits = video_order_logits + torch.stack(
                (-spatial_score, spatial_score),
                dim=-1,
            )
        return MosaicUnifiedOutput(
            text=text,
            world_state=world_state,
            modalities=tuple(modalities),
            video_order_logits=video_order_logits,
            video_object_evidence_weights=video_object_evidence_weights,
            video_object_attention=video_object_attention_output,
            video_object_trajectory_weights=video_object_trajectory_weights,
            video_object_pair_trajectory_weights=(video_object_pair_trajectory_weights),
            video_descriptor_trajectory_attention=(
                video_descriptor_trajectory_attention
            ),
            video_descriptor_visibility_logits=(video_descriptor_visibility_logits),
            video_descriptor_memory_margin=video_descriptor_memory_margin,
            video_descriptor_memory_reliability_logits=(
                video_descriptor_memory_reliability_logits
            ),
            video_camera_robustness_gate=video_camera_robustness_gate,
            video_spatial_relation_logits=video_spatial_relation_logits,
            video_egomotion_logits=video_egomotion_logits,
            video_egomotion_validity_logits=video_egomotion_validity_logits,
            video_egomotion_motion_evidence=video_egomotion_motion_evidence,
            video_egomotion_sufficient_mask=video_egomotion_sufficient_mask,
            audio_temporal_logits=(
                self.audio_temporal_head(
                    audio_event_slots[:, 1] - audio_event_slots[:, 0]
                )
                if (
                    self.audio_temporal_head is not None
                    and audio_event_slots is not None
                )
                else None
            ),
            video_embedding=video_world_summary,
            audio_embedding=(
                audio_content_summary
                if audio_content_summary is not None
                else audio_summary
            ),
            visual_embedding=(
                video_world_summary
                if video_world_summary is not None
                else image_summary
            ),
            text_retrieval_embedding=text_retrieval_summary,
            video_teacher_embedding=video_teacher_embedding,
            audio_teacher_embedding=(
                self.audio_teacher_projection(
                    audio_content_summary
                    if audio_content_summary is not None
                    else audio_summary
                )
                if (
                    audio_summary is not None
                    and self.audio_teacher_projection is not None
                )
                else None
            ),
            audio_teacher_temporal_states=(
                self.audio_teacher_projection(
                    audio_content_features
                    if audio_content_features is not None
                    else audio_temporal_features
                )
                if (
                    (
                        audio_content_features is not None
                        or audio_temporal_features is not None
                    )
                    and self.audio_teacher_projection is not None
                )
                else None
            ),
            audio_world_teacher_embedding=(
                self.audio_teacher_projection(world[:, 26:28].mean(dim=1))
                if (
                    audio_values is not None
                    and self.audio_teacher_projection is not None
                )
                else None
            ),
            audio_ctc_logits=(
                self.audio_ctc_projection(
                    audio_content_features
                    if audio_content_features is not None
                    else audio_temporal_features
                )
                if (
                    (
                        audio_content_features is not None
                        or audio_temporal_features is not None
                    )
                    and self.audio_ctc_projection is not None
                )
                else None
            ),
            audio_grapheme_ctc_logits=(
                self.audio_grapheme_ctc_projection(audio_content_features)
                if (
                    audio_content_features is not None
                    and self.audio_grapheme_ctc_projection is not None
                )
                else None
            ),
            audio_text_retrieval_embedding=(
                self.audio_text_retrieval_projection(
                    audio_content_summary
                    if audio_content_summary is not None
                    else audio_summary
                )
                if (
                    audio_summary is not None
                    and self.audio_text_retrieval_projection is not None
                )
                else None
            ),
            text_audio_retrieval_embedding=(
                self.text_audio_retrieval_projection(
                    (
                        world[:, -1]
                        if (
                            self.config.audio_text_retrieval_text_source
                            == "world_global"
                        )
                        else text_retrieval_summary
                    )
                )
                if self.text_audio_retrieval_projection is not None
                else None
            ),
            cross_modal_evidence_logits=cross_modal_evidence_logits,
            cross_modal_evidence_delta=cross_modal_evidence,
            visual_text_retrieval_embedding=(
                self.visual_text_retrieval_projection(
                    video_world_summary
                    if video_world_summary is not None
                    else image_summary
                )
                if (
                    (video_world_summary is not None or image_summary is not None)
                    and self.visual_text_retrieval_projection is not None
                )
                else None
            ),
            text_visual_retrieval_embedding=(
                self.text_visual_retrieval_projection(text_retrieval_summary)
                if self.text_visual_retrieval_projection is not None
                else None
            ),
            answerability_logits=answerability_logits,
            answerability_loss=answerability_loss,
            explicit_relation_logits=explicit_relation_logits,
            explicit_object_attention=explicit_object_attention,
        )

    def accumulate_long_video(
        self,
        clip_world_states: torch.Tensor,
        clip_mask: torch.Tensor,
        *,
        initial_state: WorldState | None = None,
    ) -> LongVideoWorldOutput:
        if self.long_video_accumulator is None:
            raise RuntimeError("long-video World accumulator is disabled")
        return self.long_video_accumulator(
            clip_world_states,
            clip_mask,
            initial_state=initial_state,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_bytes: int,
        world_input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
        video_values: torch.Tensor | None = None,
        text_rounds: int | None = None,
        use_answerability_gate: bool = False,
    ) -> torch.Tensor:
        if max_new_bytes < 0:
            raise ValueError("max_new_bytes must not be negative")
        if world_input_ids is None:
            world_input_ids = input_ids
        forced_fallback: torch.Tensor | None = None
        forced_fallback_rows: torch.Tensor | None = None
        if use_answerability_gate:
            if self.text_answerability_head is None:
                raise ValueError("answerability gate requested without a trained head")
            probe = self(
                input_ids,
                world_input_ids=world_input_ids,
                pixel_values=pixel_values,
                audio_values=audio_values,
                video_values=video_values,
                question_input_ids=input_ids,
                text_rounds=text_rounds,
            )
            if probe.answerability_logits is None:
                raise RuntimeError("answerability logits are unavailable")
            distribution = probe.answerability_logits.float().softmax(dim=-1)
            if self.config.text_answerability_classes == 2:
                unsupported = distribution[:, 1].lt(
                    self.config.text_answerability_threshold
                )
            else:
                supported_probability = distribution[
                    :, self.config.text_epistemic_supported_class
                ]
                unsupported = (1.0 - supported_probability).ge(
                    self.config.text_epistemic_output_threshold
                )
            if bool(unsupported.any()):
                forced_fallback = torch.tensor(
                    self.config.text_answerability_fallback_bytes,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                forced_fallback_rows = unsupported
        generated = input_ids
        finished = torch.zeros(
            input_ids.shape[0],
            dtype=torch.bool,
            device=input_ids.device,
        )
        for generation_step in range(max_new_bytes):
            candidate = torch.cat(
                (
                    generated,
                    torch.full(
                        (generated.shape[0], 1),
                        PAD_ID,
                        dtype=torch.long,
                        device=generated.device,
                    ),
                ),
                dim=1,
            )
            output = self(
                candidate,
                world_input_ids=world_input_ids,
                pixel_values=pixel_values,
                audio_values=audio_values,
                video_values=video_values,
                question_input_ids=input_ids,
                text_rounds=text_rounds,
            )
            position = generated.shape[1] - 1
            logits = output.logits.flatten(1, 2)[:, position].clone()
            logits[:, PAD_ID] = float("-inf")
            logits[:, BOS_ID] = float("-inf")
            next_ids = logits.argmax(dim=-1)
            if forced_fallback_rows is not None:
                forced_id = (
                    forced_fallback[generation_step]
                    if (
                        forced_fallback is not None
                        and generation_step < forced_fallback.numel()
                    )
                    else torch.tensor(EOS_ID, device=input_ids.device)
                )
                next_ids = torch.where(
                    forced_fallback_rows,
                    forced_id.expand_as(next_ids),
                    next_ids,
                )
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, EOS_ID),
                next_ids,
            )
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            finished |= next_ids.eq(EOS_ID)
            if bool(finished.all()):
                break
        return generated

    def _frontend_input(self, values: torch.Tensor) -> torch.Tensor:
        parameter = self.modality_embedding.weight
        return values.to(device=parameter.device, dtype=parameter.dtype)

    def _validate_image(
        self,
        values: torch.Tensor,
        batch: int,
        name: str,
    ) -> None:
        if (
            values.ndim != 4
            or values.shape[0] != batch
            or values.shape[1] != 3
            or min(values.shape[2:]) < self.config.vision_patch_size
        ):
            raise ValueError(f"{name} must have shape [batch, 3, height, width]")

    @staticmethod
    def _full_mask(tokens: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            tokens.shape[:2],
            dtype=torch.bool,
            device=tokens.device,
        )


class WorldToAnimaConditioning(nn.Module):
    def __init__(self, config: MosaicOmniConfig) -> None:
        super().__init__()
        self.config = config
        self.conditioning_queries = nn.Parameter(
            torch.empty(
                config.anima_conditioning_tokens,
                config.world_dim,
            )
        )
        self.cross_attention = nn.MultiheadAttention(
            config.world_dim,
            config.attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.world_dim),
            nn.Linear(
                config.world_dim,
                config.anima_conditioning_dim,
                bias=False,
            ),
        )
        nn.init.normal_(self.conditioning_queries, mean=0.0, std=0.02)

    def forward(self, world_state: WorldState) -> torch.Tensor:
        world_state.validate(self.config)
        queries = self.conditioning_queries.unsqueeze(0).expand(
            world_state.semantic_slots.shape[0],
            -1,
            -1,
        )
        conditioning, _ = self.cross_attention(
            queries,
            world_state.semantic_slots,
            world_state.semantic_slots,
            key_padding_mask=~world_state.active_mask,
            need_weights=False,
        )
        return self.output(conditioning + queries)


def edit_world_slots(
    state: WorldState,
    slot_indices: list[int],
    updates: torch.Tensor,
    config: MosaicOmniConfig,
    *,
    dirty_regions: tuple[str, ...] = (),
) -> WorldState:
    state.validate(config)
    if not slot_indices:
        raise ValueError("at least one slot must be edited")
    if min(slot_indices) < 0 or max(slot_indices) >= config.world_slots:
        raise ValueError("slot index is outside the world workspace")
    expected = (
        state.semantic_slots.shape[0],
        len(slot_indices),
        config.world_dim,
    )
    if tuple(updates.shape) != expected:
        raise ValueError(f"expected edit updates {expected}, got {updates.shape}")
    indices = torch.tensor(
        slot_indices,
        dtype=torch.long,
        device=state.semantic_slots.device,
    )
    slots = state.semantic_slots.clone()
    slots.index_copy_(1, indices, updates)
    dirty = state.dirty_mask.clone()
    dirty[:, indices] = True
    surface_refs = tuple(
        reference.mark_dirty(*dirty_regions) for reference in state.surface_refs
    )
    edited = WorldState(
        semantic_slots=slots,
        active_mask=state.active_mask.clone(),
        dirty_mask=dirty,
        source=f"{state.source}:edited",
        surface_refs=surface_refs,
    )
    edited.validate(config)
    return edited


def merge_persistent_scene_memory(
    previous: WorldState,
    observation: WorldState,
    visible_object_mask: torch.Tensor,
    config: MosaicOmniConfig,
) -> WorldState:
    previous.validate(config)
    observation.validate(config)
    if tuple(previous.semantic_slots.shape) != tuple(observation.semantic_slots.shape):
        raise ValueError("previous and observed world states must align")
    expected = (previous.semantic_slots.shape[0], config.object_slots)
    if tuple(visible_object_mask.shape) != expected:
        raise ValueError(f"visible_object_mask must have shape {expected}")
    merged = observation.semantic_slots.clone()
    merged[:, : config.object_slots] = torch.where(
        visible_object_mask.unsqueeze(-1),
        observation.semantic_slots[:, : config.object_slots],
        previous.semantic_slots[:, : config.object_slots],
    )
    result = WorldState(
        semantic_slots=merged,
        active_mask=previous.active_mask | observation.active_mask,
        dirty_mask=observation.dirty_mask.clone(),
        source=f"persistent:{observation.source}",
        surface_refs=observation.surface_refs or previous.surface_refs,
    )
    result.validate(config)
    return result


def _asset(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    exists = path.is_file()
    return {
        "path": path.as_posix(),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
    }


def _read_log(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    payload = path.read_bytes()
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16")
    return payload.decode("utf-8", errors="replace")


def inspect_local_contract(
    *,
    gemma_config: Path | None,
    gemma_processor: Path | None,
    raw_image_smoke_log: Path | None,
    anima_q4: Path | None,
    anima_bf16: Path | None,
    anima_vae: Path | None,
    anima_text_encoder: Path | None,
    gemma_q4: Path | None,
) -> dict[str, object]:
    config_data = (
        json.loads(gemma_config.read_text(encoding="utf-8"))
        if gemma_config is not None and gemma_config.is_file()
        else {}
    )
    processor_data = (
        json.loads(gemma_processor.read_text(encoding="utf-8"))
        if gemma_processor is not None and gemma_processor.is_file()
        else {}
    )
    log_text = _read_log(raw_image_smoke_log)
    hidden_match = re.search(r"image=\((\d+),\s*(\d+)\)", log_text)
    assets = {
        "anima_q4": _asset(anima_q4),
        "anima_bf16": _asset(anima_bf16),
        "anima_vae": _asset(anima_vae),
        "anima_text_encoder": _asset(anima_text_encoder),
        "gemma_q4": _asset(gemma_q4),
    }
    if assets["gemma_q4"] is not None:
        assets["gemma_q4"]["usage_boundary"] = (
            "file-size evidence only; not asserted as the valid unified "
            "raw-image runtime"
        )

    def existing_bytes(*names: str) -> int | None:
        selected = [assets[name] for name in names]
        if any(item is None or not item["exists"] for item in selected):
            return None
        return sum(int(item["bytes"]) for item in selected)

    edge_renderer_floor = existing_bytes("anima_q4", "anima_vae")
    current_image_stack_floor = existing_bytes(
        "anima_q4",
        "anima_vae",
        "anima_text_encoder",
    )
    gemma_co_resident_floor = existing_bytes(
        "anima_q4",
        "anima_vae",
        "gemma_q4",
    )
    four_gib = 4 * 1024**3
    return {
        "gemma_unified": {
            "architectures": config_data.get("architectures"),
            "model_type": config_data.get("model_type"),
            "text_hidden_size": (config_data.get("text_config") or {}).get(
                "hidden_size"
            ),
            "processor_class": processor_data.get("processor_class"),
            "image_seq_length": processor_data.get("image_seq_length"),
            "video_input_frames": (processor_data.get("video_processor") or {}).get(
                "num_frames"
            ),
            "audio_seq_length": processor_data.get("audio_seq_length"),
        },
        "raw_image_hidden_smoke": {
            "log": (
                raw_image_smoke_log.as_posix()
                if raw_image_smoke_log is not None
                else None
            ),
            "exists": bool(log_text),
            "hidden_shape": (
                [int(hidden_match.group(1)), int(hidden_match.group(2))]
                if hidden_match
                else None
            ),
            "int2_weight_path_observed": '"quanto_weights": "int2"' in log_text,
            "direct_raw_image_mode_observed": (
                '"cache_mode": "gemma4_12b_direct_raw_image_hidden"' in log_text
            ),
        },
        "assets": assets,
        "weight_file_floors": {
            "edge_renderer_anima_q4_plus_vae_bytes": edge_renderer_floor,
            "current_anima_q4_vae_te_bytes": current_image_stack_floor,
            "anima_q4_vae_plus_gemma_q4_bytes": gemma_co_resident_floor,
            "four_gib_bytes": four_gib,
            "gemma12b_co_resident_fits_4gib_by_files_only": (
                gemma_co_resident_floor <= four_gib
                if gemma_co_resident_floor is not None
                else None
            ),
            "note": (
                "file-size sums exclude activations, allocator overhead, "
                "runtime code, and operating-system memory"
            ),
        },
    }


def run_phase0_probe(
    config: MosaicOmniConfig,
    *,
    repeats: int,
    local_contract: dict[str, object],
) -> dict[str, object]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    torch.manual_seed(73)
    device = torch.device("cpu")
    text_encoder = MosaicTextEncoderProbe(
        MosaicTEConfig(
            patch_size=4,
            max_bytes=128,
            model_dim=64,
            conditioning_dim=config.world_dim,
            attention_heads=4,
            ffn_dim=128,
            local_layers=1,
            slot_count=config.world_slots,
            recurrent_rounds=2,
        )
    ).eval()
    gemma_adapter = ModalToWorldAdapter(
        config.gemma_hidden_dim,
        config,
    ).eval()
    anima_adapter = WorldToAnimaConditioning(config).eval()
    models = (text_encoder, gemma_adapter, anima_adapter)
    parameter_count = sum(
        parameter.numel() for model in models for parameter in model.parameters()
    )
    prompt = ["비 오는 골목에서 우산을 든 캐릭터가 카메라를 향해 걷는다."]
    gemma_hidden = torch.randn(
        1,
        260,
        config.gemma_hidden_dim,
    )
    gemma_mask = torch.ones(1, 260, dtype=torch.bool)

    with torch.inference_mode():
        text_output = text_encoder.encode(prompt, device=device)
        text_state = WorldState(
            semantic_slots=text_output.global_slots,
            active_mask=torch.ones(
                (1, config.world_slots),
                dtype=torch.bool,
            ),
            dirty_mask=torch.zeros(
                (1, config.world_slots),
                dtype=torch.bool,
            ),
            source="mosaic_text",
            surface_refs=(
                SurfaceResidualRef(
                    modality="image",
                    storage_key="probe://surface/image",
                    shape=(1, 4, 32, 32),
                ),
            ),
        )
        text_state.validate(config)
        gemma_state = gemma_adapter(
            gemma_hidden,
            source_mask=gemma_mask,
            source="gemma4_unified_raw_image_activation",
        )
        conditioning = anima_adapter(gemma_state)

        selected = [0, 22]
        updates = text_state.semantic_slots[:, selected] + 1.0
        edited = edit_world_slots(
            text_state,
            selected,
            updates,
            config,
            dirty_regions=("object:0", "lighting"),
        )
        unselected = [
            index for index in range(config.world_slots) if index not in selected
        ]
        edit_locality = bool(
            torch.equal(
                text_state.semantic_slots[:, unselected],
                edited.semantic_slots[:, unselected],
            )
        )

        observation_slots = text_state.semantic_slots.clone()
        observation_slots[:, : config.object_slots] += 0.5
        observation = replace(
            text_state,
            semantic_slots=observation_slots,
            source="next_clip_observation",
        )
        visible = torch.zeros(
            (1, config.object_slots),
            dtype=torch.bool,
        )
        visible[:, 0] = True
        merged = merge_persistent_scene_memory(
            text_state,
            observation,
            visible,
            config,
        )
        offscreen_identity_preserved = bool(
            torch.equal(
                merged.semantic_slots[:, 1 : config.object_slots],
                text_state.semantic_slots[:, 1 : config.object_slots],
            )
        )
        visible_identity_updated = bool(
            torch.equal(
                merged.semantic_slots[:, 0],
                observation.semantic_slots[:, 0],
            )
        )

        for _ in range(3):
            state = gemma_adapter(
                gemma_hidden,
                source_mask=gemma_mask,
                source="gemma4_unified_raw_image_activation",
            )
            output = anima_adapter(state)
        latencies_ms: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            state = gemma_adapter(
                gemma_hidden,
                source_mask=gemma_mask,
                source="gemma4_unified_raw_image_activation",
            )
            output = anima_adapter(state)
            latencies_ms.append((time.perf_counter() - started) * 1_000)

    expected_conditioning = (
        1,
        config.anima_conditioning_tokens,
        config.anima_conditioning_dim,
    )
    checks = {
        "frozen_32_slot_roles": len(SLOT_ROLES) == config.world_slots,
        "text_to_world_shape": (
            tuple(text_state.semantic_slots.shape)
            == (1, config.world_slots, config.world_dim)
        ),
        "gemma_activation_to_world_shape": (
            tuple(gemma_state.semantic_slots.shape)
            == (1, config.world_slots, config.world_dim)
        ),
        "world_to_anima_conditioning_shape": (
            tuple(conditioning.shape) == expected_conditioning
        ),
        "selective_edit_changes_only_requested_slots": edit_locality,
        "surface_residual_dirty_regions_propagate": (
            edited.surface_refs[0].dirty_regions == ("object:0", "lighting")
        ),
        "offscreen_object_identity_is_preserved": (offscreen_identity_preserved),
        "visible_object_identity_is_updated": visible_identity_updated,
        "outputs_are_finite": bool(torch.isfinite(output).all().item()),
    }
    latency_ordered = sorted(latencies_ms)
    p95_index = min(
        len(latency_ordered) - 1,
        math.ceil(len(latency_ordered) * 0.95) - 1,
    )
    local_floors = local_contract.get("weight_file_floors") or {}
    gemma_fits = local_floors.get("gemma12b_co_resident_fits_4gib_by_files_only")
    return {
        "schema_version": "mosaic-omni-phase0-v0",
        "status": "contract_probe_not_generation_quality_validation",
        "config": asdict(config),
        "slot_roles": list(SLOT_ROLES),
        "parameter_count": parameter_count,
        "parameter_storage_fp16_mib": round(
            parameter_count * 2 / 1024**2,
            3,
        ),
        "shapes": {
            "text_world": list(text_state.semantic_slots.shape),
            "gemma_raw_image_activation_input": list(gemma_hidden.shape),
            "gemma_world": list(gemma_state.semantic_slots.shape),
            "anima_conditioning": list(conditioning.shape),
        },
        "latency_ms": {
            "gemma_activation_to_anima_batch_median": round(
                statistics.median(latencies_ms),
                4,
            ),
            "gemma_activation_to_anima_batch_p95": round(
                latency_ordered[p95_index],
                4,
            ),
            "repeats": repeats,
        },
        "process_memory_mib": process_memory(),
        "local_contract": local_contract,
        "checks": checks,
        "phase0_contract_passed": all(checks.values()),
        "feasibility": {
            "world_latent_interface": "passed",
            "gemmanima_conditioning_adapter": "passed_shape_only",
            "selective_edit_state_contract": "passed",
            "persistent_identity_state_contract": "passed",
            "current_gemma4_12b_co_resident_4gb": (
                "failed_by_weight_files_before_runtime"
                if gemma_fits is False
                else "not_measured"
            ),
            "edge_route": (
                "small_cognition_core_or_remote_gemma_then_world_latent; "
                "do_not_co-reside Gemma4 12B with the edge renderer"
            ),
            "image_generation_quality": "not_tested",
            "video_generation": "no_trained_renderer_or_codec",
            "audio_generation": "no_trained_renderer_or_codec",
            "overall": ("phase0_integration_feasible_full_mosaic_omni_unverified"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--gemma-config", type=Path)
    parser.add_argument("--gemma-processor", type=Path)
    parser.add_argument("--raw-image-smoke-log", type=Path)
    parser.add_argument("--anima-q4", type=Path)
    parser.add_argument("--anima-bf16", type=Path)
    parser.add_argument("--anima-vae", type=Path)
    parser.add_argument("--anima-text-encoder", type=Path)
    parser.add_argument("--gemma-q4", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mosaic_omni_phase0.json"),
    )
    args = parser.parse_args()
    local_contract = inspect_local_contract(
        gemma_config=args.gemma_config,
        gemma_processor=args.gemma_processor,
        raw_image_smoke_log=args.raw_image_smoke_log,
        anima_q4=args.anima_q4,
        anima_bf16=args.anima_bf16,
        anima_vae=args.anima_vae,
        anima_text_encoder=args.anima_text_encoder,
        gemma_q4=args.gemma_q4,
    )
    report = run_phase0_probe(
        MosaicOmniConfig(),
        repeats=args.repeats,
        local_contract=local_contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["phase0_contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

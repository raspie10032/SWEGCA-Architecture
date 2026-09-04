from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from tinylm_slicer.mosaic_byte_retriever import encode_texts
from tinylm_slicer.mosaic_resource_profile import process_memory


@dataclass(frozen=True)
class MosaicTEConfig:
    patch_size: int = 4
    max_bytes: int = 256
    model_dim: int = 96
    conditioning_dim: int = 128
    attention_heads: int = 4
    ffn_dim: int = 192
    local_layers: int = 2
    slot_count: int = 8
    recurrent_rounds: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.patch_size,
            self.max_bytes,
            self.model_dim,
            self.conditioning_dim,
            self.attention_heads,
            self.ffn_dim,
            self.local_layers,
            self.slot_count,
            self.recurrent_rounds,
        )
        if min(positive) <= 0:
            raise ValueError("MOSAIC-TE configuration values must be positive")
        if self.max_bytes % self.patch_size:
            raise ValueError("max_bytes must be divisible by patch_size")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")


@dataclass(frozen=True)
class MosaicTEOutput:
    """Conditioning contract for a diffusion cross-attention adapter."""

    sequence_states: torch.Tensor
    global_slots: torch.Tensor
    pooled_state: torch.Tensor
    attention_mask: torch.Tensor
    byte_spans: torch.Tensor


class SharedSlotCell(nn.Module):
    def __init__(self, config: MosaicTEConfig) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            config.model_dim,
            config.attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(config.model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.model_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.model_dim),
        )
        self.output_norm = nn.LayerNorm(config.model_dim)

    def forward(
        self,
        slots: torch.Tensor,
        sequence: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        update, _ = self.cross_attention(
            query=self.cross_norm(slots),
            key=sequence,
            value=sequence,
            key_padding_mask=~sequence_mask,
            need_weights=False,
        )
        slots = slots + update
        return slots + self.feed_forward(self.output_norm(slots))


class MosaicTextEncoderProbe(nn.Module):
    """Small fixed-patch probe for the proposed MOSAIC diffusion-TE contract.

    This proves tensor and information-flow compatibility only. It is not a
    trained replacement for CLIP, T5, or another production text encoder.
    """

    def __init__(self, config: MosaicTEConfig) -> None:
        super().__init__()
        self.config = config
        patch_count = config.max_bytes // config.patch_size
        self.byte_embedding = nn.Embedding(257, config.model_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(
            torch.empty(patch_count, config.model_dim)
        )
        local_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(
            local_layer,
            num_layers=config.local_layers,
            enable_nested_tensor=False,
        )
        self.learned_slots = nn.Parameter(
            torch.empty(config.slot_count, config.model_dim)
        )
        self.slot_cell = SharedSlotCell(config)
        self.sequence_norm = nn.LayerNorm(config.model_dim)
        self.slot_norm = nn.LayerNorm(config.model_dim)
        self.sequence_projection = nn.Linear(
            config.model_dim,
            config.conditioning_dim,
            bias=False,
        )
        self.slot_projection = nn.Linear(
            config.model_dim,
            config.conditioning_dim,
            bias=False,
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.learned_slots, mean=0.0, std=0.02)

    def forward(
        self,
        byte_ids: torch.Tensor,
        byte_lengths: torch.Tensor,
        *,
        rounds: int | None = None,
    ) -> MosaicTEOutput:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, bytes]")
        if byte_lengths.shape != (byte_ids.shape[0],):
            raise ValueError("byte_lengths must have shape [batch]")
        if byte_ids.shape[1] % self.config.patch_size:
            raise ValueError("byte dimension must be divisible by patch_size")
        if (byte_lengths < 0).any() or (byte_lengths > byte_ids.shape[1]).any():
            raise ValueError("byte_lengths must fit the encoded byte dimension")

        active_rounds = self.config.recurrent_rounds if rounds is None else rounds
        if not 1 <= active_rounds <= self.config.recurrent_rounds:
            raise ValueError("rounds must be within the configured recurrent budget")

        batch, byte_count = byte_ids.shape
        patch_count = byte_count // self.config.patch_size
        byte_mask = (
            torch.arange(byte_count, device=byte_ids.device).unsqueeze(0)
            < byte_lengths.unsqueeze(1)
        )
        embedded = self.byte_embedding(byte_ids).reshape(
            batch,
            patch_count,
            self.config.patch_size,
            self.config.model_dim,
        )
        patch_byte_mask = byte_mask.reshape(
            batch,
            patch_count,
            self.config.patch_size,
        )
        patch_mask = patch_byte_mask.any(dim=2)
        # Keep one neutral patch for an empty prompt so attention stays finite.
        patch_mask[:, 0] = True
        patch_sum = (embedded * patch_byte_mask.unsqueeze(-1)).sum(dim=2)
        denominator = patch_byte_mask.sum(dim=2, keepdim=True).clamp_min(1)
        patches = patch_sum / denominator
        patches = patches + self.position_embedding[:patch_count]
        sequence = self.local_encoder(
            patches,
            src_key_padding_mask=~patch_mask,
        )

        slots = self.learned_slots.unsqueeze(0).expand(batch, -1, -1)
        valid_count = patch_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (
            sequence * patch_mask.unsqueeze(-1)
        ).sum(dim=1) / valid_count
        slots = slots + pooled.unsqueeze(1)
        for _ in range(active_rounds):
            slots = self.slot_cell(slots, sequence, patch_mask)

        sequence_states = self.sequence_projection(
            self.sequence_norm(sequence)
        )
        global_slots = self.slot_projection(self.slot_norm(slots))
        pooled_state = global_slots.mean(dim=1)
        starts = (
            torch.arange(patch_count, device=byte_ids.device)
            * self.config.patch_size
        ).expand(batch, -1)
        ends = torch.minimum(
            starts + self.config.patch_size,
            byte_lengths.unsqueeze(1),
        )
        byte_spans = torch.stack((starts, ends), dim=-1)
        byte_spans = byte_spans.masked_fill(~patch_mask.unsqueeze(-1), -1)
        return MosaicTEOutput(
            sequence_states=sequence_states,
            global_slots=global_slots,
            pooled_state=pooled_state,
            attention_mask=patch_mask,
            byte_spans=byte_spans,
        )

    def encode(
        self,
        prompts: list[str],
        *,
        device: torch.device,
        rounds: int | None = None,
    ) -> MosaicTEOutput:
        byte_ids = encode_texts(
            prompts,
            self.config.max_bytes,
            self.config.patch_size,
            device,
        )
        lengths = torch.tensor(
            [
                min(len(prompt.encode("utf-8")), byte_ids.shape[1])
                for prompt in prompts
            ],
            dtype=torch.long,
            device=device,
        )
        return self(byte_ids, lengths, rounds=rounds)


def run_probe(
    config: MosaicTEConfig,
    *,
    device: torch.device,
    repeats: int,
) -> dict[str, object]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    torch.manual_seed(47)
    prompts = [
        "붉은 우산을 든 로봇이 왼쪽의 파란 고양이를 바라본다.",
        "A glass sphere above two small copper cubes.",
        "간판에 정확히 MOSAIC-TE라고 적힌 밤거리",
        "",
    ]
    model = MosaicTextEncoderProbe(config).to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.inference_mode():
        for _ in range(3):
            output = model.encode(prompts, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies_ms: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            output = model.encode(prompts, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - started) * 1_000)

    ordered_spans = True
    for row in range(len(prompts)):
        valid = output.byte_spans[row][output.attention_mask[row]]
        if valid.shape[0] > 1:
            ordered_spans = ordered_spans and bool(
                torch.all(valid[1:, 0] >= valid[:-1, 1]).item()
            )
    checks = {
        "sequence_and_slot_dimensions_match": (
            output.sequence_states.shape[-1]
            == output.global_slots.shape[-1]
            == config.conditioning_dim
        ),
        "fixed_global_slot_count": (
            output.global_slots.shape[1] == config.slot_count
        ),
        "byte_spans_are_ordered": ordered_spans,
        "outputs_are_finite": bool(
            torch.isfinite(output.sequence_states).all().item()
            and torch.isfinite(output.global_slots).all().item()
        ),
    }
    return {
        "status": "interface_probe_not_quality_validation",
        "config": asdict(config),
        "device": str(device),
        "prompt_count": len(prompts),
        "parameter_count": parameter_count,
        "parameter_storage_fp16_mib": round(parameter_count * 2 / (1024**2), 3),
        "shapes": {
            "sequence_states": list(output.sequence_states.shape),
            "global_slots": list(output.global_slots.shape),
            "pooled_state": list(output.pooled_state.shape),
            "attention_mask": list(output.attention_mask.shape),
            "byte_spans": list(output.byte_spans.shape),
        },
        "latency_ms": {
            "batch_median": round(statistics.median(latencies_ms), 4),
            "batch_p95": round(
                sorted(latencies_ms)[
                    min(
                        len(latencies_ms) - 1,
                        math.ceil(len(latencies_ms) * 0.95) - 1,
                    )
                ],
                4,
            ),
            "repeats": repeats,
        },
        "process_memory_mib": process_memory(),
        "checks": checks,
        "interface_probe_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mosaic_te_probe_v0.json"),
    )
    args = parser.parse_args()
    report = run_probe(
        MosaicTEConfig(),
        device=torch.device(args.device),
        repeats=args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

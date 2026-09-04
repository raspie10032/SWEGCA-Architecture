"""Small causal byte-patch LM with shared recurrent depth and external context."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259
IGNORE_INDEX = -100


@dataclass(frozen=True)
class MosaicTextConfig:
    patch_size: int = 8
    byte_embedding_dim: int = 64
    model_dim: int = 128
    attention_heads: int = 4
    ffn_dim: int = 512
    physical_layers: int = 2
    workspace_slots: int = 4
    retriever_dim: int = 128
    operator_basis_count: int = 16
    operator_rank: int = 2
    max_recurrent_depth: int = 4
    maximum_operator_update: float = 0.25
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.patch_size,
            self.byte_embedding_dim,
            self.model_dim,
            self.attention_heads,
            self.ffn_dim,
            self.physical_layers,
            self.workspace_slots,
            self.retriever_dim,
            self.operator_basis_count,
            self.operator_rank,
            self.max_recurrent_depth,
        )
        if min(positive) <= 0:
            raise ValueError("model dimensions and counts must be positive")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if not 0 < self.maximum_operator_update <= 1:
            raise ValueError("maximum_operator_update must be in (0, 1]")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MosaicTextConfig:
        values = dict(values)
        schema = values.pop("schema_version", "mosaic-text-lm-config-v0")
        if schema != "mosaic-text-lm-config-v0":
            raise ValueError(f"unsupported MOSAIC text config schema: {schema}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: Path | str) -> MosaicTextConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mosaic-text-lm-config-v0",
            **asdict(self),
        }


class BytePatchCodec:
    def __init__(self, patch_size: int = 8) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        result = [BOS_ID] if add_bos else []
        result.extend(text.encode("utf-8"))
        if add_eos:
            result.append(EOS_ID)
        return result

    def decode(self, ids: list[int] | torch.Tensor, *, errors: str = "strict") -> str:
        values = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else ids
        raw = bytearray()
        for value in values:
            value = int(value)
            if value == EOS_ID:
                break
            if value in {PAD_ID, BOS_ID}:
                continue
            if not 0 <= value <= 255:
                raise ValueError(f"invalid byte token: {value}")
            raw.append(value)
        return raw.decode("utf-8", errors=errors)

    def pack(self, ids: list[int] | torch.Tensor) -> torch.Tensor:
        values = (
            ids.detach().cpu().to(torch.long).flatten()
            if isinstance(ids, torch.Tensor)
            else torch.tensor(ids, dtype=torch.long)
        )
        if values.numel() == 0:
            raise ValueError("cannot pack an empty token sequence")
        if bool(((values < 0) | (values >= VOCAB_SIZE)).any()):
            raise ValueError("token ids must be in the MOSAIC byte vocabulary")
        padded = math.ceil(values.numel() / self.patch_size) * self.patch_size
        result = torch.full((padded,), PAD_ID, dtype=torch.long)
        result[: values.numel()] = values
        return result.view(-1, self.patch_size)

    def unpack(self, patches: torch.Tensor) -> list[int]:
        if patches.ndim != 2 or patches.shape[1] != self.patch_size:
            raise ValueError("patches must have shape [patches, patch_size]")
        values = patches.detach().cpu().to(torch.long).flatten().tolist()
        while values and values[-1] == PAD_ID:
            values.pop()
        return values


@dataclass(frozen=True)
class MosaicTextOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    rounds: int
    target_mask: torch.Tensor
    decoder_states: torch.Tensor | None = None
    context_states: torch.Tensor | None = None


class SparseOperatorAdapter(nn.Module):
    def __init__(self, config: MosaicTextConfig) -> None:
        super().__init__()
        shape_u = (
            config.operator_basis_count,
            config.model_dim,
            config.operator_rank,
        )
        shape_v = (
            config.operator_basis_count,
            config.operator_rank,
            config.model_dim,
        )
        self.left = nn.Parameter(torch.empty(shape_u))
        self.right = nn.Parameter(torch.empty(shape_v))
        self.maximum_update = config.maximum_operator_update
        nn.init.normal_(self.left, std=0.02)
        nn.init.normal_(self.right, std=0.02)

    def forward(
        self,
        state: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = coefficients.to(
            device=state.device,
            dtype=state.dtype,
        ).clamp(-1.0, 1.0)
        projected = torch.einsum("btd,krd->btkr", state, self.right)
        delta = torch.einsum(
            "btkr,kdr,bk->btd",
            projected,
            self.left,
            coefficients,
        ) / math.sqrt(self.left.shape[-1])
        norm = delta.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (self.maximum_update / norm).clamp(max=1.0).to(delta.dtype)
        return state + delta * scale


class MosaicTextLM(nn.Module):
    def __init__(self, config: MosaicTextConfig) -> None:
        super().__init__()
        self.config = config
        self.byte_embedding = nn.Embedding(VOCAB_SIZE, config.byte_embedding_dim)
        self.patch_projection = nn.Linear(
            config.patch_size * config.byte_embedding_dim,
            config.model_dim,
        )
        self.patch_norm = nn.LayerNorm(config.model_dim)
        self.segment_embedding = nn.Embedding(3, config.model_dim)
        self.workspace = nn.Parameter(
            torch.empty(config.workspace_slots, config.model_dim)
        )
        self.bos_patch = nn.Parameter(torch.empty(config.model_dim))
        self.retriever_projection = nn.Linear(
            config.retriever_dim,
            config.model_dim,
        )
        self.round_embedding = nn.Parameter(
            torch.empty(config.max_recurrent_depth, config.model_dim)
        )
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.ffn_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.physical_layers)
        )
        self.operator_adapter = SparseOperatorAdapter(config)
        self.local_decoder = nn.GRUCell(
            config.byte_embedding_dim,
            config.model_dim,
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, VOCAB_SIZE)
        nn.init.normal_(self.workspace, std=0.02)
        nn.init.normal_(self.bos_patch, std=0.02)
        nn.init.normal_(self.round_embedding, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        rounds: int | None = None,
        memory_ids: torch.Tensor | None = None,
        memory_summary: torch.Tensor | None = None,
        operator_coefficients: torch.Tensor | None = None,
    ) -> MosaicTextOutput:
        self._validate_inputs(input_ids, rounds)
        rounds = 1 if rounds is None else rounds
        batch = input_ids.shape[0]
        raw_targets = input_ids[:, 1:]
        target_patches = self._pad_tokens(raw_targets, PAD_ID)
        target_patch_mask = target_patches.ne(PAD_ID).any(dim=-1)
        encoded_targets = self._encode_patches(target_patches)
        predictor = torch.cat(
            (
                self.bos_patch.view(1, 1, -1).expand(batch, -1, -1),
                encoded_targets[:, :-1],
            ),
            dim=1,
        )

        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        segments: list[torch.Tensor] = []
        if memory_ids is not None:
            self._validate_token_tensor(memory_ids, "memory_ids")
            if memory_ids.shape[0] != batch:
                raise ValueError("memory_ids batch must match input_ids")
            memory_ids = memory_ids.to(device=input_ids.device)
            memory_patches = self._pad_tokens(memory_ids, PAD_ID)
            memory_mask = memory_patches.ne(PAD_ID).any(dim=-1)
            pieces.append(self._encode_patches(memory_patches))
            masks.append(memory_mask)
            segments.append(torch.zeros_like(memory_mask, dtype=torch.long))
        if memory_summary is not None:
            if memory_summary.ndim == 2:
                memory_summary = memory_summary.unsqueeze(1)
            if (
                memory_summary.ndim != 3
                or memory_summary.shape[0] != batch
                or memory_summary.shape[-1] != self.config.retriever_dim
            ):
                raise ValueError(
                    "memory_summary must be [batch, items, retriever_dim]"
                )
            summary = self.retriever_projection(
                memory_summary.to(
                    device=input_ids.device,
                    dtype=self.workspace.dtype,
                )
            )
            summary_mask = torch.ones(
                summary.shape[:2],
                dtype=torch.bool,
                device=summary.device,
            )
            pieces.append(summary)
            masks.append(summary_mask)
            segments.append(torch.zeros_like(summary_mask, dtype=torch.long))

        workspace = self.workspace.view(1, self.config.workspace_slots, -1).expand(
            batch, -1, -1
        )
        workspace_mask = torch.ones(
            (batch, self.config.workspace_slots),
            dtype=torch.bool,
            device=input_ids.device,
        )
        pieces.extend((workspace, predictor))
        masks.extend((workspace_mask, target_patch_mask))
        segments.extend(
            (
                torch.ones_like(workspace_mask, dtype=torch.long),
                torch.full_like(target_patch_mask, 2, dtype=torch.long),
            )
        )
        state = torch.cat(pieces, dim=1)
        valid = torch.cat(masks, dim=1)
        segment_ids = torch.cat(segments, dim=1)
        state = state + self.segment_embedding(segment_ids)
        state = state + _sinusoidal_positions(
            state.shape[1],
            self.config.model_dim,
            device=state.device,
            dtype=state.dtype,
        )
        causal_mask = torch.triu(
            torch.ones(
                state.shape[1],
                state.shape[1],
                dtype=torch.bool,
                device=state.device,
            ),
            diagonal=1,
        )
        if operator_coefficients is not None:
            expected = (batch, self.config.operator_basis_count)
            if operator_coefficients.shape != expected:
                raise ValueError(
                    f"operator_coefficients must have shape {expected}"
                )

        text_start = state.shape[1] - predictor.shape[1]
        for round_index in range(rounds):
            state = state + self.round_embedding[round_index]
            for block in self.blocks:
                state = block(
                    state,
                    src_mask=causal_mask,
                    src_key_padding_mask=~valid,
                )
            if operator_coefficients is not None:
                state = self.operator_adapter(state, operator_coefficients)

        contexts = state[:, text_start:]
        logits, decoder_states = self._decode_patches_with_states(
            contexts,
            target_patches,
        )
        labels = self._labels(input_ids, targets, target_patches.shape[1])
        target_mask = labels.ne(IGNORE_INDEX)
        loss = (
            F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            if bool(target_mask.any())
            else None
        )
        return MosaicTextOutput(
            logits=logits,
            loss=loss,
            rounds=rounds,
            target_mask=target_mask,
            decoder_states=decoder_states,
            context_states=contexts,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_bytes: int,
        rounds: int | None = None,
        memory_ids: torch.Tensor | None = None,
        memory_summary: torch.Tensor | None = None,
        operator_coefficients: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if max_new_bytes < 0:
            raise ValueError("max_new_bytes must not be negative")
        self._validate_inputs(input_ids, rounds)
        generated = input_ids
        finished = torch.zeros(
            input_ids.shape[0],
            dtype=torch.bool,
            device=input_ids.device,
        )
        for _ in range(max_new_bytes):
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
            ignored = torch.full_like(candidate, IGNORE_INDEX)
            output = self(
                candidate,
                targets=ignored,
                rounds=rounds,
                memory_ids=memory_ids,
                memory_summary=memory_summary,
                operator_coefficients=operator_coefficients,
            )
            position = generated.shape[1] - 1
            next_logits = output.logits.flatten(1, 2)[:, position].clone()
            next_logits[:, PAD_ID] = float("-inf")
            next_logits[:, BOS_ID] = float("-inf")
            next_ids = next_logits.argmax(dim=-1)
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

    def _encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        embedded = self.byte_embedding(patches)
        flattened = embedded.flatten(start_dim=2)
        return self.patch_norm(self.patch_projection(flattened))

    def _decode_patches(
        self,
        contexts: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self._decode_patches_with_states(contexts, targets)
        return logits

    def _decode_patches_with_states(
        self,
        contexts: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, patch_count, _ = contexts.shape
        hidden = contexts.reshape(batch * patch_count, -1)
        flat_targets = targets.reshape(batch * patch_count, self.config.patch_size)
        previous = torch.full(
            (batch * patch_count,),
            BOS_ID,
            dtype=torch.long,
            device=targets.device,
        )
        logits = []
        states = []
        for offset in range(self.config.patch_size):
            hidden = self.local_decoder(self.byte_embedding(previous), hidden)
            logits.append(self.lm_head(self.output_norm(hidden)))
            states.append(hidden)
            previous = flat_targets[:, offset]
        return (
            torch.stack(logits, dim=1).view(
                batch,
                patch_count,
                self.config.patch_size,
                VOCAB_SIZE,
            ),
            torch.stack(states, dim=1).view(
                batch,
                patch_count,
                self.config.patch_size,
                self.config.model_dim,
            ),
        )

    def _labels(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        patch_count: int,
    ) -> torch.Tensor:
        raw = (
            input_ids[:, 1:]
            if targets is None
            else targets.to(device=input_ids.device, dtype=torch.long)
        )
        if targets is not None and targets.shape == input_ids.shape:
            raw = targets[:, 1:]
        if raw.ndim != 2 or raw.shape[0] != input_ids.shape[0]:
            raise ValueError("targets must align with input_ids or shifted targets")
        expected = patch_count * self.config.patch_size
        if raw.shape[1] > expected:
            raise ValueError("targets are longer than the input token sequence")
        labels = torch.full(
            (raw.shape[0], expected),
            IGNORE_INDEX,
            dtype=torch.long,
            device=raw.device,
        )
        labels[:, : raw.shape[1]] = raw
        labels[labels == PAD_ID] = IGNORE_INDEX
        return labels.view(raw.shape[0], patch_count, self.config.patch_size)

    def _pad_tokens(self, tokens: torch.Tensor, fill: int) -> torch.Tensor:
        padded = max(
            self.config.patch_size,
            math.ceil(tokens.shape[1] / self.config.patch_size)
            * self.config.patch_size,
        )
        result = torch.full(
            (tokens.shape[0], padded),
            fill,
            dtype=torch.long,
            device=tokens.device,
        )
        result[:, : tokens.shape[1]] = tokens
        return result.view(tokens.shape[0], -1, self.config.patch_size)

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        rounds: int | None,
    ) -> None:
        self._validate_token_tensor(input_ids, "input_ids")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain at least one BOS sequence")
        if bool((input_ids[:, 0] != BOS_ID).any()):
            raise ValueError("every input sequence must begin with BOS_ID")
        resolved = 1 if rounds is None else rounds
        if not isinstance(resolved, int):
            raise TypeError("rounds must be an integer")
        if not 1 <= resolved <= self.config.max_recurrent_depth:
            raise ValueError("rounds exceed configured recurrent depth")

    @staticmethod
    def _validate_token_tensor(tokens: torch.Tensor, name: str) -> None:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError(f"{name} must be a rank-2 torch.long tensor")
        if bool(((tokens < 0) | (tokens >= VOCAB_SIZE)).any()):
            raise ValueError(f"{name} contains ids outside the byte vocabulary")


def _sinusoidal_positions(
    length: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoded = torch.zeros((length, dimension), device=device, dtype=torch.float32)
    encoded[:, 0::2] = torch.sin(position * frequency)
    encoded[:, 1::2] = torch.cos(position * frequency[: encoded[:, 1::2].shape[1]])
    return encoded.to(dtype=dtype).unsqueeze(0)


def profile_model(config: MosaicTextConfig) -> dict[str, Any]:
    with torch.device("meta"):
        model = MosaicTextLM(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "schema_version": "mosaic-text-lm-profile-v0",
        "config": config.to_dict(),
        "parameter_count": parameter_count,
        "raw_weight_mib": {
            "bf16": parameter_count * 2 / 2**20,
            "fp32": parameter_count * 4 / 2**20,
        },
        "recurrent_depth_parameter_invariant": True,
    }

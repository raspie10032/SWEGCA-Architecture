"""Minimal legacy WorldState compatibility types used by SWEGCA.

The original product module also contains model and modality runtime code.  The
standalone architecture retains only the slot-role and WorldState contracts
required by arbitration and evidence accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import torch


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


class _WorldShape(Protocol):
    world_slots: int
    world_dim: int


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

    def validate(self, config: _WorldShape) -> None:
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

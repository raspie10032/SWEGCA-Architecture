"""Contracts and resource estimates for Rozephine Cognitive Kernel v0.2."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping

import torch


BYTES_PER_GIB = 1024**3
DEFAULT_DEPLOYMENT_BUDGET_BYTES = 8 * BYTES_PER_GIB
DEFAULT_STATE_OWNER = "rozephine_cognitive_core_v0_2"


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-compatible values") from exc
    return json.loads(encoded)


class EvidenceKind(StrEnum):
    LEARNED_PREDICTION = "learned_prediction"
    OBSERVED_EVIDENCE = "observed_evidence"
    DETERMINISTIC_SIMULATION = "deterministic_simulation"
    USER_CLAIM = "user_claim"
    EXTERNAL_MODEL_CLAIM = "external_model_claim"


@dataclass(frozen=True)
class EventSource:
    representation: str
    adapter: str
    source_ref: str

    def __post_init__(self) -> None:
        _require_text(self.representation, "representation")
        _require_text(self.adapter, "adapter")
        _require_text(self.source_ref, "source_ref")


@dataclass(frozen=True)
class EvidenceClaim:
    predicate: str
    value: Any
    confidence: float
    subject: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.predicate, "predicate")
        if self.subject is not None:
            _require_text(self.subject, "subject")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        _json_mapping({"value": self.value}, "claim value")


@dataclass(frozen=True)
class CognitiveEvent:
    event_id: str
    event_type: str
    source: EventSource
    claims: tuple[EvidenceClaim, ...]
    evidence_refs: tuple[str, ...]
    evidence_kind: EvidenceKind
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        _require_text(self.event_type, "event_type")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must preserve at least one provenance address")
        if any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError("evidence_refs must not contain empty addresses")
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_kind"] = self.evidence_kind.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CognitiveEvent":
        source = EventSource(**payload["source"])
        claims = tuple(EvidenceClaim(**claim) for claim in payload.get("claims", ()))
        return cls(
            event_id=str(payload["event_id"]),
            event_type=str(payload["event_type"]),
            source=source,
            claims=claims,
            evidence_refs=tuple(str(ref) for ref in payload["evidence_refs"]),
            evidence_kind=EvidenceKind(payload["evidence_kind"]),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class WorldEntity:
    entity_id: str
    entity_type: str
    properties: Mapping[str, Any] = field(default_factory=dict)
    spatial: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.entity_id, "entity_id")
        _require_text(self.entity_type, "entity_type")
        evidence_refs = tuple(str(ref) for ref in self.evidence_refs)
        if any(not ref for ref in evidence_refs):
            raise ValueError("entity evidence_refs must not contain empty addresses")
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(
            self,
            "properties",
            _json_mapping(self.properties, "entity properties"),
        )
        object.__setattr__(self, "spatial", _json_mapping(self.spatial, "spatial"))


@dataclass(frozen=True)
class WorldRelation:
    subject: str
    predicate: str
    object: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.subject, "subject")
        _require_text(self.predicate, "predicate")
        _require_text(self.object, "object")
        object.__setattr__(
            self,
            "properties",
            _json_mapping(self.properties, "relation properties"),
        )


@dataclass(frozen=True)
class StructuredWorldGraph:
    entities: tuple[WorldEntity, ...] = ()
    relations: tuple[WorldRelation, ...] = ()

    def __post_init__(self) -> None:
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity IDs must be unique")
        known = set(entity_ids)
        for relation in self.relations:
            if relation.subject not in known or relation.object not in known:
                raise ValueError("relations must not reference unknown entities")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuredWorldGraph":
        return cls(
            entities=tuple(WorldEntity(**item) for item in payload.get("entities", ())),
            relations=tuple(
                WorldRelation(**item) for item in payload.get("relations", ())
            ),
        )


@dataclass(frozen=True)
class CognitiveKernelConfig:
    semantic_slots: int = 256
    executive_slots: int = 32
    scratch_slots: int = 32
    hidden_dim: int = 2048

    def __post_init__(self) -> None:
        if min(
            self.semantic_slots,
            self.executive_slots,
            self.scratch_slots,
            self.hidden_dim,
        ) <= 0:
            raise ValueError("slot counts and hidden_dim must be positive")

    @property
    def total_slots(self) -> int:
        return self.semantic_slots + self.executive_slots + self.scratch_slots


_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if dtype_name not in _DTYPE_BY_NAME:
        raise ValueError(f"unsupported cognitive state dtype: {tensor.dtype}")
    return {"dtype": dtype_name, "values": tensor.detach().cpu().tolist()}


def _tensor_from_payload(payload: Mapping[str, Any]) -> torch.Tensor:
    dtype_name = str(payload["dtype"])
    try:
        dtype = _DTYPE_BY_NAME[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported cognitive state dtype: {dtype_name}") from exc
    return torch.tensor(payload["values"], dtype=dtype)


@dataclass(frozen=True)
class CognitiveState:
    semantic_slots: torch.Tensor
    executive_slots: torch.Tensor
    scratch_slots: torch.Tensor
    structured_world_graph: StructuredWorldGraph = field(
        default_factory=StructuredWorldGraph
    )
    evidence_refs: tuple[str, ...] = ()
    goal_state: Mapping[str, Any] = field(default_factory=dict)
    value_state: Mapping[str, Any] = field(default_factory=dict)
    self_state: Mapping[str, Any] = field(default_factory=dict)
    owner_id: str = DEFAULT_STATE_OWNER

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        tensors = (self.semantic_slots, self.executive_slots, self.scratch_slots)
        if any(tensor.ndim != 3 for tensor in tensors):
            raise ValueError("cognitive slot tensors must have shape [batch, slots, dim]")
        batch_dims = {(tensor.shape[0], tensor.shape[2]) for tensor in tensors}
        if len(batch_dims) != 1:
            raise ValueError("all cognitive slot tensors must share batch and hidden dims")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("all cognitive slot tensors must share dtype")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all cognitive slot tensors must share device")
        if not all(tensor.is_floating_point() for tensor in tensors):
            raise ValueError("cognitive slot tensors must use floating-point dtypes")
        object.__setattr__(self, "goal_state", _json_mapping(self.goal_state, "goal_state"))
        object.__setattr__(
            self,
            "value_state",
            _json_mapping(self.value_state, "value_state"),
        )
        object.__setattr__(self, "self_state", _json_mapping(self.self_state, "self_state"))

    @property
    def persistent_state_count(self) -> int:
        return 1

    def validate(self, config: CognitiveKernelConfig) -> None:
        expected = (
            (config.semantic_slots, config.hidden_dim),
            (config.executive_slots, config.hidden_dim),
            (config.scratch_slots, config.hidden_dim),
        )
        actual = tuple(
            (tensor.shape[1], tensor.shape[2])
            for tensor in (
                self.semantic_slots,
                self.executive_slots,
                self.scratch_slots,
            )
        )
        if actual != expected:
            raise ValueError(f"expected cognitive slot shapes {expected}, got {actual}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_slots": _tensor_payload(self.semantic_slots),
            "executive_slots": _tensor_payload(self.executive_slots),
            "scratch_slots": _tensor_payload(self.scratch_slots),
            "structured_world_graph": self.structured_world_graph.to_dict(),
            "evidence_refs": list(self.evidence_refs),
            "goal_state": dict(self.goal_state),
            "value_state": dict(self.value_state),
            "self_state": dict(self.self_state),
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CognitiveState":
        return cls(
            semantic_slots=_tensor_from_payload(payload["semantic_slots"]),
            executive_slots=_tensor_from_payload(payload["executive_slots"]),
            scratch_slots=_tensor_from_payload(payload["scratch_slots"]),
            structured_world_graph=StructuredWorldGraph.from_dict(
                payload.get("structured_world_graph", {})
            ),
            evidence_refs=tuple(payload.get("evidence_refs", ())),
            goal_state=payload.get("goal_state", {}),
            value_state=payload.get("value_state", {}),
            self_state=payload.get("self_state", {}),
            owner_id=str(payload.get("owner_id", DEFAULT_STATE_OWNER)),
        )


@dataclass(frozen=True)
class RepresentationRegistryEntry:
    id: str
    type: str
    native_shape: tuple[int | str, ...]
    encoder: str
    decoder: str | None
    bridge_in: str
    bridge_out: str | None
    device_requirements: Mapping[str, Any]
    precision: str
    license: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("id", "type", "encoder", "bridge_in", "precision", "license"):
            _require_text(getattr(self, name), name)
        if not self.native_shape:
            raise ValueError("native_shape must not be empty")
        if not self.capabilities:
            raise ValueError("capabilities must not be empty")
        object.__setattr__(
            self,
            "device_requirements",
            _json_mapping(self.device_requirements, "device_requirements"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepresentationRegistryEntry":
        values = dict(payload)
        values["native_shape"] = tuple(values["native_shape"])
        values["capabilities"] = tuple(values["capabilities"])
        return cls(**values)


@dataclass(frozen=True)
class PhysicalQuery:
    query_id: str
    query_type: str
    origin: Mapping[str, Any]
    scene_ref: str
    requirements: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.query_type, "query_type")
        _require_text(self.scene_ref, "scene_ref")
        object.__setattr__(self, "origin", _json_mapping(self.origin, "origin"))
        object.__setattr__(
            self,
            "requirements",
            _json_mapping(self.requirements, "requirements"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalQuery":
        return cls(**payload)


@dataclass(frozen=True)
class PhysicalEvidence:
    event_id: str
    query_id: str
    evidence_type: str
    claims: tuple[EvidenceClaim, ...]
    backend: str
    scene_ref: str
    deterministic: bool

    def __post_init__(self) -> None:
        for name in ("event_id", "query_id", "evidence_type", "backend", "scene_ref"):
            _require_text(getattr(self, name), name)
        if not self.deterministic:
            raise ValueError("PhysicalEvidence requires a deterministic backend result")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalEvidence":
        values = dict(payload)
        values["claims"] = tuple(EvidenceClaim(**item) for item in values["claims"])
        return cls(**values)


@dataclass(frozen=True)
class WorldBundle:
    bundle_id: str
    entity_ids: tuple[str, ...]
    representations: Mapping[str, tuple[str, ...]]
    entity_relation_metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    license: str

    def __post_init__(self) -> None:
        _require_text(self.bundle_id, "bundle_id")
        _require_text(self.license, "license")
        if not self.entity_ids or len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("entity_ids must be non-empty and unique")
        if not self.representations:
            raise ValueError("representations must not be empty")
        normalized = {
            str(kind): tuple(str(ref) for ref in refs)
            for kind, refs in self.representations.items()
        }
        if any(not kind or not refs or any(not ref for ref in refs) for kind, refs in normalized.items()):
            raise ValueError("representation kinds and references must not be empty")
        object.__setattr__(self, "representations", normalized)
        object.__setattr__(
            self,
            "entity_relation_metadata",
            _json_mapping(self.entity_relation_metadata, "entity_relation_metadata"),
        )
        object.__setattr__(
            self,
            "provenance",
            _json_mapping(self.provenance, "provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorldBundle":
        values = dict(payload)
        values["entity_ids"] = tuple(values["entity_ids"])
        values["representations"] = {
            kind: tuple(refs) for kind, refs in values["representations"].items()
        }
        return cls(**values)


@dataclass(frozen=True)
class RecurrentCoreSpec:
    hidden_dim: int = 2048
    unique_blocks: int = 16
    attention_heads: int = 16
    kv_heads: int = 4
    mlp_hidden_dim: int = 5504
    max_cycles: int = 8
    state: CognitiveKernelConfig = field(default_factory=CognitiveKernelConfig)

    def __post_init__(self) -> None:
        values = (
            self.hidden_dim,
            self.unique_blocks,
            self.attention_heads,
            self.kv_heads,
            self.mlp_hidden_dim,
            self.max_cycles,
        )
        if min(values) <= 0:
            raise ValueError("recurrent core dimensions must be positive")
        if self.hidden_dim != self.state.hidden_dim:
            raise ValueError("core and cognitive state hidden dimensions must match")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")


@dataclass(frozen=True)
class VramEstimateConfig:
    weight_bits: float = 4.0
    activation_bytes: int = 2
    batch_size: int = 1
    activation_multiplier: float = 4.0
    bridge_bytes: int = 512 * 1024**2
    specialist_bytes: int = 512 * 1024**2
    runtime_bytes: int = 1024**3
    reserve_bytes: int = 1024**3
    budget_bytes: int = DEFAULT_DEPLOYMENT_BUDGET_BYTES

    def __post_init__(self) -> None:
        if self.weight_bits <= 0 or self.activation_multiplier <= 0:
            raise ValueError("weight_bits and activation_multiplier must be positive")
        integer_values = (
            self.activation_bytes,
            self.batch_size,
            self.bridge_bytes,
            self.specialist_bytes,
            self.runtime_bytes,
            self.reserve_bytes,
            self.budget_bytes,
        )
        if min(integer_values) < 0 or self.activation_bytes == 0 or self.batch_size == 0:
            raise ValueError("VRAM estimate values must be non-negative with positive batch/dtype")


def estimate_recurrent_core_parameters(spec: RecurrentCoreSpec) -> dict[str, int]:
    head_dim = spec.hidden_dim // spec.attention_heads
    kv_dim = head_dim * spec.kv_heads
    attention_per_block = (
        spec.hidden_dim * spec.hidden_dim * 2
        + spec.hidden_dim * kv_dim * 2
    )
    mlp_per_block = 3 * spec.hidden_dim * spec.mlp_hidden_dim
    norm_per_block = 2 * spec.hidden_dim
    block_parameters = attention_per_block + mlp_per_block + norm_per_block
    state_embeddings = spec.state.total_slots * spec.hidden_dim
    final_norm = spec.hidden_dim
    total = spec.unique_blocks * block_parameters + state_embeddings + final_norm
    return {
        "attention_per_block": attention_per_block,
        "mlp_per_block": mlp_per_block,
        "norm_per_block": norm_per_block,
        "block_parameters": block_parameters,
        "state_embeddings": state_embeddings,
        "final_norm": final_norm,
        "total_parameters": total,
    }


def estimate_inference_vram(
    spec: RecurrentCoreSpec,
    config: VramEstimateConfig = VramEstimateConfig(),
) -> dict[str, int | float | bool]:
    parameters = estimate_recurrent_core_parameters(spec)["total_parameters"]
    weights = math.ceil(parameters * config.weight_bits / 8.0)
    state = (
        config.batch_size
        * spec.state.total_slots
        * spec.hidden_dim
        * config.activation_bytes
    )
    head_dim = spec.hidden_dim // spec.attention_heads
    kv = (
        config.batch_size
        * spec.unique_blocks
        * spec.state.total_slots
        * spec.kv_heads
        * head_dim
        * 2
        * config.activation_bytes
    )
    activations = math.ceil((state + kv) * config.activation_multiplier)
    total = sum(
        (
            weights,
            state,
            kv,
            activations,
            config.bridge_bytes,
            config.specialist_bytes,
            config.runtime_bytes,
            config.reserve_bytes,
        )
    )
    return {
        "parameters": parameters,
        "weight_bytes": weights,
        "state_bytes": state,
        "kv_bytes": kv,
        "activation_staging_bytes": activations,
        "bridge_bytes": config.bridge_bytes,
        "specialist_bytes": config.specialist_bytes,
        "runtime_bytes": config.runtime_bytes,
        "reserve_bytes": config.reserve_bytes,
        "total_bytes": total,
        "budget_bytes": config.budget_bytes,
        "headroom_bytes": config.budget_bytes - total,
        "fits_budget": total <= config.budget_bytes,
        "max_cycles": spec.max_cycles,
    }

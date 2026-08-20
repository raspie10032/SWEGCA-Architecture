"""Role-local slot leasing inside the sole CognitiveState."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Literal

import torch

from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_cognitive_slot_topology import cognitive_slot_topology

SlotPlanStatus = Literal[
    "insert", "refresh", "rotate", "archive_required", "capacity_exhausted"
]
_MANAGER_KEY = "cognitive_slot_manager"


def slot_tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CognitiveSlotLease:
    role: str
    pool: str
    content_id: str
    content_hash: str
    priority: float
    created_step: int
    last_access_step: int
    protected: bool
    evidence_refs: tuple[str, ...]
    archive_ref: str | None = None

    def __post_init__(self) -> None:
        for name in ("role", "pool", "content_id", "content_hash"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if not math.isfinite(self.priority):
            raise ValueError("slot priority must be finite")
        if min(self.created_step, self.last_access_step) < 0:
            raise ValueError("slot lease steps must be nonnegative")
        if self.last_access_step < self.created_step:
            raise ValueError("last access cannot precede lease creation")
        if any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError("slot evidence references must be nonempty")
        if self.archive_ref is not None and not self.archive_ref.strip():
            raise ValueError("archive reference cannot be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CognitiveSlotLease:
        values = dict(payload)
        values["evidence_refs"] = tuple(values.get("evidence_refs", ()))
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SlotAllocationRequest:
    pool: str
    content_id: str
    priority: float
    step: int
    protected: bool = False
    evidence_refs: tuple[str, ...] = ()
    archive_ref: str | None = None
    expected_content_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.pool.strip() or not self.content_id.strip():
            raise ValueError("slot request pool and content ID must be nonempty")
        if not math.isfinite(self.priority) or self.step < 0:
            raise ValueError("slot request priority/step is invalid")
        if any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError("slot request evidence references must be nonempty")
        for value in (self.archive_ref, self.expected_content_hash):
            if value is not None and not value.strip():
                raise ValueError("optional slot request strings cannot be empty")


@dataclass(frozen=True)
class SlotAllocationPlan:
    status: SlotPlanStatus
    pool: str
    role: str | None
    content_id: str
    manager_revision: int
    evicted_lease: CognitiveSlotLease | None = None
    reason: str = ""


def _manager(state: CognitiveState) -> tuple[int, dict[str, CognitiveSlotLease]]:
    payload = state.self_state.get(_MANAGER_KEY, {})
    if not isinstance(payload, dict):
        raise ValueError("cognitive slot manager metadata must be an object")
    revision = payload.get("revision", 0)
    raw_leases = payload.get("leases", [])
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("slot manager revision must be a nonnegative integer")
    if not isinstance(raw_leases, list):
        raise ValueError("slot manager leases must be a list")
    leases = [CognitiveSlotLease.from_dict(item) for item in raw_leases]
    by_role = {lease.role: lease for lease in leases}
    if len(by_role) != len(leases):
        raise ValueError("slot roles must have at most one lease")
    if len({lease.content_id for lease in leases}) != len(leases):
        raise ValueError("slot content IDs must be unique")
    topology = cognitive_slot_topology(state)
    for lease in leases:
        if lease.pool not in topology.pools:
            raise ValueError("lease pool is not registered")
        role_index = topology.roles.index(lease.role)
        if role_index not in topology.pools[lease.pool]:
            raise ValueError("lease role does not belong to its pool")
    return revision, by_role


def read_cognitive_slot_leases(state: CognitiveState) -> tuple[CognitiveSlotLease, ...]:
    _, leases = _manager(state)
    topology = cognitive_slot_topology(state)
    return tuple(
        leases[role] for role in topology.roles if role in leases
    )


def _content_is_referenced(state: CognitiveState, content_id: str) -> bool:
    if any(
        content_id in (relation.subject, relation.object)
        for relation in state.structured_world_graph.relations
    ):
        return True
    exact_metadata = json.dumps(
        {"goal": state.goal_state, "value": state.value_state},
        ensure_ascii=False,
        sort_keys=True,
    )
    return content_id in exact_metadata


def plan_cognitive_slot(
    state: CognitiveState, request: SlotAllocationRequest
) -> SlotAllocationPlan:
    if state.semantic_slots.shape[0] != 1:
        raise ValueError("persistent slot leasing currently requires batch size 1")
    topology = cognitive_slot_topology(state)
    if request.pool not in topology.pools:
        raise ValueError("requested slot pool is not registered")
    revision, by_role = _manager(state)
    pool_roles = tuple(topology.roles[index] for index in topology.pools[request.pool])
    for role in pool_roles:
        lease = by_role.get(role)
        if lease is not None and lease.content_id == request.content_id:
            return SlotAllocationPlan(
                "refresh", request.pool, role, request.content_id, revision
            )
    for role in pool_roles:
        if role not in by_role:
            return SlotAllocationPlan(
                "insert", request.pool, role, request.content_id, revision
            )
    candidates = [
        by_role[role]
        for role in pool_roles
        if not by_role[role].protected
        and not _content_is_referenced(state, by_role[role].content_id)
    ]
    if not candidates:
        return SlotAllocationPlan(
            "capacity_exhausted",
            request.pool,
            None,
            request.content_id,
            revision,
            reason="all_role_local_slots_are_protected_or_referenced",
        )
    candidate = min(
        candidates,
        key=lambda lease: (
            lease.priority,
            lease.last_access_step,
            lease.created_step,
            topology.roles.index(lease.role),
        ),
    )
    if candidate.archive_ref is None:
        return SlotAllocationPlan(
            "archive_required",
            request.pool,
            candidate.role,
            request.content_id,
            revision,
            evicted_lease=candidate,
            reason="eviction_requires_verified_external_archive",
        )
    return SlotAllocationPlan(
        "rotate",
        request.pool,
        candidate.role,
        request.content_id,
        revision,
        evicted_lease=candidate,
    )


def cognitive_slot_value(state: CognitiveState, role: str) -> torch.Tensor:
    """Return a detached copy of one role-addressed slot."""
    topology = cognitive_slot_topology(state)
    index = topology.roles.index(role)
    semantic = state.semantic_slots.shape[1]
    executive = state.executive_slots.shape[1]
    if index < semantic:
        return state.semantic_slots[:, index].detach().clone()
    if index < semantic + executive:
        return state.executive_slots[:, index - semantic].detach().clone()
    return state.scratch_slots[:, index - semantic - executive].detach().clone()


def _replace_slot(
    state: CognitiveState, role: str, value: torch.Tensor
) -> CognitiveState:
    topology = cognitive_slot_topology(state)
    index = topology.roles.index(role)
    semantic = state.semantic_slots.shape[1]
    executive = state.executive_slots.shape[1]
    if value.shape != (1, state.semantic_slots.shape[2]):
        raise ValueError("slot value must have shape [1, hidden_dim]")
    if value.device != state.semantic_slots.device or value.dtype != state.semantic_slots.dtype:
        raise ValueError("slot value must share state dtype and device")
    if index < semantic:
        slots = state.semantic_slots.clone()
        slots[:, index] = value
        return replace(state, semantic_slots=slots)
    if index < semantic + executive:
        slots = state.executive_slots.clone()
        slots[:, index - semantic] = value
        return replace(state, executive_slots=slots)
    slots = state.scratch_slots.clone()
    slots[:, index - semantic - executive] = value
    return replace(state, scratch_slots=slots)


def _replace_manager(
    state: CognitiveState,
    *,
    revision: int,
    leases: dict[str, CognitiveSlotLease],
    last_rotation_step: int | None,
) -> CognitiveState:
    topology = cognitive_slot_topology(state)
    ordered = [asdict(leases[role]) for role in topology.roles if role in leases]
    self_state = dict(state.self_state)
    self_state[_MANAGER_KEY] = {
        "policy_version": "role-local-v1",
        "revision": revision,
        "last_rotation_step": last_rotation_step,
        "leases": ordered,
    }
    return replace(state, self_state=self_state)


def archive_cognitive_slot(
    state: CognitiveState,
    *,
    content_id: str,
    archive_ref: str,
    archived_content_hash: str,
) -> CognitiveState:
    if not archive_ref.strip() or not archived_content_hash.strip():
        raise ValueError("archive receipt fields must be nonempty")
    revision, by_role = _manager(state)
    matching = [lease for lease in by_role.values() if lease.content_id == content_id]
    if len(matching) != 1:
        raise ValueError("archive receipt must match one active slot lease")
    lease = matching[0]
    current_hash = slot_tensor_hash(cognitive_slot_value(state, lease.role))
    if current_hash != lease.content_hash or current_hash != archived_content_hash:
        raise ValueError("archive receipt hash differs from active slot content")
    by_role[lease.role] = replace(lease, archive_ref=archive_ref)
    return _replace_manager(
        state,
        revision=revision + 1,
        leases=by_role,
        last_rotation_step=None,
    )


def protect_cognitive_slot(
    state: CognitiveState, *, content_id: str
) -> CognitiveState:
    """Protect an active lease without creating another persistent state."""
    revision, by_role = _manager(state)
    matching = [lease for lease in by_role.values() if lease.content_id == content_id]
    if len(matching) != 1:
        raise ValueError("protection must match one active slot lease")
    lease = matching[0]
    by_role[lease.role] = replace(lease, protected=True)
    return _replace_manager(
        state,
        revision=revision + 1,
        leases=by_role,
        last_rotation_step=None,
    )


def apply_cognitive_slot_plan(
    state: CognitiveState,
    plan: SlotAllocationPlan,
    request: SlotAllocationRequest,
    value: torch.Tensor,
) -> CognitiveState:
    if plan.status in ("archive_required", "capacity_exhausted") or plan.role is None:
        raise ValueError("non-writable slot plan cannot be applied")
    if plan.pool != request.pool or plan.content_id != request.content_id:
        raise ValueError("slot plan and request differ")
    revision, by_role = _manager(state)
    if revision != plan.manager_revision:
        raise ValueError("slot plan is stale")
    content_hash = slot_tensor_hash(value)
    if (
        request.expected_content_hash is not None
        and request.expected_content_hash != content_hash
    ):
        raise ValueError("restored slot content hash differs from archive")
    old = by_role.get(plan.role)
    if plan.status == "insert" and old is not None:
        raise ValueError("insert plan no longer points to a free role")
    if plan.status == "refresh" and (
        old is None or old.content_id != request.content_id
    ):
        raise ValueError("refresh plan no longer matches its content")
    if plan.status == "rotate" and (
        old is None
        or plan.evicted_lease is None
        or old != plan.evicted_lease
        or old.archive_ref is None
    ):
        raise ValueError("rotation plan lacks the verified evicted lease")
    refreshing = plan.status == "refresh" and old is not None
    created_step = old.created_step if refreshing else request.step
    evidence_refs = (
        request.evidence_refs or old.evidence_refs
        if refreshing
        else request.evidence_refs
    )
    preserved_archive_ref = (
        old.archive_ref if refreshing and old.content_hash == content_hash else None
    )
    by_role[plan.role] = CognitiveSlotLease(
        role=plan.role,
        pool=request.pool,
        content_id=request.content_id,
        content_hash=content_hash,
        priority=request.priority,
        created_step=created_step,
        last_access_step=request.step,
        protected=request.protected or bool(refreshing and old.protected),
        evidence_refs=evidence_refs,
        archive_ref=request.archive_ref or preserved_archive_ref,
    )
    updated = _replace_slot(state, plan.role, value)
    return _replace_manager(
        updated,
        revision=revision + 1,
        leases=by_role,
        last_rotation_step=request.step if plan.status == "rotate" else None,
    )

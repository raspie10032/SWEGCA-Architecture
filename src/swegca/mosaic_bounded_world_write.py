"""Transactional bounded writes to the sole CognitiveState verification slot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

import torch

from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_cognitive_slot_memory import (
    cognitive_slot_value,
    slot_tensor_hash,
)
from swegca.mosaic_cognitive_slot_topology import cognitive_slot_topology
from swegca.mosaic_synapse_arbiter import SingleWorldArbiter, SynapseProposal
from swegca.mosaic_evidence_accumulator import (
    AccumulatorDecision,
    is_authoritative_accumulator_decision,
)
from swegca.mosaic_evidence_revision import (
    EvidenceRevisionVerification,
    is_authoritative_evidence_revision,
)

EvidenceStatus = Literal["accept", "reject", "abstain"]
_WRITE_KEY = "bounded_verification_write"
_WORLD_WRITE_AUTHORITY = object()


@dataclass(frozen=True)
class BoundedWorldWriteConfig:
    minimum_causal_lower_bound: float = 0.55
    minimum_source_diversity: int = 2
    minimum_context_diversity: int = 4
    maximum_slot_delta: float = 0.02
    minimum_proposal_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "minimum_causal_lower_bound",
            "maximum_slot_delta",
            "minimum_proposal_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite within [0, 1]")
        if self.maximum_slot_delta <= 0:
            raise ValueError("maximum slot delta must be positive")
        if min(self.minimum_source_diversity, self.minimum_context_diversity) <= 0:
            raise ValueError("diversity minima must be positive")


@dataclass(frozen=True)
class WorldWriteGates:
    evidence_status: EvidenceStatus
    causal_lower_bound: float
    source_diversity: int
    context_diversity: int
    definitions_complete: bool
    counterfactual_support: bool
    intervention_support: bool
    regime_change_suspected: bool
    slot_gate_passed: bool
    device_gate_passed: bool
    capacity_strategy_safe: bool
    evidence_current: bool
    accumulator_revision_current: bool
    runtime_context_safe: bool
    _authority_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _authority_digest: str = field(
        default="",
        repr=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.causal_lower_bound) or not (
            0 <= self.causal_lower_bound <= 1
        ):
            raise ValueError("causal lower bound must be finite within [0, 1]")
        if min(self.source_diversity, self.context_diversity) < 0:
            raise ValueError("diversity counts must be nonnegative")


def world_write_gates_from_decision(
    decision: AccumulatorDecision,
    *,
    definitions_complete: bool,
    counterfactual_support: bool,
    intervention_support: bool,
    regime_change_suspected: bool,
    slot_gate_passed: bool,
    device_gate_passed: bool,
    capacity_strategy_safe: bool,
    evidence_current: bool,
    accumulator_revision_current: bool,
    runtime_context_safe: bool,
    revision_verification: EvidenceRevisionVerification | None = None,
) -> WorldWriteGates:
    authoritative = is_authoritative_accumulator_decision(decision)
    if not authoritative and revision_verification is not None:
        authoritative = (
            is_authoritative_evidence_revision(revision_verification)
            and revision_verification.evidence_current
            and revision_verification.accumulator_revision_current
            and dict(revision_verification.decision_payload)
            == {
                "status": decision.status,
                "reason": decision.reason,
                "posterior_mean": decision.posterior_mean,
                "causal_lower_bound": decision.causal_lower_bound,
                "overall_upper_bound": decision.overall_upper_bound,
                "effective_sample_size": decision.effective_sample_size,
                "source_diversity": decision.source_diversity,
                "context_diversity": decision.context_diversity,
                "regime_change_score": decision.regime_change_score,
                "revision": decision.revision,
            }
        )
    if not authoritative:
        raise PermissionError("World write evidence is not an authority capability")
    gates = WorldWriteGates(
        evidence_status=decision.status,
        causal_lower_bound=decision.causal_lower_bound,
        source_diversity=decision.source_diversity,
        context_diversity=decision.context_diversity,
        definitions_complete=definitions_complete,
        counterfactual_support=counterfactual_support,
        intervention_support=intervention_support,
        regime_change_suspected=regime_change_suspected,
        slot_gate_passed=slot_gate_passed,
        device_gate_passed=device_gate_passed,
        capacity_strategy_safe=capacity_strategy_safe,
        evidence_current=evidence_current,
        accumulator_revision_current=accumulator_revision_current,
        runtime_context_safe=runtime_context_safe,
        _authority_token=_WORLD_WRITE_AUTHORITY,
    )
    object.__setattr__(gates, "_authority_digest", _world_write_gate_digest(gates))
    return gates


def _world_write_gate_digest(gates: WorldWriteGates) -> str:
    payload = {
        "evidence_status": gates.evidence_status,
        "causal_lower_bound": gates.causal_lower_bound,
        "source_diversity": gates.source_diversity,
        "context_diversity": gates.context_diversity,
        "definitions_complete": gates.definitions_complete,
        "counterfactual_support": gates.counterfactual_support,
        "intervention_support": gates.intervention_support,
        "regime_change_suspected": gates.regime_change_suspected,
        "slot_gate_passed": gates.slot_gate_passed,
        "device_gate_passed": gates.device_gate_passed,
        "capacity_strategy_safe": gates.capacity_strategy_safe,
        "evidence_current": gates.evidence_current,
        "accumulator_revision_current": gates.accumulator_revision_current,
        "runtime_context_safe": gates.runtime_context_safe,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class BoundedWorldWriteReceipt:
    receipt_id: str
    revision: int
    target_role: str
    before_state_hash: str
    after_state_hash: str
    before_slot: torch.Tensor
    before_slot_hash: str
    after_slot_hash: str
    applied_delta_hash: str
    evidence_refs: tuple[str, ...]
    prior_write_metadata: Mapping[str, Any] | None


@dataclass(frozen=True)
class BoundedWorldWriteResult:
    state: CognitiveState
    authorized: bool
    committed: bool
    reason: str
    proposed_delta: torch.Tensor
    receipt: BoundedWorldWriteReceipt | None = None


def bounded_world_write_receipt_to_dict(
    receipt: BoundedWorldWriteReceipt,
) -> dict[str, Any]:
    dtype = str(receipt.before_slot.dtype).removeprefix("torch.")
    if dtype not in {"float16", "float32", "float64", "bfloat16"}:
        raise ValueError("unsupported receipt tensor dtype")
    return {
        "receipt_id": receipt.receipt_id,
        "revision": receipt.revision,
        "target_role": receipt.target_role,
        "before_state_hash": receipt.before_state_hash,
        "after_state_hash": receipt.after_state_hash,
        "before_slot": {
            "dtype": dtype,
            "values": receipt.before_slot.detach().cpu().tolist(),
        },
        "before_slot_hash": receipt.before_slot_hash,
        "after_slot_hash": receipt.after_slot_hash,
        "applied_delta_hash": receipt.applied_delta_hash,
        "evidence_refs": list(receipt.evidence_refs),
        "prior_write_metadata": (
            dict(receipt.prior_write_metadata)
            if receipt.prior_write_metadata is not None
            else None
        ),
    }


def bounded_world_write_receipt_from_dict(
    payload: Mapping[str, Any],
) -> BoundedWorldWriteReceipt:
    tensor_payload = payload["before_slot"]
    if not isinstance(tensor_payload, Mapping):
        raise ValueError("receipt before slot must be an object")
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    dtype_name = str(tensor_payload["dtype"])
    if dtype_name not in dtypes:
        raise ValueError("unsupported receipt tensor dtype")
    before_slot = torch.tensor(tensor_payload["values"], dtype=dtypes[dtype_name])
    prior = payload.get("prior_write_metadata")
    if prior is not None and not isinstance(prior, Mapping):
        raise ValueError("prior write metadata must be an object or null")
    return BoundedWorldWriteReceipt(
        receipt_id=str(payload["receipt_id"]),
        revision=int(payload["revision"]),
        target_role=str(payload["target_role"]),
        before_state_hash=str(payload["before_state_hash"]),
        after_state_hash=str(payload["after_state_hash"]),
        before_slot=before_slot,
        before_slot_hash=str(payload["before_slot_hash"]),
        after_slot_hash=str(payload["after_slot_hash"]),
        applied_delta_hash=str(payload["applied_delta_hash"]),
        evidence_refs=tuple(payload.get("evidence_refs", ())),
        prior_write_metadata=dict(prior) if prior is not None else None,
    )


def cognitive_state_hash(state: CognitiveState) -> str:
    digest = hashlib.sha256()
    for tensor in (
        state.semantic_slots,
        state.executive_slots,
        state.scratch_slots,
    ):
        digest.update(slot_tensor_hash(tensor).encode("ascii"))
    metadata = {
        "structured_world_graph": state.structured_world_graph.to_dict(),
        "evidence_refs": list(state.evidence_refs),
        "goal_state": dict(state.goal_state),
        "value_state": dict(state.value_state),
        "self_state": dict(state.self_state),
        "owner_id": state.owner_id,
    }
    digest.update(
        json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _authorization_reason(
    gates: WorldWriteGates, config: BoundedWorldWriteConfig
) -> str | None:
    checks = (
        (gates.evidence_status == "accept", "evidence_not_accepted"),
        (gates.evidence_current, "evidence_expired"),
        (gates.accumulator_revision_current, "accumulator_revision_stale"),
        (gates.runtime_context_safe, "runtime_context_unsafe"),
        (
            gates.causal_lower_bound >= config.minimum_causal_lower_bound,
            "causal_lower_bound",
        ),
        (
            gates.source_diversity >= config.minimum_source_diversity,
            "source_diversity",
        ),
        (
            gates.context_diversity >= config.minimum_context_diversity,
            "context_diversity",
        ),
        (gates.definitions_complete, "definitions_incomplete"),
        (gates.counterfactual_support, "counterfactual_support"),
        (gates.intervention_support, "intervention_support"),
        (not gates.regime_change_suspected, "regime_change_suspected"),
        (gates.slot_gate_passed, "slot_gate"),
        (gates.device_gate_passed, "device_gate"),
        (gates.capacity_strategy_safe, "capacity_strategy"),
    )
    return next((reason for passed, reason in checks if not passed), None)


def _replace_verification_slot(
    state: CognitiveState, value: torch.Tensor
) -> CognitiveState:
    topology = cognitive_slot_topology(state)
    global_index = topology.fixed_roles["verification"]
    semantic = state.semantic_slots.shape[1]
    executive = state.executive_slots.shape[1]
    if global_index < semantic + executive:
        raise ValueError("verification role must remain in the scratch partition")
    local_index = global_index - semantic - executive
    scratch = state.scratch_slots.clone()
    scratch[:, local_index] = value
    return replace(state, scratch_slots=scratch)


def bounded_verification_write(
    state: CognitiveState,
    proposal: SynapseProposal,
    gates: WorldWriteGates,
    config: BoundedWorldWriteConfig,
    *,
    commit: bool,
) -> BoundedWorldWriteResult:
    """Authorize and optionally commit one rollback-safe verification delta."""
    if state.persistent_state_count != 1 or state.semantic_slots.shape[0] != 1:
        raise ValueError("bounded World write requires one batch-one CognitiveState")
    proposal.validate(state)
    topology = cognitive_slot_topology(state)
    verification_index = topology.fixed_roles["verification"]
    targeted = proposal.target_slot_mask.nonzero(as_tuple=False)
    if targeted.shape[0] != 1 or tuple(targeted[0].tolist()) != (
        0,
        verification_index,
    ):
        raise ValueError("bounded write proposal must target only verification")
    reason = _authorization_reason(gates, config)
    arbiter = SingleWorldArbiter(
        maximum_slot_delta=config.maximum_slot_delta,
        maximum_world_delta=config.maximum_slot_delta,
        minimum_weight=config.minimum_proposal_weight,
    )
    arbitration = arbiter(state, (proposal,), commit=False)
    proposed_delta = arbitration.proposed_delta
    if reason is None and not bool(arbitration.accepted.all()):
        reason = "proposal_weight"
    if reason is not None:
        return BoundedWorldWriteResult(
            state, False, False, reason, proposed_delta
        )
    if not commit:
        return BoundedWorldWriteResult(
            state, True, False, "authorized_dry_run", proposed_delta
        )
    if (
        gates._authority_token is not _WORLD_WRITE_AUTHORITY
        or gates._authority_digest != _world_write_gate_digest(gates)
    ):
        raise PermissionError("World write gates are not an authority capability")

    before_hash = cognitive_state_hash(state)
    before_slot = cognitive_slot_value(state, "verification")
    local_delta = proposed_delta[:, verification_index]
    after_slot = before_slot + local_delta
    updated = _replace_verification_slot(state, after_slot)
    prior = state.self_state.get(_WRITE_KEY)
    if prior is not None and not isinstance(prior, dict):
        raise ValueError("bounded write metadata must be an object")
    revision = int(prior.get("revision", 0)) + 1 if prior else 1
    receipt_seed = {
        "before_state_hash": before_hash,
        "delta_hash": slot_tensor_hash(local_delta),
        "revision": revision,
        "evidence_refs": [item for batch in proposal.evidence_addresses for item in batch],
    }
    receipt_id = hashlib.sha256(
        json.dumps(receipt_seed, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    self_state = dict(updated.self_state)
    self_state[_WRITE_KEY] = {
        "policy_version": "bounded-verification-v1",
        "receipt_id": receipt_id,
        "revision": revision,
        "target_role": "verification",
        "evidence_refs": receipt_seed["evidence_refs"],
    }
    updated = replace(updated, self_state=self_state)
    after_hash = cognitive_state_hash(updated)
    receipt = BoundedWorldWriteReceipt(
        receipt_id=receipt_id,
        revision=revision,
        target_role="verification",
        before_state_hash=before_hash,
        after_state_hash=after_hash,
        before_slot=before_slot,
        before_slot_hash=slot_tensor_hash(before_slot),
        after_slot_hash=slot_tensor_hash(after_slot),
        applied_delta_hash=slot_tensor_hash(local_delta),
        evidence_refs=tuple(receipt_seed["evidence_refs"]),
        prior_write_metadata=dict(prior) if prior else None,
    )
    return BoundedWorldWriteResult(
        updated, True, True, "committed", proposed_delta, receipt
    )


def rollback_bounded_verification_write(
    state: CognitiveState, receipt: BoundedWorldWriteReceipt
) -> CognitiveState:
    if receipt.target_role != "verification":
        raise ValueError("receipt target role is invalid")
    if cognitive_state_hash(state) != receipt.after_state_hash:
        raise ValueError("state changed after bounded write; rollback is stale")
    if slot_tensor_hash(cognitive_slot_value(state, "verification")) != receipt.after_slot_hash:
        raise ValueError("verification slot differs from receipt")
    restored = _replace_verification_slot(state, receipt.before_slot)
    self_state = dict(restored.self_state)
    if receipt.prior_write_metadata is None:
        self_state.pop(_WRITE_KEY, None)
    else:
        self_state[_WRITE_KEY] = dict(receipt.prior_write_metadata)
    restored = replace(restored, self_state=self_state)
    if cognitive_state_hash(restored) != receipt.before_state_hash:
        raise ValueError("bounded write rollback was not bit-exact")
    return restored


def validate_bounded_verification_retraction(
    state: CognitiveState, receipt: BoundedWorldWriteReceipt
) -> None:
    """Verify that a receipt still owns the current verification-slot head."""
    if receipt.target_role != "verification":
        raise ValueError("receipt target role is invalid")
    current = state.self_state.get(_WRITE_KEY)
    if not isinstance(current, dict):
        raise ValueError("bounded write receipt is no longer active")
    if (
        current.get("receipt_id") != receipt.receipt_id
        or current.get("revision") != receipt.revision
        or current.get("target_role") != receipt.target_role
    ):
        raise ValueError("bounded write receipt is not the current LIFO head")
    if (
        slot_tensor_hash(cognitive_slot_value(state, "verification"))
        != receipt.after_slot_hash
    ):
        raise ValueError("verification slot has a newer or unrelated value")


def retract_bounded_verification_write(
    state: CognitiveState, receipt: BoundedWorldWriteReceipt
) -> CognitiveState:
    """Retract the current write while preserving later unrelated cognition.

    Unlike strict rollback, this operation intentionally preserves goal, value,
    graph, lease, and autonomy progress. It is valid only while the receipt still
    owns the latest verification-slot value.
    """
    validate_bounded_verification_retraction(state, receipt)
    current = cognitive_slot_value(state, "verification")
    before_slot = receipt.before_slot.to(device=current.device, dtype=current.dtype)
    if slot_tensor_hash(before_slot) != receipt.before_slot_hash:
        raise ValueError("receipt before slot differs from its content hash")
    restored = _replace_verification_slot(state, before_slot)
    self_state = dict(restored.self_state)
    if receipt.prior_write_metadata is None:
        self_state.pop(_WRITE_KEY, None)
    else:
        self_state[_WRITE_KEY] = dict(receipt.prior_write_metadata)
    return replace(restored, self_state=self_state)

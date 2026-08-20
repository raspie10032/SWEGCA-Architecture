from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from swegca.mosaic_bounded_world_write import (
    BoundedWorldWriteConfig,
    WorldWriteGates,
    bounded_verification_write,
    bounded_world_write_receipt_from_dict,
    bounded_world_write_receipt_to_dict,
    cognitive_state_hash,
    retract_bounded_verification_write,
    rollback_bounded_verification_write,
    world_write_gates_from_decision,
)
from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_evidence_accumulator import (
    EvidenceAccumulatorConfig,
    EvidenceAccumulatorState,
    EvidenceObservation,
    assess_accumulator,
    update_accumulator,
)
from swegca.mosaic_synapse_arbiter import evidence_delta_proposal


def _state() -> CognitiveState:
    return CognitiveState(
        semantic_slots=torch.zeros(1, 20, 32),
        executive_slots=torch.zeros(1, 6, 32),
        scratch_slots=torch.zeros(1, 6, 32),
    )


def _proposal(state: CognitiveState, *, target: str = "verification"):
    return evidence_delta_proposal(
        state,
        torch.ones(1, 32),
        torch.tensor([[8.0, -8.0]]),
        source="controlled-evidence",
        evidence_addresses=(("evidence:1", "counterfactual:1"),),
        target_slot=target,
    )


def _decision(status: str = "accept"):
    config = EvidenceAccumulatorConfig()
    state = EvidenceAccumulatorState.empty("world-write-test", config)
    if status == "abstain":
        return assess_accumulator(state, config)
    for index in range(16):
        state = update_accumulator(
            state,
            EvidenceObservation(
                hypothesis_id=state.hypothesis_id,
                evidence_address=f"world-write-evidence:{index}",
                source_family=f"world-write-source:{index % 2}",
                context_hash=f"world-write-context:{index}",
                axis=config.required_axes[index % len(config.required_axes)],
                outcome="support",
                observed_at=index,
                producer_id=f"world-write-producer:{index}",
            ),
            config,
            current_step=index,
        ).state
    return assess_accumulator(state, config)


def _gates(*, status: str = "accept", **changes: bool) -> WorldWriteGates:
    values = {
        "definitions_complete": True,
        "counterfactual_support": True,
        "intervention_support": True,
        "regime_change_suspected": False,
        "slot_gate_passed": True,
        "device_gate_passed": True,
        "capacity_strategy_safe": True,
        "evidence_current": True,
        "accumulator_revision_current": True,
        "runtime_context_safe": True,
    }
    values.update(changes)
    return world_write_gates_from_decision(
        _decision(status),
        **values,
    )


def test_bounded_commit_and_rollback_are_exact() -> None:
    state = _state()
    before = cognitive_state_hash(state)
    result = bounded_verification_write(
        state, _proposal(state), _gates(), BoundedWorldWriteConfig(), commit=True
    )
    assert result.authorized and result.committed and result.receipt is not None
    assert result.state.persistent_state_count == 1
    assert cognitive_state_hash(result.state) != before
    delta_norm = torch.linalg.vector_norm(result.proposed_delta[:, 30], dim=-1)
    assert bool((delta_norm <= 0.02 + 1e-7).all())
    restored = rollback_bounded_verification_write(result.state, result.receipt)
    assert cognitive_state_hash(restored) == before
    assert torch.equal(restored.scratch_slots, state.scratch_slots)


def test_receipt_json_roundtrip_retains_exact_rollback() -> None:
    state = _state()
    result = bounded_verification_write(
        state, _proposal(state), _gates(), BoundedWorldWriteConfig(), commit=True
    )
    assert result.receipt is not None
    restored_receipt = bounded_world_write_receipt_from_dict(
        bounded_world_write_receipt_to_dict(result.receipt)
    )
    restored = rollback_bounded_verification_write(result.state, restored_receipt)
    assert cognitive_state_hash(restored) == cognitive_state_hash(state)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"evidence_status": "abstain"}, "evidence_not_accepted"),
        ({"definitions_complete": False}, "definitions_incomplete"),
        ({"regime_change_suspected": True}, "regime_change_suspected"),
        ({"device_gate_passed": False}, "device_gate"),
        ({"evidence_current": False}, "evidence_expired"),
        ({"accumulator_revision_current": False}, "accumulator_revision_stale"),
        ({"runtime_context_safe": False}, "runtime_context_unsafe"),
    ],
)
def test_failed_gate_preserves_same_state(change: dict[str, object], reason: str) -> None:
    state = _state()
    result = bounded_verification_write(
        state,
        _proposal(state),
        (
            _gates(status="abstain")
            if change == {"evidence_status": "abstain"}
            else _gates(**change)  # type: ignore[arg-type]
        ),
        BoundedWorldWriteConfig(),
        commit=True,
    )
    assert result.reason == reason
    assert result.state is state
    assert not result.authorized and not result.committed
    assert result.receipt is None
    assert cognitive_state_hash(result.state) == cognitive_state_hash(state)


def test_world_write_config_can_be_stricter_than_authority_receipt() -> None:
    state = _state()
    result = bounded_verification_write(
        state,
        _proposal(state),
        _gates(),
        BoundedWorldWriteConfig(minimum_causal_lower_bound=0.99),
        commit=True,
    )

    assert result.reason == "causal_lower_bound"
    assert result.state is state


def test_plain_or_modified_gates_cannot_authorize_commit() -> None:
    state = _state()
    plain = WorldWriteGates(
        evidence_status="accept",
        causal_lower_bound=1.0,
        source_diversity=1_000,
        context_diversity=1_000,
        definitions_complete=True,
        counterfactual_support=True,
        intervention_support=True,
        regime_change_suspected=False,
        slot_gate_passed=True,
        device_gate_passed=True,
        capacity_strategy_safe=True,
        evidence_current=True,
        accumulator_revision_current=True,
        runtime_context_safe=True,
    )
    modified = replace(_gates(), causal_lower_bound=1.0)

    for gates in (plain, modified):
        with pytest.raises(PermissionError, match="authority capability"):
            bounded_verification_write(
                state,
                _proposal(state),
                gates,
                BoundedWorldWriteConfig(),
                commit=True,
            )


def test_currentness_gates_are_required_at_call_site() -> None:
    with pytest.raises(TypeError):
        WorldWriteGates(  # type: ignore[call-arg]
            evidence_status="accept",
            causal_lower_bound=0.7,
            source_diversity=3,
            context_diversity=6,
            definitions_complete=True,
            counterfactual_support=True,
            intervention_support=True,
            regime_change_suspected=False,
            slot_gate_passed=True,
            device_gate_passed=True,
            capacity_strategy_safe=True,
        )


def test_non_verification_target_and_stale_rollback_are_rejected() -> None:
    state = _state()
    with pytest.raises(ValueError, match="target only verification"):
        bounded_verification_write(
            state, _proposal(state, target="global"), _gates(),
            BoundedWorldWriteConfig(), commit=True
        )
    result = bounded_verification_write(
        state, _proposal(state), _gates(), BoundedWorldWriteConfig(), commit=True
    )
    assert result.receipt is not None
    changed = replace(result.state, goal_state={"changed": True})
    with pytest.raises(ValueError, match="rollback is stale"):
        rollback_bounded_verification_write(changed, result.receipt)


def test_scoped_retraction_preserves_unrelated_later_cognition() -> None:
    state = _state()
    result = bounded_verification_write(
        state, _proposal(state), _gates(), BoundedWorldWriteConfig(), commit=True
    )
    assert result.receipt is not None
    later = replace(
        result.state,
        goal_state={"autonomy_phase": "observe", "autonomy_step": 11},
        value_state={"current_preference": "keep"},
    )
    retracted = retract_bounded_verification_write(later, result.receipt)
    assert retracted.goal_state == later.goal_state
    assert retracted.value_state == later.value_state
    assert torch.equal(retracted.scratch_slots[:, 4], state.scratch_slots[:, 4])
    assert torch.equal(retracted.scratch_slots[:, :4], later.scratch_slots[:, :4])
    assert "bounded_verification_write" not in retracted.self_state


def test_scoped_retraction_requires_lifo_receipt_order() -> None:
    state = _state()
    first = bounded_verification_write(
        state, _proposal(state), _gates(), BoundedWorldWriteConfig(), commit=True
    )
    assert first.receipt is not None
    second = bounded_verification_write(
        first.state,
        _proposal(first.state),
        _gates(),
        BoundedWorldWriteConfig(),
        commit=True,
    )
    assert second.receipt is not None
    with pytest.raises(ValueError, match="not the current LIFO head"):
        retract_bounded_verification_write(second.state, first.receipt)
    first_head = retract_bounded_verification_write(second.state, second.receipt)
    baseline = retract_bounded_verification_write(first_head, first.receipt)
    assert torch.equal(baseline.scratch_slots, state.scratch_slots)

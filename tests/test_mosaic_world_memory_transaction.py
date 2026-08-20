from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import swegca.mosaic_world_memory_transaction as world_memory_transaction
from swegca.mosaic_bounded_world_write import (
    BoundedWorldWriteConfig,
    bounded_verification_write,
    cognitive_state_hash,
    world_write_gates_from_decision,
)
from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_evidence_accumulator import (
    AccumulatorDecision,
    EvidenceAccumulatorConfig,
    EvidenceAccumulatorState,
    EvidenceObservation,
    assess_accumulator,
    update_accumulator,
)
from swegca.mosaic_external_memory import MosaicExternalMemory
from swegca.mosaic_memory_promotion import MemoryCandidate
from swegca.mosaic_synapse_arbiter import evidence_delta_proposal
from swegca.mosaic_versioned_memory import VersionedExternalMemory
from swegca.mosaic_world_memory_transaction import (
    WorldMemoryTransactionConfig,
    promote_world_linked_semantic,
    record_uncertain_episode,
    recover_incomplete_world_memory_transactions,
    retract_world_linked_semantic,
    rollback_uncertain_episode,
    rollback_world_linked_semantic,
    world_memory_transaction_stage,
)
from tests._database import create_database


def _decision(status: str = "accept") -> AccumulatorDecision:
    config = EvidenceAccumulatorConfig()
    state = EvidenceAccumulatorState.empty("world-memory-test", config)
    if status == "abstain":
        return assess_accumulator(state, config)
    outcome = "support" if status == "accept" else "refute"
    for index in range(16):
        state = update_accumulator(
            state,
            EvidenceObservation(
                hypothesis_id=state.hypothesis_id,
                evidence_address=f"world-memory-evidence:{status}:{index}",
                source_family=f"world-memory-source:{index % 2}",
                context_hash=f"world-memory-context:{index}",
                axis=config.required_axes[index % len(config.required_axes)],
                outcome=outcome,
                observed_at=index,
                producer_id=f"world-memory-producer:{index}",
            ),
            config,
            current_step=index,
        ).state
    return assess_accumulator(state, config)


def _state() -> CognitiveState:
    return CognitiveState(
        semantic_slots=torch.zeros(1, 20, 32),
        executive_slots=torch.zeros(1, 6, 32),
        scratch_slots=torch.zeros(1, 6, 32),
    )


def _world_commit(state: CognitiveState):
    proposal = evidence_delta_proposal(
        state,
        torch.ones(1, 32),
        torch.tensor([[8.0, -8.0]]),
        source="memory-test",
        evidence_addresses=(("evidence:1", "counterfactual:1"),),
    )
    gates = world_write_gates_from_decision(
        _decision(),
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
    result = bounded_verification_write(
        state, proposal, gates, BoundedWorldWriteConfig(), commit=True
    )
    assert result.receipt is not None
    return result


def _candidate(receipt_id: str | None = None) -> MemoryCandidate:
    refs = ["evidence:1", "counterfactual:1"]
    if receipt_id:
        refs.append(f"world-write:{receipt_id}")
    return MemoryCandidate(
        hypothesis_id="test-hypothesis",
        key="새로운_지식",
        value="검증된_값",
        evidence_refs=tuple(refs),
        source_id="test",
        source_revision="1",
        timestamp="2026-08-11T00:00:00Z",
        license="CC0-1.0",
        attribution="test",
    )


def test_uncertain_episode_has_no_semantic_authority_and_rolls_back(tmp_path) -> None:
    memory = MosaicExternalMemory(create_database(tmp_path / "episodic"))
    receipt = record_uncertain_episode(
        memory,
        _candidate(),
        _decision("abstain"),
        provenance_complete=True,
        transaction_id="episode-1",
    )
    assert memory.search("새로운_지식")
    assert rollback_uncertain_episode(memory, receipt)
    assert not memory.search("새로운_지식")


def test_world_linked_semantic_update_and_joint_rollback(tmp_path) -> None:
    state = _state()
    baseline = cognitive_state_hash(state)
    world = _world_commit(state)
    assert world.receipt is not None
    memory = VersionedExternalMemory(
        MosaicExternalMemory(create_database(tmp_path / "semantic"))
    )
    memory_receipt = promote_world_linked_semantic(
        world.state,
        world.receipt,
        memory,
        _candidate(world.receipt.receipt_id),
        _decision(),
        WorldMemoryTransactionConfig(),
        valid_from="2026-08-11T00:00:01Z",
        update_id="semantic-1",
    )
    assert memory.search("새로운_지식", as_of="2026-08-11T00:00:02Z")
    assert world_memory_transaction_stage(memory, "semantic-1") == "completed"
    restored = rollback_world_linked_semantic(
        world.state,
        world.receipt,
        memory,
        memory_receipt,
        _decision(),
        rolled_back_at="2026-08-11T00:00:03Z",
    )
    assert cognitive_state_hash(restored) == baseline
    assert world_memory_transaction_stage(memory, "semantic-1") == "rolled_back"
    assert not memory.search("새로운_지식", as_of="2026-08-11T00:00:04Z")


def test_startup_recovery_rolls_back_memory_commit_before_state_handoff(
    tmp_path,
    monkeypatch,
) -> None:
    world = _world_commit(_state())
    assert world.receipt is not None
    memory = VersionedExternalMemory(
        MosaicExternalMemory(create_database(tmp_path / "semantic"))
    )
    original = world_memory_transaction._set_journal_stage

    def fail_after_memory_commit(memory_arg, transaction_id, expected, stage):
        if expected == "prepared" and stage == "memory_committed":
            raise RuntimeError("simulated process loss after memory commit")
        return original(memory_arg, transaction_id, expected, stage)

    monkeypatch.setattr(
        world_memory_transaction,
        "_set_journal_stage",
        fail_after_memory_commit,
    )
    with pytest.raises(RuntimeError, match="simulated process loss"):
        promote_world_linked_semantic(
            world.state,
            world.receipt,
            memory,
            _candidate(world.receipt.receipt_id),
            _decision(),
            WorldMemoryTransactionConfig(),
            valid_from="2026-08-11T00:00:01Z",
            update_id="crash-after-memory",
        )

    assert memory.search("새로운_지식", as_of="2026-08-11T00:00:02Z")
    assert world_memory_transaction_stage(memory, "crash-after-memory") == "prepared"

    with pytest.raises(
        RuntimeError,
        match="Incomplete World-memory transaction requires recovery",
    ):
        promote_world_linked_semantic(
            world.state,
            world.receipt,
            memory,
            _candidate(world.receipt.receipt_id),
            _decision(),
            WorldMemoryTransactionConfig(),
            valid_from="2026-08-11T00:00:02Z",
            update_id="blocked-before-recovery",
        )

    monkeypatch.setattr(world_memory_transaction, "_set_journal_stage", original)
    recovered = recover_incomplete_world_memory_transactions(
        memory,
        _decision(),
        rolled_back_at="2026-08-11T00:00:03Z",
    )

    assert recovered == ("crash-after-memory",)
    assert world_memory_transaction_stage(memory, "crash-after-memory") == "rolled_back"
    assert not memory.search("새로운_지식", as_of="2026-08-11T00:00:04Z")


def test_semantic_candidate_must_bind_world_receipt(tmp_path) -> None:
    world = _world_commit(_state())
    assert world.receipt is not None
    memory = VersionedExternalMemory(
        MosaicExternalMemory(create_database(tmp_path / "semantic"))
    )
    with pytest.raises(PermissionError, match="not bound"):
        promote_world_linked_semantic(
            world.state,
            world.receipt,
            memory,
            _candidate(),
            _decision(),
            WorldMemoryTransactionConfig(),
            valid_from="2026-08-11T00:00:01Z",
            update_id="bad",
        )


def test_world_linked_semantic_retraction_survives_unrelated_state_progress(
    tmp_path,
) -> None:
    world = _world_commit(_state())
    assert world.receipt is not None
    memory = VersionedExternalMemory(
        MosaicExternalMemory(create_database(tmp_path / "semantic"))
    )
    memory_receipt = promote_world_linked_semantic(
        world.state,
        world.receipt,
        memory,
        _candidate(world.receipt.receipt_id),
        _decision(),
        WorldMemoryTransactionConfig(),
        valid_from="2026-08-11T00:00:01Z",
        update_id="semantic-long-lived-1",
    )
    progressed = replace(
        world.state,
        goal_state={"autonomy_phase": "observe", "autonomy_step": 99},
        value_state={"must_survive": True},
    )
    retracted = retract_world_linked_semantic(
        progressed,
        world.receipt,
        memory,
        memory_receipt,
        _decision(),
        rolled_back_at="2026-08-11T00:00:03Z",
    )
    assert retracted.goal_state == progressed.goal_state
    assert retracted.value_state == progressed.value_state
    assert not memory.search("새로운_지식", as_of="2026-08-11T00:00:04Z")

from __future__ import annotations

import threading
from pathlib import Path

import torch

from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_omni import WorldState
from swegca.mosaic_unrestricted_experience import (
    build_hot_experience_index,
    discover_experience_universe,
)
from swegca.mosaic_synapse_arbiter import (
    SingleWorldArbiter,
    SynapseProposal,
    evidence_delta_proposal,
    run_dynamic_cognition,
    state_slot_tensor,
    sufficiency_gated_proposal,
)


def _world() -> WorldState:
    return WorldState(
        semantic_slots=torch.zeros(2, 32, 256),
        active_mask=torch.zeros(2, 32, dtype=torch.bool),
        dirty_mask=torch.zeros(2, 32, dtype=torch.bool),
        source="test",
    )


def _cognitive_state() -> CognitiveState:
    return CognitiveState(
        semantic_slots=torch.zeros(2, 20, 256),
        executive_slots=torch.zeros(2, 6, 256),
        scratch_slots=torch.zeros(2, 6, 256),
    )


def test_evidence_delta_becomes_verification_proposal() -> None:
    world = _world()
    proposal = evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [-8.0, 8.0]]),
        source="text-video",
        evidence_addresses=(("clip:1",), ("clip:2",)),
    )

    assert proposal.delta_candidate[:, 30].eq(1).all()
    assert proposal.delta_candidate[:, :30].eq(0).all()
    assert proposal.target_slot_mask[:, 30].all()
    assert proposal.confidence[0] > proposal.confidence[1]
    assert proposal.contradiction[0] < proposal.contradiction[1]


def test_disabled_arbiter_preserves_world_bit_exact() -> None:
    world = _world()
    proposal = evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="text-video",
    )

    result = SingleWorldArbiter()(world, (proposal,), commit=False)

    assert result.world_state is world
    assert torch.equal(result.world_state.semantic_slots, world.semantic_slots)
    assert not result.committed
    assert bool(result.accepted.all())


def test_commit_is_bounded_and_updates_only_one_world() -> None:
    world = _world()
    proposal = evidence_delta_proposal(
        world,
        torch.full((2, 256), 100.0),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="text-video",
    )
    arbiter = SingleWorldArbiter(
        maximum_slot_delta=0.1,
        maximum_world_delta=0.2,
    )

    result = arbiter(world, (proposal,), commit=True)

    assert result.committed
    assert result.world_state.semantic_slots.shape == (2, 32, 256)
    assert torch.linalg.vector_norm(result.proposed_delta[:, 30], dim=-1).le(
        0.100001
    ).all()
    assert result.proposed_delta[:, :30].eq(0).all()
    assert result.proposed_delta[:, 31].eq(0).all()
    assert result.world_state.dirty_mask[:, 30].all()
    assert torch.equal(world.semantic_slots, torch.zeros_like(world.semantic_slots))


def test_contradictory_or_uncertain_proposal_is_rejected() -> None:
    world = _world()
    contradiction = evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        torch.tensor([[-8.0, 8.0], [0.0, 0.0]]),
        source="text-video",
    )

    result = SingleWorldArbiter(minimum_weight=0.25)(
        world,
        (contradiction,),
        commit=True,
    )

    assert not bool(result.accepted.any())
    assert not result.committed
    assert result.world_state is world


def test_equal_opposing_proposals_cancel_before_world_commit() -> None:
    world = _world()
    positive = evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="text-video-a",
    )
    negative = evidence_delta_proposal(
        world,
        -torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="text-video-b",
    )

    result = SingleWorldArbiter()(world, (positive, negative), commit=True)

    assert bool(result.accepted.all())
    assert bool(result.unresolved_contradiction[:, 30].all())
    assert torch.equal(result.proposed_delta, torch.zeros_like(result.proposed_delta))
    assert not result.committed
    assert result.world_state is world


def test_unequal_opposing_proposals_abstain_instead_of_residual_commit() -> None:
    world = _world()
    positive = evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="independent-positive",
    )
    negative = evidence_delta_proposal(
        world,
        torch.full((2, 256), -0.7),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="independent-negative",
    )

    result = SingleWorldArbiter()(world, (positive, negative), commit=True)

    assert bool(result.unresolved_contradiction[:, 30].all())
    assert torch.equal(result.proposed_delta, torch.zeros_like(result.proposed_delta))
    assert result.world_state is world
    assert not result.committed


def test_same_direction_independent_proposals_aggregate_normally() -> None:
    world = _world()
    first = evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="independent-a",
    )
    second = evidence_delta_proposal(
        world,
        torch.full((2, 256), 0.7),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="independent-b",
    )

    result = SingleWorldArbiter()(world, (first, second), commit=True)

    assert not bool(result.unresolved_contradiction.any())
    assert bool(result.proposed_delta[:, 30].gt(0).all())
    assert result.committed
    assert result.world_state is not world


def test_new_contradiction_withdraws_tentative_preview() -> None:
    world = _world()
    tentative = evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="tentative",
    )
    falsifier = evidence_delta_proposal(
        world,
        -torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="new-falsifier",
    )
    arbiter = SingleWorldArbiter()

    before = arbiter(world, (tentative,), commit=False)
    after = arbiter(world, (tentative, falsifier), commit=False)

    assert bool(before.proposed_delta[:, 30].gt(0).all())
    assert not bool(before.unresolved_contradiction.any())
    assert bool(after.unresolved_contradiction[:, 30].all())
    assert torch.equal(after.proposed_delta, torch.zeros_like(after.proposed_delta))
    assert after.world_state is world


def test_proposal_rejects_any_second_state_shape() -> None:
    world = _world()
    invalid = SynapseProposal(
        source="invalid",
        delta_candidate=torch.zeros(2, 1, 256),
        confidence=torch.ones(2),
        contradiction=torch.zeros(2),
        uncertainty=torch.zeros(2),
        target_slot_mask=torch.ones(2, 32, dtype=torch.bool),
    )

    try:
        invalid.validate(world)
    except ValueError as error:
        assert "match World State" in str(error)
    else:
        raise AssertionError("a proposal shaped like a second state was accepted")


def test_sufficiency_gate_fail_closes_without_creating_state() -> None:
    world = _world()
    proposal = evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="natural-game-motion",
    )
    gated = sufficiency_gated_proposal(
        world,
        proposal,
        torch.tensor([True, False]),
    )

    result = SingleWorldArbiter()(world, (gated,), commit=False)

    assert gated.confidence[0] > 0
    assert gated.confidence[1] == 0
    assert gated.uncertainty[1] == 1
    assert result.accepted[:, 0].tolist() == [True, False]
    assert result.world_state is world
    assert not result.committed


def test_sufficiency_gate_requires_boolean_batch_mask() -> None:
    world = _world()
    proposal = evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="natural-game-motion",
    )

    for invalid in (torch.ones(2), torch.ones(2, 1, dtype=torch.bool)):
        try:
            sufficiency_gated_proposal(world, proposal, invalid)
        except ValueError as error:
            assert "sufficient mask" in str(error)
        else:
            raise AssertionError("invalid sufficiency mask was accepted")


def test_evidence_delta_rejects_negative_class_indices() -> None:
    world = _world()
    proposal_args = (world, torch.ones(2, 256), torch.tensor([[8.0, -8.0], [8.0, -8.0]]))

    for bad_consistent, bad_contradiction in [(-1, 1), (0, -1)]:
        try:
            evidence_delta_proposal(*proposal_args, source="bad-class", consistent_class=bad_consistent, contradiction_class=bad_contradiction)
        except ValueError as error:
            assert "evidence classes are invalid" in str(error)
        else:
            raise AssertionError("negative class index was accepted")


def test_cognitive_state_is_arbitrated_directly_without_a_second_state() -> None:
    state = _cognitive_state()
    proposal = evidence_delta_proposal(
        state,
        torch.ones(2, 256),
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        source="visual-state-transition",
    )

    preview = SingleWorldArbiter()(state, (proposal,), commit=False)
    committed = SingleWorldArbiter()(state, (proposal,), commit=True)

    assert state.persistent_state_count == 1
    assert state_slot_tensor(state).shape == (2, 32, 256)
    assert preview.world_state is state
    assert torch.equal(preview.proposed_delta[:, :30], torch.zeros(2, 30, 256))
    assert not preview.committed
    assert isinstance(committed.world_state, CognitiveState)
    assert committed.world_state.persistent_state_count == 1
    assert torch.equal(committed.world_state.semantic_slots, state.semantic_slots)
    assert torch.equal(committed.world_state.executive_slots, state.executive_slots)
    assert committed.world_state.scratch_slots[:, 4].abs().any()
    assert torch.equal(
        committed.world_state.scratch_slots[:, :4], state.scratch_slots[:, :4]
    )
    assert torch.equal(
        committed.world_state.scratch_slots[:, 5:], state.scratch_slots[:, 5:]
    )


def _core_proposal(
    world: WorldState,
    source: str,
    *,
    decisive: bool,
) -> SynapseProposal:
    logits = (
        torch.tensor([[8.0, -8.0], [8.0, -8.0]])
        if decisive
        else torch.zeros(2, 2)
    )
    return evidence_delta_proposal(
        world,
        torch.ones(2, 256),
        logits,
        source=source,
    )


def test_dynamic_cognition_uses_only_primary_core_when_decisive() -> None:
    world = _world()
    calls: list[str] = []

    def core(name: str, decisive: bool):
        def run(current: WorldState, _request: object) -> SynapseProposal:
            calls.append(name)
            return _core_proposal(current, name, decisive=decisive)

        return run

    result = run_dynamic_cognition(
        world,
        object(),
        operation_type="identity",
        resident_cores={
            "fast": core("fast", True),
            "shape": core("shape", True),
            "motion": core("motion", True),
        },
        routes={"identity": ("fast", "shape", "motion")},
    )

    assert calls == ["fast"]
    assert result.trace.executed_cores == ("fast",)
    assert not result.trace.fanout_used
    assert result.trace.elapsed_ns >= 0
    assert not result.trace.manager_retained
    assert not result.trace.worker_state_retained
    assert tuple(proposal.source for proposal in result.proposals) == ("fast",)
    assert result.arbitration.world_state is world
    assert not result.arbitration.committed


def test_dynamic_cognition_fans_out_relevant_workers_in_parallel() -> None:
    world = _world()
    barrier = threading.Barrier(2)
    calls: list[str] = []
    lock = threading.Lock()

    def primary(current: WorldState, _request: object) -> SynapseProposal:
        calls.append("fast")
        return _core_proposal(current, "fast", decisive=False)

    def specialist(name: str):
        def run(current: WorldState, _request: object) -> SynapseProposal:
            barrier.wait(timeout=2)
            with lock:
                calls.append(name)
            return _core_proposal(current, name, decisive=True)

        return run

    result = run_dynamic_cognition(
        world,
        {"evidence": "hot-index-view"},
        operation_type="identity",
        resident_cores={
            "fast": primary,
            "shape": specialist("shape"),
            "motion": specialist("motion"),
            "audio": specialist("audio"),
        },
        routes={
            "identity": ("fast", "shape", "motion"),
            "speech": ("fast", "audio"),
        },
    )

    assert calls[0] == "fast"
    assert set(calls[1:]) == {"shape", "motion"}
    assert "audio" not in calls
    assert result.trace.executed_cores == ("fast", "shape", "motion")
    assert result.trace.fanout_used
    assert tuple(proposal.source for proposal in result.proposals) == (
        "fast",
        "shape",
        "motion",
    )
    assert result.arbitration.world_state is world
    assert not result.arbitration.committed
    assert not any(
        thread.name.startswith("rozephine-cognition-worker")
        for thread in threading.enumerate()
    )


def test_dynamic_cognition_rejects_core_source_identity_drift() -> None:
    world = _world()

    def wrong_source(current: WorldState, _request: object) -> SynapseProposal:
        return _core_proposal(current, "other", decisive=True)

    try:
        run_dynamic_cognition(
            world,
            object(),
            operation_type="identity",
            resident_cores={"fast": wrong_source},
            routes={"identity": ("fast",)},
        )
    except ValueError as error:
        assert "source identity" in str(error)
    else:
        raise AssertionError("a worker retained a different source identity")


def test_dynamic_cognition_worker_mutation_cannot_change_main_state() -> None:
    state = _cognitive_state()
    before = tuple(
        tensor.clone()
        for tensor in (state.semantic_slots, state.executive_slots, state.scratch_slots)
    )
    observed_distinct_storage: list[bool] = []

    def malicious(current: CognitiveState, _request: object) -> SynapseProposal:
        observed_distinct_storage.append(
            current.semantic_slots.data_ptr() != state.semantic_slots.data_ptr()
        )
        current.semantic_slots.add_(99)
        current.executive_slots.fill_(-42)
        return evidence_delta_proposal(
            current,
            torch.ones(2, 256),
            torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
            source="malicious",
        )

    result = run_dynamic_cognition(
        state,
        object(),
        operation_type="identity",
        resident_cores={"malicious": malicious},
        routes={"identity": ("malicious",)},
    )

    assert observed_distinct_storage == [True]
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            before,
            (state.semantic_slots, state.executive_slots, state.scratch_slots),
            strict=True,
        )
    )
    assert state.persistent_state_count == 1
    assert result.arbitration.world_state is state
    assert not result.arbitration.committed


def test_dynamic_cognition_workers_receive_isolated_snapshots() -> None:
    world = _world()
    worker_saw_clean_state: list[bool] = []

    def primary(current: WorldState, _request: object) -> SynapseProposal:
        current.semantic_slots.fill_(7)
        return _core_proposal(current, "primary", decisive=False)

    def worker(current: WorldState, _request: object) -> SynapseProposal:
        worker_saw_clean_state.append(bool(current.semantic_slots.eq(0).all()))
        return _core_proposal(current, "worker", decisive=True)

    run_dynamic_cognition(
        world,
        object(),
        operation_type="identity",
        resident_cores={"primary": primary, "worker": worker},
        routes={"identity": ("primary", "worker")},
    )

    assert worker_saw_clean_state == [True]
    assert bool(world.semantic_slots.eq(0).all())


def test_dynamic_worker_reads_main_hot_experience_and_returns_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experience"
    root.mkdir()
    (root / "failed-identity-view.json").write_text(
        '{"status":"failed","lesson":"use temporal appearance"}\n',
        encoding="utf-8",
    )
    snapshot = discover_experience_universe({"episodic": root})
    address = snapshot.artifacts[0].address
    hot_index = build_hot_experience_index(
        snapshot,
        semantic_keys_by_address={address: ("actor_identity",)},
    )
    world = _world()

    def experience_core(
        current: WorldState,
        request: object,
    ) -> SynapseProposal:
        assert isinstance(request, dict)
        selected = request["hot_index"].lookup_semantic_key("actor_identity")
        assert selected == (address,)
        return evidence_delta_proposal(
            current,
            torch.ones(2, 256),
            torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
            source="experience-fast",
            evidence_addresses=((address,), (address,)),
        )

    result = run_dynamic_cognition(
        world,
        {"hot_index": hot_index},
        operation_type="identity",
        resident_cores={"experience-fast": experience_core},
        routes={"identity": ("experience-fast",)},
    )

    assert result.proposals[0].evidence_addresses == (
        (address,),
        (address,),
    )
    assert result.trace.executed_cores == ("experience-fast",)
    assert result.arbitration.world_state is world

from __future__ import annotations

import json

import pytest
import torch

from swegca.mosaic_evidence_accumulator import (
    EvidenceAccumulatorConfig,
    EvidenceAccumulatorState,
    EvidenceObservation,
    accumulator_gated_proposal,
    append_audit_record,
    assess_accumulator,
    update_accumulator,
)
from swegca.mosaic_omni import WorldState
from swegca.mosaic_synapse_arbiter import (
    SingleWorldArbiter,
    evidence_delta_proposal,
)


def _config() -> EvidenceAccumulatorConfig:
    return EvidenceAccumulatorConfig()


def _observation(
    index: int,
    *,
    outcome: str = "support",
    hypothesis: str = "rule-a",
    axis: str | None = None,
    source: str | None = None,
    context: str | None = None,
    address: str | None = None,
    producer: str | None = None,
    confidence: float = 0.5,
    expires_at: int | None = None,
) -> EvidenceObservation:
    axes = _config().required_axes
    return EvidenceObservation(
        hypothesis_id=hypothesis,
        evidence_address=address or f"evidence:{index}",
        source_family=source or f"source:{index % 2}",
        context_hash=context or f"context:{index}",
        axis=axis or axes[index % len(axes)],
        outcome=outcome,  # type: ignore[arg-type]
        observed_at=index,
        expires_at=expires_at,
        producer_id=producer or f"producer:{index}",
        producer_confidence=confidence,
    )


def _feed(
    state: EvidenceAccumulatorState,
    observations: list[EvidenceObservation],
) -> EvidenceAccumulatorState:
    config = _config()
    for observation in observations:
        state = update_accumulator(
            state,
            observation,
            config,
            current_step=observation.observed_at,
        ).state
    return state


def _accepted_state() -> EvidenceAccumulatorState:
    state = EvidenceAccumulatorState.empty("rule-a", _config())
    return _feed(state, [_observation(index) for index in range(16)])


def test_stable_rule_waits_for_evidence_then_accepts() -> None:
    config = _config()
    state = EvidenceAccumulatorState.empty("rule-a", config)

    state = _feed(state, [_observation(index) for index in range(15)])
    assert assess_accumulator(state, config).status == "abstain"

    state = _feed(state, [_observation(15)])
    decision = assess_accumulator(state, config)
    assert decision.status == "accept"
    assert decision.causal_lower_bound > config.chance_rate + config.accept_margin


def test_random_and_short_lucky_streams_never_accept() -> None:
    config = _config()
    lucky = EvidenceAccumulatorState.empty("rule-a", config)
    lucky = _feed(lucky, [_observation(index) for index in range(12)])
    assert assess_accumulator(lucky, config).status == "abstain"

    random = EvidenceAccumulatorState.empty("rule-a", config)
    observations = [
        _observation(index, outcome="support" if index % 5 == 0 else "refute")
        for index in range(32)
    ]
    random = _feed(random, observations)
    assert assess_accumulator(random, config).status == "reject"


def test_rule_disappears_with_bounded_delay_and_can_reappear() -> None:
    config = _config()
    state = _accepted_state()
    revoked_after = None
    for offset in range(8):
        state = _feed(
            state,
            [_observation(16 + offset, outcome="refute")],
        )
        if assess_accumulator(state, config).status != "accept":
            revoked_after = offset + 1
            break

    assert revoked_after is not None
    assert revoked_after <= 4

    start = 16 + revoked_after
    state = _feed(
        state,
        [_observation(start + offset) for offset in range(8)],
    )
    assert assess_accumulator(state, config).status == "accept"


def test_competing_rule_b_replaces_rule_a() -> None:
    config = _config()
    rule_a = _accepted_state()
    rule_b = EvidenceAccumulatorState.empty("rule-b", config)
    rule_a = _feed(
        rule_a,
        [_observation(16 + index, outcome="refute") for index in range(4)],
    )
    rule_b = _feed(
        rule_b,
        [
            _observation(index, hypothesis="rule-b", address=f"b:{index}")
            for index in range(16)
        ],
    )

    assert assess_accumulator(rule_a, config).status != "accept"
    assert assess_accumulator(rule_b, config).status == "accept"


def test_correlated_or_exact_duplicates_do_not_create_fake_sample_size() -> None:
    config = _config()
    state = EvidenceAccumulatorState.empty("rule-a", config)
    first = _observation(0, source="one", context="same", address="same")
    state = _feed(state, [first])
    baseline = assess_accumulator(state, config)
    for index in range(100):
        update = update_accumulator(
            state,
            _observation(
                index + 1,
                source="one",
                context="same",
                address="same",
            ),
            config,
            current_step=index + 1,
        )
        assert update.state is state
        assert update.reason == "duplicate"
    assert assess_accumulator(state, config).effective_sample_size == (
        baseline.effective_sample_size
    )

    correlated = EvidenceAccumulatorState.empty("rule-a", config)
    correlated = _feed(
        correlated,
        [
            _observation(index, source="one", context="same")
            for index in range(100)
        ],
    )
    assert assess_accumulator(correlated, config).effective_sample_size == 4
    assert assess_accumulator(correlated, config).status == "abstain"


def test_global_source_count_cannot_bypass_per_axis_source_diversity() -> None:
    config = EvidenceAccumulatorConfig(minimum_source_diversity_per_axis=2)
    state = EvidenceAccumulatorState.empty("rule-a", config)
    state = _feed(state, [_observation(index, source="source:a") for index in range(16)])
    state = _feed(
        state,
        [
            _observation(
                16,
                source="source:b",
                axis="observational",
                address="source-b:observational",
            )
        ],
    )

    decision = assess_accumulator(state, config)

    assert decision.source_diversity == 2
    assert decision.status == "abstain"
    assert decision.reason == "axis_source_diversity"


def test_high_producer_confidence_and_producer_name_are_not_trust_inputs() -> None:
    config = _config()
    states = []
    for producer in ("trusted-name", "unknown-name"):
        state = EvidenceAccumulatorState.empty("rule-a", config)
        state = _feed(
            state,
            [
                _observation(
                    index,
                    outcome="refute",
                    producer=f"{producer}:{index}",
                    confidence=0.999,
                )
                for index in range(16)
            ],
        )
        states.append(assess_accumulator(state, config))

    assert states[0] == states[1]
    assert states[0].status == "reject"


def test_one_producer_cannot_forge_source_or_context_diversity() -> None:
    config = _config()
    state = EvidenceAccumulatorState.empty("rule-a", config)
    state = _feed(
        state,
        [
            _observation(
                index,
                source=f"forged-source:{index}",
                context=f"forged-context:{index}",
                producer="one-producer",
            )
            for index in range(16)
        ],
    )

    decision = assess_accumulator(state, config)

    assert decision.source_diversity == 1
    assert decision.context_diversity == 1
    assert decision.status == "abstain"
    assert decision.reason == "source_diversity"


def test_observation_requires_nonempty_producer_identity() -> None:
    observation = _observation(0)
    object.__setattr__(observation, "producer_id", "")

    with pytest.raises(ValueError, match="producer_id must be nonempty"):
        observation.validate()


def test_expired_and_insufficient_evidence_are_logged_but_not_counted() -> None:
    config = _config()
    state = EvidenceAccumulatorState.empty("rule-a", config)
    expired = update_accumulator(
        state,
        _observation(0, expires_at=1),
        config,
        current_step=2,
    )
    insufficient = update_accumulator(
        state,
        _observation(1, outcome="insufficient"),
        config,
        current_step=1,
    )

    assert expired.state is state and expired.reason == "expired"
    assert insufficient.state is state and insufficient.reason == "insufficient"


def test_immutable_snapshot_is_a_local_rollback_point() -> None:
    config = _config()
    before = _accepted_state()
    after = update_accumulator(
        before,
        _observation(16, outcome="refute"),
        config,
        current_step=16,
    ).state

    assert before.revision == 16
    assert after.revision == 17
    assert assess_accumulator(before, config).status == "accept"


def test_audit_ledger_contains_provenance_and_decision(tmp_path) -> None:
    config = _config()
    state = EvidenceAccumulatorState.empty("rule-a", config)
    observation = _observation(0)
    update = update_accumulator(state, observation, config, current_step=0)
    ledger = tmp_path / "ledger.jsonl"

    append_audit_record(
        ledger,
        observation,
        update,
        world_hash_before="before",
        world_hash_after="before",
        accumulator_version="a0",
        arbiter_version="v54",
    )
    record = json.loads(ledger.read_text(encoding="utf-8"))

    assert record["proposal_hash"] == observation.proposal_hash
    assert record["world_hash_before"] == record["world_hash_after"]
    assert record["posterior_before"] == 0.5
    assert record["posterior_after"] > record["posterior_before"]
    assert record["decision"] == "abstain"


def test_accumulator_gate_preserves_disabled_world_and_order_invariance() -> None:
    config = _config()
    decision = assess_accumulator(_accepted_state(), config)
    world = WorldState(
        semantic_slots=torch.zeros(1, 32, 256),
        active_mask=torch.zeros(1, 32, dtype=torch.bool),
        dirty_mask=torch.zeros(1, 32, dtype=torch.bool),
        source="test",
    )
    first = evidence_delta_proposal(
        world,
        torch.ones(1, 256),
        torch.tensor([[8.0, -8.0]]),
        source="first",
    )
    second = evidence_delta_proposal(
        world,
        -torch.ones(1, 256),
        torch.tensor([[8.0, -8.0]]),
        source="second",
    )
    first = accumulator_gated_proposal(world, first, decision)
    second = accumulator_gated_proposal(world, second, decision)
    arbiter = SingleWorldArbiter()

    forward = arbiter(world, (first, second), commit=False)
    reverse = arbiter(world, (second, first), commit=False)

    assert forward.world_state is world
    assert reverse.world_state is world
    assert torch.equal(forward.proposed_delta, reverse.proposed_delta)
    assert torch.equal(world.semantic_slots, torch.zeros_like(world.semantic_slots))

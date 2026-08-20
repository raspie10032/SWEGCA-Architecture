from __future__ import annotations

import torch

from swegca.mosaic_autonomous_cognition import (
    AutonomyEvent,
    AutonomyEventKind,
    AutonomyPhase,
    AutonomousCognitionConfig,
    advance_autonomous_cognition,
)
from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_evidence_accumulator import AccumulatorDecision


def _state() -> CognitiveState:
    return CognitiveState(
        semantic_slots=torch.zeros(1, 2, 4),
        executive_slots=torch.zeros(1, 1, 4),
        scratch_slots=torch.zeros(1, 1, 4),
    )


def _event(
    event_id: str,
    kind: AutonomyEventKind,
    hypothesis_id: str | None = "가설-1",
    *,
    confidence: float = 1.0,
    payload: dict | None = None,
) -> AutonomyEvent:
    return AutonomyEvent(
        event_id=event_id,
        kind=kind,
        hypothesis_id=hypothesis_id,
        evidence_refs=(f"증거:{event_id}",),
        source_family="한국어-개발",
        context_hash=f"문맥:{event_id}",
        confidence=confidence,
        payload=payload or {},
    )


def _decision(status: str) -> AccumulatorDecision:
    return AccumulatorDecision(
        status=status,
        reason="test",
        posterior_mean=0.95,
        causal_lower_bound=0.8,
        overall_upper_bound=0.99,
        effective_sample_size=16,
        source_diversity=2,
        context_diversity=4,
        regime_change_score=0.0,
        revision=16,
    )


def _config() -> AutonomousCognitionConfig:
    return AutonomousCognitionConfig(
        allowed_tool_actions=("read_file", "search_memory"),
        reversible_tool_actions=("read_file", "search_memory"),
        minimum_action_confidence=0.8,
        maximum_action_failures=2,
    )


def _evidence_config() -> AutonomousCognitionConfig:
    return AutonomousCognitionConfig(
        allowed_tool_actions=("read_file",),
        reversible_tool_actions=("read_file",),
        allowed_evidence_tool_actions=("inspect_frame",),
        reversible_evidence_tool_actions=("inspect_frame",),
        maximum_evidence_action_failures=2,
    )


def test_verified_korean_loop_reaches_action_and_returns_to_observation() -> None:
    original = _state()
    state = original
    events = (
        _event("관찰", AutonomyEventKind.OBSERVATION, None),
        _event("가설", AutonomyEventKind.HYPOTHESIS),
        _event(
            "요청",
            AutonomyEventKind.EVIDENCE_REQUEST,
            payload={"requested_axes": ["observational", "counterfactual"]},
        ),
    )
    for event in events:
        transition = advance_autonomous_cognition(state, event, _config())
        assert transition.accepted
        state = transition.state
    verified = advance_autonomous_cognition(
        state,
        _event("검증", AutonomyEventKind.VERIFICATION),
        _config(),
        evidence_decision=_decision("accept"),
    )
    assert verified.memory_write_allowed
    remembered = advance_autonomous_cognition(
        verified.state,
        _event(
            "기억",
            AutonomyEventKind.MEMORY_COMMIT,
            payload={"memory_ref": "sqlite:기억-1", "content_hash": "sha256:abc"},
        ),
        _config(),
    )
    action = advance_autonomous_cognition(
        remembered.state,
        _event(
            "행동",
            AutonomyEventKind.ACTION_PROPOSAL,
            payload={"action": "read_file"},
        ),
        _config(),
    )
    assert action.tool_action_allowed and action.tool_action == "read_file"
    result = advance_autonomous_cognition(
        action.state,
        _event(
            "결과",
            AutonomyEventKind.ACTION_RESULT,
            payload={"success": True},
        ),
        _config(),
    )
    assert result.next_phase == AutonomyPhase.OBSERVE
    assert result.state.persistent_state_count == 1
    assert result.state.semantic_slots is original.semantic_slots
    assert result.state.executive_slots is original.executive_slots
    assert result.state.scratch_slots is original.scratch_slots


def test_uncertain_hypothesis_requests_more_evidence_without_memory_or_action() -> None:
    state = _state()
    for event in (
        _event("o", AutonomyEventKind.OBSERVATION, None),
        _event("h", AutonomyEventKind.HYPOTHESIS),
        _event(
            "q",
            AutonomyEventKind.EVIDENCE_REQUEST,
            payload={"requested_axes": ["cross_context"]},
        ),
    ):
        state = advance_autonomous_cognition(state, event, _config()).state
    transition = advance_autonomous_cognition(
        state,
        _event("v", AutonomyEventKind.VERIFICATION),
        _config(),
        evidence_decision=_decision("abstain"),
    )
    assert transition.next_phase == AutonomyPhase.REQUEST_EVIDENCE
    assert not transition.memory_write_allowed
    assert not transition.tool_action_allowed


def test_unlisted_or_irreversible_action_is_inert() -> None:
    state = _state()
    state = CognitiveState(
        semantic_slots=state.semantic_slots,
        executive_slots=state.executive_slots,
        scratch_slots=state.scratch_slots,
        goal_state={
            "autonomy_phase": "act",
            "active_hypothesis_id": "가설-1",
            "verified_hypothesis_id": "가설-1",
            "memory_ref": "sqlite:기억-1",
        },
    )
    transition = advance_autonomous_cognition(
        state,
        _event(
            "위험",
            AutonomyEventKind.ACTION_PROPOSAL,
            payload={"action": "delete_file"},
        ),
        _config(),
    )
    assert not transition.accepted
    assert transition.state is state
    assert not transition.tool_action_allowed


def test_action_failure_recovery_is_bounded() -> None:
    config = _config()
    state = _state()
    state = CognitiveState(
        semantic_slots=state.semantic_slots,
        executive_slots=state.executive_slots,
        scratch_slots=state.scratch_slots,
        goal_state={
            "autonomy_phase": "observe_result",
            "active_hypothesis_id": "가설-1",
            "verified_hypothesis_id": "가설-1",
            "memory_ref": "sqlite:기억-1",
        },
    )
    first = advance_autonomous_cognition(
        state,
        _event(
            "실패-1",
            AutonomyEventKind.ACTION_RESULT,
            payload={"success": False},
        ),
        config,
    )
    assert first.next_phase == AutonomyPhase.ACT
    reproposed = advance_autonomous_cognition(
        first.state,
        _event(
            "재시도",
            AutonomyEventKind.ACTION_PROPOSAL,
            payload={"action": "search_memory"},
        ),
        config,
    )
    second = advance_autonomous_cognition(
        reproposed.state,
        _event(
            "실패-2",
            AutonomyEventKind.ACTION_RESULT,
            payload={"success": False},
        ),
        config,
    )
    assert second.next_phase == AutonomyPhase.ABSTAIN
    assert second.reason == "failure_budget_exhausted"


def test_duplicate_event_is_inert() -> None:
    event = _event("same", AutonomyEventKind.OBSERVATION, None)
    first = advance_autonomous_cognition(_state(), event, _config())
    duplicate = advance_autonomous_cognition(first.state, event, _config())
    assert not duplicate.accepted
    assert duplicate.reason == "duplicate_event"
    assert duplicate.state is first.state


def test_uncertain_hypothesis_can_be_abandoned_without_memory_or_action() -> None:
    state = _state()
    for event in (
        _event("o-abandon", AutonomyEventKind.OBSERVATION, None),
        _event("h-abandon", AutonomyEventKind.HYPOTHESIS),
        _event(
            "q-abandon",
            AutonomyEventKind.EVIDENCE_REQUEST,
            payload={"requested_axes": ["cross_context"]},
        ),
    ):
        state = advance_autonomous_cognition(state, event, _config()).state
    uncertain = advance_autonomous_cognition(
        state,
        _event("v-abandon", AutonomyEventKind.VERIFICATION),
        _config(),
        evidence_decision=_decision("abstain"),
    )
    abandoned = advance_autonomous_cognition(
        uncertain.state,
        _event("abandon", AutonomyEventKind.HYPOTHESIS_ABANDON),
        _config(),
    )
    assert abandoned.accepted
    assert abandoned.next_phase == AutonomyPhase.ABSTAIN
    assert abandoned.reason == "hypothesis_abandoned"
    assert not abandoned.memory_write_allowed
    assert not abandoned.tool_action_allowed
    resumed = advance_autonomous_cognition(
        abandoned.state,
        _event("next-observation", AutonomyEventKind.OBSERVATION, None),
        _config(),
    )
    assert resumed.accepted and resumed.next_phase == AutonomyPhase.HYPOTHESIZE


def test_evidence_tool_failure_recovers_then_success_reaches_verification() -> None:
    config = _evidence_config()
    state = _state()
    for event in (
        _event("eo", AutonomyEventKind.OBSERVATION, None),
        _event("eh", AutonomyEventKind.HYPOTHESIS),
        _event(
            "eq",
            AutonomyEventKind.EVIDENCE_REQUEST,
            payload={
                "requested_axes": ["observational"],
                "collect_with_tool": True,
            },
        ),
    ):
        transition = advance_autonomous_cognition(state, event, config)
        assert transition.accepted
        state = transition.state
    assert transition.next_phase == AutonomyPhase.COLLECT_EVIDENCE

    action = advance_autonomous_cognition(
        state,
        _event(
            "ea1",
            AutonomyEventKind.EVIDENCE_ACTION,
            payload={"action": "inspect_frame"},
        ),
        config,
    )
    assert action.evidence_tool_action_allowed
    assert not action.tool_action_allowed
    assert action.next_phase == AutonomyPhase.OBSERVE_EVIDENCE_RESULT
    failed = advance_autonomous_cognition(
        action.state,
        _event(
            "er1",
            AutonomyEventKind.EVIDENCE_RESULT,
            payload={"success": False},
        ),
        config,
    )
    assert failed.next_phase == AutonomyPhase.COLLECT_EVIDENCE
    assert failed.reason == "recover_evidence_action"

    retry = advance_autonomous_cognition(
        failed.state,
        _event(
            "ea2",
            AutonomyEventKind.EVIDENCE_ACTION,
            payload={"action": "inspect_frame"},
        ),
        config,
    )
    succeeded = advance_autonomous_cognition(
        retry.state,
        _event(
            "er2",
            AutonomyEventKind.EVIDENCE_RESULT,
            payload={"success": True},
        ),
        config,
    )
    assert succeeded.next_phase == AutonomyPhase.VERIFY
    assert succeeded.state.self_state["autonomy_evidence_action_failures"] == 0


def test_evidence_tool_failure_budget_abstains_without_memory_or_action() -> None:
    config = _evidence_config()
    state = CognitiveState(
        semantic_slots=_state().semantic_slots,
        executive_slots=_state().executive_slots,
        scratch_slots=_state().scratch_slots,
        goal_state={
            "autonomy_phase": "observe_evidence_result",
            "active_hypothesis_id": "가설-1",
            "requested_evidence_axes": ["cross_context"],
        },
        self_state={"autonomy_evidence_action_failures": 1},
    )
    exhausted = advance_autonomous_cognition(
        state,
        _event(
            "er-exhausted",
            AutonomyEventKind.EVIDENCE_RESULT,
            payload={"success": False},
        ),
        config,
    )
    assert exhausted.next_phase == AutonomyPhase.ABSTAIN
    assert exhausted.reason == "evidence_failure_budget_exhausted"
    assert not exhausted.memory_write_allowed
    assert not exhausted.tool_action_allowed
    assert not exhausted.evidence_tool_action_allowed
    resumed = advance_autonomous_cognition(
        exhausted.state,
        _event("fresh-observation", AutonomyEventKind.OBSERVATION, None),
        config,
    )
    assert resumed.accepted
    assert resumed.state.self_state["autonomy_evidence_action_failures"] == 0

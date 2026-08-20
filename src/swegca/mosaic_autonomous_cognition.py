"""Stateless control contract for one-state autonomous cognition."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping

from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_evidence_accumulator import AccumulatorDecision


class AutonomyPhase(StrEnum):
    OBSERVE = "observe"
    HYPOTHESIZE = "hypothesize"
    REQUEST_EVIDENCE = "request_evidence"
    COLLECT_EVIDENCE = "collect_evidence"
    OBSERVE_EVIDENCE_RESULT = "observe_evidence_result"
    VERIFY = "verify"
    REMEMBER = "remember"
    ACT = "act"
    OBSERVE_RESULT = "observe_result"
    ABSTAIN = "abstain"


class AutonomyEventKind(StrEnum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    EVIDENCE_REQUEST = "evidence_request"
    EVIDENCE_ACTION = "evidence_action"
    EVIDENCE_RESULT = "evidence_result"
    VERIFICATION = "verification"
    MEMORY_COMMIT = "memory_commit"
    ACTION_PROPOSAL = "action_proposal"
    ACTION_RESULT = "action_result"
    HYPOTHESIS_ABANDON = "hypothesis_abandon"


@dataclass(frozen=True)
class AutonomousCognitionConfig:
    allowed_tool_actions: tuple[str, ...]
    reversible_tool_actions: tuple[str, ...]
    minimum_action_confidence: float = 0.8
    maximum_action_failures: int = 2
    allowed_evidence_tool_actions: tuple[str, ...] = ()
    reversible_evidence_tool_actions: tuple[str, ...] = ()
    minimum_evidence_action_confidence: float = 0.8
    maximum_evidence_action_failures: int = 2

    def __post_init__(self) -> None:
        allowed = tuple(dict.fromkeys(self.allowed_tool_actions))
        reversible = tuple(dict.fromkeys(self.reversible_tool_actions))
        if not allowed or any(not value.strip() for value in allowed):
            raise ValueError("allowed tool actions must be nonempty")
        if any(value not in allowed for value in reversible):
            raise ValueError("reversible actions must be a subset of allowed actions")
        if not math.isfinite(self.minimum_action_confidence) or not (
            0 <= self.minimum_action_confidence <= 1
        ):
            raise ValueError("minimum action confidence must be in [0, 1]")
        if self.maximum_action_failures <= 0:
            raise ValueError("maximum action failures must be positive")
        object.__setattr__(self, "allowed_tool_actions", allowed)
        object.__setattr__(self, "reversible_tool_actions", reversible)
        evidence_allowed = tuple(dict.fromkeys(self.allowed_evidence_tool_actions))
        evidence_reversible = tuple(
            dict.fromkeys(self.reversible_evidence_tool_actions)
        )
        if any(not value.strip() for value in evidence_allowed):
            raise ValueError("allowed evidence tool actions cannot be empty")
        if any(value not in evidence_allowed for value in evidence_reversible):
            raise ValueError(
                "reversible evidence actions must be a subset of allowed actions"
            )
        if not math.isfinite(self.minimum_evidence_action_confidence) or not (
            0 <= self.minimum_evidence_action_confidence <= 1
        ):
            raise ValueError("minimum evidence action confidence must be in [0, 1]")
        if self.maximum_evidence_action_failures <= 0:
            raise ValueError("maximum evidence action failures must be positive")
        object.__setattr__(self, "allowed_evidence_tool_actions", evidence_allowed)
        object.__setattr__(
            self, "reversible_evidence_tool_actions", evidence_reversible
        )


@dataclass(frozen=True)
class AutonomyEvent:
    event_id: str
    kind: AutonomyEventKind
    hypothesis_id: str | None
    evidence_refs: tuple[str, ...]
    source_family: str
    context_hash: str
    confidence: float
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, value in (
            ("event_id", self.event_id),
            ("source_family", self.source_family),
            ("context_hash", self.context_hash),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if self.hypothesis_id is not None and not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id cannot be empty")
        if any(not reference.strip() for reference in self.evidence_refs):
            raise ValueError("evidence references cannot be empty")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("event confidence must be in [0, 1]")
        try:
            payload = json.loads(
                json.dumps(dict(self.payload), ensure_ascii=False, sort_keys=True)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("event payload must be JSON-compatible") from exc
        object.__setattr__(self, "payload", payload)


@dataclass(frozen=True)
class AutonomyTransition:
    state: CognitiveState
    prior_phase: AutonomyPhase
    next_phase: AutonomyPhase
    accepted: bool
    reason: str
    memory_write_allowed: bool = False
    tool_action_allowed: bool = False
    tool_action: str | None = None
    evidence_tool_action_allowed: bool = False
    evidence_tool_action: str | None = None


_EXPECTED_EVENTS = {
    AutonomyPhase.OBSERVE: AutonomyEventKind.OBSERVATION,
    AutonomyPhase.HYPOTHESIZE: AutonomyEventKind.HYPOTHESIS,
    AutonomyPhase.REQUEST_EVIDENCE: AutonomyEventKind.EVIDENCE_REQUEST,
    AutonomyPhase.COLLECT_EVIDENCE: AutonomyEventKind.EVIDENCE_ACTION,
    AutonomyPhase.OBSERVE_EVIDENCE_RESULT: AutonomyEventKind.EVIDENCE_RESULT,
    AutonomyPhase.VERIFY: AutonomyEventKind.VERIFICATION,
    AutonomyPhase.REMEMBER: AutonomyEventKind.MEMORY_COMMIT,
    AutonomyPhase.ACT: AutonomyEventKind.ACTION_PROPOSAL,
    AutonomyPhase.OBSERVE_RESULT: AutonomyEventKind.ACTION_RESULT,
    AutonomyPhase.ABSTAIN: AutonomyEventKind.OBSERVATION,
}


def _phase(state: CognitiveState) -> AutonomyPhase:
    return AutonomyPhase(state.goal_state.get("autonomy_phase", AutonomyPhase.OBSERVE))


def _reject(
    state: CognitiveState, phase: AutonomyPhase, reason: str
) -> AutonomyTransition:
    return AutonomyTransition(
        state=state,
        prior_phase=phase,
        next_phase=phase,
        accepted=False,
        reason=reason,
    )


def _updated_state(
    state: CognitiveState,
    event: AutonomyEvent,
    next_phase: AutonomyPhase,
    *,
    goal_updates: Mapping[str, Any] | None = None,
    self_updates: Mapping[str, Any] | None = None,
) -> CognitiveState:
    goal_state = dict(state.goal_state)
    self_state = dict(state.self_state)
    goal_state.update(goal_updates or {})
    self_state.update(self_updates or {})
    goal_state.update(
        {
            "autonomy_phase": next_phase.value,
            "autonomy_step": int(goal_state.get("autonomy_step", 0)) + 1,
            "last_autonomy_event_id": event.event_id,
        }
    )
    return replace(state, goal_state=goal_state, self_state=self_state)


def advance_autonomous_cognition(
    state: CognitiveState,
    event: AutonomyEvent,
    config: AutonomousCognitionConfig,
    *,
    evidence_decision: AccumulatorDecision | None = None,
) -> AutonomyTransition:
    """Advance a validated event without creating a second persistent state."""

    phase = _phase(state)
    if state.goal_state.get("last_autonomy_event_id") == event.event_id:
        return _reject(state, phase, "duplicate_event")
    if event.kind == AutonomyEventKind.HYPOTHESIS_ABANDON:
        if phase not in (
            AutonomyPhase.REQUEST_EVIDENCE,
            AutonomyPhase.COLLECT_EVIDENCE,
            AutonomyPhase.VERIFY,
        ):
            return _reject(state, phase, "hypothesis_not_abandonable")
        if event.hypothesis_id != state.goal_state.get("active_hypothesis_id"):
            return _reject(state, phase, "hypothesis_mismatch")
        updated = _updated_state(
            state,
            event,
            AutonomyPhase.ABSTAIN,
            goal_updates={
                "verification_status": "abandoned",
                "requested_evidence_axes": [],
            },
        )
        return AutonomyTransition(
            state=updated,
            prior_phase=phase,
            next_phase=AutonomyPhase.ABSTAIN,
            accepted=True,
            reason="hypothesis_abandoned",
        )
    if event.kind != _EXPECTED_EVENTS[phase]:
        return _reject(state, phase, "unexpected_event_for_phase")

    hypothesis_id = state.goal_state.get("active_hypothesis_id")
    if phase not in (
        AutonomyPhase.OBSERVE,
        AutonomyPhase.ABSTAIN,
        AutonomyPhase.HYPOTHESIZE,
    ) and event.hypothesis_id != hypothesis_id:
        return _reject(state, phase, "hypothesis_mismatch")

    next_phase = phase
    reason = "accepted"
    memory_write_allowed = False
    tool_action_allowed = False
    tool_action = None
    evidence_tool_action_allowed = False
    evidence_tool_action = None
    goal_updates: dict[str, Any] = {}
    self_updates: dict[str, Any] = {}

    if phase in (AutonomyPhase.OBSERVE, AutonomyPhase.ABSTAIN):
        next_phase = AutonomyPhase.HYPOTHESIZE
        goal_updates = {
            "active_hypothesis_id": None,
            "verified_hypothesis_id": None,
            "memory_ref": None,
        }
        self_updates = {
            "autonomy_action_failures": 0,
            "autonomy_evidence_action_failures": 0,
        }
    elif phase == AutonomyPhase.HYPOTHESIZE:
        if event.hypothesis_id is None or not event.evidence_refs:
            return _reject(state, phase, "hypothesis_requires_provenance")
        next_phase = AutonomyPhase.REQUEST_EVIDENCE
        goal_updates = {
            "active_hypothesis_id": event.hypothesis_id,
            "hypothesis_confidence": event.confidence,
        }
    elif phase == AutonomyPhase.REQUEST_EVIDENCE:
        axes = event.payload.get("requested_axes")
        if not isinstance(axes, list) or not axes or any(
            not isinstance(axis, str) or not axis for axis in axes
        ):
            return _reject(state, phase, "evidence_request_requires_axes")
        collect_with_tool = event.payload.get("collect_with_tool", False)
        if not isinstance(collect_with_tool, bool):
            return _reject(state, phase, "collect_with_tool_requires_boolean")
        if collect_with_tool and not config.allowed_evidence_tool_actions:
            return _reject(state, phase, "evidence_tool_collection_not_configured")
        next_phase = (
            AutonomyPhase.COLLECT_EVIDENCE
            if collect_with_tool
            else AutonomyPhase.VERIFY
        )
        goal_updates = {"requested_evidence_axes": axes}
    elif phase == AutonomyPhase.COLLECT_EVIDENCE:
        action = event.payload.get("action")
        if (
            not isinstance(action, str)
            or action not in config.allowed_evidence_tool_actions
        ):
            return _reject(state, phase, "evidence_action_not_allowed")
        if action not in config.reversible_evidence_tool_actions:
            return _reject(state, phase, "evidence_action_not_reversible")
        if event.confidence < config.minimum_evidence_action_confidence:
            return _reject(state, phase, "evidence_action_confidence_too_low")
        if not state.goal_state.get("requested_evidence_axes"):
            return _reject(state, phase, "evidence_action_requires_requested_axes")
        next_phase = AutonomyPhase.OBSERVE_EVIDENCE_RESULT
        evidence_tool_action_allowed = True
        evidence_tool_action = action
        goal_updates = {"pending_evidence_tool_action": action}
    elif phase == AutonomyPhase.OBSERVE_EVIDENCE_RESULT:
        success = event.payload.get("success")
        if not isinstance(success, bool):
            return _reject(
                state, phase, "evidence_result_requires_boolean_success"
            )
        if success:
            next_phase = AutonomyPhase.VERIFY
            goal_updates = {"pending_evidence_tool_action": None}
            self_updates = {"autonomy_evidence_action_failures": 0}
        else:
            failures = int(
                state.self_state.get("autonomy_evidence_action_failures", 0)
            ) + 1
            next_phase = (
                AutonomyPhase.COLLECT_EVIDENCE
                if failures < config.maximum_evidence_action_failures
                else AutonomyPhase.ABSTAIN
            )
            reason = (
                "recover_evidence_action"
                if next_phase == AutonomyPhase.COLLECT_EVIDENCE
                else "evidence_failure_budget_exhausted"
            )
            goal_updates = {"pending_evidence_tool_action": None}
            self_updates = {"autonomy_evidence_action_failures": failures}
    elif phase == AutonomyPhase.VERIFY:
        if evidence_decision is None:
            return _reject(state, phase, "verification_requires_accumulator_decision")
        if evidence_decision.status == "accept":
            next_phase = AutonomyPhase.REMEMBER
            memory_write_allowed = True
            goal_updates = {
                "verified_hypothesis_id": event.hypothesis_id,
                "verification_status": "accept",
                "verification_lcb": evidence_decision.causal_lower_bound,
            }
        elif evidence_decision.status == "reject":
            next_phase = AutonomyPhase.ABSTAIN
            reason = "hypothesis_rejected"
            goal_updates = {"verification_status": "reject"}
        else:
            next_phase = AutonomyPhase.REQUEST_EVIDENCE
            reason = "additional_evidence_required"
            goal_updates = {"verification_status": "abstain"}
    elif phase == AutonomyPhase.REMEMBER:
        memory_ref = event.payload.get("memory_ref")
        content_hash = event.payload.get("content_hash")
        if (
            event.hypothesis_id != state.goal_state.get("verified_hypothesis_id")
            or not isinstance(memory_ref, str)
            or not memory_ref
            or not isinstance(content_hash, str)
            or not content_hash
            or not event.evidence_refs
        ):
            return _reject(state, phase, "memory_commit_requires_verified_provenance")
        next_phase = AutonomyPhase.ACT
        goal_updates = {"memory_ref": memory_ref, "memory_content_hash": content_hash}
    elif phase == AutonomyPhase.ACT:
        action = event.payload.get("action")
        if not isinstance(action, str) or action not in config.allowed_tool_actions:
            return _reject(state, phase, "action_not_allowed")
        if action not in config.reversible_tool_actions:
            return _reject(state, phase, "action_not_reversible")
        if event.confidence < config.minimum_action_confidence:
            return _reject(state, phase, "action_confidence_too_low")
        if state.goal_state.get("memory_ref") is None:
            return _reject(state, phase, "action_requires_verified_memory")
        next_phase = AutonomyPhase.OBSERVE_RESULT
        tool_action_allowed = True
        tool_action = action
        goal_updates = {"pending_tool_action": action}
    elif phase == AutonomyPhase.OBSERVE_RESULT:
        success = event.payload.get("success")
        if not isinstance(success, bool):
            return _reject(state, phase, "action_result_requires_boolean_success")
        if success:
            next_phase = AutonomyPhase.OBSERVE
            goal_updates = {"pending_tool_action": None}
            self_updates = {"autonomy_action_failures": 0}
        else:
            failures = int(state.self_state.get("autonomy_action_failures", 0)) + 1
            next_phase = (
                AutonomyPhase.ACT
                if failures < config.maximum_action_failures
                else AutonomyPhase.ABSTAIN
            )
            reason = "recover_action" if next_phase == AutonomyPhase.ACT else "failure_budget_exhausted"
            goal_updates = {"pending_tool_action": None}
            self_updates = {"autonomy_action_failures": failures}

    updated = _updated_state(
        state,
        event,
        next_phase,
        goal_updates=goal_updates,
        self_updates=self_updates,
    )
    return AutonomyTransition(
        state=updated,
        prior_phase=phase,
        next_phase=next_phase,
        accepted=True,
        reason=reason,
        memory_write_allowed=memory_write_allowed,
        tool_action_allowed=tool_action_allowed,
        tool_action=tool_action,
        evidence_tool_action_allowed=evidence_tool_action_allowed,
        evidence_tool_action=evidence_tool_action,
    )

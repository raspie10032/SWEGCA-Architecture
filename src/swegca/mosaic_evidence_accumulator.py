"""Uncertainty-aware evidence accounting for one persistent MOSAIC World."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Literal

import torch

from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_omni import WorldState
from swegca.mosaic_synapse_arbiter import (
    SynapseProposal,
    sufficiency_gated_proposal,
)

EvidenceOutcome = Literal["support", "refute", "insufficient"]
AccumulatorStatus = Literal["accept", "reject", "abstain"]
_STATE_AUTHORITY = object()
_DECISION_AUTHORITY = object()


def _mark_authoritative(value: object, token: object) -> None:
    object.__setattr__(value, "_authority_token", token)


def _require_authoritative_state(state: EvidenceAccumulatorState) -> None:
    if getattr(state, "_authority_token", None) is not _STATE_AUTHORITY:
        raise PermissionError("accumulator state was not issued by this accumulator")


def is_authoritative_accumulator_decision(decision: AccumulatorDecision) -> bool:
    return getattr(decision, "_authority_token", None) is _DECISION_AUTHORITY


def require_authoritative_accumulator_decision(
    decision: AccumulatorDecision,
) -> None:
    if not is_authoritative_accumulator_decision(decision):
        raise PermissionError("accumulator decision is not an authority capability")


@dataclass(frozen=True)
class EvidenceAccumulatorConfig:
    chance_rate: float = 0.2
    accept_margin: float = 0.25
    confidence_level: float = 0.9
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0
    minimum_effective_samples_per_axis: int = 4
    minimum_source_diversity: int = 2
    minimum_source_diversity_per_axis: int = 1
    minimum_context_diversity: int = 4
    recent_window: int = 6
    minimum_recent_samples: int = 4
    regime_change_threshold: float = 0.3
    required_axes: tuple[str, ...] = (
        "observational",
        "counterfactual",
        "intervention",
        "cross_context",
    )

    def __post_init__(self) -> None:
        if not 0 <= self.chance_rate < 1:
            raise ValueError("chance rate must be in [0, 1)")
        if not 0 <= self.accept_margin <= 1 - self.chance_rate:
            raise ValueError("accept margin is outside the probability range")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence level must be in (0, 1)")
        if min(self.beta_prior_alpha, self.beta_prior_beta) <= 0:
            raise ValueError("Beta prior values must be positive")
        if min(
            self.minimum_effective_samples_per_axis,
            self.minimum_source_diversity,
            self.minimum_source_diversity_per_axis,
            self.minimum_context_diversity,
            self.recent_window,
            self.minimum_recent_samples,
        ) <= 0:
            raise ValueError("sample, diversity, and window limits must be positive")
        if self.minimum_recent_samples > self.recent_window:
            raise ValueError("minimum recent samples exceed the recent window")
        if not 0 <= self.regime_change_threshold <= 1:
            raise ValueError("regime threshold must be in [0, 1]")
        if not self.required_axes or len(set(self.required_axes)) != len(
            self.required_axes
        ):
            raise ValueError("required evidence axes must be unique and nonempty")


@dataclass(frozen=True)
class EvidenceObservation:
    hypothesis_id: str
    evidence_address: str
    source_family: str
    context_hash: str
    axis: str
    outcome: EvidenceOutcome
    observed_at: int
    expires_at: int | None = None
    producer_id: str = ""
    producer_confidence: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("hypothesis_id", self.hypothesis_id),
            ("evidence_address", self.evidence_address),
            ("source_family", self.source_family),
            ("context_hash", self.context_hash),
            ("axis", self.axis),
            ("producer_id", self.producer_id),
        ):
            if not value:
                raise ValueError(f"{name} must be nonempty")
        if self.outcome not in ("support", "refute", "insufficient"):
            raise ValueError("unsupported evidence outcome")
        if self.observed_at < 0:
            raise ValueError("observed_at must be nonnegative")
        if self.expires_at is not None and self.expires_at < self.observed_at:
            raise ValueError("expires_at precedes the observation")
        if not math.isfinite(self.producer_confidence) or not (
            0 <= self.producer_confidence <= 1
        ):
            raise ValueError("producer confidence must be finite and in [0, 1]")

    @property
    def proposal_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceGroup:
    source_family: str
    context_hash: str
    supports: int = 0
    refutes: int = 0
    producer_ids: frozenset[str] = frozenset()

    @property
    def effective_support(self) -> float:
        total = self.supports + self.refutes
        return self.supports / total if total else 0.0

    @property
    def effective_refute(self) -> float:
        total = self.supports + self.refutes
        return self.refutes / total if total else 0.0


@dataclass(frozen=True)
class EvidenceAxisState:
    name: str
    groups: tuple[EvidenceGroup, ...] = ()

    @property
    def effective_support(self) -> float:
        return sum(group.effective_support for group in self.groups)

    @property
    def effective_refute(self) -> float:
        return sum(group.effective_refute for group in self.groups)

    @property
    def effective_samples(self) -> float:
        return self.effective_support + self.effective_refute

    @property
    def source_diversity(self) -> int:
        sources = {group.source_family for group in self.groups}
        producers = {
            producer_id
            for group in self.groups
            for producer_id in group.producer_ids
        }
        return min(len(sources), len(producers))

    def record(self, observation: EvidenceObservation) -> EvidenceAxisState:
        groups = list(self.groups)
        key = (observation.source_family, observation.context_hash)
        for index, group in enumerate(groups):
            if (group.source_family, group.context_hash) != key:
                continue
            groups[index] = replace(
                group,
                supports=group.supports + (observation.outcome == "support"),
                refutes=group.refutes + (observation.outcome == "refute"),
                producer_ids=group.producer_ids | {observation.producer_id},
            )
            break
        else:
            groups.append(
                EvidenceGroup(
                    source_family=observation.source_family,
                    context_hash=observation.context_hash,
                    supports=int(observation.outcome == "support"),
                    refutes=int(observation.outcome == "refute"),
                    producer_ids=frozenset({observation.producer_id}),
                )
            )
        return replace(self, groups=tuple(groups))


@dataclass(frozen=True)
class EvidenceAccumulatorState:
    hypothesis_id: str
    axes: tuple[EvidenceAxisState, ...]
    seen_addresses: frozenset[str] = frozenset()
    source_families: frozenset[str] = frozenset()
    context_hashes: frozenset[str] = frozenset()
    producer_ids: frozenset[str] = frozenset()
    recent_outcomes: tuple[int, ...] = ()
    revision: int = 0

    @classmethod
    def empty(
        cls,
        hypothesis_id: str,
        config: EvidenceAccumulatorConfig,
    ) -> EvidenceAccumulatorState:
        if not hypothesis_id:
            raise ValueError("hypothesis id must be nonempty")
        state = cls(
            hypothesis_id=hypothesis_id,
            axes=tuple(EvidenceAxisState(name) for name in config.required_axes),
        )
        _mark_authoritative(state, _STATE_AUTHORITY)
        return state


@dataclass(frozen=True)
class AccumulatorDecision:
    status: AccumulatorStatus
    reason: str
    posterior_mean: float
    causal_lower_bound: float
    overall_upper_bound: float
    effective_sample_size: float
    source_diversity: int
    context_diversity: int
    regime_change_score: float
    revision: int


@dataclass(frozen=True)
class AccumulatorUpdate:
    state: EvidenceAccumulatorState
    previous_decision: AccumulatorDecision
    decision: AccumulatorDecision
    applied: bool
    reason: str
    proposal_hash: str


def wilson_interval(
    supports: float,
    refutes: float,
    confidence_level: float,
) -> tuple[float, float]:
    samples = supports + refutes
    if samples <= 0:
        return 0.0, 1.0
    probability = supports / samples
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    z_squared = z_score * z_score
    denominator = 1 + z_squared / samples
    center = (probability + z_squared / (2 * samples)) / denominator
    radius = (
        z_score
        * math.sqrt(
            probability * (1 - probability) / samples
            + z_squared / (4 * samples * samples)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def assess_accumulator(
    state: EvidenceAccumulatorState,
    config: EvidenceAccumulatorConfig,
) -> AccumulatorDecision:
    _require_authoritative_state(state)
    axes = {axis.name: axis for axis in state.axes}
    required = [axes[name] for name in config.required_axes]
    supports = sum(axis.effective_support for axis in required)
    refutes = sum(axis.effective_refute for axis in required)
    samples = supports + refutes
    posterior_mean = (supports + config.beta_prior_alpha) / (
        samples + config.beta_prior_alpha + config.beta_prior_beta
    )
    intervals = [
        wilson_interval(
            axis.effective_support,
            axis.effective_refute,
            config.confidence_level,
        )
        for axis in required
    ]
    causal_lower_bound = min(lower for lower, _ in intervals)
    _, overall_upper_bound = wilson_interval(
        supports,
        refutes,
        config.confidence_level,
    )
    if len(state.recent_outcomes) >= config.minimum_recent_samples:
        recent_mean = sum(state.recent_outcomes) / len(state.recent_outcomes)
        regime_change_score = abs(recent_mean - posterior_mean)
    else:
        regime_change_score = 0.0

    threshold = config.chance_rate + config.accept_margin
    source_diversity = min(len(state.source_families), len(state.producer_ids))
    context_diversity = min(len(state.context_hashes), len(state.producer_ids))
    if any(
        axis.effective_samples < config.minimum_effective_samples_per_axis
        for axis in required
    ):
        status, reason = "abstain", "minimum_effective_samples"
    elif source_diversity < config.minimum_source_diversity:
        status, reason = "abstain", "source_diversity"
    elif any(
        axis.source_diversity < config.minimum_source_diversity_per_axis
        for axis in required
    ):
        status, reason = "abstain", "axis_source_diversity"
    elif context_diversity < config.minimum_context_diversity:
        status, reason = "abstain", "context_diversity"
    elif regime_change_score >= config.regime_change_threshold:
        status, reason = "abstain", "regime_change_suspected"
    elif causal_lower_bound > threshold:
        status, reason = "accept", "causal_lower_bound"
    elif overall_upper_bound <= threshold:
        status, reason = "reject", "upper_bound_below_threshold"
    else:
        status, reason = "abstain", "uncertain"
    decision = AccumulatorDecision(
        status=status,
        reason=reason,
        posterior_mean=posterior_mean,
        causal_lower_bound=causal_lower_bound,
        overall_upper_bound=overall_upper_bound,
        effective_sample_size=samples,
        source_diversity=source_diversity,
        context_diversity=context_diversity,
        regime_change_score=regime_change_score,
        revision=state.revision,
    )
    _mark_authoritative(decision, _DECISION_AUTHORITY)
    return decision


def update_accumulator(
    state: EvidenceAccumulatorState,
    observation: EvidenceObservation,
    config: EvidenceAccumulatorConfig,
    *,
    current_step: int,
) -> AccumulatorUpdate:
    _require_authoritative_state(state)
    observation.validate()
    if observation.hypothesis_id != state.hypothesis_id:
        raise ValueError("observation and accumulator hypothesis differ")
    proposal_hash = observation.proposal_hash
    previous_decision = assess_accumulator(state, config)
    if observation.axis not in config.required_axes:
        raise ValueError("observation axis is not registered")
    if observation.expires_at is not None and current_step > observation.expires_at:
        return AccumulatorUpdate(
            state,
            previous_decision,
            previous_decision,
            False,
            "expired",
            proposal_hash,
        )
    if observation.outcome == "insufficient":
        return AccumulatorUpdate(
            state,
            previous_decision,
            previous_decision,
            False,
            "insufficient",
            proposal_hash,
        )
    if observation.evidence_address in state.seen_addresses:
        return AccumulatorUpdate(
            state,
            previous_decision,
            previous_decision,
            False,
            "duplicate",
            proposal_hash,
        )

    axes = list(state.axes)
    for index, axis in enumerate(axes):
        if axis.name == observation.axis:
            axes[index] = axis.record(observation)
            break
    recent = (
        state.recent_outcomes + (int(observation.outcome == "support"),)
    )[-config.recent_window :]
    updated = EvidenceAccumulatorState(
        hypothesis_id=state.hypothesis_id,
        axes=tuple(axes),
        seen_addresses=state.seen_addresses | {observation.evidence_address},
        source_families=state.source_families | {observation.source_family},
        context_hashes=state.context_hashes | {observation.context_hash},
        producer_ids=state.producer_ids | {observation.producer_id},
        recent_outcomes=recent,
        revision=state.revision + 1,
    )
    _mark_authoritative(updated, _STATE_AUTHORITY)
    return AccumulatorUpdate(
        updated,
        previous_decision,
        assess_accumulator(updated, config),
        True,
        "applied",
        proposal_hash,
    )


def accumulator_gated_proposal(
    world: WorldState | CognitiveState,
    proposal: SynapseProposal,
    decision: AccumulatorDecision,
) -> SynapseProposal:
    """Let only an accepted external evidence history reach the arbiter."""
    sufficient = torch.full(
        (world.semantic_slots.shape[0],),
        decision.status == "accept",
        dtype=torch.bool,
        device=world.semantic_slots.device,
    )
    return sufficiency_gated_proposal(world, proposal, sufficient)


def append_audit_record(
    path: Path,
    observation: EvidenceObservation,
    update: AccumulatorUpdate,
    *,
    world_hash_before: str,
    world_hash_after: str,
    accumulator_version: str,
    arbiter_version: str,
) -> None:
    """Append one provenance-rich JSON record to an external audit ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "world_hash_before": world_hash_before,
        "hypothesis_id": observation.hypothesis_id,
        "proposal_hash": update.proposal_hash,
        "evidence_addresses": [observation.evidence_address],
        "source_family": observation.source_family,
        "context_hash": observation.context_hash,
        "producer_id": observation.producer_id,
        "outcome": observation.outcome,
        "axis": observation.axis,
        "counterfactual_type": observation.axis,
        "posterior_before": update.previous_decision.posterior_mean,
        "posterior_after": update.decision.posterior_mean,
        "short_horizon_state": (
            sum(update.state.recent_outcomes) / len(update.state.recent_outcomes)
            if update.state.recent_outcomes
            else None
        ),
        "long_horizon_state": update.decision.posterior_mean,
        "causal_lower_bound": update.decision.causal_lower_bound,
        "regime_change_score": update.decision.regime_change_score,
        "decision": update.decision.status,
        "update_applied": update.applied,
        "update_reason": update.reason,
        "world_hash_after": world_hash_after,
        "accumulator_version": accumulator_version,
        "arbiter_version": arbiter_version,
        "revision": update.state.revision,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

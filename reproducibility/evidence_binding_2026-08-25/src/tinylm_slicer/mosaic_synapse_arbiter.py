"""Stateless specialist proposals for one persistent MOSAIC World State."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Mapping, TypeAlias

import torch
from torch import nn

from tinylm_slicer.mosaic_cognitive_kernel import CognitiveState
from tinylm_slicer.mosaic_omni import SLOT_ROLES, WorldState

ArbitratedState = WorldState | CognitiveState


def state_slot_tensor(state: ArbitratedState) -> torch.Tensor:
    """Return one transient slot view without creating another persistent state."""
    if isinstance(state, CognitiveState):
        return torch.cat(
            (state.semantic_slots, state.executive_slots, state.scratch_slots),
            dim=1,
        )
    return state.semantic_slots


def _transient_state_snapshot(state: ArbitratedState) -> ArbitratedState:
    """Clone mutable tensors so a request-local core cannot mutate main state."""

    if isinstance(state, CognitiveState):
        return replace(
            state,
            semantic_slots=state.semantic_slots.detach().clone(),
            executive_slots=state.executive_slots.detach().clone(),
            scratch_slots=state.scratch_slots.detach().clone(),
        )
    return replace(
        state,
        semantic_slots=state.semantic_slots.detach().clone(),
        active_mask=state.active_mask.detach().clone(),
        dirty_mask=state.dirty_mask.detach().clone(),
    )


def _state_tensors(state: ArbitratedState) -> tuple[torch.Tensor, ...]:
    if isinstance(state, CognitiveState):
        return state.semantic_slots, state.executive_slots, state.scratch_slots
    return state.semantic_slots, state.active_mask, state.dirty_mask


@dataclass(frozen=True)
class SynapseProposal:
    """One stateless bounded candidate; hypothesis/addresses grant no authority."""

    source: str
    delta_candidate: torch.Tensor
    confidence: torch.Tensor
    contradiction: torch.Tensor
    uncertainty: torch.Tensor
    target_slot_mask: torch.Tensor
    evidence_addresses: tuple[tuple[str, ...], ...] = ()
    hypothesis_id: str = ""

    def validate(self, world: ArbitratedState) -> None:
        batch, slots, _ = state_slot_tensor(world).shape
        if not self.source:
            raise ValueError("proposal source must be nonempty")
        if tuple(self.delta_candidate.shape) != tuple(state_slot_tensor(world).shape):
            raise ValueError("proposal delta must match World State")
        if tuple(self.target_slot_mask.shape) != (batch, slots):
            raise ValueError("proposal target mask must match World slots")
        if self.target_slot_mask.dtype != torch.bool:
            raise ValueError("proposal target mask must be boolean")
        for name, value in (
            ("confidence", self.confidence),
            ("contradiction", self.contradiction),
            ("uncertainty", self.uncertainty),
        ):
            if tuple(value.shape) != (batch,):
                raise ValueError(f"proposal {name} must have shape [batch]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"proposal {name} must be finite")
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"proposal {name} must be in [0, 1]")
        if not bool(torch.isfinite(self.delta_candidate).all()):
            raise ValueError("proposal delta must be finite")
        if self.evidence_addresses and len(self.evidence_addresses) != batch:
            raise ValueError("proposal evidence addresses must match batch")


@dataclass(frozen=True)
class ArbitrationResult:
    world_state: ArbitratedState
    proposed_delta: torch.Tensor
    proposal_weights: torch.Tensor
    accepted: torch.Tensor
    unresolved_contradiction: torch.Tensor
    sources: tuple[str, ...]
    committed: bool


CognitionCore: TypeAlias = Callable[[ArbitratedState, object], SynapseProposal]


@dataclass(frozen=True)
class DynamicCognitionTrace:
    operation_type: str
    executed_cores: tuple[str, ...]
    fanout_used: bool
    elapsed_ns: int
    primary_weights: torch.Tensor
    manager_retained: bool = False
    worker_state_retained: bool = False


@dataclass(frozen=True)
class DynamicCognitionResult:
    arbitration: ArbitrationResult
    proposals: tuple[SynapseProposal, ...]
    trace: DynamicCognitionTrace


def evidence_delta_proposal(
    world: ArbitratedState,
    evidence_delta: torch.Tensor,
    evidence_logits: torch.Tensor,
    *,
    source: str,
    evidence_addresses: tuple[tuple[str, ...], ...] = (),
    hypothesis_id: str = "",
    target_slot: int | str = "verification",
    consistent_class: int = 0,
    contradiction_class: int = 1,
) -> SynapseProposal:
    """Convert an existing binary evidence readout into a stateless proposal."""
    slot_tensor = state_slot_tensor(world)
    batch, slots, dimension = slot_tensor.shape
    if tuple(evidence_delta.shape) != (batch, dimension):
        raise ValueError("evidence delta must have shape [batch, World dimension]")
    if evidence_logits.ndim != 2 or evidence_logits.shape[0] != batch:
        raise ValueError("evidence logits must have shape [batch, classes]")
    classes = evidence_logits.shape[1]
    if classes < 2 or any(index < 0 for index in (consistent_class, contradiction_class)) or not all(
        0 <= index < classes for index in (consistent_class, contradiction_class)
    ):
        raise ValueError("evidence classes are invalid")
    if isinstance(target_slot, str):
        try:
            target_slot = SLOT_ROLES.index(target_slot)
        except ValueError as error:
            raise ValueError(f"unknown World slot role: {target_slot}") from error
    if not 0 <= target_slot < slots:
        raise ValueError("target slot is outside World State")

    probabilities = evidence_logits.softmax(dim=-1)
    entropy = -(
        probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    ).sum(dim=-1) / math.log(classes)
    delta = torch.zeros_like(slot_tensor)
    delta[:, target_slot] = evidence_delta
    mask = torch.zeros(
        (batch, slots),
        dtype=torch.bool,
        device=slot_tensor.device,
    )
    mask[:, target_slot] = True
    proposal = SynapseProposal(
        source=source,
        delta_candidate=delta,
        confidence=probabilities[:, consistent_class],
        contradiction=probabilities[:, contradiction_class],
        uncertainty=entropy,
        target_slot_mask=mask,
        evidence_addresses=evidence_addresses,
        hypothesis_id=hypothesis_id,
    )
    proposal.validate(world)
    return proposal


def sufficiency_gated_proposal(
    world: ArbitratedState,
    proposal: SynapseProposal,
    sufficient_mask: torch.Tensor,
) -> SynapseProposal:
    """Fail closed when a specialist has no usable input evidence.

    This does not make a direction or semantic claim. It only prevents an
    otherwise valid transient proposal from reaching the arbiter when the
    specialist's independently validated evidence gate reports insufficiency.
    """
    proposal.validate(world)
    batch = state_slot_tensor(world).shape[0]
    if tuple(sufficient_mask.shape) != (batch,):
        raise ValueError("sufficient mask must have shape [batch]")
    if sufficient_mask.dtype != torch.bool:
        raise ValueError("sufficient mask must be boolean")
    gated = SynapseProposal(
        source=proposal.source,
        delta_candidate=proposal.delta_candidate,
        confidence=torch.where(
            sufficient_mask,
            proposal.confidence,
            torch.zeros_like(proposal.confidence),
        ),
        contradiction=proposal.contradiction,
        uncertainty=torch.where(
            sufficient_mask,
            proposal.uncertainty,
            torch.ones_like(proposal.uncertainty),
        ),
        target_slot_mask=proposal.target_slot_mask,
        evidence_addresses=proposal.evidence_addresses,
        hypothesis_id=proposal.hypothesis_id,
    )
    gated.validate(world)
    return gated


class SingleWorldArbiter(nn.Module):
    """Combine transient proposals and optionally commit exactly one World update."""

    def __init__(
        self,
        *,
        maximum_slot_delta: float = 0.1,
        maximum_world_delta: float = 0.5,
        minimum_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if min(maximum_slot_delta, maximum_world_delta) <= 0:
            raise ValueError("delta limits must be positive")
        if not 0 <= minimum_weight <= 1:
            raise ValueError("minimum weight must be in [0, 1]")
        self.maximum_slot_delta = maximum_slot_delta
        self.maximum_world_delta = maximum_world_delta
        self.minimum_weight = minimum_weight

    def forward(
        self,
        world: ArbitratedState,
        proposals: tuple[SynapseProposal, ...],
        *,
        commit: bool = False,
    ) -> ArbitrationResult:
        slot_tensor = state_slot_tensor(world)
        if not proposals:
            empty = slot_tensor.new_zeros((slot_tensor.shape[0], 0))
            return ArbitrationResult(
                world_state=world,
                proposed_delta=torch.zeros_like(slot_tensor),
                proposal_weights=empty,
                accepted=empty.bool(),
                unresolved_contradiction=torch.zeros(
                    slot_tensor.shape[:2],
                    dtype=torch.bool,
                    device=slot_tensor.device,
                ),
                sources=(),
                committed=False,
            )
        for proposal in proposals:
            proposal.validate(world)

        weights = torch.stack(
            [
                proposal.confidence
                * (1 - proposal.contradiction)
                * (1 - proposal.uncertainty)
                for proposal in proposals
            ],
            dim=1,
        )
        accepted = weights >= self.minimum_weight
        bounded = []
        slot_weights = []
        for index, proposal in enumerate(proposals):
            masked = proposal.delta_candidate * proposal.target_slot_mask.unsqueeze(-1)
            slot_norm = torch.linalg.vector_norm(masked, dim=-1, keepdim=True)
            scale = (self.maximum_slot_delta / slot_norm.clamp_min(1e-12)).clamp_max(1)
            bounded.append(masked * scale)
            slot_weights.append(
                weights[:, index : index + 1]
                * accepted[:, index : index + 1]
                * proposal.target_slot_mask
            )
        deltas = torch.stack(bounded, dim=1)
        per_slot_weights = torch.stack(slot_weights, dim=1)
        unresolved_contradiction = torch.zeros(
            slot_tensor.shape[:2],
            dtype=torch.bool,
            device=slot_tensor.device,
        )
        for left_index, left in enumerate(proposals):
            for right_index in range(left_index + 1, len(proposals)):
                right = proposals[right_index]
                if left.source == right.source:
                    continue
                shared = (
                    left.target_slot_mask
                    & right.target_slot_mask
                    & accepted[:, left_index : left_index + 1]
                    & accepted[:, right_index : right_index + 1]
                )
                directional_product = (
                    deltas[:, left_index] * deltas[:, right_index]
                ).sum(dim=-1)
                unresolved_contradiction |= shared & (directional_product < 0)
        proposed_delta = (deltas * per_slot_weights.unsqueeze(-1)).sum(dim=1)
        proposed_delta = proposed_delta / per_slot_weights.sum(dim=1).clamp_min(
            1e-12
        ).unsqueeze(-1)
        proposed_delta = torch.where(
            unresolved_contradiction.unsqueeze(-1),
            torch.zeros_like(proposed_delta),
            proposed_delta,
        )
        world_norm = torch.linalg.vector_norm(proposed_delta.flatten(1), dim=-1)
        world_scale = (
            self.maximum_world_delta / world_norm.clamp_min(1e-12)
        ).clamp_max(1)
        proposed_delta = proposed_delta * world_scale[:, None, None]

        dirty = proposed_delta.abs().any(dim=-1)
        if not commit or not bool(dirty.any()):
            output_world = world
            committed = False
        elif isinstance(world, WorldState):
            output_world = WorldState(
                semantic_slots=world.semantic_slots + proposed_delta,
                active_mask=world.active_mask | dirty,
                dirty_mask=world.dirty_mask | dirty,
                source=f"{world.source}+synapse_arbiter",
                surface_refs=world.surface_refs,
            )
            committed = True
        else:
            semantic_count = world.semantic_slots.shape[1]
            executive_count = world.executive_slots.shape[1]
            semantic_delta, executive_delta, scratch_delta = proposed_delta.split(
                (
                    semantic_count,
                    executive_count,
                    world.scratch_slots.shape[1],
                ),
                dim=1,
            )
            output_world = replace(
                world,
                semantic_slots=world.semantic_slots + semantic_delta,
                executive_slots=world.executive_slots + executive_delta,
                scratch_slots=world.scratch_slots + scratch_delta,
            )
            committed = True
        return ArbitrationResult(
            world_state=output_world,
            proposed_delta=proposed_delta,
            proposal_weights=weights,
            accepted=accepted,
            unresolved_contradiction=unresolved_contradiction,
            sources=tuple(proposal.source for proposal in proposals),
            committed=committed,
        )


def run_dynamic_cognition(
    world: ArbitratedState,
    request: object,
    *,
    operation_type: str,
    resident_cores: Mapping[str, CognitionCore],
    routes: Mapping[str, tuple[str, ...]],
    decisive_weight: float = 0.75,
    arbiter: SingleWorldArbiter | None = None,
) -> DynamicCognitionResult:
    """Run transient manager/workers over main-owned read-only core weights.

    The first core in a route is the fast path. Remaining cores run in
    parallel only when the primary proposal is not decisive. The executor is
    closed before return, and arbitration is preview-only: only main may
    commit the returned proposal.
    """

    if not operation_type:
        raise ValueError("operation_type must be nonempty")
    if not resident_cores:
        raise ValueError("at least one resident cognition core is required")
    if not 0 <= decisive_weight <= 1:
        raise ValueError("decisive_weight must be within [0, 1]")
    try:
        route = routes[operation_type]
    except KeyError as exc:
        raise KeyError(operation_type) from exc
    if not route or len(route) != len(set(route)):
        raise ValueError("a cognition route must contain unique core names")
    unknown = set(route) - set(resident_cores)
    if unknown:
        raise KeyError(sorted(unknown)[0])

    started_ns = time.perf_counter_ns()
    main_before = tuple(tensor.detach().clone() for tensor in _state_tensors(world))

    def execute(core_name: str) -> SynapseProposal:
        snapshot = _transient_state_snapshot(world)
        with torch.inference_mode():
            proposal = resident_cores[core_name](snapshot, request)
        proposal.validate(world)
        if proposal.source != core_name:
            raise ValueError("cognition core source identity changed")
        return proposal

    primary_name = route[0]
    primary = execute(primary_name)
    primary_weights = (
        primary.confidence
        * (1 - primary.contradiction)
        * (1 - primary.uncertainty)
    )
    additional_names = route[1:]
    fanout_used = bool(additional_names) and not bool(
        (primary_weights >= decisive_weight).all()
    )
    proposals = (primary,)
    if fanout_used:
        # Workers and their executor are request-local and die at this boundary.
        with ThreadPoolExecutor(
            max_workers=len(additional_names),
            thread_name_prefix="rozephine-cognition-worker",
        ) as workers:
            futures = {
                name: workers.submit(execute, name) for name in additional_names
            }
            proposals += tuple(futures[name].result() for name in additional_names)

    if any(
        not torch.equal(before, after)
        for before, after in zip(main_before, _state_tensors(world), strict=True)
    ):
        raise RuntimeError("dynamic cognition mutated main persistent state")

    arbitration = (arbiter or SingleWorldArbiter())(
        world,
        proposals,
        commit=False,
    )
    return DynamicCognitionResult(
        arbitration=arbitration,
        proposals=proposals,
        trace=DynamicCognitionTrace(
            operation_type=operation_type,
            executed_cores=tuple(proposal.source for proposal in proposals),
            fanout_used=fanout_used,
            elapsed_ns=time.perf_counter_ns() - started_ns,
            primary_weights=primary_weights.detach().clone(),
        ),
    )

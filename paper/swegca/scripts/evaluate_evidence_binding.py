"""Falsify evidence-address binding on the guarded verification-slot path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from swegca.mosaic_bounded_world_write import (
    BoundedWorldWriteConfig,
    bounded_verification_write,
    cognitive_state_hash,
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BOUND_SOURCES = (
    "paper/swegca/scripts/evaluate_evidence_binding.py",
    "src/swegca/mosaic_evidence_accumulator.py",
    "src/swegca/mosaic_evidence_revision.py",
    "src/swegca/mosaic_synapse_arbiter.py",
    "src/swegca/mosaic_bounded_world_write.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_decision():
    config = EvidenceAccumulatorConfig()
    state = EvidenceAccumulatorState.empty("binding-hypothesis-a", config)
    for index in range(16):
        state = update_accumulator(
            state,
            EvidenceObservation(
                hypothesis_id=state.hypothesis_id,
                evidence_address=f"hypothesis-a:evidence:{index}",
                source_family=f"source:{index % 2}",
                context_hash=f"context:{index}",
                axis=config.required_axes[index % len(config.required_axes)],
                outcome="support",
                observed_at=index,
                producer_id=f"producer:{index}",
            ),
            config,
            current_step=index,
        ).state
    return assess_accumulator(state, config)


def evaluate_case(
    state: CognitiveState,
    gates,
    name: str,
    addresses: tuple[tuple[str, ...], ...],
) -> dict[str, object]:
    proposal = evidence_delta_proposal(
        state,
        torch.ones(1, state.semantic_slots.shape[2]),
        torch.tensor([[8.0, -8.0]]),
        source=f"binding-probe:{name}",
        evidence_addresses=addresses,
        target_slot="verification",
    )
    before = cognitive_state_hash(state)
    result = bounded_verification_write(
        state,
        proposal,
        gates,
        BoundedWorldWriteConfig(),
        commit=True,
    )
    return {
        "authorized": result.authorized,
        "committed": result.committed,
        "reason": result.reason,
        "state_hash_changed": cognitive_state_hash(result.state) != before,
        "receipt_evidence_refs": (
            list(result.receipt.evidence_refs) if result.receipt is not None else []
        ),
    }


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    decision = accepted_decision()
    gates = world_write_gates_from_decision(
        decision,
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
    state = CognitiveState(
        semantic_slots=torch.zeros(1, 20, 32),
        executive_slots=torch.zeros(1, 6, 32),
        scratch_slots=torch.zeros(1, 6, 32),
    )
    cases = {
        "matching_address": evaluate_case(
            state, gates, "matching", (("hypothesis-a:evidence:0",),)
        ),
        "empty_address": evaluate_case(state, gates, "empty", ()),
        "mismatched_address": evaluate_case(
            state, gates, "mismatched", (("hypothesis-b:evidence:0",),)
        ),
    }
    checks = {
        "matching_address_commits": cases["matching_address"]["committed"] is True,
        "empty_address_no_commit": cases["empty_address"]["committed"] is False,
        "mismatched_address_no_commit": cases["mismatched_address"]["committed"]
        is False,
    }
    payload = {
        "artifact": "swegca-evidence-binding-evaluation-v1",
        "commit": git_commit(),
        "source_sha256": {
            relative: sha256_file(REPOSITORY_ROOT / relative)
            for relative in BOUND_SOURCES
        },
        "scope": "audited guarded verification-slot path",
        "decision_status": decision.status,
        "cases": cases,
        "checks": checks,
        "contract_passed": all(checks.values()),
        "interpretation": (
            "A false contract result narrows the evidence-gated mutation claim; "
            "it is not a task-performance score."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return int(args.require_pass and not payload["contract_passed"])


if __name__ == "__main__":
    raise SystemExit(main())

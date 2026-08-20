"""Run fixed synthetic SWEGCA conflict and evidence-discipline cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_cognitive_slot_topology import cognitive_slot_topology
from swegca.mosaic_evidence_accumulator import (
    EvidenceAccumulatorConfig,
    EvidenceAccumulatorState,
    EvidenceObservation,
    assess_accumulator,
    update_accumulator,
)
from swegca.mosaic_synapse_arbiter import (
    SingleWorldArbiter,
    evidence_delta_proposal,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BOUND_SOURCES = (
    "paper/swegca/scripts/evaluate_contract_matrix.py",
    "src/swegca/mosaic_evidence_accumulator.py",
    "src/swegca/mosaic_synapse_arbiter.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def cognitive_state() -> CognitiveState:
    return CognitiveState(
        semantic_slots=torch.zeros(1, 20, 32),
        executive_slots=torch.zeros(1, 6, 32),
        scratch_slots=torch.zeros(1, 6, 32),
    )


def proposal(state, value: float, source: str, target: str):
    return evidence_delta_proposal(
        state,
        torch.full((1, 32), value),
        torch.tensor([[8.0, -8.0]]),
        source=source,
        target_slot=target,
    )


def conflict_matrix() -> dict[str, object]:
    state = cognitive_state()
    arbiter = SingleWorldArbiter(
        maximum_slot_delta=10.0,
        maximum_world_delta=20.0,
        minimum_weight=0.01,
    )
    topology = cognitive_slot_topology(state)
    verification = topology.fixed_roles["verification"]
    global_slot = topology.fixed_roles["global"]
    cases = {
        "opposing_distinct_sources": (
            proposal(state, 1.0, "source-a", "verification"),
            proposal(state, -0.5, "source-b", "verification"),
        ),
        "opposing_same_source": (
            proposal(state, 1.0, "source-a", "verification"),
            proposal(state, -0.5, "source-a", "verification"),
        ),
        "disjoint_targets": (
            proposal(state, 1.0, "source-a", "verification"),
            proposal(state, -0.5, "source-b", "global"),
        ),
    }
    results: dict[str, object] = {}
    for name, proposals in cases.items():
        result = arbiter(state, proposals, commit=False)
        results[name] = {
            "verification_unresolved": bool(
                result.unresolved_contradiction[0, verification]
            ),
            "verification_delta_norm": float(
                torch.linalg.vector_norm(result.proposed_delta[0, verification]).item()
            ),
            "global_delta_norm": float(
                torch.linalg.vector_norm(result.proposed_delta[0, global_slot]).item()
            ),
            "committed": result.committed,
        }
    checks = {
        "distinct_opposition_no_commit_on_slot": (
            results["opposing_distinct_sources"]["verification_unresolved"] is True
            and results["opposing_distinct_sources"]["verification_delta_norm"] == 0.0
        ),
        "same_source_is_not_independence": (
            results["opposing_same_source"]["verification_unresolved"] is False
            and results["opposing_same_source"]["verification_delta_norm"] > 0.0
        ),
        "disjoint_targets_survive": (
            results["disjoint_targets"]["verification_delta_norm"] > 0.0
            and results["disjoint_targets"]["global_delta_norm"] > 0.0
        ),
        "evaluation_is_dry_run": all(
            result["committed"] is False for result in results.values()
        ),
    }
    return {"cases": results, "checks": checks, "passed": all(checks.values())}


def observation(
    hypothesis_id: str,
    index: int,
    config: EvidenceAccumulatorConfig,
    *,
    outcome: str,
    source_family: str | None = None,
    evidence_address: str | None = None,
    expires_at: int | None = None,
) -> EvidenceObservation:
    return EvidenceObservation(
        hypothesis_id=hypothesis_id,
        evidence_address=evidence_address or f"evidence:{index}",
        source_family=source_family or f"source:{index % 2}",
        context_hash=f"context:{index}",
        axis=config.required_axes[index % len(config.required_axes)],
        outcome=outcome,
        observed_at=index,
        expires_at=expires_at,
        producer_id=f"producer:{index}",
    )


def accumulated_decision(
    name: str,
    outcomes: list[str],
    config: EvidenceAccumulatorConfig,
    *,
    one_source: bool = False,
):
    state = EvidenceAccumulatorState.empty(name, config)
    for index, outcome in enumerate(outcomes):
        update = update_accumulator(
            state,
            observation(
                name,
                index,
                config,
                outcome=outcome,
                source_family="one-source" if one_source else None,
            ),
            config,
            current_step=index,
        )
        state = update.state
    return assess_accumulator(state, config)


def accumulator_matrix() -> dict[str, object]:
    default = EvidenceAccumulatorConfig()
    empty = EvidenceAccumulatorState.empty("empty", default)
    accepted = accumulated_decision("accepted", ["support"] * 18, default)
    rejected = accumulated_decision("rejected", ["refute"] * 18, default)
    one_source = accumulated_decision(
        "one-source", ["support"] * 18, default, one_source=True
    )

    duplicate_state = EvidenceAccumulatorState.empty("duplicate", default)
    first = observation("duplicate", 0, default, outcome="support")
    duplicate_state = update_accumulator(
        duplicate_state, first, default, current_step=0
    ).state
    duplicate = update_accumulator(duplicate_state, first, default, current_step=0)

    expired_state = EvidenceAccumulatorState.empty("expired", default)
    expired = update_accumulator(
        expired_state,
        observation("expired", 0, default, outcome="support", expires_at=0),
        default,
        current_step=1,
    )

    regime_config = EvidenceAccumulatorConfig(
        recent_window=4,
        minimum_recent_samples=4,
        regime_change_threshold=0.2,
    )
    regime = accumulated_decision(
        "regime",
        ["support"] * 18 + ["refute"] * 4,
        regime_config,
    )
    cases = {
        "empty": {
            "status": assess_accumulator(empty, default).status,
            "reason": assess_accumulator(empty, default).reason,
        },
        "accepted": {"status": accepted.status, "reason": accepted.reason},
        "rejected": {"status": rejected.status, "reason": rejected.reason},
        "one_source": {"status": one_source.status, "reason": one_source.reason},
        "duplicate": {"applied": duplicate.applied, "reason": duplicate.reason},
        "expired": {"applied": expired.applied, "reason": expired.reason},
        "regime_change": {"status": regime.status, "reason": regime.reason},
    }
    checks = {
        "empty_abstains": cases["empty"]
        == {"status": "abstain", "reason": "minimum_effective_samples"},
        "diverse_support_accepts": cases["accepted"]
        == {"status": "accept", "reason": "causal_lower_bound"},
        "diverse_refutation_rejects": cases["rejected"]
        == {"status": "reject", "reason": "upper_bound_below_threshold"},
        "one_source_abstains": cases["one_source"]
        == {"status": "abstain", "reason": "source_diversity"},
        "duplicate_not_applied": cases["duplicate"]
        == {"applied": False, "reason": "duplicate"},
        "expired_not_applied": cases["expired"]
        == {"applied": False, "reason": "expired"},
        "regime_change_abstains": cases["regime_change"]
        == {"status": "abstain", "reason": "regime_change_suspected"},
    }
    return {"cases": cases, "checks": checks, "passed": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    conflict = conflict_matrix()
    accumulator = accumulator_matrix()
    payload = {
        "artifact": "swegca-contract-matrix-v1",
        "commit": git_commit(),
        "source_sha256": {
            relative: sha256_file(REPOSITORY_ROOT / relative)
            for relative in BOUND_SOURCES
        },
        "conflict_matrix": conflict,
        "accumulator_matrix": accumulator,
        "contract_passed": conflict["passed"] and accumulator["passed"],
        "scope": "fixed synthetic development evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return int(args.require_pass and not payload["contract_passed"])


if __name__ == "__main__":
    raise SystemExit(main())

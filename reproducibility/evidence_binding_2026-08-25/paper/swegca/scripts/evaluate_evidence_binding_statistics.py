"""Source-diverse statistical evaluation of exact evidence-bound World writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import torch

from tinylm_slicer.mosaic_bounded_world_write import (
    BoundedWorldWriteConfig,
    WorldWriteGates,
    bounded_verification_write,
    cognitive_state_hash,
    world_write_gates_from_decision,
)
from tinylm_slicer.mosaic_cognitive_kernel import CognitiveState
from tinylm_slicer.mosaic_evidence_accumulator import (
    AccumulatorDecision,
    EvidenceAccumulatorConfig,
    EvidenceAccumulatorState,
    EvidenceObservation,
    assess_accumulator,
    update_accumulator,
)
from tinylm_slicer.mosaic_synapse_arbiter import (
    SynapseProposal,
    evidence_delta_proposal,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BOUND_SOURCES = (
    "paper/swegca/scripts/evaluate_evidence_binding_statistics.py",
    "src/tinylm_slicer/mosaic_evidence_accumulator.py",
    "src/tinylm_slicer/mosaic_evidence_revision.py",
    "src/tinylm_slicer/mosaic_synapse_arbiter.py",
    "src/tinylm_slicer/mosaic_bounded_world_write.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*values: object) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "reviewer-supplement-without-git-metadata"


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    z = NormalDist().inv_cdf(0.975)
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def state() -> CognitiveState:
    return CognitiveState(
        semantic_slots=torch.zeros(1, 20, 32),
        executive_slots=torch.zeros(1, 6, 32),
        scratch_slots=torch.zeros(1, 6, 32),
    )


def accepted_decision(
    split: str,
    split_seed: int,
    group_index: int,
    generation: dict[str, object],
) -> tuple[AccumulatorDecision, dict[str, int]]:
    rng = random.Random(split_seed + group_index * 104729)
    config = EvidenceAccumulatorConfig()
    hypothesis_id = f"binding:{split}:{stable_id(split_seed, group_index)[:24]}"
    accumulator = EvidenceAccumulatorState.empty(hypothesis_id, config)
    source_min, source_max = generation["source_family_count_range"]
    context_min, context_max = generation["context_count_range"]
    producer_min, producer_max = generation["producer_count_range"]
    source_count = rng.randint(int(source_min), int(source_max))
    context_count = rng.randint(int(context_min), int(context_max))
    producer_count = rng.randint(int(producer_min), int(producer_max))
    observations = int(generation["observations_per_history"])
    axes = tuple(generation["required_axes"])
    for index in range(observations):
        axis_index = index % len(axes)
        axis_round = index // len(axes)
        source = f"source:{group_index}:{index % source_count}"
        context = (
            f"context:{group_index}:"
            f"{(axis_index * (observations // len(axes)) + axis_round) % context_count}"
        )
        producer = f"producer:{group_index}:{index % producer_count}"
        axis = str(axes[axis_index])
        address = "evidence:sha256:" + stable_id(
            split, hypothesis_id, source, context, producer, axis, index
        )
        accumulator = update_accumulator(
            accumulator,
            EvidenceObservation(
                hypothesis_id=hypothesis_id,
                evidence_address=address,
                source_family=source,
                context_hash=context,
                axis=axis,
                outcome="support",
                observed_at=index,
                producer_id=producer,
            ),
            config,
            current_step=index,
        ).state
    decision = assess_accumulator(accumulator, config)
    if decision.status != "accept":
        raise AssertionError(f"generated history did not accept: {decision.reason}")
    return decision, {
        "source_families": source_count,
        "contexts": context_count,
        "producers": producer_count,
    }


def proposal(
    world: CognitiveState,
    decision: AccumulatorDecision,
    case_id: str,
    addresses: tuple[str, ...],
) -> SynapseProposal:
    generator = torch.Generator().manual_seed(int(case_id[:16], 16))
    delta = torch.randn(1, 32, generator=generator)
    return evidence_delta_proposal(
        world,
        delta,
        torch.tensor([[8.0, -8.0]]),
        source=f"statistical-binding:{case_id}",
        evidence_addresses=(addresses,),
        hypothesis_id=decision.hypothesis_id,
        target_slot="verification",
    )


def gates(
    decision: AccumulatorDecision, candidate: SynapseProposal
) -> WorldWriteGates:
    return world_write_gates_from_decision(
        decision,
        proposal=candidate,
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


def plain_gates(decision: AccumulatorDecision) -> WorldWriteGates:
    return WorldWriteGates(
        evidence_status=decision.status,
        causal_lower_bound=decision.causal_lower_bound,
        source_diversity=decision.source_diversity,
        context_diversity=decision.context_diversity,
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


def execute_case(
    case_id: str,
    family: str,
    variant: int,
    decision: AccumulatorDecision,
) -> dict[str, object]:
    world = state()
    before_hash = cognitive_state_hash(world)
    accepted = decision.evidence_addresses
    single = (accepted[(variant * 3) % len(accepted)],)
    multiple = tuple(
        accepted[(variant + offset) % len(accepted)] for offset in range(2 + variant)
    )
    base = proposal(world, decision, case_id, single)
    expected_commit = family in {"valid_single_address", "valid_multi_address_subset"}
    result = None
    issued_gates = None
    started = time.perf_counter_ns()
    try:
        candidate = base
        if family == "valid_single_address":
            pass
        elif family == "valid_multi_address_subset":
            candidate = proposal(world, decision, case_id, multiple)
        elif family == "empty_addresses":
            candidate = proposal(world, decision, case_id, ())
        elif family == "foreign_address_only":
            candidate = proposal(world, decision, case_id, (f"foreign:{case_id}",))
        elif family == "mixed_accepted_and_foreign_addresses":
            candidate = proposal(
                world, decision, case_id, (single[0], f"foreign:{case_id}")
            )
        elif family == "wrong_hypothesis":
            candidate = replace(base, hypothesis_id=f"wrong:{case_id}")
        elif family == "delta_changed_after_binding":
            issued_gates = gates(decision, base)
            changed = base.delta_candidate.clone()
            changed[:, 30, variant] += 0.001
            candidate = replace(base, delta_candidate=changed)
        elif family == "target_changed_after_binding":
            issued_gates = gates(decision, base)
            changed = base.target_slot_mask.clone()
            changed[:, 30] = False
            changed[:, variant] = True
            candidate = replace(base, target_slot_mask=changed)
        elif family == "proposal_metadata_changed_after_binding":
            issued_gates = gates(decision, base)
            candidate = replace(base, source=f"changed:{base.source}:{variant}")
        elif family == "stale_or_forged_authority":
            if variant == 3:
                gates(replace(decision, revision=decision.revision + 1), base)
            issued_gates = gates(decision, base)
            if variant == 0:
                issued_gates = replace(
                    issued_gates,
                    causal_lower_bound=min(1.0, issued_gates.causal_lower_bound + 0.001),
                )
            elif variant == 1:
                issued_gates = replace(
                    issued_gates, _proposal_binding_digest="0" * 64
                )
            elif variant == 2:
                issued_gates = plain_gates(decision)
        else:
            raise ValueError(f"unknown case family: {family}")
        if issued_gates is None:
            issued_gates = gates(decision, candidate)
        result = bounded_verification_write(
            world,
            candidate,
            issued_gates,
            BoundedWorldWriteConfig(),
            commit=True,
        )
        reason = result.reason
    except (PermissionError, ValueError) as error:
        reason = f"{type(error).__name__}: {error}"
    elapsed_ns = time.perf_counter_ns() - started
    committed = bool(result is not None and result.committed)
    authorized = bool(result is not None and result.authorized)
    returned_state = result.state if result is not None else world
    state_changed = cognitive_state_hash(returned_state) != before_hash
    receipt = result.receipt if result is not None else None
    receipt_binding_match = (
        receipt is not None
        and issued_gates is not None
        and receipt.hypothesis_id == candidate.hypothesis_id
        and receipt.evidence_refs
        == tuple(address for batch in candidate.evidence_addresses for address in batch)
        and receipt.proposal_binding_digest == issued_gates._proposal_binding_digest
    )
    return {
        "case_id": case_id,
        "family": family,
        "variant": variant,
        "expected_commit": expected_commit,
        "authorized": authorized,
        "committed": committed,
        "state_changed": state_changed,
        "receipt_binding_match": receipt_binding_match if committed else None,
        "reason": reason,
        "elapsed_ns": elapsed_ns,
    }


def verify_heldout_manifest(
    manifest_path: Path | None,
    config_path: Path,
    source_hashes: dict[str, str],
) -> None:
    if manifest_path is None:
        raise PermissionError("heldout requires a frozen implementation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise PermissionError("heldout config differs from frozen manifest")
    if manifest.get("source_sha256") != source_hashes:
        raise PermissionError("heldout sources differ from frozen manifest")


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(args.output)
    ledger_path = args.output.with_name("cases.jsonl")
    if ledger_path.exists():
        raise FileExistsError(ledger_path)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split = config["splits"][args.split]
    families = [item["name"] for item in config["case_families"]]
    expected_cases = int(split["expected_cases"])
    calculated_cases = (
        int(split["history_groups"])
        * len(families)
        * int(split["variants_per_family_per_group"])
    )
    if expected_cases != calculated_cases:
        raise ValueError("configured expected case count is inconsistent")
    source_hashes = {
        relative: sha256_file(REPOSITORY_ROOT / relative) for relative in BOUND_SOURCES
    }
    if args.split == "heldout":
        verify_heldout_manifest(args.frozen_manifest, args.config, source_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    family_rows: dict[str, list[dict[str, object]]] = {name: [] for name in families}
    diversity = {"source_families": [], "contexts": [], "producers": []}
    total_started = time.perf_counter_ns()
    history_groups = int(split["history_groups"])
    progress_interval = max(1, history_groups // 10)
    with ledger_path.open("x", encoding="utf-8", newline="\n") as ledger:
        for group_index in range(history_groups):
            decision, observed_diversity = accepted_decision(
                args.split,
                int(split["seed"]),
                group_index,
                config["history_generation"],
            )
            for key, value in observed_diversity.items():
                diversity[key].append(value)
            for family in families:
                for variant in range(int(split["variants_per_family_per_group"])):
                    case_id = stable_id(
                        args.split, split["seed"], group_index, family, variant
                    )
                    row = execute_case(case_id, family, variant, decision)
                    family_rows[family].append(row)
                    ledger.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
            completed_groups = group_index + 1
            if completed_groups % progress_interval == 0:
                print(
                    f"progress {completed_groups}/{history_groups} histories",
                    flush=True,
                )
    total_elapsed_ns = time.perf_counter_ns() - total_started
    summaries = {}
    all_rows = [row for rows in family_rows.values() for row in rows]
    for family, rows in family_rows.items():
        expected_commit = bool(rows[0]["expected_commit"])
        expected_outcomes = sum(
            int(bool(row["committed"]) == expected_commit) for row in rows
        )
        failures = len(rows) - expected_outcomes
        state_mutations_on_rejection = sum(
            int(bool(row["state_changed"])) for row in rows if not expected_commit
        )
        receipt_mismatches = sum(
            int(row["receipt_binding_match"] is False) for row in rows
        )
        latencies = [int(row["elapsed_ns"]) for row in rows]
        summaries[family] = {
            "cases": len(rows),
            "expected_commit": expected_commit,
            "expected_outcomes": expected_outcomes,
            "expected_outcome_rate": expected_outcomes / len(rows),
            "expected_outcome_wilson_95": wilson_interval(expected_outcomes, len(rows)),
            "failures": failures,
            "failure_rate": failures / len(rows),
            "failure_wilson_95": wilson_interval(failures, len(rows)),
            "rejected_state_mutations": state_mutations_on_rejection,
            "receipt_binding_mismatches": receipt_mismatches,
            "latency_ns": {
                "median": percentile(latencies, 0.5),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
            },
        }
    invalid_rows = [row for row in all_rows if not row["expected_commit"]]
    valid_rows = [row for row in all_rows if row["expected_commit"]]
    invalid_commits = sum(int(bool(row["committed"])) for row in invalid_rows)
    valid_rejections = sum(int(not bool(row["committed"])) for row in valid_rows)
    rejected_mutations = sum(int(bool(row["state_changed"])) for row in invalid_rows)
    receipt_mismatches = sum(
        int(row["receipt_binding_match"] is False) for row in all_rows
    )
    latencies = [int(row["elapsed_ns"]) for row in all_rows]
    report = {
        "schema_version": "swegca-evidence-binding-statistics-v1",
        "split": args.split,
        "single_pass_fail_label": None,
        "git_commit": git_commit(),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "source_sha256": source_hashes,
        "case_ledger": str(ledger_path.resolve()),
        "case_ledger_sha256": sha256_file(ledger_path),
        "history_groups": int(split["history_groups"]),
        "cases": len(all_rows),
        "family_statistics": summaries,
        "aggregate_statistics": {
            "invalid_cases": len(invalid_rows),
            "unauthorized_commits": invalid_commits,
            "unauthorized_commit_rate": invalid_commits / len(invalid_rows),
            "unauthorized_commit_wilson_95": wilson_interval(
                invalid_commits, len(invalid_rows)
            ),
            "valid_cases": len(valid_rows),
            "valid_rejections": valid_rejections,
            "valid_rejection_rate": valid_rejections / len(valid_rows),
            "valid_rejection_wilson_95": wilson_interval(
                valid_rejections, len(valid_rows)
            ),
            "rejected_case_state_mutations": rejected_mutations,
            "rejected_case_state_mutation_rate": rejected_mutations
            / len(invalid_rows),
            "receipt_binding_mismatches": receipt_mismatches,
            "receipt_binding_mismatch_rate": receipt_mismatches / len(all_rows),
            "latency_ns": {
                "median": percentile(latencies, 0.5),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
            },
        },
        "observed_history_diversity": {
            key: {"minimum": min(values), "maximum": max(values)}
            for key, values in diversity.items()
        },
        "total_elapsed_ns": total_elapsed_ns,
        "claim_boundary": config["zero_observed_failures_interpretation"][
            "claim_boundary"
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "stress", "heldout"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

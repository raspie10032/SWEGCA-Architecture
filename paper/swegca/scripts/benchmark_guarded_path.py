"""Measure selected SWEGCA guarded-path operations separately."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path
from time import perf_counter_ns

import torch

from swegca.mosaic_bounded_world_write import (
    BoundedWorldWriteConfig,
    bounded_verification_write,
    rollback_bounded_verification_write,
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
from swegca.mosaic_synapse_arbiter import (
    SingleWorldArbiter,
    evidence_delta_proposal,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BOUND_SOURCES = (
    "paper/swegca/scripts/benchmark_guarded_path.py",
    "src/swegca/mosaic_evidence_accumulator.py",
    "src/swegca/mosaic_synapse_arbiter.py",
    "src/swegca/mosaic_bounded_world_write.py",
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


def fixtures():
    evidence_config = EvidenceAccumulatorConfig()
    accumulator = EvidenceAccumulatorState.empty(
        "benchmark-hypothesis", evidence_config
    )
    for index in range(18):
        accumulator = update_accumulator(
            accumulator,
            EvidenceObservation(
                hypothesis_id=accumulator.hypothesis_id,
                evidence_address=f"benchmark:evidence:{index}",
                source_family=f"benchmark:source:{index % 2}",
                context_hash=f"benchmark:context:{index}",
                axis=evidence_config.required_axes[
                    index % len(evidence_config.required_axes)
                ],
                outcome="support",
                observed_at=index,
                producer_id=f"benchmark:producer:{index}",
            ),
            evidence_config,
            current_step=index,
        ).state
    decision = assess_accumulator(accumulator, evidence_config)
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
    proposal = evidence_delta_proposal(
        state,
        torch.ones(1, 32),
        torch.tensor([[8.0, -8.0]]),
        source="benchmark-producer",
        evidence_addresses=(("benchmark:evidence:0",),),
        target_slot="verification",
    )
    write_config = BoundedWorldWriteConfig()
    arbiter = SingleWorldArbiter(
        maximum_slot_delta=write_config.maximum_slot_delta,
        maximum_world_delta=write_config.maximum_slot_delta,
        minimum_weight=write_config.minimum_proposal_weight,
    )
    committed = bounded_verification_write(
        state, proposal, gates, write_config, commit=True
    )
    if committed.receipt is None:
        raise RuntimeError("benchmark fixture did not commit")
    return (
        accumulator,
        evidence_config,
        state,
        proposal,
        gates,
        write_config,
        arbiter,
        committed,
    )


def measure(operation, warmups: int, samples: int) -> dict[str, float | int]:
    for _ in range(warmups):
        operation()
    timings: list[int] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(samples):
            started = perf_counter_ns()
            operation()
            timings.append(perf_counter_ns() - started)
    finally:
        if was_enabled:
            gc.enable()
    ordered = sorted(timings)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples": samples,
        "median_ns": int(statistics.median(ordered)),
        "p95_ns": ordered[p95_index],
        "minimum_ns": ordered[0],
        "maximum_ns": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.warmups, args.samples) <= 0:
        raise ValueError("warmups and samples must be positive")

    (
        accumulator,
        evidence_config,
        state,
        proposal,
        gates,
        write_config,
        arbiter,
        committed,
    ) = fixtures()
    operations = {
        "assess_accumulator": lambda: assess_accumulator(accumulator, evidence_config),
        "single_proposal_arbitration_dry_run": lambda: arbiter(
            state, (proposal,), commit=False
        ),
        "guarded_write_dry_run": lambda: bounded_verification_write(
            state, proposal, gates, write_config, commit=False
        ),
        "guarded_write_commit": lambda: bounded_verification_write(
            state, proposal, gates, write_config, commit=True
        ),
        "receipt_checked_rollback": lambda: rollback_bounded_verification_write(
            committed.state, committed.receipt
        ),
    }
    measurements = {
        name: measure(operation, args.warmups, args.samples)
        for name, operation in operations.items()
    }
    payload = {
        "artifact": "swegca-guarded-path-benchmark-v1",
        "commit": git_commit(),
        "source_sha256": {
            relative: sha256_file(REPOSITORY_ROOT / relative)
            for relative in BOUND_SOURCES
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "warmups": args.warmups,
        "measurements": measurements,
        "scope": (
            "single-process development microbenchmark; operations are isolated "
            "and do not represent end-to-end cognition"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

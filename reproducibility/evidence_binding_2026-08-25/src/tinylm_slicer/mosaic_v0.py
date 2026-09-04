from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tinylm_slicer.conversation_memory import ConversationMemory


Vector = tuple[float, ...]


@dataclass(frozen=True)
class MosaicConfig:
    workspace_dim: int = 4
    operator_top_k: int = 4
    halt_tolerance: float = 1e-6
    max_update_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.workspace_dim <= 0 or self.operator_top_k <= 0:
            raise ValueError("workspace_dim and operator_top_k must be positive")
        if self.halt_tolerance <= 0 or self.max_update_norm <= 0:
            raise ValueError("halt_tolerance and max_update_norm must be positive")


@dataclass(frozen=True)
class LowRankBasis:
    name: str
    left: Vector
    right: Vector
    bias: float = 0.0

    def validate(self, dimension: int) -> None:
        if not self.name:
            raise ValueError("basis name must not be empty")
        if len(self.left) != dimension or len(self.right) != dimension:
            raise ValueError(f"basis {self.name} does not match workspace dimension")


@dataclass(frozen=True)
class OperatorCode:
    address: str
    tags: frozenset[str]
    coefficients: tuple[tuple[str, float], ...]
    confidence: float = 1.0
    priority: int = 0
    conflicts: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.address or not self.tags or not self.coefficients:
            raise ValueError("operator address, tags, and coefficients are required")
        if not 0 <= self.confidence <= 1:
            raise ValueError("operator confidence must be between 0 and 1")


@dataclass(frozen=True)
class SynthesizedOperator:
    coefficients: tuple[tuple[str, float], ...]
    selected: tuple[str, ...]
    disabled: tuple[str, ...]
    conflicts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MosaicRun:
    success: bool
    halted: bool
    stalled: bool
    steps: int
    final_state: Vector
    final_error: float
    operator: SynthesizedOperator
    trace: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "halted": self.halted,
            "stalled": self.stalled,
            "steps": self.steps,
            "final_state": list(self.final_state),
            "final_error": round(self.final_error, 9),
            "operator": {
                "coefficients": dict(self.operator.coefficients),
                "selected": list(self.operator.selected),
                "disabled": list(self.operator.disabled),
                "conflicts": [list(pair) for pair in self.operator.conflicts],
            },
            "trace": list(self.trace),
        }


class OperatorArchive:
    def __init__(self, operators: Iterable[OperatorCode]):
        self.operators = tuple(operators)
        addresses = [operator.address for operator in self.operators]
        if len(addresses) != len(set(addresses)):
            raise ValueError("operator addresses must be unique")

    def search(self, query: str, *, top_k: int) -> tuple[OperatorCode, ...]:
        if top_k <= 0:
            return ()
        tokens = set(re.findall(r"\w+", query.casefold()))
        ranked: list[tuple[int, OperatorCode]] = []
        for operator in self.operators:
            overlap = len(tokens & {tag.casefold() for tag in operator.tags})
            if overlap:
                ranked.append((overlap, operator))
        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[1].priority,
                -item[1].confidence,
                item[1].address,
            )
        )
        return tuple(operator for _, operator in ranked[:top_k])


def synthesize_operators(
    candidates: Iterable[OperatorCode],
) -> SynthesizedOperator:
    selected: list[OperatorCode] = []
    disabled: list[str] = []
    conflicts: list[tuple[str, str]] = []
    coefficients: dict[str, float] = {}

    for candidate in candidates:
        blocker = next(
            (
                active
                for active in selected
                if candidate.address in active.conflicts
                or active.address in candidate.conflicts
            ),
            None,
        )
        if blocker is not None:
            disabled.append(candidate.address)
            conflicts.append((blocker.address, candidate.address))
            continue
        selected.append(candidate)
        for basis, coefficient in candidate.coefficients:
            coefficients[basis] = coefficients.get(basis, 0.0) + coefficient

    return SynthesizedOperator(
        coefficients=tuple(sorted(coefficients.items())),
        selected=tuple(operator.address for operator in selected),
        disabled=tuple(disabled),
        conflicts=tuple(conflicts),
    )


class RecurrentCell:
    def __init__(
        self,
        bases: Iterable[LowRankBasis],
        *,
        dimension: int,
        max_update_norm: float,
    ):
        self.bases = {basis.name: basis for basis in bases}
        if not self.bases:
            raise ValueError("at least one operator basis is required")
        for basis in self.bases.values():
            basis.validate(dimension)
        self.dimension = dimension
        self.max_update_norm = max_update_norm

    def apply(
        self,
        state: Vector,
        operator: SynthesizedOperator,
    ) -> tuple[Vector, float]:
        if len(state) != self.dimension:
            raise ValueError("state does not match workspace dimension")
        delta = [0.0] * self.dimension
        for basis_name, coefficient in operator.coefficients:
            try:
                basis = self.bases[basis_name]
            except KeyError as exc:
                raise ValueError(f"unknown operator basis: {basis_name}") from exc
            activation = sum(
                weight * value for weight, value in zip(basis.right, state)
            ) + basis.bias
            for index, value in enumerate(basis.left):
                delta[index] += coefficient * activation * value

        norm = math.sqrt(sum(value * value for value in delta))
        if norm > self.max_update_norm:
            scale = self.max_update_norm / norm
            delta = [value * scale for value in delta]
            norm = self.max_update_norm
        return tuple(value + delta[index] for index, value in enumerate(state)), norm


class MosaicSimulator:
    def __init__(
        self,
        config: MosaicConfig,
        *,
        bases: Iterable[LowRankBasis],
        operators: Iterable[OperatorCode],
    ):
        self.config = config
        self.archive = OperatorArchive(operators)
        self.cell = RecurrentCell(
            bases,
            dimension=config.workspace_dim,
            max_update_norm=config.max_update_norm,
        )

    def solve(
        self,
        query: str,
        *,
        initial_state: Vector,
        goal: Vector,
        max_steps: int,
    ) -> MosaicRun:
        if max_steps < 0:
            raise ValueError("max_steps must not be negative")
        if len(initial_state) != self.config.workspace_dim or len(goal) != len(
            initial_state
        ):
            raise ValueError("initial state and goal must match workspace dimension")

        operator = synthesize_operators(
            self.archive.search(query, top_k=self.config.operator_top_k)
        )
        state = initial_state
        error = _distance(state, goal)
        trace: list[dict[str, object]] = []
        stalled = False

        for step in range(1, max_steps + 1):
            if error <= self.config.halt_tolerance:
                break
            previous = state
            state, update_norm = self.cell.apply(state, operator)
            error = _distance(state, goal)
            trace.append(
                {
                    "step": step,
                    "state": [round(value, 9) for value in state],
                    "error": round(error, 9),
                    "update_norm": round(update_norm, 9),
                }
            )
            if _distance(previous, state) <= self.config.halt_tolerance:
                stalled = True
                break

        success = error <= self.config.halt_tolerance
        return MosaicRun(
            success=success,
            halted=success,
            stalled=stalled,
            steps=len(trace),
            final_state=state,
            final_error=error,
            operator=operator,
            trace=tuple(trace),
        )


def run_demo(memory_path: Path) -> dict[str, object]:
    namespace = "mosaic-v0"
    memory = ConversationMemory(memory_path)
    memory.remember_fact(
        namespace=namespace,
        subject="task",
        predicate="target_steps",
        value="3",
        source_turn="initial-pack",
    )
    memory.remember_fact(
        namespace=namespace,
        subject="task",
        predicate="target_steps",
        value="4",
        source_turn="replacement-pack",
    )
    swapped_value = memory.active_facts(namespace)[0]["value"]
    deleted_rows = memory.forget_fact(
        namespace=namespace,
        subject="task",
        predicate="target_steps",
    )
    deletion_ok = not memory.active_facts(namespace)

    config = MosaicConfig()
    simulator = MosaicSimulator(
        config,
        bases=[
            LowRankBasis(
                name="progress",
                left=(1.0, 0.0, 0.0, 0.0),
                right=(0.0, 0.0, 0.0, 0.0),
                bias=1.0,
            )
        ],
        operators=[
            OperatorCode(
                address="procedure.forward",
                tags=frozenset({"advance", "progress"}),
                coefficients=(("progress", 1.0),),
                priority=10,
                conflicts=frozenset({"procedure.reverse"}),
            ),
            OperatorCode(
                address="procedure.reverse",
                tags=frozenset({"advance", "progress"}),
                coefficients=(("progress", -1.0),),
                priority=1,
                conflicts=frozenset({"procedure.forward"}),
            ),
        ],
    )
    initial_state = (0.0, 0.0, 0.0, 0.0)
    goal = (float(swapped_value), 0.0, 0.0, 0.0)
    depths = [
        {
            "max_steps": max_steps,
            **simulator.solve(
                "advance progress",
                initial_state=initial_state,
                goal=goal,
                max_steps=max_steps,
            ).to_dict(),
        }
        for max_steps in (1, 2, 4, 8)
    ]
    unknown = simulator.solve(
        "unseen operation",
        initial_state=initial_state,
        goal=goal,
        max_steps=8,
    )
    acceptance = {
        "knowledge_swap": swapped_value == "4",
        "knowledge_delete": deletion_ok and deleted_rows == 2,
        "operator_conflict": depths[-1]["operator"]["disabled"]
        == ["procedure.reverse"],
        "recurrent_scaling": not depths[0]["success"] and depths[2]["success"],
        "unknown_stalls_without_operator": unknown.stalled
        and not unknown.operator.selected,
    }
    return {
        "schema_version": "mosaic-v0",
        "scope": "deterministic architecture simulator; not an LM quality result",
        "memory": {
            "swapped_value": swapped_value,
            "deleted_rows": deleted_rows,
            "active_after_delete": memory.active_facts(namespace),
        },
        "recurrent_depths": depths,
        "unknown_query": unknown.to_dict(),
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
    }


def _distance(left: Vector, right: Vector) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the dependency-free MOSAIC v0 architecture simulator."
    )
    parser.add_argument("--memory-db", type=Path)
    args = parser.parse_args(argv)
    if args.memory_db is not None:
        report = run_demo(args.memory_db)
    else:
        with tempfile.TemporaryDirectory(prefix="mosaic-v0-") as directory:
            report = run_demo(Path(directory) / "memory.sqlite3")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

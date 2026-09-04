from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from tinylm_slicer.conversation_memory import ConversationMemory
from tinylm_slicer.mosaic_v0 import (
    OperatorArchive,
    OperatorCode,
    synthesize_operators,
)


@dataclass(frozen=True)
class Phase1Config:
    seed: int = 29
    node_count: int = 32
    workspace_slots: int = 4
    model_dim: int = 64
    attention_heads: int = 4
    ffn_dim: int = 128
    physical_layers: int = 1
    reencode_interval: int = 1
    train_depth: int = 4
    eval_depth: int = 8
    batch_size: int = 256
    train_steps: int = 1500
    learning_rate: float = 2e-3
    min_seen_accuracy: float = 0.95
    min_unseen_accuracy: float = 0.8
    min_unseen_gain: float = 0.05
    min_operator_gain: float = 0.03
    min_operator_vs_rag_gain: float = 0.03
    min_operator_latency_improvement: float = 0.20
    rag_success_tolerance: float = 0.01

    def __post_init__(self) -> None:
        sizes = (
            self.node_count,
            self.workspace_slots,
            self.model_dim,
            self.attention_heads,
            self.ffn_dim,
            self.physical_layers,
            self.train_depth,
            self.eval_depth,
            self.batch_size,
            self.train_steps,
        )
        if any(value <= 0 for value in sizes):
            raise ValueError("model and training sizes must be positive")
        if self.reencode_interval < 0:
            raise ValueError("reencode_interval must not be negative")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.eval_depth <= self.train_depth:
            raise ValueError("eval_depth must exceed train_depth")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if (
            not 0 <= self.min_seen_accuracy <= 1
            or not 0 <= self.min_unseen_accuracy <= 1
            or self.min_unseen_gain < 0
            or self.min_operator_gain < 0
            or self.min_operator_vs_rag_gain < 0
            or not 0 <= self.min_operator_latency_improvement <= 1
            or self.rag_success_tolerance < 0
        ):
            raise ValueError("invalid acceptance thresholds")

    @classmethod
    def target(cls) -> Phase1Config:
        return cls(
            node_count=256,
            workspace_slots=16,
            model_dim=896,
            attention_heads=14,
            ffn_dim=3584,
            physical_layers=2,
            batch_size=64,
            train_steps=1_000,
            learning_rate=3e-4,
        )


class SharedDepthSequenceModel(nn.Module):
    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config
        self.node_embedding = nn.Embedding(config.node_count, config.model_dim)
        self.operator_embedding = nn.Embedding(2, config.model_dim)
        self.workspace = nn.Parameter(
            torch.empty(config.workspace_slots, config.model_dim)
        )
        self.depth_projection = nn.Sequential(
            nn.Linear(2, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.ffn_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.physical_layers)
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.node_count)
        nn.init.normal_(self.workspace, std=0.02)

    def forward(
        self,
        start_nodes: torch.Tensor,
        operator_ids: torch.Tensor,
        depths: torch.Tensor,
        *,
        recurrent: bool,
    ) -> torch.Tensor:
        return self.forward_trace(
            start_nodes,
            operator_ids,
            depths,
            recurrent=recurrent,
        )[-1]

    def encode_operator(self, operator_ids: torch.Tensor) -> torch.Tensor:
        return self.operator_embedding(operator_ids)

    def extra_context_tokens(
        self,
        operator_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        del operator_ids
        return None

    def forward_trace(
        self,
        start_nodes: torch.Tensor,
        operator_ids: torch.Tensor,
        depths: torch.Tensor,
        *,
        recurrent: bool,
    ) -> tuple[torch.Tensor, ...]:
        if start_nodes.ndim != 1 or operator_ids.shape != start_nodes.shape:
            raise ValueError("start_nodes and operator_ids must be one-dimensional")
        if depths.shape != start_nodes.shape or bool((depths < 1).any()):
            raise ValueError("depths must be positive and match the batch")

        batch = start_nodes.shape[0]
        # The recurrent scheduler already expresses requested depth through the
        # number of cell applications. Conditioning the shared cell on an
        # unseen target depth makes its one-step transition distribution shift.
        feature_depth = torch.ones_like(depths) if recurrent else depths
        normalized_depth = feature_depth.float() / self.config.eval_depth
        depth_features = torch.stack(
            (normalized_depth, torch.sin(math.pi * normalized_depth)),
            dim=-1,
        )
        context = self.encode_operator(operator_ids) + self.depth_projection(
            depth_features
        )
        extra_context = self.extra_context_tokens(operator_ids)
        query = self.node_embedding(start_nodes) + context
        state = self.workspace.view(1, self.config.workspace_slots, -1).expand(
            batch, -1, -1
        ).clone()
        state[:, 0] = state[:, 0] + query
        if extra_context is not None:
            state = torch.cat((state, extra_context), dim=1)

        rounds = int(depths.max()) if recurrent else 1
        trace = []
        for round_index in range(rounds):
            updated = state
            for block in self.blocks:
                updated = block(updated)
            if recurrent:
                active = (depths > round_index).view(batch, 1, 1)
                state = torch.where(active, updated, state)
            else:
                state = updated
            logits = self.output(self.output_norm(state[:, 0]))
            trace.append(logits)
            completed_rounds = round_index + 1
            if (
                recurrent
                and self.config.reencode_interval
                and completed_rounds < rounds
                and completed_rounds % self.config.reencode_interval == 0
            ):
                node_state = logits.softmax(dim=-1) @ self.node_embedding.weight
                reencoded = self.workspace.view(
                    1, self.config.workspace_slots, -1
                ).expand(batch, -1, -1).clone()
                reencoded[:, 0] = reencoded[:, 0] + node_state + context
                if extra_context is not None:
                    reencoded = torch.cat((reencoded, extra_context), dim=1)
                continuing = (depths > completed_rounds).view(batch, 1, 1)
                state = torch.where(continuing, reencoded, state)
        return tuple(trace)


class TextRagSharedDepthSequenceModel(SharedDepthSequenceModel):
    procedure_texts = (
        (
            "Source: ring procedure manual. Confidence: 1.0. "
            "Instruction: to advance around the ring, move exactly one node "
            "forward for every requested reasoning step."
        ),
        (
            "Source: ring procedure manual. Confidence: 1.0. "
            "Instruction: to retreat around the ring, move exactly one node "
            "backward for every requested reasoning step."
        ),
    )

    def __init__(self, config: Phase1Config):
        super().__init__(config)
        text_dim = max(8, config.model_dim // 4)
        encoded = [text.encode("utf-8") for text in self.procedure_texts]
        patch_width = 8
        max_patches = max(
            math.ceil(len(value) / patch_width) for value in encoded
        )
        documents = torch.zeros(
            (len(encoded), max_patches, patch_width),
            dtype=torch.long,
        )
        for index, value in enumerate(encoded):
            flattened = documents[index].view(-1)
            flattened[: len(value)] = torch.tensor(
                [byte + 1 for byte in value],
                dtype=torch.long,
            )
        self.register_buffer("procedure_documents", documents)
        self.byte_embedding = nn.Embedding(257, text_dim, padding_idx=0)
        self.text_projection = nn.Linear(text_dim, config.model_dim)

    def encode_operator(self, operator_ids: torch.Tensor) -> torch.Tensor:
        return self.workspace.new_zeros(
            (operator_ids.shape[0], self.config.model_dim)
        )

    def extra_context_tokens(
        self,
        operator_ids: torch.Tensor,
    ) -> torch.Tensor:
        documents = self.procedure_documents[operator_ids]
        embedded = self.byte_embedding(documents)
        mask = documents.ne(0).unsqueeze(-1)
        denominator = mask.sum(dim=2).clamp_min(1)
        patches = (embedded * mask).sum(dim=2) / denominator
        return self.text_projection(patches)


def ring_targets(
    start_nodes: torch.Tensor,
    operator_ids: torch.Tensor,
    depths: torch.Tensor,
    *,
    node_count: int,
) -> torch.Tensor:
    direction = torch.where(operator_ids == 0, 1, -1)
    return (start_nodes + direction * depths) % node_count


def compare_models(
    config: Phase1Config,
    *,
    device: str = "auto",
    checkpoint: Path | None = None,
) -> dict[str, object]:
    resolved_device = _resolve_device(device)
    recurrent, recurrent_model = _train_variant(
        config,
        recurrent=True,
        device=resolved_device,
    )
    baseline, _ = _train_variant(config, recurrent=False, device=resolved_device)
    no_operator, _ = _train_variant(
        config,
        recurrent=True,
        operator_visible=False,
        device=resolved_device,
    )
    text_rag, text_rag_model = _train_variant(
        config,
        recurrent=True,
        model_kind="text_rag",
        device=resolved_device,
    )
    memory_operator_suite = evaluate_memory_operator_suite(
        recurrent_model,
        config,
        resolved_device,
    )
    recurrent_unseen = _unseen_mean(recurrent["depth_metrics"], config.train_depth)
    baseline_unseen = _unseen_mean(baseline["depth_metrics"], config.train_depth)
    no_operator_unseen = _unseen_mean(
        no_operator["depth_metrics"],
        config.train_depth,
    )
    text_rag_unseen = _unseen_mean(
        text_rag["depth_metrics"],
        config.train_depth,
    )
    latency_comparison = compare_inference_latency(
        recurrent_model,
        text_rag_model,
        config,
        resolved_device,
    )
    rag_matched_success = (
        text_rag_unseen + config.rag_success_tolerance >= recurrent_unseen
    )
    operator_vs_rag_passed = (
        recurrent_unseen
        >= text_rag_unseen + config.min_operator_vs_rag_gain
        or (
            rag_matched_success
            and latency_comparison["operator_latency_improvement"]
            >= config.min_operator_latency_improvement
        )
    )
    seen_min = min(
        metric["accuracy"]
        for metric in recurrent["depth_metrics"]
        if metric["depth"] <= config.train_depth
    )
    recurrent_unseen_min = min(
        metric["accuracy"]
        for metric in recurrent["depth_metrics"]
        if metric["depth"] > config.train_depth
    )
    acceptance = {
        "equal_parameter_count": recurrent["parameter_count"]
        == baseline["parameter_count"],
        "recurrent_seen_accuracy": seen_min >= config.min_seen_accuracy,
        "recurrent_absolute_unseen_accuracy": recurrent_unseen_min
        >= config.min_unseen_accuracy,
        "recurrent_unseen_gain": recurrent_unseen
        >= baseline_unseen + config.min_unseen_gain,
        "operator_information_gain": recurrent_unseen
        >= no_operator_unseen + config.min_operator_gain,
        "operator_vs_text_rag": operator_vs_rag_passed,
        "finite_losses": all(
            math.isfinite(float(report["final_loss"]))
            for report in (recurrent, baseline, no_operator, text_rag)
        ),
        "memory_operator_suite": memory_operator_suite["passed"],
    }
    result: dict[str, object] = {
        "schema_version": "mosaic-phase1-v0",
        "scope": "synthetic token-sequence recurrence comparison; not LM quality",
        "device": str(resolved_device),
        "config": asdict(config),
        "recurrent": recurrent,
        "single_pass_baseline": baseline,
        "no_operator_ablation": no_operator,
        "text_rag_baseline": text_rag,
        "memory_operator_suite": memory_operator_suite,
        "comparison": {
            "recurrent_unseen_mean_accuracy": round(recurrent_unseen, 6),
            "baseline_unseen_mean_accuracy": round(baseline_unseen, 6),
            "unseen_gain": round(recurrent_unseen - baseline_unseen, 6),
            "no_operator_unseen_mean_accuracy": round(no_operator_unseen, 6),
            "operator_information_gain": round(
                recurrent_unseen - no_operator_unseen,
                6,
            ),
            "text_rag_unseen_mean_accuracy": round(text_rag_unseen, 6),
            "text_rag_matched_success": rag_matched_success,
            "operator_vs_text_rag_accuracy_gain": round(
                recurrent_unseen - text_rag_unseen,
                6,
            ),
            "latency": latency_comparison,
        },
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
    }
    if checkpoint is not None:
        result["checkpoint"] = _save_checkpoint(
            checkpoint,
            recurrent_model,
            config,
            "recurrent",
        )
    return result


def reevaluate_report(report: dict[str, object]) -> dict[str, object]:
    updated = copy.deepcopy(report)
    runs = updated.get("runs")
    if isinstance(runs, list):
        updated_runs = [
            reevaluate_report(run)
            for run in runs
            if isinstance(run, dict)
        ]
        if len(updated_runs) != len(runs):
            raise ValueError("multi-seed report contains an invalid run")
        updated["runs"] = updated_runs
        aggregate = updated.get("aggregate")
        if not isinstance(aggregate, dict):
            raise ValueError("multi-seed report does not contain aggregate data")
        aggregate["minimum_operator_information_gain"] = round(
            min(
                float(run["comparison"]["operator_information_gain"])
                for run in updated_runs
            ),
            6,
        )
        aggregate["minimum_operator_vs_text_rag_latency_improvement"] = round(
            min(
                float(
                    run["comparison"]["latency"][
                        "operator_latency_improvement"
                    ]
                )
                for run in updated_runs
            ),
            6,
        )
        aggregate["all_runs_passed"] = all(
            bool(run["passed"]) for run in updated_runs
        )
        updated["passed"] = aggregate["all_runs_passed"]
        updated["reevaluation"] = {
            "reason": "nested Phase 1 gate semantics updated",
            "source_passed": bool(report.get("passed")),
        }
        return updated
    acceptance = updated.get("acceptance")
    comparison = updated.get("comparison")
    if not isinstance(acceptance, dict) or not isinstance(comparison, dict):
        raise ValueError("report does not contain Phase 1 acceptance data")
    matched = acceptance.pop("text_rag_matched_success", None)
    if matched is not None:
        comparison["text_rag_matched_success"] = bool(matched)
    updated["passed"] = all(bool(value) for value in acceptance.values())
    updated["reevaluation"] = {
        "reason": (
            "matched text-RAG success is diagnostic; the registered gate is "
            "operator accuracy gain OR latency gain at matched success"
        ),
        "source_passed": bool(report.get("passed")),
    }
    return updated


def train_one_variant(
    config: Phase1Config,
    variant: str,
    *,
    device: str = "auto",
    checkpoint: Path | None = None,
) -> dict[str, object]:
    variants = {
        "recurrent": {
            "recurrent": True,
            "operator_visible": True,
            "model_kind": "operator",
        },
        "single_pass": {
            "recurrent": False,
            "operator_visible": True,
            "model_kind": "operator",
        },
        "no_operator": {
            "recurrent": True,
            "operator_visible": False,
            "model_kind": "operator",
        },
        "text_rag": {
            "recurrent": True,
            "operator_visible": True,
            "model_kind": "text_rag",
        },
    }
    if variant not in variants:
        raise ValueError(f"unsupported variant: {variant}")
    resolved_device = _resolve_device(device)
    report, model = _train_variant(
        config,
        device=resolved_device,
        **variants[variant],
    )
    seen_min = min(
        float(metric["accuracy"])
        for metric in report["depth_metrics"]
        if int(metric["depth"]) <= config.train_depth
    )
    unseen_min = min(
        float(metric["accuracy"])
        for metric in report["depth_metrics"]
        if int(metric["depth"]) > config.train_depth
    )
    acceptance = {
        "finite_loss": math.isfinite(float(report["final_loss"])),
    }
    if variant == "recurrent":
        acceptance.update(
            {
                "seen_accuracy": seen_min >= config.min_seen_accuracy,
                "unseen_accuracy": unseen_min >= config.min_unseen_accuracy,
            }
        )
    result: dict[str, object] = {
        "schema_version": "mosaic-phase1-variant-v0",
        "scope": "single trained Phase 1 variant; synthetic ring task only",
        "device": str(resolved_device),
        "variant": variant,
        "config": asdict(config),
        "training": report,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
    }
    if checkpoint is not None:
        result["checkpoint"] = _save_checkpoint(
            checkpoint,
            model,
            config,
            variant,
        )
    return result


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    config: Phase1Config,
    variant: str,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "schema_version": "mosaic-phase1-checkpoint-v0",
            "variant": variant,
            "config": asdict(config),
            "model_class": type(model).__name__,
            "state_dict": state_dict,
        },
        path,
    )
    digest = _sha256(path)
    manifest = {
        "schema_version": "mosaic-phase1-artifact-manifest-v0",
        "checkpoint": path.name,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "source": "random initialization trained on locally generated ring targets",
        "external_datasets": [],
        "external_checkpoints": [],
        "license_status": "project license not declared",
        "redistribution": (
            "not authorized by this manifest; choose and document a project "
            "license before publishing"
        ),
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "sha256": manifest["sha256"],
        "bytes": manifest["bytes"],
    }


def verify_checkpoint(
    path: Path,
    *,
    device: str = "auto",
) -> dict[str, object]:
    resolved_device = _resolve_device(device)
    payload = torch.load(
        path,
        map_location=resolved_device,
        weights_only=True,
    )
    if payload.get("schema_version") != "mosaic-phase1-checkpoint-v0":
        raise ValueError("unsupported checkpoint schema")
    config = Phase1Config(**payload["config"])
    variant = str(payload["variant"])
    model_type = (
        TextRagSharedDepthSequenceModel
        if variant == "text_rag"
        else SharedDepthSequenceModel
    )
    model = model_type(config).to(resolved_device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)

    starts = torch.arange(
        config.node_count,
        device=resolved_device,
    ).repeat_interleave(2)
    operators = torch.arange(2, device=resolved_device).repeat(
        config.node_count
    )
    model_operators = (
        torch.zeros_like(operators) if variant == "no_operator" else operators
    )
    depth_metrics = []
    recurrent = variant != "single_pass"
    with torch.inference_mode():
        for depth_value in range(1, config.eval_depth + 1):
            depths = torch.full_like(starts, depth_value)
            targets = ring_targets(
                starts,
                operators,
                depths,
                node_count=config.node_count,
            )
            predictions = model(
                starts,
                model_operators,
                depths,
                recurrent=recurrent,
            ).argmax(dim=-1)
            depth_metrics.append(
                {
                    "depth": depth_value,
                    "cases": starts.numel(),
                    "accuracy": round(
                        float((predictions == targets).float().mean()),
                        6,
                    ),
                }
            )

    digest = _sha256(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "manifest_digest": digest == manifest.get("sha256"),
        "all_tensors_loaded": True,
    }
    if variant == "recurrent":
        checks["seen_accuracy"] = min(
            metric["accuracy"]
            for metric in depth_metrics
            if metric["depth"] <= config.train_depth
        ) >= config.min_seen_accuracy
        checks["unseen_accuracy"] = min(
            metric["accuracy"]
            for metric in depth_metrics
            if metric["depth"] > config.train_depth
        ) >= config.min_unseen_accuracy
    return {
        "schema_version": "mosaic-phase1-checkpoint-verification-v0",
        "checkpoint": str(path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "sha256": digest,
        "device": str(resolved_device),
        "variant": variant,
        "config": asdict(config),
        "total_cases": config.node_count * 2 * config.eval_depth,
        "depth_metrics": depth_metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_inference_latency(
    operator_model: nn.Module,
    text_rag_model: nn.Module,
    config: Phase1Config,
    device: torch.device,
    *,
    depth_value: int = 2,
    warmups: int = 10,
    repeats: int = 100,
) -> dict[str, float | int]:
    if depth_value <= 0 or warmups < 0 or repeats <= 0:
        raise ValueError("latency profile values are invalid")
    start = torch.tensor([7], device=device)
    operator_id = torch.tensor([0], device=device)
    depth = torch.tensor([depth_value], device=device)
    models = (operator_model.eval(), text_rag_model.eval())
    with torch.inference_mode():
        for _ in range(warmups):
            for model in models:
                model(start, operator_id, depth, recurrent=True)
        _synchronize(device)
        samples: list[list[float]] = [[], []]
        for repeat in range(repeats):
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            for index in order:
                _synchronize(device)
                started = time.perf_counter()
                models[index](start, operator_id, depth, recurrent=True)
                _synchronize(device)
                samples[index].append((time.perf_counter() - started) * 1000)
    operator_p50 = statistics.median(samples[0])
    text_rag_p50 = statistics.median(samples[1])
    improvement = (
        0.0
        if text_rag_p50 <= 0
        else 1.0 - operator_p50 / text_rag_p50
    )
    return {
        "depth": depth_value,
        "batch_size": 1,
        "samples_per_model": repeats,
        "operator_p50_ms": round(operator_p50, 6),
        "text_rag_p50_ms": round(text_rag_p50, 6),
        "operator_latency_improvement": round(improvement, 6),
    }


def evaluate_memory_operator_suite(
    model: nn.Module,
    config: Phase1Config,
    device: torch.device,
) -> dict[str, object]:
    namespace = f"mosaic-phase1-{config.seed}"
    subject = "ring"
    predicate = "start_node"
    model_requests = 0
    with tempfile.TemporaryDirectory(prefix="mosaic-phase1-") as directory:
        memory = ConversationMemory(Path(directory) / "memory.sqlite3")
        first_id = memory.remember_fact(
            namespace=namespace,
            subject=subject,
            predicate=predicate,
            value="3",
            source_turn="before-swap",
        )
        second_id = memory.remember_fact(
            namespace=namespace,
            subject=subject,
            predicate=predicate,
            value="7",
            source_turn="after-swap",
        )
        memory.remember_fact(
            namespace=namespace,
            subject=subject,
            predicate="depth",
            value="2",
            source_turn="independent-depth-fact",
        )
        active = memory.active_facts(namespace)
        active_by_predicate = {row["predicate"]: row for row in active}
        history = memory.history(
            namespace=namespace,
            subject=subject,
            predicate=predicate,
        )

        archive = OperatorArchive(
            (
                OperatorCode(
                    address="procedure.forward",
                    tags=frozenset({"advance", "progress"}),
                    coefficients=(("direction", 1.0),),
                    priority=10,
                    conflicts=frozenset({"procedure.reverse"}),
                ),
                OperatorCode(
                    address="procedure.reverse",
                    tags=frozenset({"advance", "progress"}),
                    coefficients=(("direction", -1.0),),
                    priority=1,
                    conflicts=frozenset({"procedure.forward"}),
                ),
            )
        )
        operator = synthesize_operators(
            archive.search("advance progress", top_k=2)
        )

        predicted = None
        expected = None
        if (
            {"start_node", "depth"} <= active_by_predicate.keys()
            and operator.selected == ("procedure.forward",)
        ):
            start_value = int(active_by_predicate["start_node"]["value"])
            depth_value = int(active_by_predicate["depth"]["value"])
            start = torch.tensor([start_value], device=device)
            operator_id = torch.tensor([0], device=device)
            depth = torch.tensor([depth_value], device=device)
            expected = int(
                ring_targets(
                    start,
                    operator_id,
                    depth,
                    node_count=config.node_count,
                ).item()
            )
            model.eval()
            with torch.no_grad():
                predicted = int(
                    model(
                        start,
                        operator_id,
                        depth,
                        recurrent=True,
                    ).argmax(dim=-1).item()
                )
            model_requests += 1

        deleted_rows = memory.forget_fact(
            namespace=namespace,
            subject=subject,
            predicate=predicate,
        )
        active_after_delete = memory.active_facts(namespace)
        remaining_by_predicate = {
            row["predicate"]: row for row in active_after_delete
        }
        requests_before_unknown = model_requests
        if {"start_node", "depth"} <= remaining_by_predicate.keys():
            model_requests += 1

    acceptance = {
        "knowledge_swap": (
            first_id != second_id
            and len(history) == 2
            and active_by_predicate["start_node"]["value"] == "7"
        ),
        "independent_two_fact_composition": (
            predicted == expected
            and predicted is not None
            and active_by_predicate["start_node"]["value"] == "7"
            and active_by_predicate["depth"]["value"] == "2"
        ),
        "knowledge_delete": (
            deleted_rows == 2
            and "start_node" not in remaining_by_predicate
            and remaining_by_predicate["depth"]["value"] == "2"
        ),
        "operator_conflict": (
            operator.selected == ("procedure.forward",)
            and operator.disabled == ("procedure.reverse",)
            and operator.conflicts
            == (("procedure.forward", "procedure.reverse"),)
        ),
        "unknown_without_model_call": (
            model_requests == requests_before_unknown == 1
        ),
    }
    return {
        "scope": (
            "independent external start/depth fact composition, start-fact "
            "swap/delete, and retrieved procedure conflict resolution"
        ),
        "active_value_after_swap": (
            active_by_predicate.get("start_node", {}).get("value")
        ),
        "composed_depth_value": (
            active_by_predicate.get("depth", {}).get("value")
        ),
        "history_values": [row["value"] for row in history],
        "selected_operators": list(operator.selected),
        "disabled_operators": list(operator.disabled),
        "predicted_node": predicted,
        "expected_node": expected,
        "deleted_rows": deleted_rows,
        "active_fact_count_after_delete": len(active_after_delete),
        "remaining_predicates_after_delete": sorted(
            remaining_by_predicate
        ),
        "model_requests": model_requests,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
    }


def profile_model(config: Phase1Config) -> dict[str, object]:
    model = SharedDepthSequenceModel(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "schema_version": "mosaic-phase1-profile-v0",
        "config": asdict(config),
        "parameter_count": parameter_count,
        "raw_weight_mib": {
            "bf16": round(parameter_count * 2 / 2**20, 3),
            "int8": round(parameter_count / 2**20, 3),
            "2bit": round(parameter_count / 4 / 2**20, 3),
        },
        "target_parameter_range": [20_000_000, 50_000_000],
        "in_target_range": 20_000_000 <= parameter_count <= 50_000_000,
    }


def compare_seeds(
    config: Phase1Config,
    seeds: list[int],
    *,
    device: str = "auto",
    checkpoint_dir: Path | None = None,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    runs = []
    for seed in seeds:
        run_config = Phase1Config(**{**asdict(config), "seed": seed})
        runs.append(
            compare_models(
                run_config,
                device=device,
                checkpoint=(
                    checkpoint_dir / f"recurrent_seed_{seed}.pt"
                    if checkpoint_dir is not None
                    else None
                ),
            )
        )
    recurrent_unseen = [
        float(run["comparison"]["recurrent_unseen_mean_accuracy"])
        for run in runs
    ]
    recurrent_depth8 = [
        float(run["recurrent"]["depth_metrics"][-1]["accuracy"])
        for run in runs
    ]
    rag_latency_improvements = [
        float(run["comparison"]["latency"]["operator_latency_improvement"])
        for run in runs
    ]
    operator_information_gains = [
        float(run["comparison"]["operator_information_gain"])
        for run in runs
    ]
    return {
        "schema_version": "mosaic-phase1-multiseed-v0",
        "scope": "three-seed synthetic recurrence gate; not LM quality",
        "seeds": seeds,
        "runs": runs,
        "aggregate": {
            "minimum_recurrent_unseen_mean_accuracy": round(
                min(recurrent_unseen), 6
            ),
            "minimum_recurrent_depth8_accuracy": round(min(recurrent_depth8), 6),
            "minimum_operator_information_gain": round(
                min(operator_information_gains),
                6,
            ),
            "minimum_operator_vs_text_rag_latency_improvement": round(
                min(rag_latency_improvements),
                6,
            ),
            "all_runs_passed": all(bool(run["passed"]) for run in runs),
        },
        "passed": all(bool(run["passed"]) for run in runs),
    }


def _train_variant(
    config: Phase1Config,
    *,
    recurrent: bool,
    device: torch.device,
    operator_visible: bool = True,
    model_kind: str = "operator",
) -> tuple[dict[str, object], nn.Module]:
    if model_kind not in {"operator", "text_rag"}:
        raise ValueError(f"unsupported model kind: {model_kind}")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    model_type = (
        TextRagSharedDepthSequenceModel
        if model_kind == "text_rag"
        else SharedDepthSequenceModel
    )
    model = model_type(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    cuda_allocated_before_train = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        cuda_allocated_before_train = torch.cuda.memory_allocated(device)
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for _ in range(config.train_steps):
        start, operator, depth = _batch(
            config,
            generator=generator,
            device=device,
            max_depth=config.train_depth,
        )
        target = ring_targets(
            start,
            operator,
            depth,
            node_count=config.node_count,
        )
        model_operator = operator if operator_visible else torch.zeros_like(operator)
        if recurrent:
            step_losses = []
            for step, logits in enumerate(
                model.forward_trace(start, model_operator, depth, recurrent=True),
                start=1,
            ):
                active = depth >= step
                step_depth = torch.full_like(depth[active], step)
                step_target = ring_targets(
                    start[active],
                    operator[active],
                    step_depth,
                    node_count=config.node_count,
                )
                step_losses.append(
                    nn.functional.cross_entropy(logits[active], step_target)
                )
            loss = torch.stack(step_losses).mean()
        else:
            logits = model(start, model_operator, depth, recurrent=False)
            loss = nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cuda_peak_allocated = 0
    cuda_peak_reserved = 0
    if device.type == "cuda":
        cuda_peak_allocated = max(
            0,
            torch.cuda.max_memory_allocated(device)
            - cuda_allocated_before_train,
        )
        cuda_peak_reserved = torch.cuda.max_memory_reserved(device)

    metrics = []
    model.eval()
    with torch.no_grad():
        for depth_value in range(1, config.eval_depth + 1):
            start, operator, depth = _batch(
                config,
                generator=generator,
                device=device,
                max_depth=depth_value,
                fixed_depth=depth_value,
                batch_size=config.batch_size * 4,
            )
            target = ring_targets(
                start,
                operator,
                depth,
                node_count=config.node_count,
            )
            prediction = model(
                start,
                operator if operator_visible else torch.zeros_like(operator),
                depth,
                recurrent=recurrent,
            ).argmax(dim=-1)
            metrics.append(
                {
                    "depth": depth_value,
                    "seen_in_training": depth_value <= config.train_depth,
                    "accuracy": round(float((prediction == target).float().mean()), 6),
                }
            )
    return (
        {
            "mode": "recurrent" if recurrent else "single_pass",
            "operator_visible": operator_visible,
            "input_mode": model_kind,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "initial_loss": round(losses[0], 9),
            "final_loss": round(losses[-1], 9),
            "minimum_loss": round(min(losses), 9),
            "mean_last_100_loss": round(
                statistics.fmean(losses[-100:]),
                9,
            ),
            "train_elapsed_sec": round(elapsed, 3),
            "train_steps_per_sec": round(config.train_steps / elapsed, 3),
            "incremental_cuda_peak_allocated_mib": round(
                cuda_peak_allocated / 2**20,
                3,
            ),
            "process_cuda_peak_reserved_mib": round(
                cuda_peak_reserved / 2**20,
                3,
            ),
            "depth_metrics": metrics,
        },
        model,
    )


def _batch(
    config: Phase1Config,
    *,
    generator: torch.Generator,
    device: torch.device,
    max_depth: int,
    fixed_depth: int | None = None,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = batch_size or config.batch_size
    start = torch.randint(
        config.node_count,
        (size,),
        generator=generator,
        device=device,
    )
    operator = torch.randint(2, (size,), generator=generator, device=device)
    if fixed_depth is None:
        depth = torch.randint(
            1,
            max_depth + 1,
            (size,),
            generator=generator,
            device=device,
        )
    else:
        depth = torch.full((size,), fixed_depth, device=device, dtype=torch.long)
    return start, operator, depth


def _unseen_mean(metrics: list[dict[str, object]], train_depth: int) -> float:
    unseen = [
        float(metric["accuracy"])
        for metric in metrics
        if int(metric["depth"]) > train_depth
    ]
    return sum(unseen) / len(unseen)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile or train the MOSAIC Phase 1 sequence model."
    )
    parser.add_argument("--preset", choices=("smoke", "target"), default="smoke")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--variant",
        choices=("recurrent", "single_pass", "no_operator", "text_rag"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--verify-checkpoint", type=Path)
    parser.add_argument("--reevaluate-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = Phase1Config.target() if args.preset == "target" else Phase1Config()
    if args.steps is not None:
        config = Phase1Config(**{**asdict(config), "train_steps": args.steps})
    if args.batch_size is not None:
        config = Phase1Config(**{**asdict(config), "batch_size": args.batch_size})
    if args.profile_only and args.seeds:
        parser.error("--seeds cannot be combined with --profile-only")
    if args.checkpoint and not args.variant:
        parser.error("--checkpoint requires --variant")
    if args.checkpoint_dir and not args.seeds:
        parser.error("--checkpoint-dir requires --seeds")
    if args.verify_checkpoint and (
        args.variant
        or args.profile_only
        or args.seeds
        or args.reevaluate_report
    ):
        parser.error(
            "--verify-checkpoint cannot be combined with training or reevaluation"
        )
    if args.variant and (args.profile_only or args.seeds or args.reevaluate_report):
        parser.error("--variant cannot be combined with profiling, seeds, or reevaluation")
    if args.reevaluate_report and (args.profile_only or args.seeds):
        parser.error(
            "--reevaluate-report cannot be combined with profiling or seeds"
        )
    if args.verify_checkpoint:
        report = verify_checkpoint(
            args.verify_checkpoint,
            device=args.device,
        )
    elif args.reevaluate_report:
        report = reevaluate_report(
            json.loads(args.reevaluate_report.read_text(encoding="utf-8"))
        )
    elif args.variant:
        report = train_one_variant(
            config,
            args.variant,
            device=args.device,
            checkpoint=args.checkpoint,
        )
    elif args.profile_only:
        report = profile_model(config)
    elif args.seeds:
        report = compare_seeds(
            config,
            args.seeds,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        report = compare_models(config, device=args.device)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("passed", report.get("in_target_range", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())

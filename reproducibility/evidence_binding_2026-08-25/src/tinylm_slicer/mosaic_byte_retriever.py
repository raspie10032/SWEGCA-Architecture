from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ByteRetrieverConfig:
    seed: int = 41
    patch_size: int = 4
    model_dim: int = 192
    attention_heads: int = 6
    ffn_dim: int = 384
    layers: int = 2
    embedding_dim: int = 128
    max_query_bytes: int = 192
    max_document_bytes: int = 768
    batch_size: int = 32
    train_steps: int = 1_000
    learning_rate: float = 3e-4
    temperature: float = 0.07

    def __post_init__(self) -> None:
        if min(
            self.patch_size,
            self.model_dim,
            self.attention_heads,
            self.ffn_dim,
            self.embedding_dim,
            self.max_query_bytes,
            self.max_document_bytes,
            self.batch_size,
            self.train_steps,
        ) <= 0:
            raise ValueError("retriever configuration values must be positive")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.max_query_bytes % self.patch_size:
            raise ValueError("max_query_bytes must be divisible by patch_size")
        if self.max_document_bytes % self.patch_size:
            raise ValueError("max_document_bytes must be divisible by patch_size")


class BytePatchRetriever(nn.Module):
    def __init__(self, config: ByteRetrieverConfig) -> None:
        super().__init__()
        self.config = config
        maximum_patches = max(
            config.max_query_bytes,
            config.max_document_bytes,
        ) // config.patch_size
        self.byte_embedding = nn.Embedding(257, config.model_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(
            torch.empty(maximum_patches, config.model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.projection = nn.Linear(
            config.model_dim,
            config.embedding_dim,
            bias=False,
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, bytes]")
        if byte_ids.shape[1] % self.config.patch_size:
            raise ValueError("byte dimension must be divisible by patch_size")
        byte_mask = byte_ids.ne(0)
        embedded = self.byte_embedding(byte_ids)
        batch, byte_count, dimension = embedded.shape
        patch_count = byte_count // self.config.patch_size
        embedded = embedded.reshape(
            batch,
            patch_count,
            self.config.patch_size,
            dimension,
        )
        patch_byte_mask = byte_mask.reshape(
            batch,
            patch_count,
            self.config.patch_size,
        )
        patch_mask = patch_byte_mask.any(dim=2)
        patch_sum = (
            embedded * patch_byte_mask.unsqueeze(-1)
        ).sum(dim=2)
        patch_denominator = patch_byte_mask.sum(dim=2, keepdim=True).clamp_min(1)
        patches = patch_sum / patch_denominator
        patches = patches + self.position_embedding[:patch_count]
        encoded = self.encoder(
            patches,
            src_key_padding_mask=~patch_mask,
        )
        pooled = (
            encoded * patch_mask.unsqueeze(-1)
        ).sum(dim=1) / patch_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return F.normalize(
            self.projection(self.output_norm(pooled)),
            dim=-1,
        )


def encode_texts(
    texts: list[str],
    max_bytes: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not texts:
        raise ValueError("texts cannot be empty")
    encoded = [text.encode("utf-8")[:max_bytes] for text in texts]
    used_bytes = max(max(len(value), 1) for value in encoded)
    padded_bytes = min(
        max_bytes,
        math.ceil(used_bytes / patch_size) * patch_size,
    )
    result = torch.zeros(
        (len(texts), padded_bytes),
        dtype=torch.long,
        device=device,
    )
    for row, value in enumerate(encoded):
        if value:
            result[row, : len(value)] = (
                torch.tensor(list(value), dtype=torch.long, device=device) + 1
            )
    return result


def load_retrieval_split(
    topics_path: Path,
    qrels_path: Path,
    corpus_path: Path,
) -> dict[str, object]:
    topics: dict[str, str] = {}
    for line in topics_path.read_text(encoding="utf-8").splitlines():
        query_id, query = line.split("\t", 1)
        topics[query_id] = query

    positives: dict[str, set[str]] = defaultdict(set)
    negatives: dict[str, set[str]] = defaultdict(set)
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        query_id, _iteration, document_id, relevance = line.split("\t")
        target = positives if int(relevance) > 0 else negatives
        target[query_id].add(document_id)

    documents: dict[str, str] = {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        document_id = record["docid"]
        if document_id in documents:
            raise ValueError(f"duplicate document: {document_id}")
        documents[document_id] = f"{record['title']}\n{record['text']}"

    missing = {
        document_id
        for items in list(positives.values()) + list(negatives.values())
        for document_id in items
        if document_id not in documents
    }
    if missing:
        raise ValueError(f"qrels reference missing documents: {len(missing)}")
    return {
        "topics": topics,
        "positives": positives,
        "negatives": negatives,
        "documents": documents,
    }


def train_retriever(
    model: BytePatchRetriever,
    split: dict[str, object],
    config: ByteRetrieverConfig,
    device: torch.device,
) -> dict[str, object]:
    eligible = sorted(
        query_id
        for query_id in split["topics"]
        if split["positives"][query_id] and split["negatives"][query_id]
    )
    if not eligible:
        raise ValueError("training split has no eligible queries")
    rng = random.Random(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    losses: list[float] = []
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _step in range(config.train_steps):
        query_ids = [rng.choice(eligible) for _ in range(config.batch_size)]
        query_texts = [split["topics"][query_id] for query_id in query_ids]
        document_texts: list[str] = []
        for query_id in query_ids:
            positive = rng.choice(sorted(split["positives"][query_id]))
            negative = rng.choice(sorted(split["negatives"][query_id]))
            document_texts.extend(
                [split["documents"][positive], split["documents"][negative]]
            )
        query_bytes = encode_texts(
            query_texts,
            config.max_query_bytes,
            config.patch_size,
            device,
        )
        document_bytes = encode_texts(
            document_texts,
            config.max_document_bytes,
            config.patch_size,
            device,
        )
        query_embeddings = model(query_bytes)
        document_embeddings = model(document_bytes)
        logits = (
            query_embeddings @ document_embeddings.transpose(0, 1)
        ) / config.temperature
        targets = torch.arange(
            config.batch_size,
            device=device,
        ) * 2
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started
    return {
        "eligible_queries": len(eligible),
        "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6),
        "mean_last_100_loss": round(
            sum(losses[-100:]) / min(len(losses), 100),
            6,
        ),
        "elapsed_sec": round(elapsed, 6),
        "steps_per_sec": round(config.train_steps / elapsed, 6),
        "cuda_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / (1024**2),
            3,
        )
        if device.type == "cuda"
        else 0.0,
        "cuda_peak_reserved_mib": round(
            torch.cuda.max_memory_reserved(device) / (1024**2),
            3,
        )
        if device.type == "cuda"
        else 0.0,
    }


def evaluate_retriever(
    model: BytePatchRetriever,
    split: dict[str, object],
    config: ByteRetrieverConfig,
    device: torch.device,
    *,
    top_k: int = 100,
    encode_batch_size: int = 64,
) -> dict[str, object]:
    model.eval()
    document_ids = sorted(split["documents"])
    document_embeddings: list[torch.Tensor] = []
    corpus_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(document_ids), encode_batch_size):
            batch_ids = document_ids[start : start + encode_batch_size]
            byte_ids = encode_texts(
                [split["documents"][item] for item in batch_ids],
                config.max_document_bytes,
                config.patch_size,
                device,
            )
            document_embeddings.append(model(byte_ids))
        corpus_matrix = torch.cat(document_embeddings)
    corpus_elapsed = time.perf_counter() - corpus_started

    query_ids = sorted(split["topics"])
    query_started = time.perf_counter()
    with torch.inference_mode():
        query_bytes = encode_texts(
            [split["topics"][item] for item in query_ids],
            config.max_query_bytes,
            config.patch_size,
            device,
        )
        query_embeddings = model(query_bytes)
        scores = query_embeddings @ corpus_matrix.transpose(0, 1)
        ranking_indices = torch.topk(
            scores,
            k=min(top_k, len(document_ids)),
            dim=1,
        ).indices.cpu()
    query_elapsed = time.perf_counter() - query_started

    rankings = [
        [document_ids[index] for index in row.tolist()]
        for row in ranking_indices
    ]
    metrics = _metrics(
        query_ids,
        rankings,
        split["positives"],
    )
    return {
        "counts": {
            "queries": len(query_ids),
            "documents": len(document_ids),
            "positive_qrels": sum(
                len(split["positives"][query_id]) for query_id in query_ids
            ),
        },
        "metrics": metrics,
        "timing": {
            "corpus_encode_sec": round(corpus_elapsed, 6),
            "query_batch_and_search_sec": round(query_elapsed, 6),
            "query_mean_ms": round(query_elapsed * 1000 / len(query_ids), 6),
        },
    }


def _metrics(
    query_ids: list[str],
    rankings: list[list[str]],
    positives: dict[str, set[str]],
) -> dict[str, float]:
    reciprocal_ranks: list[float] = []
    recalls_10: list[float] = []
    recalls_100: list[float] = []
    ndcgs_10: list[float] = []
    for query_id, ranking in zip(query_ids, rankings, strict=True):
        relevant = positives[query_id]
        relevant_ranks = [
            rank
            for rank, document_id in enumerate(ranking, start=1)
            if document_id in relevant
        ]
        first_rank = relevant_ranks[0] if relevant_ranks else None
        reciprocal_ranks.append(
            1 / first_rank if first_rank is not None and first_rank <= 10 else 0
        )
        recalls_10.append(len(relevant & set(ranking[:10])) / len(relevant))
        recalls_100.append(len(relevant & set(ranking[:100])) / len(relevant))
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, document_id in enumerate(ranking[:10], start=1)
            if document_id in relevant
        )
        ideal = sum(
            1 / math.log2(rank + 1)
            for rank in range(1, min(len(relevant), 10) + 1)
        )
        ndcgs_10.append(dcg / ideal)
    return {
        "mrr_at_10": round(sum(reciprocal_ranks) / len(query_ids), 6),
        "ndcg_at_10": round(sum(ndcgs_10) / len(query_ids), 6),
        "recall_at_10": round(sum(recalls_10) / len(query_ids), 6),
        "recall_at_100": round(sum(recalls_100) / len(query_ids), 6),
    }


def run_experiment(
    config: ByteRetrieverConfig,
    *,
    train_split: dict[str, object],
    dev_split: dict[str, object],
    device: torch.device,
    checkpoint: Path,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = BytePatchRetriever(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    untrained = evaluate_retriever(model, dev_split, config, device)
    training = train_retriever(model, train_split, config, device)
    trained = evaluate_retriever(model, dev_split, config, device)
    checkpoint_info = save_checkpoint(checkpoint, model, config)

    mrr_gain = (
        trained["metrics"]["mrr_at_10"] - untrained["metrics"]["mrr_at_10"]
    )
    acceptance = {
        "parameter_count_at_most_5m": parameter_count <= 5_000_000,
        "checkpoint_at_most_25mb": checkpoint_info["bytes"] <= 25_000_000,
        "mrr_at_10_at_least_0_10": trained["metrics"]["mrr_at_10"] >= 0.10,
        "recall_at_100_at_least_0_50": (
            trained["metrics"]["recall_at_100"] >= 0.50
        ),
        "mrr_gain_at_least_0_05": mrr_gain >= 0.05,
        "loss_fell": training["final_loss"] < training["initial_loss"],
    }
    return {
        "schema_version": "mosaic-byte-retriever-v0",
        "scope": (
            "MIRACL-ko judged-candidate mechanism gate; not full-corpus retrieval"
        ),
        "device": str(device),
        "config": asdict(config),
        "parameter_count": parameter_count,
        "untrained_dev": untrained,
        "training": training,
        "trained_dev": trained,
        "comparison": {
            "mrr_at_10_gain": round(mrr_gain, 6),
            "frozen_bm25_dev_mrr_at_10": 0.574204,
            "frozen_bm25_dev_recall_at_100": 0.939045,
        },
        "checkpoint": checkpoint_info,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "limitations": [
            "Development is evaluated for this pre-registered run and must not be used for retuning.",
            "The model sees only fixed four-byte patches and judged candidate documents.",
            "No recurrent workspace or operator synthesis is connected yet.",
        ],
    }


def save_checkpoint(
    path: Path,
    model: BytePatchRetriever,
    config: ByteRetrieverConfig,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "mosaic-byte-retriever-checkpoint-v0",
            "config": asdict(config),
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
        },
        path,
    )
    digest = _sha256(path)
    manifest = {
        "schema_version": "mosaic-byte-retriever-manifest-v0",
        "checkpoint": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "source": (
            "random initialization trained only on MIRACL-ko train "
            "annotations and judged corpus documents"
        ),
        "external_datasets": [
            {
                "id": "miracl-ko-train-annotations",
                "revision": "5be20db9509754dadad47689368639fcec739c00",
                "license": "Apache-2.0",
            },
            {
                "id": "miracl-ko-corpus",
                "revision": "d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
                "license": "Apache-2.0 packaging; CC-BY-SA-4.0 Wikipedia text",
            },
        ],
        "license_status": "trained-weight redistribution review required",
        "redistribution": "not authorized by this manifest",
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "bytes": manifest["bytes"],
        "sha256": digest,
    }


def verify_checkpoint(
    path: Path,
    *,
    dev_split: dict[str, object],
    device: torch.device,
    expected_report: Path | None = None,
) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != "mosaic-byte-retriever-checkpoint-v0":
        raise ValueError("unsupported byte retriever checkpoint schema")
    config = ByteRetrieverConfig(**payload["config"])
    model = BytePatchRetriever(config).to(device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    evaluation = evaluate_retriever(model, dev_split, config, device)

    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha256(path)
    checks = {
        "manifest_digest": digest == manifest.get("sha256"),
        "manifest_size": path.stat().st_size == manifest.get("bytes"),
        "parameter_count_at_most_5m": (
            sum(parameter.numel() for parameter in model.parameters())
            <= 5_000_000
        ),
        "mrr_at_10_at_least_0_10": evaluation["metrics"]["mrr_at_10"] >= 0.10,
        "recall_at_100_at_least_0_50": (
            evaluation["metrics"]["recall_at_100"] >= 0.50
        ),
    }
    expected_metrics = None
    if expected_report is not None:
        expected = json.loads(expected_report.read_text(encoding="utf-8"))
        expected_metrics = expected["trained_dev"]["metrics"]
        checks["matches_training_report"] = (
            evaluation["metrics"] == expected_metrics
        )
    return {
        "schema_version": "mosaic-byte-retriever-verification-v0",
        "checkpoint": str(path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "device": str(device),
        "config": asdict(config),
        "evaluation": evaluation,
        "expected_metrics": expected_metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MOSAIC byte retriever")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/mosaic_sources_v1"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/mosaic_byte_retriever_v0.pt"),
    )
    parser.add_argument("--verify-checkpoint", type=Path)
    parser.add_argument("--expected-report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mosaic_byte_retriever_v0.json"),
    )
    args = parser.parse_args()
    raw = args.source_root / "raw"
    dev = load_retrieval_split(
        raw / "miracl-ko-dev-annotations" / "topics.ko.dev.tsv",
        raw / "miracl-ko-dev-annotations" / "qrels.ko.dev.tsv",
        args.source_root / "miracl_ko_dev_pilot" / "corpus.jsonl",
    )
    device = _resolve_device(args.device)
    if args.verify_checkpoint:
        report = verify_checkpoint(
            args.verify_checkpoint,
            dev_split=dev,
            device=device,
            expected_report=args.expected_report,
        )
    else:
        config = ByteRetrieverConfig(
            seed=args.seed,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
        )
        train = load_retrieval_split(
            raw
            / "miracl-ko-train-annotations"
            / "topics.ko.train.tsv",
            raw
            / "miracl-ko-train-annotations"
            / "qrels.ko.train.tsv",
            args.source_root / "miracl_ko_train" / "corpus.jsonl",
        )
        report = run_experiment(
            config,
            train_split=train,
            dev_split=dev,
            device=device,
            checkpoint=args.checkpoint,
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

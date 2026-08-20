"""Read-only access to every Rozephine-owned experience and knowledge artifact.

The universe deliberately does not filter by verification state, success, file
type, or memory tier.  Rozephine chooses what is relevant.  Consequential
writes and promoted claims remain separate concerns.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


@dataclass(frozen=True)
class ExperienceArtifact:
    root_id: str
    relative_path: str
    byte_count: int
    modified_ns: int

    def __post_init__(self) -> None:
        _require_text(self.root_id, "root_id")
        relative = Path(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("relative_path must stay inside its experience root")
        if self.byte_count < 0 or self.modified_ns < 0:
            raise ValueError("artifact metadata must be nonnegative")

    @property
    def address(self) -> str:
        return f"experience-artifact:{self.root_id}:{self.relative_path}"


@dataclass(frozen=True)
class ExperienceUniverseSnapshot:
    roots: tuple[tuple[str, str], ...]
    artifacts: tuple[ExperienceArtifact, ...]
    total_bytes: int
    metadata_snapshot_sha256: str
    status_filter_applied: bool = False
    verification_filter_applied: bool = False
    file_type_filter_applied: bool = False

    def __post_init__(self) -> None:
        if not self.roots:
            raise ValueError("at least one experience root is required")
        if len({name for name, _path in self.roots}) != len(self.roots):
            raise ValueError("experience root IDs must be unique")
        addresses = [artifact.address for artifact in self.artifacts]
        if len(addresses) != len(set(addresses)):
            raise ValueError("experience artifact addresses must be unique")
        if self.total_bytes != sum(row.byte_count for row in self.artifacts):
            raise ValueError("experience universe byte count changed")
        if any(
            (
                self.status_filter_applied,
                self.verification_filter_applied,
                self.file_type_filter_applied,
            )
        ):
            raise ValueError("an unrestricted experience universe cannot filter records")

    def root_path(self, root_id: str) -> Path:
        for name, path in self.roots:
            if name == root_id:
                return Path(path)
        raise KeyError(root_id)

    def artifact(self, address: str) -> ExperienceArtifact:
        for row in self.artifacts:
            if row.address == address:
                return row
        raise KeyError(address)

    def path(self, artifact: ExperienceArtifact) -> Path:
        if self.artifact(artifact.address) != artifact:
            raise ValueError("artifact is not part of this experience snapshot")
        root = self.root_path(artifact.root_id).resolve(strict=True)
        path = (root / artifact.relative_path).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("experience artifact escaped its owning root")
        stat = path.stat()
        if stat.st_size != artifact.byte_count or stat.st_mtime_ns != artifact.modified_ns:
            raise ValueError("experience artifact changed after discovery")
        return path

    def read_bytes(self, artifact: ExperienceArtifact) -> bytes:
        payload = self.path(artifact).read_bytes()
        if len(payload) != artifact.byte_count:
            raise ValueError("experience artifact changed while being read")
        return payload

    def read_text(self, artifact: ExperienceArtifact) -> str:
        payload = self.read_bytes(artifact)
        try:
            return payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("selected experience artifact is not UTF-8 text") from exc

    def json_lines(self, artifact: ExperienceArtifact) -> tuple[Any, ...]:
        rows = []
        for line_number, line in enumerate(self.read_text(artifact).splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"experience JSONL changed at line {line_number}"
                ) from exc
        return tuple(rows)

    def sqlite_tables(self, artifact: ExperienceArtifact) -> tuple[str, ...]:
        path = self.path(artifact)
        uri = f"file:{quote(path.as_posix(), safe='/:')}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def sqlite_rows(
        self, artifact: ExperienceArtifact, table: str
    ) -> tuple[dict[str, Any], ...]:
        table_name = _require_text(table, "table")
        if table_name not in self.sqlite_tables(artifact):
            raise KeyError(table_name)
        path = self.path(artifact)
        uri = f"file:{quote(path.as_posix(), safe='/:')}?mode=ro&immutable=1"
        quoted = '"' + table_name.replace('"', '""') + '"'
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(f"SELECT * FROM {quoted}").fetchall()
        return tuple(dict(row) for row in rows)


@dataclass(frozen=True)
class ExperienceSelectionReceipt:
    method: str
    rationale: str
    universe_artifact_count: int
    universe_total_bytes: int
    universe_metadata_snapshot_sha256: str
    selected_artifacts: tuple[dict[str, object], ...]
    rozephine_selected: bool = True
    codex_per_item_approval_used: bool = False
    external_action_authorized: bool = False
    memory_write_authorized: bool = False
    world_write_authorized: bool = False
    training_write_authorized: bool = False
    p3_promotion_authorized: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExperienceCandidateJudgment:
    """One runtime cognition judgment over a retrieved experience candidate."""

    address: str
    selected: bool
    relevance: float
    contradiction: float
    verification_state: str
    revision: str
    rationale: str
    rejection_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.address, "address")
        _require_text(self.verification_state, "verification_state")
        _require_text(self.revision, "revision")
        _require_text(self.rationale, "rationale")
        for name, value in (
            ("relevance", self.relevance),
            ("contradiction", self.contradiction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if any(not str(item).strip() for item in self.rejection_evidence):
            raise ValueError("rejection evidence must be nonempty")


@dataclass(frozen=True)
class RuntimeExperienceSelectionReceipt:
    """Replayable result of Rozephine judging every retrieved candidate."""

    query: str
    context: Mapping[str, Any]
    candidate_judgments: tuple[ExperienceCandidateJudgment, ...]
    selected_artifacts: tuple[Mapping[str, object], ...]
    selection_rationale: str
    universe_artifact_count: int
    universe_total_bytes: int
    universe_metadata_snapshot_sha256: str
    rozephine_selected: bool = True
    codex_per_item_approval_used: bool = False
    external_action_authorized: bool = False
    memory_write_authorized: bool = False
    world_write_authorized: bool = False
    training_write_authorized: bool = False
    p3_promotion_authorized: bool = False

    def __post_init__(self) -> None:
        _require_text(self.query, "query")
        _require_text(self.selection_rationale, "selection_rationale")
        if min(self.universe_artifact_count, self.universe_total_bytes) < 0:
            raise ValueError("experience universe metrics must be nonnegative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.universe_metadata_snapshot_sha256):
            raise ValueError("experience universe snapshot digest changed")
        candidate_addresses = tuple(item.address for item in self.candidate_judgments)
        selected_addresses = tuple(str(item["address"]) for item in self.selected_artifacts)
        expected_selected = tuple(
            item.address for item in self.candidate_judgments if item.selected
        )
        if (
            not candidate_addresses
            or len(candidate_addresses) != len(set(candidate_addresses))
            or selected_addresses != expected_selected
        ):
            raise ValueError("runtime experience selection receipt changed")
        if (
            not self.rozephine_selected
            or self.codex_per_item_approval_used
            or self.external_action_authorized
            or self.memory_write_authorized
            or self.world_write_authorized
            or self.training_write_authorized
            or self.p3_promotion_authorized
        ):
            raise ValueError("experience read authority changed")
        object.__setattr__(
            self,
            "context",
            MappingProxyType(json.loads(json.dumps(dict(self.context), ensure_ascii=False))),
        )
        object.__setattr__(
            self,
            "selected_artifacts",
            tuple(MappingProxyType(dict(item)) for item in self.selected_artifacts),
        )

    @property
    def method(self) -> str:
        return "runtime_cognition_relevance_and_contradiction_judgment"

    @property
    def rationale(self) -> str:
        return self.selection_rationale

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "context": dict(self.context),
            "candidate_judgments": [asdict(item) for item in self.candidate_judgments],
            "selected_artifacts": [dict(item) for item in self.selected_artifacts],
            "selection_rationale": self.selection_rationale,
            "universe_artifact_count": self.universe_artifact_count,
            "universe_total_bytes": self.universe_total_bytes,
            "universe_metadata_snapshot_sha256": self.universe_metadata_snapshot_sha256,
            "rozephine_selected": self.rozephine_selected,
            "codex_per_item_approval_used": self.codex_per_item_approval_used,
            "external_action_authorized": self.external_action_authorized,
            "memory_write_authorized": self.memory_write_authorized,
            "world_write_authorized": self.world_write_authorized,
            "training_write_authorized": self.training_write_authorized,
            "p3_promotion_authorized": self.p3_promotion_authorized,
        }


@dataclass(frozen=True)
class HotExperienceIndex:
    """Prebuilt O(1) lookup view for the live cognition path.

    Building the index is deliberately separate from lookup.  The hot methods
    perform no filesystem, JSON, hashing, or SQLite work.
    """

    universe_metadata_snapshot_sha256: str
    artifact_count: int
    _artifacts_by_address: Mapping[str, ExperienceArtifact]
    _semantic_postings: Mapping[str, tuple[str, ...]]
    timing_unit: str = "nanoseconds"
    requires_io_on_lookup: bool = False

    def __post_init__(self) -> None:
        if self.artifact_count != len(self._artifacts_by_address):
            raise ValueError("hot experience artifact count changed")
        if self.timing_unit != "nanoseconds" or self.requires_io_on_lookup:
            raise ValueError("hot experience lookup contract changed")

    def lookup_address(self, address: str) -> ExperienceArtifact:
        try:
            return self._artifacts_by_address[address]
        except KeyError as exc:
            raise KeyError(address) from exc

    def lookup_semantic_key(self, semantic_key: str) -> tuple[str, ...]:
        return self._semantic_postings.get(semantic_key, ())

    def timed_address_lookup(
        self, address: str
    ) -> tuple[ExperienceArtifact, int]:
        started_ns = time.perf_counter_ns()
        artifact = self.lookup_address(address)
        elapsed_ns = time.perf_counter_ns() - started_ns
        return artifact, elapsed_ns

    def timed_semantic_lookup(self, semantic_key: str) -> tuple[tuple[str, ...], int]:
        started_ns = time.perf_counter_ns()
        addresses = self.lookup_semantic_key(semantic_key)
        elapsed_ns = time.perf_counter_ns() - started_ns
        return addresses, elapsed_ns

    @property
    def all_addresses(self) -> tuple[str, ...]:
        return tuple(self._artifacts_by_address)


def discover_experience_universe(
    roots: Mapping[str, Path],
) -> ExperienceUniverseSnapshot:
    """Discover every regular file under the supplied Rozephine-owned roots."""

    normalized = []
    artifacts = []
    root_ids = set()
    for raw_name, raw_path in sorted(roots.items()):
        name = _require_text(raw_name, "root_id")
        if name in root_ids:
            raise ValueError("experience root IDs must be unique")
        root_ids.add(name)
        root = Path(raw_path).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("experience roots must be directories")
        normalized.append((name, str(root)))
        for directory, directories, files in os.walk(root, followlinks=False):
            directories.sort()
            files.sort()
            base = Path(directory)
            for filename in files:
                path = base / filename
                if not path.is_file():
                    continue
                stat = path.stat()
                artifacts.append(
                    ExperienceArtifact(
                        root_id=name,
                        relative_path=path.relative_to(root).as_posix(),
                        byte_count=stat.st_size,
                        modified_ns=stat.st_mtime_ns,
                    )
                )
    artifacts.sort(key=lambda row: (row.root_id, row.relative_path))
    digest = hashlib.sha256()
    for name, path in normalized:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    for row in artifacts:
        digest.update(row.address.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.byte_count).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row.modified_ns).encode("ascii"))
        digest.update(b"\0")
    return ExperienceUniverseSnapshot(
        roots=tuple(normalized),
        artifacts=tuple(artifacts),
        total_bytes=sum(row.byte_count for row in artifacts),
        metadata_snapshot_sha256=digest.hexdigest(),
    )


def build_hot_experience_index(
    snapshot: ExperienceUniverseSnapshot,
    *,
    semantic_keys_by_address: Mapping[str, Iterable[str]] | None = None,
) -> HotExperienceIndex:
    """Build a complete address index and optional Rozephine-authored postings."""

    artifacts = {row.address: row for row in snapshot.artifacts}
    supplied = semantic_keys_by_address or {}
    unknown = set(supplied) - set(artifacts)
    if unknown:
        raise KeyError(sorted(unknown)[0])
    postings: dict[str, set[str]] = {}
    for address, artifact in artifacts.items():
        path_tokens = tuple(
            re.findall(
                r"n\d+|r\d+|[a-z]+|\d+|[^\W\d_]+",
                artifact.relative_path.lower(),
            )
        )
        automatic_keys = (
            artifact.root_id,
            artifact.relative_path,
            Path(artifact.relative_path).name,
            Path(artifact.relative_path).stem,
            *path_tokens,
        )
        for raw_key in (*automatic_keys, *tuple(supplied.get(address, ()))):
            key = _require_text(raw_key, "semantic_key")
            postings.setdefault(key, set()).add(address)
    frozen_postings = {
        key: tuple(sorted(addresses)) for key, addresses in postings.items()
    }
    return HotExperienceIndex(
        universe_metadata_snapshot_sha256=snapshot.metadata_snapshot_sha256,
        artifact_count=len(artifacts),
        _artifacts_by_address=MappingProxyType(artifacts),
        _semantic_postings=MappingProxyType(frozen_postings),
    )


def record_rozephine_selection(
    snapshot: ExperienceUniverseSnapshot,
    selected_addresses: Iterable[str],
    *,
    method: str,
    rationale: str,
) -> ExperienceSelectionReceipt:
    """Record Rozephine's choice without applying a Codex item allowlist."""

    selected = []
    seen = set()
    for address in selected_addresses:
        if address in seen:
            raise ValueError("selected experience addresses must be unique")
        seen.add(address)
        artifact = snapshot.artifact(address)
        payload = snapshot.read_bytes(artifact)
        selected.append(
            {
                "address": artifact.address,
                "byte_count": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    if not selected:
        raise ValueError("Rozephine must select at least one experience artifact")
    return ExperienceSelectionReceipt(
        method=_require_text(method, "method"),
        rationale=_require_text(rationale, "rationale"),
        universe_artifact_count=len(snapshot.artifacts),
        universe_total_bytes=snapshot.total_bytes,
        universe_metadata_snapshot_sha256=snapshot.metadata_snapshot_sha256,
        selected_artifacts=tuple(selected),
    )


def select_experience_for_cognition(
    snapshot: ExperienceUniverseSnapshot,
    hot: HotExperienceIndex,
    *,
    query: str,
    context: Mapping[str, Any],
    judge: Callable[[ExperienceArtifact, Mapping[str, Any]], ExperienceCandidateJudgment],
) -> RuntimeExperienceSelectionReceipt:
    """Let runtime cognition choose from fast-retrieved, unfiltered experience.

    Retrieval derives candidates from the query text and falls back to the complete
    hot universe.  The caller supplies a cognition judgment, not a selected-address
    allowlist.  Every candidate receives a replayable selected/rejected audit.
    """

    query_text = _require_text(query, "query")
    if hot.universe_metadata_snapshot_sha256 != snapshot.metadata_snapshot_sha256:
        raise ValueError("hot experience index does not match the universe snapshot")
    keys = tuple(
        dict.fromkeys(
            re.findall(r"n\d+|r\d+|[a-z]+|\d+|[^\W\d_]+", query_text.lower())
        )
    )
    retrieved = {
        address for key in keys for address in hot.lookup_semantic_key(key)
    }
    candidate_addresses = tuple(sorted(retrieved)) or hot.all_addresses
    if not candidate_addresses:
        raise ValueError("experience universe has no candidate artifacts")

    judgments = []
    selected = []
    for address in candidate_addresses:
        artifact = hot.lookup_address(address)
        judgment = judge(artifact, context)
        if judgment.address != address:
            raise ValueError("runtime cognition judgment address changed")
        judgments.append(judgment)
        if judgment.selected:
            payload = snapshot.read_bytes(artifact)
            selected.append(
                {
                    "address": address,
                    "byte_count": len(payload),
                    "sha256": _sha256_bytes(payload),
                    "verification_state": judgment.verification_state,
                    "revision": judgment.revision,
                }
            )
    if not selected:
        raise ValueError("runtime cognition selected no experience")
    return RuntimeExperienceSelectionReceipt(
        query=query_text,
        context=json.loads(json.dumps(dict(context), ensure_ascii=False)),
        candidate_judgments=tuple(judgments),
        selected_artifacts=tuple(selected),
        selection_rationale=(
            "Rozephine selected relevant experience after explicit contradiction review"
        ),
        universe_artifact_count=len(snapshot.artifacts),
        universe_total_bytes=snapshot.total_bytes,
        universe_metadata_snapshot_sha256=snapshot.metadata_snapshot_sha256,
    )

"""Evidence-gated validity metadata for external MOSAIC memory documents."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from swegca.mosaic_evidence_accumulator import (
    AccumulatorDecision,
    require_authoritative_accumulator_decision,
)
from swegca.mosaic_external_memory import (
    ExternalMemoryDocument,
    MosaicExternalMemory,
)


def _utc_timestamp(value: str) -> str:
    if not value.endswith("Z"):
        raise ValueError("memory validity timestamps must use UTC Z format")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo != timezone.utc:
        raise ValueError("memory validity timestamps must be UTC")
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _require_accept(decision: AccumulatorDecision) -> None:
    require_authoritative_accumulator_decision(decision)
    if decision.status != "accept":
        raise PermissionError("external memory mutation requires accepted evidence")


@dataclass(frozen=True)
class MemoryMutation:
    update_id: str
    operation: str
    docid: str
    supersedes_docid: str | None
    applied_at: str


class VersionedExternalMemory:
    """Adds verified validity, supersession, expiry, and rollback to a memory DB."""

    def __init__(
        self,
        memory: MosaicExternalMemory,
        *,
        maximum_search_candidates: int = 256,
    ) -> None:
        if maximum_search_candidates <= 0:
            raise ValueError("maximum_search_candidates must be positive")
        self.memory = memory
        self.database = memory.database
        self.tokenizer = memory.tokenizer
        self.maximum_search_candidates = maximum_search_candidates
        self._ensure_schema()

    @property
    def persistent_state_count(self) -> int:
        return 0

    def _connect(self) -> sqlite3.Connection:
        return self.memory._connect(readonly=False)

    def _ensure_schema(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS verified_document_validity (
                    docid TEXT PRIMARY KEY,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    supersedes_docid TEXT,
                    confidence REAL NOT NULL,
                    evidence_revision INTEGER NOT NULL,
                    update_id TEXT NOT NULL UNIQUE,
                    FOREIGN KEY (docid) REFERENCES documents(docid)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS verified_document_validity_window
                ON verified_document_validity(valid_from, valid_until);
                CREATE TABLE IF NOT EXISTS verified_memory_audit (
                    update_id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    docid TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    rolled_back_at TEXT
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _decision_payload(decision: AccumulatorDecision) -> dict[str, object]:
        return asdict(decision)

    def register_existing_verified(
        self,
        *,
        decision: AccumulatorDecision,
        valid_from: str,
        update_prefix: str,
    ) -> int:
        """Register an already evidence-approved immutable DB snapshot."""
        _require_accept(decision)
        valid_from = _utc_timestamp(valid_from)
        connection = self._connect()
        try:
            rows = connection.execute("SELECT docid FROM documents ORDER BY docid").fetchall()
            inserted = 0
            for index, row in enumerate(rows):
                docid = str(row["docid"])
                update_id = f"{update_prefix}:{index:08d}"
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO verified_document_validity("
                    "docid, valid_from, valid_until, supersedes_docid, confidence, "
                    "evidence_revision, update_id) VALUES (?, ?, NULL, NULL, ?, ?, ?)",
                    (
                        docid,
                        valid_from,
                        decision.posterior_mean,
                        decision.revision,
                        update_id,
                    ),
                )
                inserted += cursor.rowcount
            connection.commit()
            return inserted
        finally:
            connection.close()

    def upsert_verified(
        self,
        document: ExternalMemoryDocument,
        *,
        decision: AccumulatorDecision,
        valid_from: str,
        update_id: str,
        supersedes_docid: str | None = None,
        valid_until: str | None = None,
    ) -> MemoryMutation:
        _require_accept(decision)
        valid_from = _utc_timestamp(valid_from)
        valid_until = _utc_timestamp(valid_until) if valid_until is not None else None
        if valid_until is not None and valid_until <= valid_from:
            raise ValueError("memory expiry must follow validity start")
        if not update_id:
            raise ValueError("update_id must be nonempty")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM verified_memory_audit WHERE update_id=?", (update_id,)
            ).fetchone():
                raise ValueError("update_id already exists")
            if connection.execute(
                "SELECT 1 FROM verified_document_validity WHERE docid=?",
                (document.docid,),
            ).fetchone():
                raise ValueError("versioned documents require a new docid")
            previous_valid_until = None
            if supersedes_docid is not None:
                predecessor = connection.execute(
                    "SELECT valid_from, valid_until FROM verified_document_validity "
                    "WHERE docid=?",
                    (supersedes_docid,),
                ).fetchone()
                if predecessor is None:
                    raise ValueError("superseded document is not verified")
                if predecessor["valid_from"] > valid_from:
                    raise ValueError("replacement predates superseded document")
                previous_valid_until = predecessor["valid_until"]
            self.memory._upsert(connection, document)
            if supersedes_docid is not None:
                connection.execute(
                    "UPDATE verified_document_validity SET valid_until=? WHERE docid=?",
                    (valid_from, supersedes_docid),
                )
            connection.execute(
                "INSERT INTO verified_document_validity("
                "docid, valid_from, valid_until, supersedes_docid, confidence, "
                "evidence_revision, update_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document.docid,
                    valid_from,
                    valid_until,
                    supersedes_docid,
                    decision.posterior_mean,
                    decision.revision,
                    update_id,
                ),
            )
            payload = {
                "supersedes_docid": supersedes_docid,
                "previous_valid_until": previous_valid_until,
                "decision": self._decision_payload(decision),
            }
            connection.execute(
                "INSERT INTO verified_memory_audit("
                "update_id, operation, docid, applied_at, payload_json"
                ") VALUES (?, 'upsert', ?, ?, ?)",
                (
                    update_id,
                    document.docid,
                    valid_from,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return MemoryMutation(
            update_id, "upsert", document.docid, supersedes_docid, valid_from
        )

    def due_for_revalidation(
        self,
        *,
        as_of: str,
        horizon_seconds: int,
        limit: int = 32,
    ) -> list[dict[str, object]]:
        """Return currently readable documents whose explicit TTL is nearly due."""

        instant = _utc_timestamp(as_of)
        if horizon_seconds < 0 or limit <= 0:
            raise ValueError("revalidation horizon must be nonnegative and limit positive")
        deadline = (
            datetime.fromisoformat(instant[:-1] + "+00:00")
            + timedelta(seconds=horizon_seconds)
        ).isoformat(timespec="seconds").replace("+00:00", "Z")
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT d.*, v.valid_from, v.valid_until, v.confidence, "
                "v.evidence_revision, v.update_id FROM documents d "
                "JOIN verified_document_validity v ON v.docid=d.docid "
                "WHERE v.valid_from <= ? AND v.valid_until IS NOT NULL "
                "AND v.valid_until > ? AND v.valid_until <= ? "
                "ORDER BY v.valid_until, d.docid LIMIT ?",
                (instant, instant, deadline, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def expire_verified(
        self,
        docid: str,
        *,
        decision: AccumulatorDecision,
        valid_until: str,
        update_id: str,
    ) -> MemoryMutation:
        _require_accept(decision)
        valid_until = _utc_timestamp(valid_until)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT valid_from, valid_until FROM verified_document_validity "
                "WHERE docid=?",
                (docid,),
            ).fetchone()
            if row is None:
                raise KeyError(docid)
            if row["valid_from"] > valid_until:
                raise ValueError("expiry predates document validity")
            if connection.execute(
                "SELECT 1 FROM verified_memory_audit WHERE update_id=?", (update_id,)
            ).fetchone():
                raise ValueError("update_id already exists")
            payload = {
                "previous_valid_until": row["valid_until"],
                "decision": self._decision_payload(decision),
            }
            connection.execute(
                "UPDATE verified_document_validity SET valid_until=? WHERE docid=?",
                (valid_until, docid),
            )
            connection.execute(
                "INSERT INTO verified_memory_audit("
                "update_id, operation, docid, applied_at, payload_json"
                ") VALUES (?, 'expire', ?, ?, ?)",
                (
                    update_id,
                    docid,
                    valid_until,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return MemoryMutation(update_id, "expire", docid, None, valid_until)

    def rollback_verified(
        self,
        update_id: str,
        *,
        decision: AccumulatorDecision,
        rolled_back_at: str,
    ) -> bool:
        _require_accept(decision)
        rolled_back_at = _utc_timestamp(rolled_back_at)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT operation, docid, payload_json, rolled_back_at "
                "FROM verified_memory_audit WHERE update_id=?",
                (update_id,),
            ).fetchone()
            if row is None:
                raise KeyError(update_id)
            if row["rolled_back_at"] is not None:
                return False
            payload = json.loads(str(row["payload_json"]))
            if row["operation"] == "upsert":
                delete_docid = str(row["docid"])
                predecessor = payload.get("supersedes_docid")
                if predecessor is not None:
                    connection.execute(
                        "UPDATE verified_document_validity SET valid_until=? "
                        "WHERE docid=?",
                        (payload.get("previous_valid_until"), predecessor),
                    )
                connection.execute(
                    "DELETE FROM verified_document_validity WHERE docid=?",
                    (delete_docid,),
                )
                if not self.memory._delete(connection, delete_docid):
                    raise RuntimeError("verified document disappeared during rollback")
            elif row["operation"] == "expire":
                connection.execute(
                    "UPDATE verified_document_validity SET valid_until=? WHERE docid=?",
                    (payload.get("previous_valid_until"), row["docid"]),
                )
            else:
                raise ValueError("unsupported audit operation")
            connection.execute(
                "UPDATE verified_memory_audit SET rolled_back_at=? WHERE update_id=?",
                (rolled_back_at, update_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return True

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        as_of: str | None = None,
    ) -> list[dict[str, object]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        instant = _utc_timestamp(
            as_of
            or datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            )
        )
        candidates = self.memory.search(
            query,
            limit=max(limit, self.maximum_search_candidates),
        )
        if not candidates:
            return []
        docids = [str(row["docid"]) for row in candidates]
        placeholders = ",".join("?" for _ in docids)
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT docid, valid_from, valid_until, confidence, "
                "evidence_revision, update_id FROM verified_document_validity "
                f"WHERE docid IN ({placeholders}) AND valid_from <= ? "
                "AND (valid_until IS NULL OR valid_until > ?)",
                (*docids, instant, instant),
            ).fetchall()
        finally:
            connection.close()
        active = {str(row["docid"]): dict(row) for row in rows}
        output = []
        for candidate in candidates:
            metadata = active.get(str(candidate["docid"]))
            if metadata is None:
                continue
            merged = dict(candidate)
            merged["validity"] = metadata
            output.append(merged)
            if len(output) == limit:
                break
        return output

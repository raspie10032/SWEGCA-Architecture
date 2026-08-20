"""Incremental provenance memory used by the unified World Latent runtime."""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from swegca.mosaic_sqlite_memory import _fts_query


@dataclass(frozen=True)
class ExternalMemoryDocument:
    docid: str
    source_id: str
    source_revision: str
    page_id: str
    revision_id: str
    namespace: str
    timestamp: str
    redirect: bool
    title: str
    text: str
    content_sha256: str
    duplicate_status: str
    source_url: str
    license: str
    attribution: str


@dataclass(frozen=True)
class ExternalMemoryResource:
    docid: str
    resource_id: str
    modality: str
    storage_path: str
    storage_sha256: str
    item_index: int
    metadata: dict[str, object]


def _query(text: str, tokenizer: str) -> str:
    if tokenizer == "trigram":
        return _fts_query(text)
    if tokenizer == "unicode61":
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"'
            for term in normalized.split()
            if term
        )
    raise ValueError(f"unsupported tokenizer: {tokenizer}")


class MosaicExternalMemory:
    def __init__(self, database: Path, *, tokenizer: str = "unicode61") -> None:
        if tokenizer not in {"trigram", "unicode61"}:
            raise ValueError(f"unsupported tokenizer: {tokenizer}")
        if not database.exists():
            raise FileNotFoundError(database)
        self.database = database.resolve()
        self.tokenizer = tokenizer

    def _connect(self, *, readonly: bool) -> sqlite3.Connection:
        if readonly:
            connection = sqlite3.connect(
                f"file:{self.database.as_posix()}?mode=ro",
                uri=True,
            )
            connection.execute("PRAGMA query_only=ON")
        else:
            connection = sqlite3.connect(self.database)
            connection.execute("PRAGMA foreign_keys=ON")
            self._ensure_update_triggers(connection)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _ensure_update_triggers(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS documents_after_insert
            AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, title, text)
                VALUES (new.rowid, new.title, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS documents_after_delete
            AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(
                    documents_fts, rowid, title, text
                ) VALUES ('delete', old.rowid, old.title, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS documents_after_update
            AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(
                    documents_fts, rowid, title, text
                ) VALUES ('delete', old.rowid, old.title, old.text);
                INSERT INTO documents_fts(rowid, title, text)
                VALUES (new.rowid, new.title, new.text);
            END;
            CREATE TABLE IF NOT EXISTS document_resources (
                docid TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                storage_sha256 TEXT NOT NULL,
                item_index INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (docid, resource_id),
                FOREIGN KEY (docid) REFERENCES documents(docid)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS document_resources_modality
            ON document_resources(modality);
            """
        )

    def search(self, query: str, *, limit: int = 8) -> list[dict[str, object]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        compiled = _query(query, self.tokenizer)
        if not compiled:
            return []
        connection = self._connect(readonly=True)
        try:
            rows = connection.execute(
                "SELECT d.docid, d.source_id, d.source_revision, d.page_id, "
                "d.revision_id, d.namespace, d.timestamp, d.redirect, "
                "d.title, d.text, d.content_sha256, d.duplicate_status, "
                "d.source_url, d.license, d.attribution, "
                "bm25(documents_fts, 4.0, 1.0) AS score "
                "FROM documents_fts AS f "
                "JOIN documents AS d ON d.rowid=f.rowid "
                "WHERE documents_fts MATCH ? "
                "ORDER BY bm25(documents_fts, 4.0, 1.0) LIMIT ?",
                (compiled, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def _refresh_duplicate_group(
        self,
        connection: sqlite3.Connection,
        content_sha256: str,
    ) -> None:
        connection.execute(
            "DELETE FROM duplicate_groups WHERE content_sha256=?",
            (content_sha256,),
        )
        occurrences = connection.execute(
            "SELECT count(*) FROM documents WHERE content_sha256=?",
            (content_sha256,),
        ).fetchone()[0]
        if occurrences > 1:
            connection.execute(
                "INSERT INTO duplicate_groups(content_sha256, occurrences) "
                "VALUES (?, ?)",
                (content_sha256, occurrences),
            )

    def upsert(self, document: ExternalMemoryDocument) -> None:
        connection = self._connect(readonly=False)
        try:
            self._upsert(connection, document)
            connection.commit()
        finally:
            connection.close()

    def _upsert(
        self,
        connection: sqlite3.Connection,
        document: ExternalMemoryDocument,
    ) -> None:
        existing = connection.execute(
            "SELECT content_sha256 FROM documents WHERE docid=?",
            (document.docid,),
        ).fetchone()
        values = (
            document.docid,
            document.source_id,
            document.source_revision,
            document.page_id,
            document.revision_id,
            document.namespace,
            document.timestamp,
            int(document.redirect),
            document.title,
            document.text,
            document.content_sha256,
            document.duplicate_status,
            document.source_url,
            document.license,
            document.attribution,
        )
        connection.execute(
            "INSERT INTO documents("
            "docid, source_id, source_revision, page_id, revision_id, "
            "namespace, timestamp, redirect, title, text, content_sha256, "
            "duplicate_status, source_url, license, attribution"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(docid) DO UPDATE SET "
            "source_id=excluded.source_id, "
            "source_revision=excluded.source_revision, "
            "page_id=excluded.page_id, "
            "revision_id=excluded.revision_id, "
            "namespace=excluded.namespace, "
            "timestamp=excluded.timestamp, "
            "redirect=excluded.redirect, "
            "title=excluded.title, text=excluded.text, "
            "content_sha256=excluded.content_sha256, "
            "duplicate_status=excluded.duplicate_status, "
            "source_url=excluded.source_url, license=excluded.license, "
            "attribution=excluded.attribution",
            values,
        )
        self._refresh_duplicate_group(connection, document.content_sha256)
        if existing is not None and existing[0] != document.content_sha256:
            self._refresh_duplicate_group(connection, str(existing[0]))

    def upsert_resource(self, resource: ExternalMemoryResource) -> None:
        if resource.item_index < 0:
            raise ValueError("resource item index must be non-negative")
        connection = self._connect(readonly=False)
        try:
            connection.execute(
                "INSERT INTO document_resources("
                "docid, resource_id, modality, storage_path, storage_sha256, "
                "item_index, metadata_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(docid, resource_id) DO UPDATE SET "
                "modality=excluded.modality, "
                "storage_path=excluded.storage_path, "
                "storage_sha256=excluded.storage_sha256, "
                "item_index=excluded.item_index, "
                "metadata_json=excluded.metadata_json",
                (
                    resource.docid,
                    resource.resource_id,
                    resource.modality,
                    resource.storage_path,
                    resource.storage_sha256,
                    resource.item_index,
                    json.dumps(
                        resource.metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def resources(self, docid: str) -> list[dict[str, object]]:
        connection = self._connect(readonly=True)
        try:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='document_resources'"
            ).fetchone()
            if exists is None:
                return []
            rows = connection.execute(
                "SELECT docid, resource_id, modality, storage_path, "
                "storage_sha256, item_index, metadata_json "
                "FROM document_resources WHERE docid=? "
                "ORDER BY resource_id",
                (docid,),
            ).fetchall()
            resources = []
            for row in rows:
                resource = dict(row)
                resource["metadata"] = json.loads(
                    str(resource.pop("metadata_json"))
                )
                resources.append(resource)
            return resources
        finally:
            connection.close()

    def documents(
        self,
        *,
        namespace: str | None = None,
    ) -> list[dict[str, object]]:
        connection = self._connect(readonly=True)
        try:
            where = "WHERE namespace=?" if namespace is not None else ""
            parameters = (namespace,) if namespace is not None else ()
            rows = connection.execute(
                "SELECT docid, source_id, source_revision, page_id, revision_id, "
                "namespace, timestamp, redirect, title, text, content_sha256, "
                "duplicate_status, source_url, license, attribution "
                f"FROM documents {where} ORDER BY timestamp, docid",
                parameters,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def delete(self, docid: str) -> bool:
        connection = self._connect(readonly=False)
        try:
            deleted = self._delete(connection, docid)
            connection.commit()
            return deleted
        finally:
            connection.close()

    def _delete(self, connection: sqlite3.Connection, docid: str) -> bool:
        existing = connection.execute(
            "SELECT content_sha256 FROM documents WHERE docid=?",
            (docid,),
        ).fetchone()
        if existing is None:
            return False
        connection.execute("DELETE FROM documents WHERE docid=?", (docid,))
        self._refresh_duplicate_group(connection, str(existing[0]))
        return True

    @staticmethod
    def evidence_text(
        query: str,
        matches: list[dict[str, object]],
        *,
        maximum_bytes: int,
        per_document_characters: int = 2_000,
    ) -> str:
        if min(maximum_bytes, per_document_characters) <= 0:
            raise ValueError("evidence budgets must be positive")
        sections = [f"Query:\n{query}\n\nRetrieved evidence:"]
        for index, match in enumerate(matches, start=1):
            sections.append(
                "\n".join(
                    (
                        f"[{index}] {match['title']}",
                        str(match["text"])[:per_document_characters],
                        (
                            "Provenance: "
                            f"{match['source_id']} revision={match['revision_id']} "
                            f"license={match['license']} url={match['source_url']}"
                        ),
                    )
                )
            )
        encoded = "\n\n".join(sections).encode("utf-8")
        return encoded[:maximum_bytes].decode("utf-8", errors="ignore")

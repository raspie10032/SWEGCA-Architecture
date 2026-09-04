from __future__ import annotations

import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any


FACT_SEARCH_ALIASES = {
    "user_name": "name 이름 성함",
    "favorite_color": "favorite color 좋아하는 색 색상",
    "user_goal": "goal 목표",
    "remembered_word": "remember word 기억 기억할 기억해 단어 단어를",
}


class ConversationMemory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source_turn TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    supersedes_id INTEGER,
                    created_ns INTEGER NOT NULL,
                    FOREIGN KEY (supersedes_id) REFERENCES memory_events(id)
                );
                CREATE INDEX IF NOT EXISTS memory_lookup
                    ON memory_events(namespace, kind, subject, predicate, id);
                CREATE INDEX IF NOT EXISTS memory_supersedes
                    ON memory_events(supersedes_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def remember_fact(
        self,
        *,
        namespace: str,
        subject: str,
        predicate: str,
        value: str,
        source_turn: str,
        confidence: float = 1.0,
    ) -> int:
        fields = [namespace, subject, predicate, value, source_turn]
        if any(not str(field).strip() for field in fields):
            raise ValueError("memory fact fields must not be empty")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._remember_fact(
                connection,
                namespace=namespace,
                subject=subject,
                predicate=predicate,
                value=value,
                source_turn=source_turn,
                confidence=confidence,
            )

    def append_episode(
        self,
        *,
        namespace: str,
        subject: str,
        value: str,
        source_turn: str,
        confidence: float = 1.0,
    ) -> int:
        if any(not str(field).strip() for field in (namespace, subject, value, source_turn)):
            raise ValueError("episode fields must not be empty")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        with closing(self._connect()) as connection, connection:
            return self._append_episode(
                connection,
                namespace=namespace,
                subject=subject,
                value=value,
                source_turn=source_turn,
                confidence=confidence,
            )

    def record_turn(
        self,
        *,
        namespace: str,
        episode_subject: str,
        episode_value: str,
        source_turn: str,
        facts: tuple[tuple[str, str, str], ...] = (),
        confidence: float = 1.0,
    ) -> tuple[int, tuple[int, ...]]:
        if any(
            not str(field).strip()
            for field in (namespace, episode_subject, episode_value, source_turn)
        ):
            raise ValueError("turn episode fields must not be empty")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        for subject, predicate, value in facts:
            if any(not str(field).strip() for field in (subject, predicate, value)):
                raise ValueError("memory fact fields must not be empty")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            episode_id = self._append_episode(
                connection,
                namespace=namespace,
                subject=episode_subject,
                value=episode_value,
                source_turn=source_turn,
                confidence=confidence,
            )
            fact_ids = tuple(
                self._remember_fact(
                    connection,
                    namespace=namespace,
                    subject=subject,
                    predicate=predicate,
                    value=value,
                    source_turn=source_turn,
                    confidence=confidence,
                )
                for subject, predicate, value in facts
            )
        return episode_id, fact_ids

    @staticmethod
    def _append_episode(
        connection: sqlite3.Connection,
        *,
        namespace: str,
        subject: str,
        value: str,
        source_turn: str,
        confidence: float,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO memory_events (
                namespace, kind, subject, predicate, value, source_turn,
                confidence, supersedes_id, created_ns
            ) VALUES (?, 'episode', ?, 'event', ?, ?, ?, NULL, ?)
            """,
            (namespace, subject, value, source_turn, confidence, time.time_ns()),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _remember_fact(
        connection: sqlite3.Connection,
        *,
        namespace: str,
        subject: str,
        predicate: str,
        value: str,
        source_turn: str,
        confidence: float,
    ) -> int:
        previous = connection.execute(
            """
            SELECT id, value
            FROM memory_events
            WHERE namespace = ? AND kind = 'fact'
              AND subject = ? AND predicate = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (namespace, subject, predicate),
        ).fetchone()
        if previous is not None and previous["value"] == value:
            return int(previous["id"])
        cursor = connection.execute(
            """
            INSERT INTO memory_events (
                namespace, kind, subject, predicate, value, source_turn,
                confidence, supersedes_id, created_ns
            ) VALUES (?, 'fact', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                namespace,
                subject,
                predicate,
                value,
                source_turn,
                confidence,
                int(previous["id"]) if previous is not None else None,
                time.time_ns(),
            ),
        )
        return int(cursor.lastrowid)

    def active_facts(self, namespace: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT current.*
                FROM memory_events AS current
                WHERE current.namespace = ? AND current.kind = 'fact'
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_events AS newer
                      WHERE newer.supersedes_id = current.id
                  )
                ORDER BY current.id
                """,
                (namespace,),
            ).fetchall()
        return [dict(row) for row in rows]

    def forget_fact(
        self,
        *,
        namespace: str,
        subject: str,
        predicate: str,
    ) -> int:
        if any(not str(field).strip() for field in (namespace, subject, predicate)):
            raise ValueError("memory fact identity fields must not be empty")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                DELETE FROM memory_events
                WHERE namespace = ? AND kind = 'fact'
                  AND subject = ? AND predicate = ?
                """,
                (namespace, subject, predicate),
            )
        return max(0, int(cursor.rowcount))

    def history(
        self,
        *,
        namespace: str,
        subject: str,
        predicate: str,
    ) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT *
                FROM memory_events
                WHERE namespace = ? AND kind = 'fact'
                  AND subject = ? AND predicate = ?
                ORDER BY id
                """,
                (namespace, subject, predicate),
            ).fetchall()
        return [dict(row) for row in rows]

    def recall(
        self,
        *,
        namespace: str,
        query: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with closing(self._connect()) as connection, connection:
            facts = connection.execute(
                """
                SELECT current.*
                FROM memory_events AS current
                WHERE current.namespace = ? AND current.kind = 'fact'
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_events AS newer
                      WHERE newer.supersedes_id = current.id
                  )
                ORDER BY current.id DESC
                """,
                (namespace,),
            ).fetchall()
            episodes = connection.execute(
                """
                SELECT *
                FROM memory_events
                WHERE namespace = ? AND kind = 'episode'
                ORDER BY id DESC
                LIMIT ?
                """,
                (namespace, max(limit * 32, 128)),
            ).fetchall()
        rows = [*facts, *episodes]
        tokens = set(re.findall(r"\w+", query.casefold()))

        def overlap(row: sqlite3.Row) -> int:
            text = " ".join(
                str(row[key])
                for key in ("subject", "predicate", "value")
            ).casefold()
            if row["kind"] == "fact":
                text += " " + FACT_SEARCH_ALIASES.get(str(row["predicate"]), "")
            searchable = set(re.findall(r"\w+", text))
            return sum(
                token in text
                or any(
                    len(candidate) >= 2 and candidate in token
                    for candidate in searchable
                )
                for token in tokens
            )

        ranked = sorted(
            rows,
            key=lambda row: (
                row["kind"] == "fact",
                overlap(row),
                int(row["id"]),
            ),
            reverse=True,
        )
        if tokens:
            ranked = [row for row in ranked if overlap(row) > 0]
        else:
            ranked = []
        return [dict(row) for row in ranked[:limit]]

    def build_capsule(
        self,
        namespace: str,
        *,
        char_limit: int = 3575,
        overlays: tuple[dict[str, Any], ...] = (),
    ) -> str:
        if char_limit <= 0:
            return ""
        facts = self.active_facts(namespace)
        if overlays:
            by_identity = {
                (str(fact["subject"]), str(fact["predicate"])): fact
                for fact in facts
            }
            for fact in overlays:
                by_identity[(str(fact["subject"]), str(fact["predicate"]))] = fact
            facts = list(by_identity.values())
        header = "[Persistent identity and user memory]"
        lines = [header]
        for fact in facts:
            line = (
                f"- {fact['subject']}.{fact['predicate']}: {fact['value']} "
                f"(source={fact['source_turn']})"
            )
            candidate = "\n".join([*lines, line])
            if len(candidate) > char_limit:
                break
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""

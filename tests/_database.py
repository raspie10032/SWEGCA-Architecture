"""Minimal SQLite fixture for standalone verified-memory tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def create_database(output: Path) -> Path:
    output.mkdir(parents=True)
    database = output / "memory.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE documents (
                docid TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                page_id TEXT NOT NULL,
                revision_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                redirect INTEGER NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                duplicate_status TEXT NOT NULL,
                source_url TEXT NOT NULL,
                license TEXT NOT NULL,
                attribution TEXT NOT NULL
            );
            CREATE INDEX documents_source_id ON documents(source_id);
            CREATE INDEX documents_content_sha256 ON documents(content_sha256);
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                title,
                text,
                content='documents',
                content_rowid='rowid',
                tokenize='unicode61'
            );
            CREATE TABLE duplicate_groups (
                content_sha256 TEXT PRIMARY KEY,
                occurrences INTEGER NOT NULL
            );
            """
        )
        connection.commit()
    finally:
        connection.close()
    return database

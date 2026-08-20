"""Minimal SQLite FTS query helper required by external memory."""

from __future__ import annotations

import unicodedata


def _fts_query(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    trigrams: list[str] = []
    seen: set[str] = set()
    for index in range(max(len(normalized) - 2, 0)):
        trigram = normalized[index : index + 3]
        if '"' in trigram or trigram.isspace() or trigram in seen:
            continue
        seen.add(trigram)
        trigrams.append(f'"{trigram}"')
    return " OR ".join(trigrams[:96])

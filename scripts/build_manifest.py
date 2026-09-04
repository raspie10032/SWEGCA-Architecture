"""Build a deterministic SHA-256 manifest for the standalone repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.json"
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != OUTPUT
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and not path.name.endswith((".pyc", ".pyo"))
    )


def main() -> None:
    entries = []
    for path in sorted(ROOT.rglob("*"), key=lambda item: item.as_posix()):
        if not included(path):
            continue
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "artifact": "swegca-public-release-file-manifest-v2",
        "algorithm": "SHA-256",
        "files": entries,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUTPUT} with {len(entries)} entries")


if __name__ == "__main__":
    main()

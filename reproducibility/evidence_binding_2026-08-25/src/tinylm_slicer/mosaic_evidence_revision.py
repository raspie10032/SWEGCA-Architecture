"""Bind an evidence decision to immutable report, ledger, and World artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REVISION_AUTHORITY = object()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class EvidenceRevisionContract:
    """Expected immutable identity for one accepted accumulator revision."""

    report_sha256: str
    ledger_sha256: str
    state_sha256: str
    accumulator_revision: int
    hypothesis_id: str

    def __post_init__(self) -> None:
        _validate_sha256(self.report_sha256, "report_sha256")
        _validate_sha256(self.ledger_sha256, "ledger_sha256")
        _validate_sha256(self.state_sha256, "state_sha256")
        if self.accumulator_revision <= 0:
            raise ValueError("accumulator_revision must be positive")
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must be nonempty")


@dataclass(frozen=True)
class EvidenceRevisionVerification:
    """Derived currentness signals consumed by the bounded World-write gate."""

    evidence_current: bool
    accumulator_revision_current: bool
    checks: dict[str, bool]
    observed_report_sha256: str
    observed_ledger_sha256: str
    observed_state_sha256: str
    observed_revision: int | None
    hypothesis_id: str
    evidence_addresses: tuple[str, ...]
    accepted_evidence_addresses: tuple[str, ...]
    decision_payload: tuple[tuple[str, object], ...]


def is_authoritative_evidence_revision(
    verification: EvidenceRevisionVerification,
) -> bool:
    return getattr(verification, "_authority_token", None) is _REVISION_AUTHORITY


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("evidence ledger must contain JSON objects")
    return rows


def verify_evidence_revision(
    *,
    report_path: Path,
    ledger_path: Path,
    state_path: Path,
    contract: EvidenceRevisionContract,
) -> EvidenceRevisionVerification:
    """Derive fail-closed evidence and revision currentness from pinned artifacts."""
    report_hash = sha256_file(report_path)
    ledger_hash = sha256_file(ledger_path)
    state_hash = sha256_file(state_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _load_ledger(ledger_path)
    accumulator_section = report.get("accumulator", {})
    accumulator = (
        accumulator_section.get("decision", accumulator_section)
        if isinstance(accumulator_section, dict)
        else {}
    )
    revisions = [row.get("revision") for row in rows]
    observed_revision = revisions[-1] if isinstance(revisions[-1], int) else None
    final = rows[-1]
    report_revision = accumulator.get("revision")
    expected_revisions = list(range(1, len(rows) + 1))
    world_hashes = {
        str(row.get(key, ""))
        for row in rows
        for key in ("world_hash_before", "world_hash_after")
    }
    report_world_hash = report.get("lineage", {}).get("state_sha256")
    metrics_match = (
        observed_revision is not None
        and report_revision == observed_revision
        and final.get("decision") == accumulator.get("status")
        and math.isclose(
            float(final.get("posterior_after", -1)),
            float(accumulator.get("posterior_mean", -2)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(final.get("causal_lower_bound", -1)),
            float(accumulator.get("causal_lower_bound", -2)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    checks = {
        "report_hash_matches": report_hash == contract.report_sha256,
        "ledger_hash_matches": ledger_hash == contract.ledger_sha256,
        "state_hash_matches": state_hash == contract.state_sha256,
        "report_passed_and_accepted": (
            report.get("passed") is True and accumulator.get("status") == "accept"
        ),
        "hypothesis_matches": all(
            row.get("hypothesis_id") == contract.hypothesis_id for row in rows
        ),
        "ledger_updates_applied": all(
            row.get("update_applied") is True and row.get("update_reason") == "applied"
            for row in rows
        ),
        "ledger_world_hashes_match_state": world_hashes == {contract.state_sha256},
        "report_world_hash_matches_state": report_world_hash == contract.state_sha256,
        "ledger_revisions_contiguous": revisions == expected_revisions,
        "revision_matches_contract": (
            observed_revision == contract.accumulator_revision
            and report_revision == contract.accumulator_revision
        ),
        "final_ledger_metrics_match_report": metrics_match,
    }
    evidence_current_keys = (
        "report_hash_matches",
        "ledger_hash_matches",
        "state_hash_matches",
        "report_passed_and_accepted",
        "hypothesis_matches",
        "ledger_updates_applied",
        "ledger_world_hashes_match_state",
        "report_world_hash_matches_state",
    )
    revision_current_keys = (
        "ledger_revisions_contiguous",
        "revision_matches_contract",
        "final_ledger_metrics_match_report",
    )
    decision_fields = (
        "status",
        "reason",
        "posterior_mean",
        "causal_lower_bound",
        "overall_upper_bound",
        "effective_sample_size",
        "source_diversity",
        "context_diversity",
        "regime_change_score",
        "revision",
    )
    accepted_evidence_addresses = tuple(
        sorted(
            {
                str(address)
                for row in rows
                for address in row.get("evidence_addresses", ())
                if isinstance(address, str) and address
            }
        )
    )
    verification = EvidenceRevisionVerification(
        evidence_current=all(checks[key] for key in evidence_current_keys),
        accumulator_revision_current=all(
            checks[key] for key in revision_current_keys
        ),
        checks=checks,
        observed_report_sha256=report_hash,
        observed_ledger_sha256=ledger_hash,
        observed_state_sha256=state_hash,
        observed_revision=observed_revision,
        hypothesis_id=contract.hypothesis_id,
        evidence_addresses=(
            f"sha256:{report_hash}",
            f"sha256:{ledger_hash}",
            f"sha256:{state_hash}",
        ),
        accepted_evidence_addresses=accepted_evidence_addresses,
        decision_payload=(
            tuple((key, accumulator.get(key)) for key in decision_fields)
            + (
                ("hypothesis_id", contract.hypothesis_id),
                ("evidence_addresses", accepted_evidence_addresses),
            )
        ),
    )
    object.__setattr__(verification, "_authority_token", _REVISION_AUTHORITY)
    return verification

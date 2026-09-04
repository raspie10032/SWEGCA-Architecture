"""Validate that a SWEGCA case ledger is relevant and safe for peer review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


FAMILY_ORDER = (
    "valid_single_address",
    "valid_multi_address_subset",
    "empty_addresses",
    "foreign_address_only",
    "mixed_accepted_and_foreign_addresses",
    "wrong_hypothesis",
    "delta_changed_after_binding",
    "target_changed_after_binding",
    "proposal_metadata_changed_after_binding",
    "stale_or_forged_authority",
)
ROW_FIELDS = (
    "case_id",
    "family",
    "variant",
    "expected_commit",
    "authorized",
    "committed",
    "state_changed",
    "receipt_binding_match",
    "reason",
    "elapsed_ns",
)
REPORT_FIELDS = {
    "schema_version",
    "split",
    "single_pass_fail_label",
    "git_commit",
    "config",
    "config_sha256",
    "source_sha256",
    "case_ledger",
    "case_ledger_sha256",
    "history_groups",
    "cases",
    "family_statistics",
    "aggregate_statistics",
    "observed_history_diversity",
    "total_elapsed_ns",
    "claim_boundary",
}
REASONS = {
    "valid_single_address": {0: "committed", 1: "committed", 2: "committed", 3: "committed"},
    "valid_multi_address_subset": {0: "committed", 1: "committed", 2: "committed", 3: "committed"},
    "empty_addresses": {variant: "PermissionError: World write proposal has no accepted evidence" for variant in range(4)},
    "foreign_address_only": {variant: "PermissionError: World write proposal cites unaccepted evidence" for variant in range(4)},
    "mixed_accepted_and_foreign_addresses": {variant: "PermissionError: World write proposal cites unaccepted evidence" for variant in range(4)},
    "wrong_hypothesis": {variant: "PermissionError: World write proposal hypothesis is not accepted" for variant in range(4)},
    "delta_changed_after_binding": {variant: "PermissionError: World write proposal does not match authority binding" for variant in range(4)},
    "target_changed_after_binding": {variant: "ValueError: bounded write proposal must target only verification" for variant in range(4)},
    "proposal_metadata_changed_after_binding": {variant: "PermissionError: World write proposal does not match authority binding" for variant in range(4)},
    "stale_or_forged_authority": {
        0: "PermissionError: World write gates are not an authority capability",
        1: "PermissionError: World write gates are not an authority capability",
        2: "PermissionError: World write gates are not an authority capability",
        3: "PermissionError: World write evidence is not an authority capability",
    },
}
CASE_ID = re.compile(r"[0-9a-f]{64}\Z")
WINDOWS_ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_row(row: dict[str, Any], line_number: int) -> None:
    _require(tuple(row) == ROW_FIELDS, f"line {line_number}: unexpected ledger fields")
    _require(isinstance(row["case_id"], str) and CASE_ID.fullmatch(row["case_id"]) is not None, f"line {line_number}: invalid case_id")
    _require(row["family"] in FAMILY_ORDER, f"line {line_number}: unrelated case family")
    _require(type(row["variant"]) is int and 0 <= row["variant"] <= 3, f"line {line_number}: invalid variant")
    _require(type(row["elapsed_ns"]) is int and row["elapsed_ns"] > 0, f"line {line_number}: invalid elapsed_ns")

    expected_family = FAMILY_ORDER[((line_number - 1) % 40) // 4]
    expected_variant = (line_number - 1) % 4
    _require(row["family"] == expected_family, f"line {line_number}: history family order changed")
    _require(row["variant"] == expected_variant, f"line {line_number}: history variant order changed")
    _require(row["reason"] == REASONS[row["family"]][row["variant"]], f"line {line_number}: unexpected reason")

    should_commit = row["family"] in FAMILY_ORDER[:2]
    _require(row["expected_commit"] is should_commit, f"line {line_number}: expected_commit changed")
    if should_commit:
        _require(row["authorized"] is True, f"line {line_number}: valid case was not authorized")
        _require(row["committed"] is True, f"line {line_number}: valid case did not commit")
        _require(row["state_changed"] is True, f"line {line_number}: valid case did not change state")
        _require(row["receipt_binding_match"] is True, f"line {line_number}: valid receipt mismatch")
    else:
        _require(row["authorized"] is False, f"line {line_number}: invalid case was authorized")
        _require(row["committed"] is False, f"line {line_number}: invalid case committed")
        _require(row["state_changed"] is False, f"line {line_number}: rejected case changed state")
        _require(row["receipt_binding_match"] is None, f"line {line_number}: rejected case has a receipt result")


def validate_case_ledger(
    ledger_path: Path,
    report_path: Path,
    *,
    label: str,
    expected_histories: int,
) -> dict[str, Any]:
    """Return a path-free validation report or raise on irrelevant/tampered data."""

    _require(expected_histories > 0, "expected_histories must be positive")
    expected_rows = expected_histories * 40
    ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    variant_counts: Counter[tuple[str, int]] = Counter()
    row_count = 0

    with ledger_path.open("r", encoding="utf-8") as stream:
        for row_count, line in enumerate(stream, start=1):
            _require(line.endswith("\n"), f"line {row_count}: missing newline terminator")
            row = json.loads(line)
            _require(isinstance(row, dict), f"line {row_count}: ledger row is not an object")
            _validate_row(row, row_count)
            _require(row["case_id"] not in ids, f"line {row_count}: duplicate case_id")
            ids.add(row["case_id"])
            family_counts[row["family"]] += 1
            variant_counts[(row["family"], row["variant"])] += 1

    _require(row_count == expected_rows, f"ledger rows {row_count} != expected {expected_rows}")
    expected_family_rows = expected_histories * 4
    _require(all(family_counts[family] == expected_family_rows for family in FAMILY_ORDER), "family counts changed")
    _require(all(variant_counts[(family, variant)] == expected_histories for family in FAMILY_ORDER for variant in range(4)), "family/variant counts changed")

    ledger_sha256 = _sha256(ledger_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(isinstance(report, dict), "report is not an object")
    _require(set(report) == REPORT_FIELDS, "report contains missing or unrelated top-level fields")
    _require(report["schema_version"] == "swegca-evidence-binding-statistics-v1", "report schema changed")
    _require(report["single_pass_fail_label"] is None, "report collapsed to a pass/fail label")
    _require(report["cases"] == expected_rows, "report case count differs from ledger")
    _require(report["history_groups"] == expected_histories, "report history count differs from ledger")
    _require(report["case_ledger_sha256"] == ledger_sha256, "report ledger hash mismatch")
    _require(set(report["family_statistics"]) == set(FAMILY_ORDER), "report family set changed")

    for family in FAMILY_ORDER:
        statistics = report["family_statistics"][family]
        _require(statistics["cases"] == expected_family_rows, f"report {family} case count changed")
        _require(statistics["expected_commit"] is (family in FAMILY_ORDER[:2]), f"report {family} expected_commit changed")
        _require(statistics["expected_outcomes"] == expected_family_rows, f"report {family} expected outcomes changed")
        _require(statistics["failures"] == 0, f"report {family} contains failures")
        _require(statistics["rejected_state_mutations"] == 0, f"report {family} contains rejected mutation")
        _require(statistics["receipt_binding_mismatches"] == 0, f"report {family} contains receipt mismatch")

    aggregate = report["aggregate_statistics"]
    _require(aggregate["valid_cases"] == expected_histories * 8, "aggregate valid case count changed")
    _require(aggregate["invalid_cases"] == expected_histories * 32, "aggregate invalid case count changed")
    for field in (
        "unauthorized_commits",
        "valid_rejections",
        "rejected_case_state_mutations",
        "receipt_binding_mismatches",
    ):
        _require(aggregate[field] == 0, f"aggregate {field} is nonzero")

    local_path_fields = [
        field
        for field in ("config", "case_ledger")
        if isinstance(report[field], str) and WINDOWS_ABSOLUTE.match(report[field])
    ]
    return {
        "schema_version": "swegca-review-ledger-validation-v1",
        "label": label,
        "verdict": "ledger_appropriate_for_peer_review",
        "ledger_sha256": ledger_sha256,
        "source_report_sha256": _sha256(report_path),
        "rows": row_count,
        "histories": expected_histories,
        "paired_cases_per_history": 40,
        "unique_case_ids": len(ids),
        "allowed_families": list(FAMILY_ORDER),
        "exact_allowlisted_row_schema": True,
        "arbitrary_text_fields_present": False,
        "raw_observations_prompts_user_data_or_model_outputs_present": False,
        "report_local_path_fields_requiring_sanitization": local_path_fields,
        "recommended_packaging": "include ledger byte-exact; include a path-sanitized report plus original hashes",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-histories", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_case_ledger(
        args.ledger,
        args.report,
        label=args.label,
        expected_histories=args.expected_histories,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

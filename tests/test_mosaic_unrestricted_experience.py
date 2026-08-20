from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

import pytest

from swegca.mosaic_unrestricted_experience import (
    ExperienceCandidateJudgment,
    build_hot_experience_index,
    discover_experience_universe,
    record_rozephine_selection,
    select_experience_for_cognition,
)


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE experiences (experience_id TEXT, status TEXT, payload TEXT)"
        )
        connection.executemany(
            "INSERT INTO experiences VALUES (?, ?, ?)",
            (
                ("one", "promoted", "success"),
                ("two", "candidate", "not yet verified"),
                ("three", "rejected", "failed experience remains useful"),
            ),
        )


def test_universe_exposes_every_file_and_every_experience_status(tmp_path: Path) -> None:
    memories = tmp_path / "memories"
    knowledge = tmp_path / "knowledge"
    memories.mkdir()
    knowledge.mkdir()
    (memories / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"status": status, "value": value}) + "\n"
            for status, value in (
                ("verified", "known"),
                ("unverified", "possible"),
                ("failed", "counterexample"),
            )
        ),
        encoding="utf-8",
    )
    (knowledge / "raw.bin").write_bytes(b"\x00\x01unclassified")
    _database(memories / "autonomy.sqlite3")

    universe = discover_experience_universe(
        {"episodic": memories, "knowledge": knowledge}
    )

    assert len(universe.artifacts) == 3
    assert universe.status_filter_applied is False
    assert universe.verification_filter_applied is False
    assert universe.file_type_filter_applied is False
    jsonl = next(row for row in universe.artifacts if row.relative_path.endswith(".jsonl"))
    assert [row["status"] for row in universe.json_lines(jsonl)] == [
        "verified",
        "unverified",
        "failed",
    ]
    database = next(
        row for row in universe.artifacts if row.relative_path.endswith(".sqlite3")
    )
    assert universe.sqlite_tables(database) == ("experiences",)
    assert [row["status"] for row in universe.sqlite_rows(database, "experiences")] == [
        "promoted",
        "candidate",
        "rejected",
    ]


def test_rozephine_selection_records_provenance_without_granting_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experience"
    root.mkdir()
    selected = root / "failed-but-useful.json"
    selected.write_text('{"status":"failed","lesson":"reverse the method"}\n', encoding="utf-8")
    (root / "unused.txt").write_text("still accessible", encoding="utf-8")
    universe = discover_experience_universe({"all": root})
    artifact = next(row for row in universe.artifacts if row.relative_path == selected.name)

    receipt = record_rozephine_selection(
        universe,
        (artifact.address,),
        method="rozephine_self_selected_counterexample_reuse",
        rationale="a failed experience reveals which assumption to reverse",
    )

    assert receipt.universe_artifact_count == 2
    assert receipt.rozephine_selected is True
    assert receipt.codex_per_item_approval_used is False
    assert receipt.selected_artifacts[0]["address"] == artifact.address
    assert receipt.external_action_authorized is False
    assert receipt.memory_write_authorized is False
    assert receipt.world_write_authorized is False
    assert receipt.training_write_authorized is False
    assert receipt.p3_promotion_authorized is False


def test_snapshot_rejects_artifact_mutation(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    root.mkdir()
    path = root / "episode.json"
    path.write_text("{}\n", encoding="utf-8")
    universe = discover_experience_universe({"all": root})
    artifact = universe.artifacts[0]
    path.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="changed after discovery"):
        universe.read_bytes(artifact)


def test_selection_rejects_unknown_and_duplicate_addresses(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    root.mkdir()
    (root / "episode.json").write_text("{}\n", encoding="utf-8")
    universe = discover_experience_universe({"all": root})
    address = universe.artifacts[0].address

    with pytest.raises(KeyError):
        record_rozephine_selection(
            universe,
            ("experience-artifact:all:missing.json",),
            method="self_selected",
            rationale="relevant",
        )
    with pytest.raises(ValueError, match="must be unique"):
        record_rozephine_selection(
            universe,
            (address, address),
            method="self_selected",
            rationale="relevant",
        )


def test_hot_index_exposes_every_artifact_without_io_on_lookup(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    root.mkdir()
    for index, status in enumerate(("promoted", "candidate", "rejected")):
        (root / f"episode-{index}.json").write_text(
            json.dumps({"status": status}) + "\n", encoding="utf-8"
        )
    universe = discover_experience_universe({"all": root})
    rejected = next(
        row for row in universe.artifacts if row.relative_path == "episode-2.json"
    )
    hot = build_hot_experience_index(
        universe,
        semantic_keys_by_address={
            rejected.address: ("failed_experience", "useful_counterexample")
        },
    )

    assert hot.artifact_count == len(universe.artifacts)
    assert hot.requires_io_on_lookup is False
    assert hot.lookup_address(rejected.address) is rejected
    assert hot.lookup_semantic_key("failed_experience") == (rejected.address,)
    assert hot.lookup_semantic_key("episode-2") == (rejected.address,)
    assert hot.lookup_semantic_key("episode") == tuple(
        row.address for row in universe.artifacts
    )


def test_hot_lookup_is_measured_in_nanoseconds_and_does_not_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "experience"
    root.mkdir()
    (root / "episode.json").write_text("{}\n", encoding="utf-8")
    universe = discover_experience_universe({"all": root})
    artifact = universe.artifacts[0]
    hot = build_hot_experience_index(universe)
    monkeypatch.setattr(
        type(universe),
        "path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hot lookup attempted filesystem access")
        ),
    )

    elapsed = []
    for _ in range(2_000):
        found, elapsed_ns = hot.timed_address_lookup(artifact.address)
        assert found is artifact
        elapsed.append(elapsed_ns)

    assert hot.timing_unit == "nanoseconds"
    assert min(elapsed) >= 0
    assert statistics.median(elapsed) < 50_000


def test_runtime_cognition_selects_and_rejects_from_unfiltered_candidates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experience"
    root.mkdir()
    for status in ("verified", "failed", "rejected", "candidate", "unverified"):
        (root / f"actor-identity-{status}.json").write_text(
            json.dumps({"status": status}) + "\n", encoding="utf-8"
        )
    universe = discover_experience_universe({"history": root})
    hot = build_hot_experience_index(universe)
    observed: list[str] = []

    def judge(artifact, context):
        status = artifact.relative_path.removesuffix(".json").rsplit("-", 1)[-1]
        observed.append(status)
        selected = status in {"verified", "failed", "unverified"}
        return ExperienceCandidateJudgment(
            address=artifact.address,
            selected=selected,
            relevance=0.9 if selected else 0.4,
            contradiction=0.1 if selected else 0.95,
            verification_state=status,
            revision=str(context["revision"]),
            rationale=(
                "use this experience including failure history"
                if selected
                else "reject because independent contradiction remains unresolved"
            ),
            rejection_evidence=("counterexample:opposite-direction",) if not selected else (),
        )

    receipt = select_experience_for_cognition(
        universe,
        hot,
        query="actor identity",
        context={"scene": "sanabi", "revision": "episode-before-outcome"},
        judge=judge,
    )

    assert set(observed) == {
        "verified",
        "failed",
        "rejected",
        "candidate",
        "unverified",
    }
    assert {row["verification_state"] for row in receipt.selected_artifacts} == {
        "verified",
        "failed",
        "unverified",
    }
    rejected = [row for row in receipt.candidate_judgments if not row.selected]
    assert {row.verification_state for row in rejected} == {"candidate", "rejected"}
    assert all(
        row.rejection_evidence == ("counterexample:opposite-direction",)
        for row in rejected
    )
    assert receipt.context == {
        "scene": "sanabi",
        "revision": "episode-before-outcome",
    }
    assert receipt.rozephine_selected is True
    assert receipt.codex_per_item_approval_used is False
    assert receipt.world_write_authorized is False
    assert receipt.memory_write_authorized is False
    assert receipt.external_action_authorized is False
    assert receipt.p3_promotion_authorized is False


def test_runtime_cognition_selection_rejects_wrong_judgment_address(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experience"
    root.mkdir()
    (root / "episode.json").write_text("{}\n", encoding="utf-8")
    universe = discover_experience_universe({"history": root})
    hot = build_hot_experience_index(universe)

    with pytest.raises(ValueError, match="judgment address"):
        select_experience_for_cognition(
            universe,
            hot,
            query="episode",
            context={},
            judge=lambda _artifact, _context: ExperienceCandidateJudgment(
                address="experience-artifact:history:other.json",
                selected=True,
                relevance=1.0,
                contradiction=0.0,
                verification_state="candidate",
                revision="test",
                rationale="wrong address",
            ),
        )

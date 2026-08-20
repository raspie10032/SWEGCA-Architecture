"""Uncertainty-gated promotion between episodic and semantic memory stores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from swegca.mosaic_evidence_accumulator import (
    AccumulatorDecision,
    is_authoritative_accumulator_decision,
    require_authoritative_accumulator_decision,
)
from swegca.mosaic_external_memory import (
    ExternalMemoryDocument,
    MosaicExternalMemory,
)

if TYPE_CHECKING:
    from swegca.mosaic_versioned_memory import (
        MemoryMutation,
        VersionedExternalMemory,
    )


class MemoryTier(StrEnum):
    NONE = "none"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    QUARANTINED = "quarantined"
    RETRACTED = "retracted"


@dataclass(frozen=True)
class MemoryCandidate:
    hypothesis_id: str
    key: str
    value: str
    evidence_refs: tuple[str, ...]
    source_id: str
    source_revision: str
    timestamp: str
    license: str
    attribution: str
    retrieval_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = (
            self.hypothesis_id,
            self.key,
            self.value,
            self.source_id,
            self.source_revision,
            self.timestamp,
            self.license,
            self.attribution,
        )
        if any(not value.strip() for value in required):
            raise ValueError("memory candidate fields must be nonempty")
        if not self.evidence_refs or any(not value.strip() for value in self.evidence_refs):
            raise ValueError("memory candidate requires provenance references")
        if any(not alias.strip() for alias in self.retrieval_aliases):
            raise ValueError("memory retrieval aliases cannot be empty")
        object.__setattr__(
            self,
            "retrieval_aliases",
            tuple(dict.fromkeys(self.retrieval_aliases)),
        )


@dataclass(frozen=True)
class MemoryPromotionDecision:
    previous_tier: MemoryTier
    next_tier: MemoryTier
    action: str
    reason: str
    semantic_read_allowed: bool


_PROMOTION_AUTHORITY = object()


def _promotion_decision(
    evidence: AccumulatorDecision,
    previous_tier: MemoryTier,
    next_tier: MemoryTier,
    action: str,
    reason: str,
    semantic_read_allowed: bool,
) -> MemoryPromotionDecision:
    decision = MemoryPromotionDecision(
        previous_tier,
        next_tier,
        action,
        reason,
        semantic_read_allowed,
    )
    if is_authoritative_accumulator_decision(evidence):
        object.__setattr__(decision, "_authority_token", _PROMOTION_AUTHORITY)
    return decision


def _require_authoritative_promotion(decision: MemoryPromotionDecision) -> None:
    if getattr(decision, "_authority_token", None) is not _PROMOTION_AUTHORITY:
        raise PermissionError("memory promotion is not an authority capability")


def decide_memory_promotion(
    current_tier: MemoryTier,
    evidence: AccumulatorDecision,
    *,
    counterfactual_verified: bool,
    provenance_complete: bool,
) -> MemoryPromotionDecision:
    if not provenance_complete:
        return _promotion_decision(
            evidence,
            current_tier,
            MemoryTier.QUARANTINED,
            "quarantine",
            "incomplete_provenance",
            False,
        )
    if evidence.status == "reject":
        return _promotion_decision(
            evidence,
            current_tier,
            MemoryTier.RETRACTED,
            "retract",
            evidence.reason,
            False,
        )
    if evidence.status == "accept" and counterfactual_verified:
        action = "refresh_semantic" if current_tier == MemoryTier.SEMANTIC else "promote"
        return _promotion_decision(
            evidence,
            current_tier,
            MemoryTier.SEMANTIC,
            action,
            evidence.reason,
            True,
        )
    if current_tier == MemoryTier.SEMANTIC and (
        evidence.reason == "regime_change_suspected" or not counterfactual_verified
    ):
        return _promotion_decision(
            evidence,
            current_tier,
            MemoryTier.QUARANTINED,
            "quarantine",
            evidence.reason,
            False,
        )
    return _promotion_decision(
        evidence,
        current_tier,
        MemoryTier.EPISODIC,
        "record_episode",
        evidence.reason,
        False,
    )


def memory_candidate_document(
    candidate: MemoryCandidate,
    tier: MemoryTier,
    *,
    docid: str | None = None,
) -> ExternalMemoryDocument:
    alias_text = ""
    if candidate.retrieval_aliases:
        alias_text = " aliases=" + json.dumps(
            candidate.retrieval_aliases,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    text = f"FACT key={candidate.key} value={candidate.value}{alias_text}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ExternalMemoryDocument(
        docid=docid or f"{tier.value}:{candidate.hypothesis_id}",
        source_id=candidate.source_id,
        source_revision=candidate.source_revision,
        page_id=candidate.key,
        revision_id=candidate.source_revision,
        namespace=f"{tier.value}_memory",
        timestamp=candidate.timestamp,
        redirect=False,
        title=f"{tier.value.title()} memory {candidate.key}",
        text=text,
        content_sha256=digest,
        duplicate_status="unique",
        source_url="evidence://" + ",".join(candidate.evidence_refs),
        license=candidate.license,
        attribution=candidate.attribution,
    )


def apply_memory_promotion(
    episodic_memory: MosaicExternalMemory,
    semantic_memory: MosaicExternalMemory,
    candidate: MemoryCandidate,
    decision: MemoryPromotionDecision,
) -> dict[str, object]:
    _require_authoritative_promotion(decision)
    episodic_docid = f"{MemoryTier.EPISODIC.value}:{candidate.hypothesis_id}"
    semantic_docid = f"{MemoryTier.SEMANTIC.value}:{candidate.hypothesis_id}"
    if decision.next_tier == MemoryTier.EPISODIC:
        episodic_memory.upsert(memory_candidate_document(candidate, MemoryTier.EPISODIC))
        semantic_memory.delete(semantic_docid)
    elif decision.next_tier == MemoryTier.SEMANTIC:
        for document in semantic_memory.documents(namespace="semantic_memory"):
            if (
                document["page_id"] == candidate.key
                and document["docid"] != semantic_docid
            ):
                semantic_memory.delete(str(document["docid"]))
        semantic_memory.upsert(memory_candidate_document(candidate, MemoryTier.SEMANTIC))
        episodic_memory.delete(episodic_docid)
    else:
        episodic_memory.delete(episodic_docid)
        semantic_memory.delete(semantic_docid)
    return {
        "action": decision.action,
        "next_tier": decision.next_tier.value,
        "episodic_present": any(
            row["docid"] == episodic_docid for row in episodic_memory.documents()
        ),
        "semantic_present": any(
            row["docid"] == semantic_docid for row in semantic_memory.documents()
        ),
        "semantic_read_allowed": decision.semantic_read_allowed,
    }


def apply_verified_memory_update(
    memory: VersionedExternalMemory,
    candidate: MemoryCandidate,
    evidence: AccumulatorDecision,
    promotion: MemoryPromotionDecision,
    *,
    valid_from: str,
    update_id: str,
    supersedes_docid: str | None = None,
    valid_until: str | None = None,
) -> MemoryMutation:
    """Commit an accepted semantic candidate without destroying prior versions."""
    require_authoritative_accumulator_decision(evidence)
    _require_authoritative_promotion(promotion)
    if promotion.next_tier != MemoryTier.SEMANTIC or not promotion.semantic_read_allowed:
        raise PermissionError("candidate is not authorized for semantic memory")
    docid = f"semantic:{candidate.hypothesis_id}:{candidate.source_revision}"
    return memory.upsert_verified(
        memory_candidate_document(candidate, MemoryTier.SEMANTIC, docid=docid),
        decision=evidence,
        valid_from=valid_from,
        update_id=update_id,
        supersedes_docid=supersedes_docid,
        valid_until=valid_until,
    )

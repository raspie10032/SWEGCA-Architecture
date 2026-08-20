"""Link external memory mutations to bounded one-state World-write receipts."""

from __future__ import annotations

from dataclasses import dataclass

from swegca.mosaic_bounded_world_write import (
    BoundedWorldWriteReceipt,
    cognitive_state_hash,
    retract_bounded_verification_write,
    rollback_bounded_verification_write,
    validate_bounded_verification_retraction,
)
from swegca.mosaic_cognitive_kernel import CognitiveState
from swegca.mosaic_evidence_accumulator import AccumulatorDecision
from swegca.mosaic_external_memory import MosaicExternalMemory
from swegca.mosaic_memory_promotion import (
    MemoryCandidate,
    MemoryTier,
    apply_verified_memory_update,
    decide_memory_promotion,
    memory_candidate_document,
)
from swegca.mosaic_versioned_memory import VersionedExternalMemory


@dataclass(frozen=True)
class WorldMemoryTransactionConfig:
    minimum_causal_lower_bound: float = 0.55
    minimum_source_diversity: int = 2
    minimum_context_diversity: int = 4

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_causal_lower_bound <= 1:
            raise ValueError("memory causal lower bound must be within [0, 1]")
        if min(self.minimum_source_diversity, self.minimum_context_diversity) <= 0:
            raise ValueError("memory diversity minima must be positive")


@dataclass(frozen=True)
class MemoryTransactionReceipt:
    transaction_id: str
    tier: MemoryTier
    docid: str
    world_write_receipt_id: str | None
    update_id: str | None


def _ensure_journal(memory: VersionedExternalMemory) -> None:
    with memory._connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS world_memory_transaction_journal (
                transaction_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                tier TEXT NOT NULL,
                docid TEXT NOT NULL,
                update_id TEXT,
                world_write_receipt_id TEXT,
                world_before_hash TEXT NOT NULL,
                world_after_hash TEXT NOT NULL
            )
            """
        )


def _prepare_journal(
    memory: VersionedExternalMemory,
    receipt: MemoryTransactionReceipt,
    world_receipt: BoundedWorldWriteReceipt,
) -> None:
    _ensure_journal(memory)
    with memory._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        incomplete = connection.execute(
            "SELECT transaction_id, stage FROM world_memory_transaction_journal "
            "WHERE stage NOT IN ('completed', 'rolled_back') "
            "ORDER BY transaction_id LIMIT 1"
        ).fetchone()
        if incomplete is not None:
            raise RuntimeError(
                "Incomplete World-memory transaction requires recovery: "
                f"{incomplete[0]} ({incomplete[1]})"
            )
        existing = connection.execute(
            "SELECT stage FROM world_memory_transaction_journal "
            "WHERE transaction_id=?",
            (receipt.transaction_id,),
        ).fetchone()
        if existing is not None:
            raise ValueError(
                f"World-memory transaction is already {existing[0]}"
            )
        connection.execute(
            """
            INSERT INTO world_memory_transaction_journal (
                transaction_id, stage, tier, docid, update_id,
                world_write_receipt_id, world_before_hash, world_after_hash
            ) VALUES (?, 'prepared', ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.transaction_id,
                receipt.tier.value,
                receipt.docid,
                receipt.update_id,
                receipt.world_write_receipt_id,
                world_receipt.before_state_hash,
                world_receipt.after_state_hash,
            ),
        )
        connection.commit()


def _set_journal_stage(
    memory: VersionedExternalMemory,
    transaction_id: str,
    expected: str,
    stage: str,
) -> None:
    with memory._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            "UPDATE world_memory_transaction_journal SET stage=? "
            "WHERE transaction_id=? AND stage=?",
            (stage, transaction_id, expected),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            raise ValueError("World-memory journal stage changed concurrently")
        connection.commit()


def recover_incomplete_world_memory_transactions(
    memory: VersionedExternalMemory,
    decision: AccumulatorDecision,
    *,
    rolled_back_at: str,
) -> tuple[str, ...]:
    """Rollback memory commits whose linked CognitiveState handoff did not finish."""
    _ensure_journal(memory)
    with memory._connect() as connection:
        rows = connection.execute(
            "SELECT transaction_id, stage, update_id "
            "FROM world_memory_transaction_journal "
            "WHERE stage NOT IN ('completed', 'rolled_back') "
            "ORDER BY transaction_id"
        ).fetchall()
    recovered: list[str] = []
    for transaction_id, stage, update_id in rows:
        if update_id is not None:
            try:
                memory.rollback_verified(
                    str(update_id),
                    decision=decision,
                    rolled_back_at=rolled_back_at,
                )
            except (KeyError, ValueError):
                if stage != "prepared":
                    raise
        with memory._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE world_memory_transaction_journal SET stage='rolled_back' "
                "WHERE transaction_id=? AND stage=?",
                (transaction_id, stage),
            )
            connection.commit()
        recovered.append(str(transaction_id))
    return tuple(recovered)


def world_memory_transaction_stage(
    memory: VersionedExternalMemory,
    transaction_id: str,
) -> str | None:
    _ensure_journal(memory)
    with memory._connect() as connection:
        row = connection.execute(
            "SELECT stage FROM world_memory_transaction_journal "
            "WHERE transaction_id=?",
            (transaction_id,),
        ).fetchone()
    return None if row is None else str(row[0])


def record_uncertain_episode(
    episodic_memory: MosaicExternalMemory,
    candidate: MemoryCandidate,
    decision: AccumulatorDecision,
    *,
    provenance_complete: bool,
    transaction_id: str,
) -> MemoryTransactionReceipt:
    """Record uncertainty externally without granting semantic read authority."""
    promotion = decide_memory_promotion(
        MemoryTier.NONE,
        decision,
        counterfactual_verified=False,
        provenance_complete=provenance_complete,
    )
    if promotion.next_tier != MemoryTier.EPISODIC:
        raise PermissionError("uncertain candidate is not eligible for episodic memory")
    if not transaction_id.strip():
        raise ValueError("memory transaction ID must be nonempty")
    document = memory_candidate_document(candidate, MemoryTier.EPISODIC)
    episodic_memory.upsert(document)
    return MemoryTransactionReceipt(
        transaction_id, MemoryTier.EPISODIC, document.docid, None, None
    )


def promote_world_linked_semantic(
    state: CognitiveState,
    world_receipt: BoundedWorldWriteReceipt,
    semantic_memory: VersionedExternalMemory,
    candidate: MemoryCandidate,
    decision: AccumulatorDecision,
    config: WorldMemoryTransactionConfig,
    *,
    valid_from: str,
    update_id: str,
) -> MemoryTransactionReceipt:
    """Promote only memory that is bound to the currently committed World receipt."""
    if cognitive_state_hash(state) != world_receipt.after_state_hash:
        raise ValueError("World state no longer matches the memory transaction receipt")
    if decision.status != "accept":
        raise PermissionError("semantic promotion requires accepted evidence")
    if decision.causal_lower_bound < config.minimum_causal_lower_bound:
        raise PermissionError("semantic promotion causal lower bound is insufficient")
    if decision.source_diversity < config.minimum_source_diversity:
        raise PermissionError("semantic promotion source diversity is insufficient")
    if decision.context_diversity < config.minimum_context_diversity:
        raise PermissionError("semantic promotion context diversity is insufficient")
    required_refs = {
        *world_receipt.evidence_refs,
        f"world-write:{world_receipt.receipt_id}",
    }
    if not required_refs.issubset(candidate.evidence_refs):
        raise PermissionError("memory candidate is not bound to the World receipt")
    promotion = decide_memory_promotion(
        MemoryTier.EPISODIC,
        decision,
        counterfactual_verified=True,
        provenance_complete=True,
    )
    docid = f"semantic:{candidate.hypothesis_id}:{candidate.source_revision}"
    receipt = MemoryTransactionReceipt(
        transaction_id=update_id,
        tier=MemoryTier.SEMANTIC,
        docid=docid,
        world_write_receipt_id=world_receipt.receipt_id,
        update_id=update_id,
    )
    _prepare_journal(semantic_memory, receipt, world_receipt)
    mutation = apply_verified_memory_update(
        semantic_memory,
        candidate,
        decision,
        promotion,
        valid_from=valid_from,
        update_id=update_id,
    )
    if mutation.docid != receipt.docid or mutation.update_id != receipt.update_id:
        raise ValueError("memory mutation differs from prepared journal")
    _set_journal_stage(semantic_memory, update_id, "prepared", "memory_committed")
    _set_journal_stage(
        semantic_memory,
        update_id,
        "memory_committed",
        "state_committed",
    )
    _set_journal_stage(semantic_memory, update_id, "state_committed", "completed")
    return receipt


def rollback_world_linked_semantic(
    state: CognitiveState,
    world_receipt: BoundedWorldWriteReceipt,
    semantic_memory: VersionedExternalMemory,
    memory_receipt: MemoryTransactionReceipt,
    decision: AccumulatorDecision,
    *,
    rolled_back_at: str,
) -> CognitiveState:
    """Rollback memory first and then restore the exact prior CognitiveState."""
    if cognitive_state_hash(state) != world_receipt.after_state_hash:
        raise ValueError("World state changed after linked memory mutation")
    if (
        memory_receipt.tier != MemoryTier.SEMANTIC
        or memory_receipt.world_write_receipt_id != world_receipt.receipt_id
        or memory_receipt.update_id is None
    ):
        raise ValueError("memory receipt does not match the World transaction")
    restored = rollback_bounded_verification_write(state, world_receipt)
    _set_journal_stage(
        semantic_memory,
        memory_receipt.transaction_id,
        "completed",
        "rollback_pending",
    )
    if not semantic_memory.rollback_verified(
        memory_receipt.update_id,
        decision=decision,
        rolled_back_at=rolled_back_at,
    ):
        raise ValueError("memory transaction was already rolled back")
    _set_journal_stage(
        semantic_memory,
        memory_receipt.transaction_id,
        "rollback_pending",
        "rolled_back",
    )
    return restored


def rollback_uncertain_episode(
    episodic_memory: MosaicExternalMemory,
    receipt: MemoryTransactionReceipt,
) -> bool:
    if receipt.tier != MemoryTier.EPISODIC or receipt.world_write_receipt_id is not None:
        raise ValueError("receipt is not an independent episodic transaction")
    return episodic_memory.delete(receipt.docid)


def retract_world_linked_semantic(
    state: CognitiveState,
    world_receipt: BoundedWorldWriteReceipt,
    semantic_memory: VersionedExternalMemory,
    memory_receipt: MemoryTransactionReceipt,
    decision: AccumulatorDecision,
    *,
    rolled_back_at: str,
) -> CognitiveState:
    """Retract a live semantic fact without erasing unrelated later cognition."""
    if (
        memory_receipt.tier != MemoryTier.SEMANTIC
        or memory_receipt.world_write_receipt_id != world_receipt.receipt_id
        or memory_receipt.update_id is None
    ):
        raise ValueError("memory receipt does not match the World transaction")
    validate_bounded_verification_retraction(state, world_receipt)
    restored = retract_bounded_verification_write(state, world_receipt)
    _set_journal_stage(
        semantic_memory,
        memory_receipt.transaction_id,
        "completed",
        "rollback_pending",
    )
    if not semantic_memory.rollback_verified(
        memory_receipt.update_id,
        decision=decision,
        rolled_back_at=rolled_back_at,
    ):
        raise ValueError("memory transaction was already rolled back")
    _set_journal_stage(
        semantic_memory,
        memory_receipt.transaction_id,
        "rollback_pending",
        "rolled_back",
    )
    return restored

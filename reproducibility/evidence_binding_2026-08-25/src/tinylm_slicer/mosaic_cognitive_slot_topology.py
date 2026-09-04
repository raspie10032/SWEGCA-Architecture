"""Role-aware capacity audit for the 32-slot CognitiveState compatibility profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from tinylm_slicer.mosaic_cognitive_kernel import CognitiveState
from tinylm_slicer.mosaic_omni import SLOT_ROLES


@dataclass(frozen=True)
class CognitiveSlotTopology:
    roles: tuple[str, ...]
    pools: Mapping[str, tuple[int, ...]]
    fixed_roles: Mapping[str, int]
    partition_sizes: tuple[int, int, int]

    @property
    def total_slots(self) -> int:
        return len(self.roles)


@dataclass(frozen=True)
class SlotCapacityAudit:
    topology: CognitiveSlotTopology
    requested_capacity: Mapping[str, int]
    overflow: Mapping[str, int]
    lease_metadata_present: bool
    rotation_metadata_present: bool
    archive_receipts_present: bool
    string_addressable_extensions: bool
    g5_ready: bool
    missing_definitions: tuple[str, ...]


_POOL_PREFIXES = (
    "object",
    "relation",
    "action",
    "camera",
    "lighting",
    "environment",
    "audio_event",
)
_FIXED = ("narrative", "constraints", "verification", "global")


def cognitive_slot_topology(state: CognitiveState) -> CognitiveSlotTopology:
    partition_sizes = (
        state.semantic_slots.shape[1],
        state.executive_slots.shape[1],
        state.scratch_slots.shape[1],
    )
    if sum(partition_sizes) != len(SLOT_ROLES):
        raise ValueError(
            "compatibility slot topology requires exactly the registered 32 roles"
        )
    pools = {
        prefix: tuple(
            index
            for index, role in enumerate(SLOT_ROLES)
            if role.startswith(f"{prefix}_")
        )
        for prefix in _POOL_PREFIXES
    }
    fixed = {role: SLOT_ROLES.index(role) for role in _FIXED}
    return CognitiveSlotTopology(
        roles=tuple(SLOT_ROLES),
        pools=pools,
        fixed_roles=fixed,
        partition_sizes=partition_sizes,
    )


def audit_slot_capacity(
    state: CognitiveState,
    requested_capacity: Mapping[str, int],
) -> SlotCapacityAudit:
    topology = cognitive_slot_topology(state)
    unknown = set(requested_capacity) - set(topology.pools)
    if unknown:
        raise ValueError(f"unknown slot pools: {sorted(unknown)}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in requested_capacity.values()
    ):
        raise ValueError("requested slot capacity must use nonnegative integers")
    overflow = {
        pool: max(0, requested_capacity.get(pool, 0) - len(indices))
        for pool, indices in topology.pools.items()
    }
    metadata = state.self_state.get("cognitive_slot_manager")
    lease_metadata_present = isinstance(metadata, dict) and isinstance(
        metadata.get("leases"), list
    )
    rotation_metadata_present = isinstance(metadata, dict) and all(
        key in metadata for key in ("revision", "last_rotation_step", "policy_version")
    )
    archive_receipts_present = (
        lease_metadata_present
        and all(
            isinstance(lease, dict) and "archive_ref" in lease
            for lease in metadata["leases"]
        )
    )
    string_addressable_extensions = len(topology.roles) > len(SLOT_ROLES)
    missing = []
    if not lease_metadata_present:
        missing.append("slot_lease_identity_age_priority")
    if not rotation_metadata_present:
        missing.append("role_local_rotation_policy")
    if not archive_receipts_present:
        missing.append("eviction_archive_and_recovery_receipts")
    if any(overflow.values()):
        missing.append("overflow_handling_for_requested_workload")
    if not string_addressable_extensions:
        missing.append("checkpoint_and_role_compatible_extension_profile")
    return SlotCapacityAudit(
        topology=topology,
        requested_capacity=dict(requested_capacity),
        overflow=overflow,
        lease_metadata_present=lease_metadata_present,
        rotation_metadata_present=rotation_metadata_present,
        archive_receipts_present=archive_receipts_present,
        string_addressable_extensions=string_addressable_extensions,
        g5_ready=not missing,
        missing_definitions=tuple(missing),
    )

import json

import pytest
import torch

from swegca.mosaic_cognitive_kernel import (
    BYTES_PER_GIB,
    CognitiveEvent,
    CognitiveKernelConfig,
    CognitiveState,
    EventSource,
    EvidenceClaim,
    EvidenceKind,
    PhysicalEvidence,
    PhysicalQuery,
    RecurrentCoreSpec,
    RepresentationRegistryEntry,
    StructuredWorldGraph,
    VramEstimateConfig,
    WorldBundle,
    WorldEntity,
    WorldRelation,
    estimate_inference_vram,
    estimate_recurrent_core_parameters,
)


def _graph() -> StructuredWorldGraph:
    return StructuredWorldGraph(
        entities=(
            WorldEntity(
                entity_id="actor_01",
                entity_type="character",
                properties={"hair": "blonde_twintail"},
                evidence_refs=("mesh://actor_01.glb",),
            ),
            WorldEntity(
                entity_id="umbrella_01",
                entity_type="prop",
                properties={"state": "open"},
            ),
        ),
        relations=(
            WorldRelation(
                subject="actor_01",
                predicate="holds",
                object="umbrella_01",
            ),
        ),
    )


def test_cognitive_state_is_one_state_and_round_trips() -> None:
    config = CognitiveKernelConfig(
        semantic_slots=4,
        executive_slots=2,
        scratch_slots=2,
        hidden_dim=8,
    )
    state = CognitiveState(
        semantic_slots=torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
        executive_slots=torch.zeros(1, 2, 8),
        scratch_slots=torch.ones(1, 2, 8),
        structured_world_graph=_graph(),
        evidence_refs=("image://input/001.png",),
        goal_state={"active_goal": "track_actor"},
        value_state={"consistency": 0.9},
        self_state={"identity": "rozephine"},
    )

    state.validate(config)
    restored = CognitiveState.from_dict(json.loads(json.dumps(state.to_dict())))
    restored.validate(config)

    assert state.persistent_state_count == 1
    assert restored.owner_id == state.owner_id
    assert torch.equal(restored.semantic_slots, state.semantic_slots)
    assert torch.equal(restored.executive_slots, state.executive_slots)
    assert torch.equal(restored.scratch_slots, state.scratch_slots)
    assert restored.structured_world_graph == state.structured_world_graph


def test_cognitive_state_rejects_split_or_mismatched_slot_shapes() -> None:
    with pytest.raises(ValueError, match="share batch and hidden dims"):
        CognitiveState(
            semantic_slots=torch.zeros(1, 4, 8),
            executive_slots=torch.zeros(1, 2, 4),
            scratch_slots=torch.zeros(1, 2, 8),
        )

    state = CognitiveState(
        semantic_slots=torch.zeros(1, 4, 8),
        executive_slots=torch.zeros(1, 2, 8),
        scratch_slots=torch.zeros(1, 2, 8),
    )
    with pytest.raises(ValueError, match="expected cognitive slot shapes"):
        state.validate(CognitiveKernelConfig(hidden_dim=8))


def test_event_and_world_graph_preserve_provenance_and_exact_relations() -> None:
    event = CognitiveEvent(
        event_id="evt_0001",
        event_type="observation",
        source=EventSource(
            representation="image",
            adapter="vision_bridge_v0",
            source_ref="image://input/001.png",
        ),
        claims=(EvidenceClaim("has_color", "red", 0.97, "umbrella_01"),),
        evidence_refs=("image://input/001.png",),
        evidence_kind=EvidenceKind.OBSERVED_EVIDENCE,
    )

    assert CognitiveEvent.from_dict(json.loads(json.dumps(event.to_dict()))) == event
    assert StructuredWorldGraph.from_dict(
        json.loads(json.dumps(_graph().to_dict()))
    ) == _graph()

    with pytest.raises(ValueError, match="unknown entities"):
        StructuredWorldGraph(
            entities=(WorldEntity("actor_01", "character"),),
            relations=(WorldRelation("actor_01", "holds", "missing_prop"),),
        )

    with pytest.raises(ValueError, match="provenance"):
        CognitiveEvent(
            event_id="evt_invalid",
            event_type="observation",
            source=event.source,
            claims=(),
            evidence_refs=(),
            evidence_kind=EvidenceKind.OBSERVED_EVIDENCE,
        )


def test_registry_physical_evidence_and_world_bundle_round_trip() -> None:
    registry = RepresentationRegistryEntry(
        id="simple_mesh_v0",
        type="mesh",
        native_shape=("vertices", 3),
        encoder="simple_3d_parser_v0",
        decoder=None,
        bridge_in="mesh_bridge_v0",
        bridge_out=None,
        device_requirements={"device": "cpu"},
        precision="float32",
        license="local_research",
        capabilities=("3d_understanding", "ray_query"),
    )
    query = PhysicalQuery(
        query_id="rayq_0001",
        query_type="visibility",
        origin={"source": "camera_primary"},
        scene_ref="scene://room_17",
        requirements={"return": ["hit_entity", "distance"]},
    )
    evidence = PhysicalEvidence(
        event_id="phys_evt_0001",
        query_id=query.query_id,
        evidence_type="visibility",
        claims=(EvidenceClaim("is_visible", True, 1.0, "actor_01"),),
        backend="engine_ray_query",
        scene_ref=query.scene_ref,
        deterministic=True,
    )
    bundle = WorldBundle(
        bundle_id="world_0001",
        entity_ids=("actor_01", "umbrella_01"),
        representations={
            "text": ("text://world_0001.txt",),
            "image": ("image://world_0001/front.png",),
            "mesh": ("mesh://world_0001/actor.glb",),
        },
        entity_relation_metadata={"actor_01": {"holds": "umbrella_01"}},
        provenance={"generator": "procedural_scene_v0", "sha256": "abc123"},
        license="user_owned",
    )

    assert RepresentationRegistryEntry.from_dict(
        json.loads(json.dumps(registry.to_dict()))
    ) == registry
    assert PhysicalQuery.from_dict(json.loads(json.dumps(query.to_dict()))) == query
    assert PhysicalEvidence.from_dict(
        json.loads(json.dumps(evidence.to_dict()))
    ) == evidence
    assert WorldBundle.from_dict(json.loads(json.dumps(bundle.to_dict()))) == bundle

    with pytest.raises(ValueError, match="deterministic"):
        PhysicalEvidence(
            event_id="phys_evt_invalid",
            query_id=query.query_id,
            evidence_type="visibility",
            claims=(),
            backend="nondeterministic_model",
            scene_ref=query.scene_ref,
            deterministic=False,
        )


def test_parameter_and_vram_estimates_are_configurable_and_cycle_shared() -> None:
    spec = RecurrentCoreSpec()
    parameters = estimate_recurrent_core_parameters(spec)
    default_report = estimate_inference_vram(spec)
    deeper_cycles = estimate_recurrent_core_parameters(
        RecurrentCoreSpec(max_cycles=64)
    )
    constrained_report = estimate_inference_vram(
        spec,
        VramEstimateConfig(
            weight_bits=4,
            bridge_bytes=0,
            specialist_bytes=0,
            runtime_bytes=0,
            reserve_bytes=0,
            budget_bytes=1,
        ),
    )

    assert 500_000_000 < parameters["total_parameters"] < 1_000_000_000
    assert deeper_cycles["total_parameters"] == parameters["total_parameters"]
    assert default_report["budget_bytes"] == 8 * BYTES_PER_GIB
    assert default_report["total_bytes"] == sum(
        default_report[key]
        for key in (
            "weight_bytes",
            "state_bytes",
            "kv_bytes",
            "activation_staging_bytes",
            "bridge_bytes",
            "specialist_bytes",
            "runtime_bytes",
            "reserve_bytes",
        )
    )
    assert not constrained_report["fits_budget"]
    assert constrained_report["headroom_bytes"] < 0

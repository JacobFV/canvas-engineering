"""Tests for canvas schema: semantic types, transfer distance, serialization."""
import json
import math
import torch
import pytest
from pathlib import Path

from canvas_engine import (
    CanvasLayout, RegionSpec, CanvasSchema,
    Connection, CanvasTopology, transfer_distance,
)


# --- Semantic type fields ---


def test_region_spec_semantic_defaults():
    spec = RegionSpec(bounds=(0, 4, 0, 8, 0, 8))
    assert spec.semantic_type is None
    assert spec.semantic_embedding is None
    assert spec.embedding_model == "openai/text-embedding-3-small"


def test_region_spec_semantic_fields():
    emb = (0.1, 0.2, 0.3, -0.1)
    spec = RegionSpec(
        bounds=(0, 4, 0, 8, 0, 8),
        semantic_type="RGB video 224x224 30fps",
        semantic_embedding=emb,
        embedding_model="custom/my-model-v1",
    )
    assert spec.semantic_type == "RGB video 224x224 30fps"
    assert spec.semantic_embedding == emb
    assert spec.embedding_model == "custom/my-model-v1"


def test_region_spec_semantic_frozen():
    spec = RegionSpec(bounds=(0, 1, 0, 1, 0, 1), semantic_type="test")
    with pytest.raises(AttributeError):
        spec.semantic_type = "changed"


# --- Transfer distance ---


def _make_spec(embedding, model="openai/text-embedding-3-small"):
    return RegionSpec(
        bounds=(0, 1, 0, 1, 0, 1),
        semantic_embedding=tuple(embedding),
        embedding_model=model,
    )


def test_transfer_distance_identical():
    a = _make_spec([1.0, 0.0, 0.0])
    assert transfer_distance(a, a) == pytest.approx(0.0, abs=1e-6)


def test_transfer_distance_orthogonal():
    a = _make_spec([1.0, 0.0, 0.0])
    b = _make_spec([0.0, 1.0, 0.0])
    assert transfer_distance(a, b) == pytest.approx(1.0, abs=1e-6)


def test_transfer_distance_opposite():
    a = _make_spec([1.0, 0.0, 0.0])
    b = _make_spec([-1.0, 0.0, 0.0])
    assert transfer_distance(a, b) == pytest.approx(2.0, abs=1e-6)


def test_transfer_distance_similar():
    a = _make_spec([1.0, 0.1, 0.0])
    b = _make_spec([1.0, 0.2, 0.0])
    dist = transfer_distance(a, b)
    assert 0.0 < dist < 0.1  # very similar vectors


def test_transfer_distance_symmetric():
    a = _make_spec([0.5, 0.3, -0.1, 0.8])
    b = _make_spec([0.1, -0.2, 0.7, 0.4])
    assert transfer_distance(a, b) == pytest.approx(transfer_distance(b, a), abs=1e-6)


def test_transfer_distance_no_embedding_raises():
    a = RegionSpec(bounds=(0, 1, 0, 1, 0, 1))
    b = _make_spec([1.0, 0.0])
    with pytest.raises(ValueError, match="semantic_embedding"):
        transfer_distance(a, b)


def test_transfer_distance_model_mismatch_raises():
    a = _make_spec([1.0, 0.0], model="model-a")
    b = _make_spec([1.0, 0.0], model="model-b")
    with pytest.raises(ValueError, match="Embedding model mismatch"):
        transfer_distance(a, b)


def test_transfer_distance_dim_mismatch_raises():
    a = _make_spec([1.0, 0.0, 0.0])
    b = _make_spec([1.0, 0.0])
    with pytest.raises(ValueError, match="dimension mismatch"):
        transfer_distance(a, b)


# --- CanvasSchema serialization ---


def _sample_layout():
    return CanvasLayout(
        T=8, H=16, W=16, d_model=256,
        regions={
            "visual": (0, 8, 0, 12, 0, 12),
            "action": RegionSpec(
                bounds=(0, 8, 12, 14, 0, 2),
                period=1,
                loss_weight=2.0,
                semantic_type="6-DOF end-effector delta pose + gripper",
                semantic_embedding=(0.1, -0.2, 0.3),
            ),
            "thought": RegionSpec(
                bounds=(0, 2, 14, 16, 0, 4),
                period=4,
                is_output=False,
            ),
        },
        t_current=3,
    )


def _sample_topology():
    return CanvasTopology(connections=[
        Connection(src="visual", dst="visual"),
        Connection(src="action", dst="visual"),
        Connection(src="action", dst="action"),
        Connection(src="thought", dst="visual", weight=0.5, t_src=0, t_dst=0),
    ])


def test_schema_to_dict_basic():
    schema = CanvasSchema(layout=_sample_layout())
    d = schema.to_dict()
    assert d["schema_version"] == "0.2.0"
    assert d["layout"]["T"] == 8
    assert d["layout"]["d_model"] == 256
    assert d["layout"]["t_current"] == 3
    assert "visual" in d["regions"]
    assert "action" in d["regions"]
    assert "thought" in d["regions"]


def test_schema_to_dict_regions():
    schema = CanvasSchema(layout=_sample_layout())
    d = schema.to_dict()
    # raw tuple region — just bounds
    assert d["regions"]["visual"] == {"bounds": [0, 8, 0, 12, 0, 12]}
    # action region with non-default fields
    action = d["regions"]["action"]
    assert action["bounds"] == [0, 8, 12, 14, 0, 2]
    assert action["loss_weight"] == 2.0
    assert action["semantic_type"] == "6-DOF end-effector delta pose + gripper"
    assert action["semantic_embedding"] == [0.1, -0.2, 0.3]
    assert "period" not in action  # period=1 is default, omitted
    assert "is_output" not in action  # True is default, omitted
    # thought region
    thought = d["regions"]["thought"]
    assert thought["period"] == 4
    assert thought["is_output"] is False


def test_schema_to_dict_topology():
    schema = CanvasSchema(layout=_sample_layout(), topology=_sample_topology())
    d = schema.to_dict()
    assert "topology" in d
    assert len(d["topology"]) == 4
    # Check the thought→visual connection preserves weight and temporal
    tv = [c for c in d["topology"] if c["src"] == "thought"][0]
    assert tv["weight"] == 0.5
    assert tv["t_src"] == 0
    assert tv["t_dst"] == 0


def test_schema_to_dict_metadata():
    schema = CanvasSchema(
        layout=_sample_layout(),
        metadata={"model": "CogVideoX-2B", "data": "bridge_v2"},
    )
    d = schema.to_dict()
    assert d["metadata"]["model"] == "CogVideoX-2B"


def test_schema_roundtrip_dict():
    original = CanvasSchema(
        layout=_sample_layout(),
        topology=_sample_topology(),
        metadata={"note": "test"},
    )
    d = original.to_dict()
    loaded = CanvasSchema.from_dict(d)

    assert loaded.layout.T == 8
    assert loaded.layout.H == 16
    assert loaded.layout.d_model == 256
    assert loaded.layout.t_current == 3
    assert loaded.version == "0.2.0"
    assert loaded.metadata == {"note": "test"}

    # Check regions survived
    assert set(loaded.layout.regions.keys()) == {"visual", "action", "thought"}

    # Raw tuple stays as tuple
    assert loaded.layout.regions["visual"] == (0, 8, 0, 12, 0, 12)

    # RegionSpec fields survive
    action = loaded.layout.region_spec("action")
    assert action.bounds == (0, 8, 12, 14, 0, 2)
    assert action.loss_weight == 2.0
    assert action.semantic_type == "6-DOF end-effector delta pose + gripper"
    assert action.semantic_embedding == (0.1, -0.2, 0.3)
    assert action.embedding_model == "openai/text-embedding-3-small"

    thought = loaded.layout.region_spec("thought")
    assert thought.period == 4
    assert thought.is_output is False

    # Topology survived
    assert loaded.topology is not None
    assert len(loaded.topology.connections) == 4
    tv = [c for c in loaded.topology.connections if c.src == "thought"][0]
    assert tv.weight == 0.5
    assert tv.t_src == 0
    assert tv.t_dst == 0


def test_schema_roundtrip_json(tmp_path):
    path = tmp_path / "test_schema.json"
    original = CanvasSchema(
        layout=_sample_layout(),
        topology=_sample_topology(),
        metadata={"round": "trip"},
    )
    original.to_json(path)

    # Verify it's valid JSON
    with open(path) as f:
        raw = json.load(f)
    assert raw["schema_version"] == "0.2.0"

    loaded = CanvasSchema.from_json(path)
    assert loaded.layout.T == 8
    assert loaded.layout.region_spec("action").semantic_type == "6-DOF end-effector delta pose + gripper"
    assert loaded.topology is not None
    assert loaded.metadata == {"round": "trip"}


def test_schema_no_topology():
    schema = CanvasSchema(layout=_sample_layout())
    d = schema.to_dict()
    assert "topology" not in d
    loaded = CanvasSchema.from_dict(d)
    assert loaded.topology is None


def test_schema_no_metadata():
    schema = CanvasSchema(layout=_sample_layout())
    d = schema.to_dict()
    assert "metadata" not in d


# --- Compatible regions ---


def test_compatible_regions_basic():
    emb_rgb = (1.0, 0.0, 0.0, 0.0)
    emb_depth = (0.95, 0.1, 0.0, 0.0)  # close to RGB
    emb_joints = (0.0, 0.0, 1.0, 0.0)   # far from RGB

    schema_a = CanvasSchema(layout=CanvasLayout(
        T=4, H=8, W=8, d_model=64,
        regions={
            "cam": RegionSpec(
                bounds=(0, 4, 0, 6, 0, 6),
                semantic_type="RGB video",
                semantic_embedding=emb_rgb,
            ),
        },
    ))

    schema_b = CanvasSchema(layout=CanvasLayout(
        T=4, H=8, W=8, d_model=64,
        regions={
            "depth": RegionSpec(
                bounds=(0, 4, 0, 6, 0, 6),
                semantic_type="depth map",
                semantic_embedding=emb_depth,
            ),
            "joints": RegionSpec(
                bounds=(0, 4, 6, 7, 0, 1),
                semantic_type="joint angles",
                semantic_embedding=emb_joints,
            ),
        },
    ))

    pairs = schema_a.compatible_regions(schema_b, threshold=0.3)
    assert len(pairs) == 1
    assert pairs[0][0] == "cam"
    assert pairs[0][1] == "depth"
    assert pairs[0][2] < 0.1  # very close


def test_compatible_regions_no_embedding_skipped():
    schema_a = CanvasSchema(layout=CanvasLayout(
        T=2, H=2, W=2, d_model=16,
        regions={"a": (0, 2, 0, 2, 0, 2)},  # no semantic_embedding
    ))
    schema_b = CanvasSchema(layout=CanvasLayout(
        T=2, H=2, W=2, d_model=16,
        regions={"b": RegionSpec(
            bounds=(0, 2, 0, 2, 0, 2),
            semantic_embedding=(1.0, 0.0),
        )},
    ))
    pairs = schema_a.compatible_regions(schema_b)
    assert pairs == []


def test_compatible_regions_different_models_skipped():
    schema_a = CanvasSchema(layout=CanvasLayout(
        T=2, H=2, W=2, d_model=16,
        regions={"a": RegionSpec(
            bounds=(0, 2, 0, 2, 0, 2),
            semantic_embedding=(1.0, 0.0),
            embedding_model="model-x",
        )},
    ))
    schema_b = CanvasSchema(layout=CanvasLayout(
        T=2, H=2, W=2, d_model=16,
        regions={"b": RegionSpec(
            bounds=(0, 2, 0, 2, 0, 2),
            semantic_embedding=(1.0, 0.0),
            embedding_model="model-y",
        )},
    ))
    pairs = schema_a.compatible_regions(schema_b)
    assert pairs == []


def test_compatible_regions_sorted_by_distance():
    emb_base = (1.0, 0.0, 0.0)
    emb_close = (0.98, 0.1, 0.0)
    emb_medium = (0.7, 0.7, 0.0)

    schema_a = CanvasSchema(layout=CanvasLayout(
        T=2, H=4, W=4, d_model=32,
        regions={"x": RegionSpec(
            bounds=(0, 2, 0, 2, 0, 2),
            semantic_embedding=emb_base,
        )},
    ))
    schema_b = CanvasSchema(layout=CanvasLayout(
        T=2, H=4, W=4, d_model=32,
        regions={
            "close": RegionSpec(
                bounds=(0, 2, 0, 2, 0, 2),
                semantic_embedding=emb_close,
            ),
            "medium": RegionSpec(
                bounds=(0, 2, 2, 4, 0, 2),
                semantic_embedding=emb_medium,
            ),
        },
    ))
    pairs = schema_a.compatible_regions(schema_b, threshold=1.0)
    assert len(pairs) == 2
    assert pairs[0][1] == "close"
    assert pairs[1][1] == "medium"
    assert pairs[0][2] < pairs[1][2]


def test_compatible_regions_self():
    """A schema is perfectly compatible with itself."""
    emb = (0.5, 0.3, -0.1, 0.8)
    schema = CanvasSchema(layout=CanvasLayout(
        T=2, H=4, W=4, d_model=32,
        regions={
            "a": RegionSpec(bounds=(0, 2, 0, 2, 0, 2), semantic_embedding=emb),
            "b": RegionSpec(bounds=(0, 2, 2, 4, 0, 2), semantic_embedding=emb),
        },
    ))
    pairs = schema.compatible_regions(schema, threshold=0.0)
    # a↔a, a↔b, b↔a, b↔b all have distance 0
    assert len(pairs) == 4
    for _, _, dist in pairs:
        assert dist == pytest.approx(0.0, abs=1e-6)


# --- Integration: schema with transfer distance ---


def test_schema_preserves_transfer_distance(tmp_path):
    """Transfer distance is preserved across JSON round-trip."""
    emb_a = (0.5, 0.3, -0.1, 0.8)
    emb_b = (0.1, -0.2, 0.7, 0.4)

    layout = CanvasLayout(
        T=4, H=8, W=8, d_model=128,
        regions={
            "cam": RegionSpec(
                bounds=(0, 4, 0, 6, 0, 6),
                semantic_type="RGB video",
                semantic_embedding=emb_a,
            ),
            "action": RegionSpec(
                bounds=(0, 4, 6, 7, 0, 1),
                semantic_type="joint angles",
                semantic_embedding=emb_b,
            ),
        },
    )
    schema = CanvasSchema(layout=layout)

    # Compute distance before serialization
    dist_before = transfer_distance(
        layout.region_spec("cam"),
        layout.region_spec("action"),
    )

    # Round-trip through JSON
    path = tmp_path / "test.json"
    schema.to_json(path)
    loaded = CanvasSchema.from_json(path)

    # Distance after round-trip
    dist_after = transfer_distance(
        loaded.layout.region_spec("cam"),
        loaded.layout.region_spec("action"),
    )

    assert dist_before == pytest.approx(dist_after, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

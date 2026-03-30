"""Tests for declarative canvas topology, including temporal connections."""
import torch
import pytest
from canvas_engineering import CanvasLayout, Connection, CanvasTopology, TemporalFill


def make_layout():
    return CanvasLayout(T=2, H=4, W=4, d_model=16, regions={
        "a": (0, 2, 0, 2, 0, 2),  # 8 positions
        "b": (0, 2, 2, 4, 0, 2),  # 8 positions
        "c": (0, 1, 0, 4, 2, 4),  # 8 positions
    })


def make_temporal_layout():
    """Layout with 4 timesteps for temporal tests."""
    return CanvasLayout(T=4, H=2, W=2, d_model=16, regions={
        "obs": (0, 4, 0, 1, 0, 1),    # 4 positions (1 per frame)
        "action": (0, 4, 1, 2, 0, 1),  # 4 positions (1 per frame)
        "thought": (0, 4, 0, 2, 1, 2), # 8 positions (2 per frame)
    })


# ── Original spatial tests ──────────────────────────────────────────────

def test_isolated():
    layout = make_layout()
    topo = CanvasTopology.isolated(list(layout.regions.keys()))
    mask = topo.to_attention_mask(layout)
    a_idx = layout.region_indices("a")
    b_idx = layout.region_indices("b")
    assert mask[a_idx[0], a_idx[1]].item() == 1.0
    assert mask[a_idx[0], b_idx[0]].item() == 0.0


def test_directed_connection():
    layout = make_layout()
    topo = CanvasTopology(connections=[Connection(src="a", dst="b", weight=0.7)])
    mask = topo.to_attention_mask(layout)
    a_idx = layout.region_indices("a")
    b_idx = layout.region_indices("b")
    assert abs(mask[a_idx[0], b_idx[0]].item() - 0.7) < 1e-5
    assert mask[b_idx[0], a_idx[0]].item() == 0.0  # no reverse


def test_dense():
    layout = make_layout()
    topo = CanvasTopology.dense(list(layout.regions.keys()))
    mask = topo.to_attention_mask(layout)
    all_idx = sorted(set().union(*(layout.region_indices(n) for n in layout.regions)))
    sub = mask[all_idx][:, all_idx]
    assert (sub > 0).all()


def test_hub_spoke():
    layout = make_layout()
    topo = CanvasTopology.hub_spoke("c", ["a", "b"])
    mask = topo.to_attention_mask(layout)
    a_idx = layout.region_indices("a")
    c_idx = layout.region_indices("c")
    assert mask[c_idx[0], a_idx[0]].item() == 1.0  # hub → spoke
    assert mask[a_idx[0], c_idx[0]].item() == 1.0  # spoke → hub (bidirectional)


def test_causal_chain():
    layout = make_layout()
    topo = CanvasTopology.causal_chain(["a", "b", "c"])
    mask = topo.to_attention_mask(layout)
    a_idx = layout.region_indices("a")
    b_idx = layout.region_indices("b")
    c_idx = layout.region_indices("c")
    assert mask[b_idx[0], a_idx[0]].item() == 1.0  # b attends to a
    assert mask[c_idx[0], b_idx[0]].item() == 1.0  # c attends to b
    assert mask[a_idx[0], b_idx[0]].item() == 0.0  # a does NOT attend to b


def test_neighbors_and_attended():
    topo = CanvasTopology(connections=[
        Connection(src="a", dst="a"),
        Connection(src="a", dst="b"),
        Connection(src="b", dst="a"),
    ])
    assert set(topo.neighbors_of("a")) == {"a", "b"}
    assert set(topo.attended_by("a")) == {"a", "b"}
    assert topo.neighbors_of("b") == ["a"]


def test_attention_ops():
    topo = CanvasTopology(connections=[
        Connection(src="x", dst="y", weight=0.5),
        Connection(src="x", dst="x"),
    ])
    ops = topo.attention_ops()
    assert len(ops) == 2
    assert ("x", "y", 0.5, "cross_attention") in ops


def test_summary():
    topo = CanvasTopology(connections=[
        Connection(src="a", dst="a"),
        Connection(src="a", dst="b"),
        Connection(src="b", dst="a"),
    ])
    s = topo.summary()
    assert "↺ self" in s
    assert "↔" in s


def test_repr():
    topo = CanvasTopology(connections=[Connection(src="a", dst="b")])
    assert "connections=1" in repr(topo)


# ── Temporal topology tests ─────────────────────────────────────────────

def test_connection_temporal_defaults():
    """Default connection has no temporal constraints."""
    c = Connection(src="a", dst="b")
    assert c.t_src is None
    assert c.t_dst is None


def test_connection_temporal_fields():
    """Temporal offsets are stored correctly."""
    c = Connection(src="a", dst="b", t_src=0, t_dst=-1)
    assert c.t_src == 0
    assert c.t_dst == -1


def test_has_temporal_constraints():
    """Topology reports temporal constraints correctly."""
    topo_no = CanvasTopology(connections=[Connection(src="a", dst="b")])
    assert not topo_no.has_temporal_constraints

    topo_yes = CanvasTopology(connections=[
        Connection(src="a", dst="b", t_src=0, t_dst=0),
    ])
    assert topo_yes.has_temporal_constraints


def test_temporal_none_none_backward_compat():
    """t_src=None, t_dst=None produces same mask as original behavior."""
    layout = make_layout()
    topo_old = CanvasTopology(connections=[Connection(src="a", dst="b")])
    topo_explicit = CanvasTopology(connections=[
        Connection(src="a", dst="b", t_src=None, t_dst=None)
    ])
    mask_old = topo_old.to_attention_mask(layout)
    mask_new = topo_explicit.to_attention_mask(layout)
    assert torch.equal(mask_old, mask_new)


def test_same_frame_only():
    """t_src=0, t_dst=0: src at frame t only attends to dst at frame t."""
    layout = make_temporal_layout()
    topo = CanvasTopology(connections=[
        Connection(src="obs", dst="action", t_src=0, t_dst=0)
    ])
    mask = topo.to_attention_mask(layout)

    # obs at t=0 should attend to action at t=0
    obs_t0 = layout.region_indices_at_t("obs", 0)
    act_t0 = layout.region_indices_at_t("action", 0)
    assert mask[obs_t0[0], act_t0[0]].item() == 1.0

    # obs at t=0 should NOT attend to action at t=1
    act_t1 = layout.region_indices_at_t("action", 1)
    assert mask[obs_t0[0], act_t1[0]].item() == 0.0

    # obs at t=2 should attend to action at t=2
    obs_t2 = layout.region_indices_at_t("obs", 2)
    act_t2 = layout.region_indices_at_t("action", 2)
    assert mask[obs_t2[0], act_t2[0]].item() == 1.0


def test_prev_frame_cross_attention():
    """t_src=0, t_dst=-1: src at frame t attends to dst at frame t-1."""
    layout = make_temporal_layout()
    topo = CanvasTopology(connections=[
        Connection(src="action", dst="obs", t_src=0, t_dst=-1)
    ])
    mask = topo.to_attention_mask(layout)

    # action at t=1 should attend to obs at t=0
    act_t1 = layout.region_indices_at_t("action", 1)
    obs_t0 = layout.region_indices_at_t("obs", 0)
    assert mask[act_t1[0], obs_t0[0]].item() == 1.0

    # action at t=0 should NOT attend to obs at t=-1 (out of bounds)
    act_t0 = layout.region_indices_at_t("action", 0)
    # Since t=-1 is out of bounds, no connections from act_t0 to any obs
    obs_all = layout.region_indices("obs")
    assert mask[act_t0[0], obs_all[0]].item() == 0.0

    # action at t=1 should NOT attend to obs at t=1 (wrong frame)
    obs_t1 = layout.region_indices_at_t("obs", 1)
    assert mask[act_t1[0], obs_t1[0]].item() == 0.0


def test_next_frame_cross_attention():
    """t_src=0, t_dst=1: src at frame t attends to dst at frame t+1."""
    layout = make_temporal_layout()
    topo = CanvasTopology(connections=[
        Connection(src="obs", dst="obs", t_src=0, t_dst=1)
    ])
    mask = topo.to_attention_mask(layout)

    # obs at t=0 attends to obs at t=1
    obs_t0 = layout.region_indices_at_t("obs", 0)
    obs_t1 = layout.region_indices_at_t("obs", 1)
    assert mask[obs_t0[0], obs_t1[0]].item() == 1.0

    # obs at t=3: t_dst=1 targets t=4 (out of bounds). With default HOLD,
    # falls back to most recent available (t=3). With DROP, would be zero.
    obs_t3 = layout.region_indices_at_t("obs", 3)
    assert mask[obs_t3[0], obs_t3[0]].item() == 1.0  # held to t=3

    # Verify DROP mode gives the old behavior (no connection)
    topo_drop = CanvasTopology(connections=[
        Connection(src="obs", dst="obs", t_src=0, t_dst=1, temporal_fill=TemporalFill.DROP)
    ])
    mask_drop = topo_drop.to_attention_mask(layout)
    assert sum(mask_drop[obs_t3[0]].tolist()) == 0.0


def test_mixed_src_constrained_dst_all():
    """t_src=0, t_dst=None: src at frame ref queries ALL dst timesteps."""
    layout = make_temporal_layout()
    topo = CanvasTopology(connections=[
        Connection(src="action", dst="obs", t_src=0, t_dst=None)
    ])
    mask = topo.to_attention_mask(layout)

    # action at t=1 should attend to obs at ALL timesteps
    act_t1 = layout.region_indices_at_t("action", 1)
    obs_all = layout.region_indices("obs")
    for oi in obs_all:
        assert mask[act_t1[0], oi].item() == 1.0


def test_mixed_src_all_dst_constrained():
    """t_src=None, t_dst=0: ALL src timesteps query dst at frame ref."""
    layout = make_temporal_layout()
    topo = CanvasTopology(connections=[
        Connection(src="obs", dst="action", t_src=None, t_dst=0)
    ])
    mask = topo.to_attention_mask(layout)

    # ALL obs timesteps should attend to action at some frames
    obs_all = layout.region_indices("obs")
    act_t0 = layout.region_indices_at_t("action", 0)
    # obs (all timesteps) attend to action at ref+0, for each ref frame
    # So all obs positions should attend to at least one action position
    for oi in obs_all:
        assert sum(mask[oi, ai].item() for ai in layout.region_indices("action")) > 0


def test_causal_temporal_constructor():
    """causal_temporal: same-frame self + prev-frame cross, no future leakage."""
    layout = make_temporal_layout()
    topo = CanvasTopology.causal_temporal(["obs", "action"])
    mask = topo.to_attention_mask(layout)

    # obs at t=1 attends to obs at t=1 (same-frame self)
    obs_t1 = layout.region_indices_at_t("obs", 1)
    assert mask[obs_t1[0], obs_t1[0]].item() == 1.0

    # obs at t=1 attends to obs at t=0 (prev-frame self)
    obs_t0 = layout.region_indices_at_t("obs", 0)
    assert mask[obs_t1[0], obs_t0[0]].item() == 1.0

    # obs at t=1 attends to action at t=0 (prev-frame cross)
    act_t0 = layout.region_indices_at_t("action", 0)
    assert mask[obs_t1[0], act_t0[0]].item() == 1.0

    # obs at t=0 does NOT attend to obs at t=1 (no future leakage)
    assert mask[obs_t0[0], obs_t1[0]].item() == 0.0

    # obs at t=0 does NOT attend to action at t=1 (no future leakage)
    act_t1 = layout.region_indices_at_t("action", 1)
    assert mask[obs_t0[0], act_t1[0]].item() == 0.0


def test_temporal_weight():
    """Temporal connections respect weight parameter."""
    layout = make_temporal_layout()
    topo = CanvasTopology(connections=[
        Connection(src="obs", dst="action", t_src=0, t_dst=0, weight=0.3)
    ])
    mask = topo.to_attention_mask(layout)

    obs_t0 = layout.region_indices_at_t("obs", 0)
    act_t0 = layout.region_indices_at_t("action", 0)
    assert abs(mask[obs_t0[0], act_t0[0]].item() - 0.3) < 1e-5


def test_dense_no_temporal_vs_temporal_same():
    """Dense topology with no temporal = dense with t_src=None, t_dst=None."""
    layout = make_temporal_layout()
    regions = list(layout.regions.keys())
    topo_dense = CanvasTopology.dense(regions)
    mask_dense = topo_dense.to_attention_mask(layout)

    # All covered positions should be connected
    all_idx = sorted(set().union(*(layout.region_indices(n) for n in regions)))
    sub = mask_dense[all_idx][:, all_idx]
    assert (sub > 0).all()


def test_temporal_summary():
    """Summary reports temporal constraints."""
    topo = CanvasTopology(connections=[
        Connection(src="a", dst="b", t_src=0, t_dst=-1),
    ])
    s = topo.summary()
    assert "temporal" in s.lower()


def test_layout_region_indices_at_t():
    """CanvasLayout.region_indices_at_t returns correct indices."""
    layout = make_temporal_layout()
    # obs is at (0, 4, 0, 1, 0, 1) = h=0, w=0 for all t
    # Flat index = t * (H * W) + h * W + w = t * 4 + 0
    obs_t0 = layout.region_indices_at_t("obs", 0)
    assert obs_t0 == [0]
    obs_t1 = layout.region_indices_at_t("obs", 1)
    assert obs_t1 == [4]
    obs_t2 = layout.region_indices_at_t("obs", 2)
    assert obs_t2 == [8]

    # Out of bounds returns empty
    assert layout.region_indices_at_t("obs", 5) == []
    assert layout.region_indices_at_t("obs", -1) == []


def test_layout_region_timesteps():
    """CanvasLayout.region_timesteps returns correct timesteps."""
    layout = make_temporal_layout()
    assert layout.region_timesteps("obs") == [0, 1, 2, 3]
    assert layout.region_timesteps("action") == [0, 1, 2, 3]


# --- Attention function type tests ---


from canvas_engineering import RegionSpec, ATTENTION_TYPES


def test_connection_fn_default_none():
    c = Connection(src="a", dst="b")
    assert c.fn is None


def test_connection_fn_explicit():
    c = Connection(src="a", dst="b", fn="gated")
    assert c.fn == "gated"


def test_resolve_fn_explicit_overrides_region():
    """Connection.fn takes priority over region default_attn."""
    layout = CanvasLayout(T=2, H=4, W=4, d_model=16, regions={
        "a": RegionSpec(bounds=(0, 2, 0, 2, 0, 2), default_attn="linear_attention"),
        "b": (0, 2, 2, 4, 0, 2),
    })
    topo = CanvasTopology(connections=[
        Connection(src="a", dst="b", fn="gated"),
    ])
    assert topo.resolve_fn(topo.connections[0], layout) == "gated"


def test_resolve_fn_falls_back_to_region_default():
    """When fn is None, resolve uses src region's default_attn."""
    layout = CanvasLayout(T=2, H=4, W=4, d_model=16, regions={
        "a": RegionSpec(bounds=(0, 2, 0, 2, 0, 2), default_attn="mamba"),
        "b": (0, 2, 2, 4, 0, 2),
    })
    topo = CanvasTopology(connections=[
        Connection(src="a", dst="b"),
    ])
    assert topo.resolve_fn(topo.connections[0], layout) == "mamba"


def test_resolve_fn_no_layout_gives_global_default():
    """Without layout, resolve defaults to cross_attention."""
    topo = CanvasTopology(connections=[Connection(src="a", dst="b")])
    assert topo.resolve_fn(topo.connections[0]) == "cross_attention"


def test_resolve_fn_tuple_region_gives_global_default():
    """Raw tuple regions (no RegionSpec) resolve to cross_attention."""
    layout = CanvasLayout(T=2, H=4, W=4, d_model=16, regions={
        "a": (0, 2, 0, 2, 0, 2),
        "b": (0, 2, 2, 4, 0, 2),
    })
    topo = CanvasTopology(connections=[Connection(src="a", dst="b")])
    assert topo.resolve_fn(topo.connections[0], layout) == "cross_attention"


def test_attention_ops_includes_fn():
    """attention_ops returns 4-tuples with resolved fn."""
    layout = CanvasLayout(T=2, H=4, W=4, d_model=16, regions={
        "cam": RegionSpec(bounds=(0, 2, 0, 2, 0, 2), default_attn="linear_attention"),
        "act": RegionSpec(bounds=(0, 2, 2, 4, 0, 2), default_attn="cross_attention"),
        "goal": (0, 1, 0, 1, 0, 1),
    })
    topo = CanvasTopology(connections=[
        Connection(src="cam", dst="cam"),           # from region default → linear
        Connection(src="act", dst="cam"),            # from region default → cross
        Connection(src="act", dst="goal", fn="gated"),  # explicit → gated
        Connection(src="goal", dst="goal"),          # tuple region → cross
    ])
    ops = topo.attention_ops(layout)
    assert len(ops) == 4
    assert ops[0] == ("cam", "cam", 1.0, "linear_attention")
    assert ops[1] == ("act", "cam", 1.0, "cross_attention")
    assert ops[2] == ("act", "goal", 1.0, "gated")
    assert ops[3] == ("goal", "goal", 1.0, "cross_attention")


def test_attention_ops_without_layout():
    """attention_ops without layout resolves everything to cross_attention."""
    topo = CanvasTopology(connections=[
        Connection(src="a", dst="b"),
        Connection(src="a", dst="b", fn="pooling"),
    ])
    ops = topo.attention_ops()
    assert ops[0][3] == "cross_attention"
    assert ops[1][3] == "pooling"


def test_region_spec_default_attn_default():
    spec = RegionSpec(bounds=(0, 1, 0, 1, 0, 1))
    assert spec.default_attn == "cross_attention"


def test_region_spec_default_attn_custom():
    spec = RegionSpec(bounds=(0, 1, 0, 1, 0, 1), default_attn="mamba")
    assert spec.default_attn == "mamba"


def test_attention_types_registry():
    """ATTENTION_TYPES has descriptions for all declared types."""
    expected = {
        "cross_attention", "linear_attention", "cosine_attention",
        "sigmoid_attention", "gated", "perceiver", "pooling", "copy",
        "mamba", "rwkv", "hyena", "sparse_attention", "local_attention",
        "none", "random_fixed", "mixture", "cogvideox",
    }
    assert set(ATTENTION_TYPES.keys()) == expected
    for name, desc in ATTENTION_TYPES.items():
        assert isinstance(desc, str) and len(desc) > 10, f"{name} has no description"


def test_mixed_fn_topology():
    """A realistic topology mixing multiple attention function types."""
    layout = CanvasLayout(T=4, H=8, W=8, d_model=256, regions={
        "visual": RegionSpec(bounds=(0, 4, 0, 6, 0, 6), default_attn="cross_attention"),
        "proprio": RegionSpec(bounds=(0, 4, 6, 7, 0, 2), default_attn="linear_attention"),
        "thought": RegionSpec(bounds=(0, 2, 7, 8, 0, 4), default_attn="mamba"),
        "goal": RegionSpec(bounds=(0, 1, 7, 8, 4, 8), default_attn="cross_attention",
                           is_output=False),
    })
    topo = CanvasTopology(connections=[
        # Self-attention per region (uses region defaults)
        Connection(src="visual", dst="visual"),
        Connection(src="proprio", dst="proprio"),
        Connection(src="thought", dst="thought"),
        Connection(src="goal", dst="goal"),
        # Cross-region with explicit overrides
        Connection(src="visual", dst="goal", fn="gated"),       # optional conditioning
        Connection(src="thought", dst="visual", fn="perceiver"),  # compress visual
        Connection(src="proprio", dst="visual", fn="pooling"),   # cheap summary
        # Multi-agent broadcast
        Connection(src="thought", dst="thought", fn="copy", t_src=0, t_dst=-1),
    ])
    ops = topo.attention_ops(layout)
    fns = [op[3] for op in ops]
    assert fns == [
        "cross_attention",   # visual self (region default)
        "linear_attention",  # proprio self (region default)
        "mamba",             # thought self (region default)
        "cross_attention",   # goal self (region default)
        "gated",             # visual ← goal (explicit)
        "perceiver",         # thought ← visual (explicit)
        "pooling",           # proprio ← visual (explicit)
        "copy",              # thought ← thought t-1 (explicit)
    ]


def test_additive_mask_no_nan():
    """to_additive_mask: unused positions get self-attention, no all-inf rows."""
    # Layout with unused positions: 2*4*4=32 positions, regions use 8+8=16
    layout = make_layout()
    topo = CanvasTopology.isolated(["a", "b"])
    mask = topo.to_additive_mask(layout)

    # No row should be all -inf (would cause NaN in softmax)
    for i in range(layout.num_positions):
        assert not mask[i].eq(float('-inf')).all(), f"row {i} is all -inf"

    # Region positions should attend within their region
    a_idx = layout.region_indices("a")
    assert mask[a_idx[0], a_idx[1]].item() == 0.0  # can attend

    # Cross-region should be blocked (isolated topology)
    b_idx = layout.region_indices("b")
    assert mask[a_idx[0], b_idx[0]].item() == float('-inf')


def test_additive_mask_dense():
    """to_additive_mask with dense topology: all region positions attend."""
    layout = make_layout()
    topo = CanvasTopology.dense(["a", "b"])
    mask = topo.to_additive_mask(layout)

    a_idx = layout.region_indices("a")
    b_idx = layout.region_indices("b")
    # Cross-region should be allowed
    assert mask[a_idx[0], b_idx[0]].item() == 0.0


# ── Temporal fill mode tests ──────────────────────────────────────────


def make_sparse_layout():
    """Layout where regions only exist at certain timesteps, for fill testing.

    'fast' is at every timestep (0-7), 'slow' is at 0 and 4 only.
    This simulates a period-4 slow region and a period-1 fast region.
    """
    return CanvasLayout(T=8, H=2, W=2, d_model=16, regions={
        "fast": (0, 8, 0, 1, 0, 1),   # 8 positions, 1 per frame
        "slow": RegionSpec(bounds=(0, 2, 1, 2, 0, 1), period=4),  # 2 positions at t=0,1
    })


def test_temporal_fill_hold_basic():
    """HOLD: when dst missing at target timestep, use most recent past."""
    layout = make_temporal_layout()
    # fast queries slow at t_dst=0 for each ref. At ref=1, abs_dst=1.
    # If slow has no position at t=1, HOLD grabs t=0.
    # In make_temporal_layout, all regions span all 4 timesteps, so this
    # doesn't trigger hold. Use a layout where slow is sparse.
    layout = CanvasLayout(T=4, H=2, W=2, d_model=16, regions={
        "fast": (0, 4, 0, 1, 0, 1),   # t=0,1,2,3
        "slow": (0, 1, 1, 2, 0, 1),   # t=0 only
    })
    topo = CanvasTopology(connections=[
        Connection(src="fast", dst="slow", t_src=0, t_dst=0,
                   temporal_fill=TemporalFill.HOLD)
    ])
    mask = topo.to_attention_mask(layout)

    fast_t0 = layout.region_indices_at_t("fast", 0)
    slow_t0 = layout.region_indices_at_t("slow", 0)

    # ref=0: abs_dst=0, slow exists at t=0 → direct connection
    assert mask[fast_t0[0], slow_t0[0]].item() == 1.0

    # ref=1: abs_dst=1, slow doesn't exist at t=1 → HOLD to t=0
    fast_t1 = layout.region_indices_at_t("fast", 1)
    assert mask[fast_t1[0], slow_t0[0]].item() == 1.0

    # ref=2: abs_dst=2, slow doesn't exist → HOLD to t=0
    fast_t2 = layout.region_indices_at_t("fast", 2)
    assert mask[fast_t2[0], slow_t0[0]].item() == 1.0


def test_temporal_fill_drop():
    """DROP: no connection when dst is missing."""
    layout = CanvasLayout(T=4, H=2, W=2, d_model=16, regions={
        "fast": (0, 4, 0, 1, 0, 1),
        "slow": (0, 1, 1, 2, 0, 1),
    })
    topo = CanvasTopology(connections=[
        Connection(src="fast", dst="slow", t_src=0, t_dst=0,
                   temporal_fill=TemporalFill.DROP)
    ])
    mask = topo.to_attention_mask(layout)

    fast_t0 = layout.region_indices_at_t("fast", 0)
    slow_t0 = layout.region_indices_at_t("slow", 0)

    # ref=0: direct connection exists
    assert mask[fast_t0[0], slow_t0[0]].item() == 1.0

    # ref=1: DROP → no connection
    fast_t1 = layout.region_indices_at_t("fast", 1)
    assert mask[fast_t1[0], slow_t0[0]].item() == 0.0



def test_temporal_fill_interpolate_both_endpoints():
    """INTERPOLATE: lerp between past and future when both available."""
    layout = CanvasLayout(T=6, H=2, W=2, d_model=16, regions={
        "fast": (0, 6, 0, 1, 0, 1),
        "slow": (0, 1, 1, 2, 0, 1),  # only at t=0
    })
    # Also add slow at t=4 by using a bigger region
    layout = CanvasLayout(T=6, H=2, W=2, d_model=16, regions={
        "fast": (0, 6, 0, 1, 0, 1),
        # slow at t=0 and t=4
        "slow_0": (0, 1, 1, 2, 0, 1),
        "slow_4": (4, 5, 1, 2, 0, 1),
    })
    # For a clean test, let's use a single 'slow' region that spans t=0..5
    # but only has positions at specific timesteps. Actually CanvasLayout
    # gives positions at all timesteps in the range. So let's just test
    # with the _resolve_temporal_fill helper directly.
    from canvas_engineering.connectivity import _resolve_temporal_fill

    # Simulate: slow has positions at t=0 and t=4, query wants t=2
    dst_t_cache = {0: [10], 4: [14]}
    conn = Connection(src="fast", dst="slow", t_src=0, t_dst=0,
                      temporal_fill=TemporalFill.INTERPOLATE)

    resolved = _resolve_temporal_fill(conn, dst_t_cache, abs_dst=2, max_T=6)
    # Should get two pairs: past (t=0, dist=2) and future (t=4, dist=2)
    assert len(resolved) == 2
    # Equal distance → equal weight (0.5 each)
    assert abs(resolved[0][1] - 0.5) < 1e-5  # w_past
    assert abs(resolved[1][1] - 0.5) < 1e-5  # w_future
    assert resolved[0][0] == [10]  # t=0 indices
    assert resolved[1][0] == [14]  # t=4 indices


def test_temporal_fill_interpolate_asymmetric():
    """INTERPOLATE: weights are proportional to distance from other endpoint."""
    from canvas_engineering.connectivity import _resolve_temporal_fill

    # slow at t=0 and t=4, query at t=1 (closer to past)
    dst_t_cache = {0: [10], 4: [14]}
    conn = Connection(src="fast", dst="slow", t_src=0, t_dst=0,
                      temporal_fill=TemporalFill.INTERPOLATE)

    resolved = _resolve_temporal_fill(conn, dst_t_cache, abs_dst=1, max_T=6)
    assert len(resolved) == 2
    # past_d=1, future_d=3, span=4
    # w_past = future_d/span = 3/4 = 0.75 (closer to past → more weight on past)
    # w_future = past_d/span = 1/4 = 0.25
    assert abs(resolved[0][1] - 0.75) < 1e-5
    assert abs(resolved[1][1] - 0.25) < 1e-5


def test_temporal_fill_interpolate_only_past():
    """INTERPOLATE: falls back to HOLD when only past is available."""
    from canvas_engineering.connectivity import _resolve_temporal_fill

    dst_t_cache = {0: [10]}  # only past
    conn = Connection(src="fast", dst="slow", t_src=0, t_dst=0,
                      temporal_fill=TemporalFill.INTERPOLATE)

    resolved = _resolve_temporal_fill(conn, dst_t_cache, abs_dst=3, max_T=6)
    assert len(resolved) == 1
    assert resolved[0][0] == [10]
    assert resolved[0][1] == 1.0  # full weight (hold fallback)



def test_temporal_fill_interpolate_period_mismatch():
    """INTERPOLATE with period-mismatched regions: real-time gaps enable lerp.

    fast (period=1) queries slow (period=4). slow has canvas frames 0,1
    mapping to real times 0,4. At fast frame 2 (real time 2), INTERPOLATE
    should lerp between slow real t=0 and real t=4 with 50/50 weights.
    """
    layout = CanvasLayout(T=8, H=2, W=1, d_model=16, regions={
        "fast": (0, 8, 0, 1, 0, 1),  # period=1, real times 0-7
        "slow": RegionSpec(bounds=(0, 2, 1, 2, 0, 1), period=4),  # real times 0,4
    })
    topo = CanvasTopology(connections=[
        Connection(src="fast", dst="slow", t_src=0, t_dst=0,
                   temporal_fill=TemporalFill.INTERPOLATE)
    ])
    mask = topo.to_attention_mask(layout)

    fast_t0 = layout.region_indices_at_t("fast", 0)
    fast_t2 = layout.region_indices_at_t("fast", 2)
    fast_t3 = layout.region_indices_at_t("fast", 3)
    slow_t0 = layout.region_indices_at_t("slow", 0)  # real time 0
    slow_t1 = layout.region_indices_at_t("slow", 1)  # real time 4

    # fast t=0 (real 0) → slow at real 0 exists → direct, weight=1.0
    assert mask[fast_t0[0], slow_t0[0]].item() == 1.0

    # fast t=2 (real 2) → slow at real 2 doesn't exist
    # INTERPOLATE: past=real 0 (dist=2), future=real 4 (dist=2)
    # w_past = 2/4 = 0.5, w_future = 2/4 = 0.5
    assert abs(mask[fast_t2[0], slow_t0[0]].item() - 0.5) < 1e-5
    assert abs(mask[fast_t2[0], slow_t1[0]].item() - 0.5) < 1e-5

    # fast t=3 (real 3) → slow at real 3 doesn't exist
    # INTERPOLATE: past=real 0 (dist=3), future=real 4 (dist=1)
    # w_past = 1/4 = 0.25, w_future = 3/4 = 0.75
    assert abs(mask[fast_t3[0], slow_t0[0]].item() - 0.25) < 1e-5
    assert abs(mask[fast_t3[0], slow_t1[0]].item() - 0.75) < 1e-5


def test_temporal_fill_hold_period_mismatch():
    """HOLD with period-mismatched regions: holds most recent real-time value."""
    layout = CanvasLayout(T=8, H=2, W=1, d_model=16, regions={
        "fast": (0, 8, 0, 1, 0, 1),
        "slow": RegionSpec(bounds=(0, 2, 1, 2, 0, 1), period=4),
    })
    topo = CanvasTopology(connections=[
        Connection(src="fast", dst="slow", t_src=0, t_dst=0,
                   temporal_fill=TemporalFill.HOLD)
    ])
    mask = topo.to_attention_mask(layout)

    fast_t2 = layout.region_indices_at_t("fast", 2)
    slow_t0 = layout.region_indices_at_t("slow", 0)  # real time 0
    slow_t1 = layout.region_indices_at_t("slow", 1)  # real time 4

    # fast t=2 (real 2): slow at real 2 doesn't exist → HOLD to real 0
    assert mask[fast_t2[0], slow_t0[0]].item() == 1.0
    assert mask[fast_t2[0], slow_t1[0]].item() == 0.0  # future, not held

    # fast t=5 (real 5): slow at real 5 doesn't exist → HOLD to real 4
    fast_t5 = layout.region_indices_at_t("fast", 5)
    assert mask[fast_t5[0], slow_t1[0]].item() == 1.0  # held to slow canvas t=1 (real 4)



def test_temporal_fill_connection_defaults():
    """Connection defaults to HOLD fill and interpolation_order=1; no decay_halflife."""
    c = Connection(src="a", dst="b")
    assert c.temporal_fill == TemporalFill.HOLD
    assert c.interpolation_order == 1
    assert not hasattr(c, "decay_halflife")


def test_temporal_fill_summary_reports_fill_modes():
    """Summary should report fill mode counts for temporal connections."""
    topo = CanvasTopology(connections=[
        Connection(src="a", dst="b", t_src=0, t_dst=-1, temporal_fill=TemporalFill.HOLD),
        Connection(src="a", dst="b", t_src=0, t_dst=0, temporal_fill=TemporalFill.INTERPOLATE),
        Connection(src="b", dst="a", t_src=0, t_dst=-1, temporal_fill=TemporalFill.HOLD),
    ])
    s = topo.summary()
    assert "hold=2" in s
    assert "interpolate=1" in s


def test_causal_temporal_with_fill_mode():
    """causal_temporal constructor propagates fill mode."""
    topo = CanvasTopology.causal_temporal(
        ["a", "b"], temporal_fill=TemporalFill.INTERPOLATE
    )
    for c in topo.connections:
        assert c.temporal_fill == TemporalFill.INTERPOLATE


def test_interpolation_order2_idw_weights():
    """INTERPOLATE with order=2: IDW uses 1/dist^2 weights over nearest anchors.

    slow region has 3 anchors at real times 0, 4, 8. order=2 means we take
    up to order+1=3 nearest anchors and weight them by 1/dist^2, normalized.

    For order=2, at real time 2 (missing):
      - all 3 anchors available: t=0 (dist=2), t=4 (dist=2), t=8 (dist=6)
      - raw weights: 1/4, 1/4, 1/36
      - total = 9/36 + 9/36 + 1/36 = 19/36
      - normalized: 9/19, 9/19, 1/19

    For order=2, at real time 6 (missing):
      - all 3 anchors: t=4 (dist=2), t=8 (dist=2), t=0 (dist=6)
      - same weight pattern: 9/19, 9/19, 1/19
    """
    from canvas_engineering.connectivity import _resolve_temporal_fill

    # slow has real times 0, 4, 8
    dst_t_cache = {0: [10], 4: [14], 8: [18]}
    conn = Connection(
        src="fast", dst="slow", t_src=0, t_dst=0,
        temporal_fill=TemporalFill.INTERPOLATE,
        interpolation_order=2,
    )

    # Query at real time 2 (missing): all 3 anchors within order+1=3 slots
    # sorted by dist: t=0(dist=2), t=4(dist=2), t=8(dist=6)
    # raw_weights: 1/4, 1/4, 1/36 → total=19/36 → normalized: 9/19, 9/19, 1/19
    resolved = _resolve_temporal_fill(conn, dst_t_cache, abs_dst=2, max_T=12)
    assert len(resolved) == 3
    weights = {tuple(idxs): w for idxs, w in resolved}
    expected_close = 9.0 / 19.0
    expected_far = 1.0 / 19.0
    assert abs(weights[(10,)] - expected_close) < 1e-5, weights
    assert abs(weights[(14,)] - expected_close) < 1e-5, weights
    assert abs(weights[(18,)] - expected_far) < 1e-5, weights

    # Query at real time 6 (missing): sorted: t=4(dist=2), t=8(dist=2), t=0(dist=6)
    # same weight pattern
    resolved6 = _resolve_temporal_fill(conn, dst_t_cache, abs_dst=6, max_T=12)
    assert len(resolved6) == 3
    w_map = {tuple(idxs): w for idxs, w in resolved6}
    assert abs(w_map[(14,)] - expected_close) < 1e-5, w_map
    assert abs(w_map[(18,)] - expected_close) < 1e-5, w_map
    assert abs(w_map[(10,)] - expected_far) < 1e-5, w_map

    # Query at existing real time 4 → direct connection, weight=1.0
    resolved_direct = _resolve_temporal_fill(conn, dst_t_cache, abs_dst=4, max_T=12)
    assert len(resolved_direct) == 1
    assert resolved_direct[0][1] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

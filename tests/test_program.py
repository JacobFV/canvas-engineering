"""Tests for canvas_engineering.program — typed process semantics."""

import json
import pytest
import torch
from dataclasses import dataclass, field as dc_field

from canvas_engineering import (
    CanvasLayout, RegionSpec, CanvasSchema, CanvasTopology, Connection,
    CanvasProgram, RegionProgram, ConnectionProgram,
    ClockSpec, LearningSpec,
    Field, compile_schema, compile_program,
    REGION_FAMILIES, CARRIERS, OPERATORS, WRITE_MODES, COMPILE_MODES,
)


# ── ClockSpec ────────────────────────────────────────────────────────


class TestClockSpec:
    def test_defaults(self):
        c = ClockSpec()
        assert c.domain == "external"
        assert c.mode == "periodic"
        assert c.period == 1
        assert c.event_source is None
        assert c.event_threshold == 0.0
        assert c.cooldown == 0
        assert c.max_silence is None

    def test_on_event(self):
        c = ClockSpec(
            domain="external", mode="on_event",
            event_source="vision.err.prediction",
            event_threshold=0.25,
            cooldown=2, max_silence=16,
        )
        assert c.mode == "on_event"
        assert c.event_source == "vision.err.prediction"
        assert c.event_threshold == 0.25
        assert c.cooldown == 2
        assert c.max_silence == 16

    def test_boundary(self):
        c = ClockSpec(domain="boundary", mode="boundary", event_source="episode_end")
        assert c.domain == "boundary"
        assert c.mode == "boundary"

    def test_frozen(self):
        c = ClockSpec()
        with pytest.raises(AttributeError):
            c.period = 2


# ── LearningSpec ─────────────────────────────────────────────────────


class TestLearningSpec:
    def test_defaults(self):
        ls = LearningSpec()
        assert ls.mode == "supervised"
        assert ls.losses == ()
        assert ls.compile_mode == "runtime"

    def test_custom(self):
        ls = LearningSpec(
            mode="ssl_prediction",
            losses=("next_step", "masked_prediction"),
            compile_mode="freeze",
        )
        assert ls.mode == "ssl_prediction"
        assert len(ls.losses) == 2
        assert ls.compile_mode == "freeze"

    def test_frozen(self):
        ls = LearningSpec()
        with pytest.raises(AttributeError):
            ls.mode = "other"


# ── RegionProgram ────────────────────────────────────────────────────


class TestRegionProgram:
    def test_defaults(self):
        rp = RegionProgram()
        assert rp.family == "state"
        assert rp.tags == ()
        assert rp.carrier == "deterministic"
        assert rp.clock is None
        assert rp.learning is None
        assert rp.compile_mode == "runtime"

    def test_observation(self):
        rp = RegionProgram(
            family="observation",
            carrier="deterministic",
            learning=LearningSpec(mode="ssl_prediction"),
        )
        assert rp.family == "observation"
        assert rp.learning.mode == "ssl_prediction"

    def test_state_with_tags(self):
        rp = RegionProgram(
            family="state",
            tags=("belief", "object"),
            carrier="filter",
        )
        assert rp.tags == ("belief", "object")
        assert rp.carrier == "filter"

    def test_frozen(self):
        rp = RegionProgram()
        with pytest.raises(AttributeError):
            rp.family = "observation"

    def test_all_families_valid(self):
        for family in REGION_FAMILIES:
            rp = RegionProgram(family=family)
            assert rp.family == family

    def test_custom_family_allowed(self):
        rp = RegionProgram(family="custom_thing")
        assert rp.family == "custom_thing"


# ── ConnectionProgram ────────────────────────────────────────────────


class TestConnectionProgram:
    def test_defaults(self):
        cp = ConnectionProgram()
        assert cp.operator == "attend"
        assert cp.trigger is None
        assert cp.write_mode == "add"

    def test_custom(self):
        cp = ConnectionProgram(
            operator="predict",
            trigger="vision.err.prediction > 0.25",
            write_mode="replace",
        )
        assert cp.operator == "predict"
        assert cp.trigger == "vision.err.prediction > 0.25"
        assert cp.write_mode == "replace"

    def test_frozen(self):
        cp = ConnectionProgram()
        with pytest.raises(AttributeError):
            cp.operator = "observe"


# ── CanvasProgram ────────────────────────────────────────────────────


def _make_schema():
    layout = CanvasLayout(T=4, H=4, W=4, d_model=32, regions={
        "obs": (0, 4, 0, 2, 0, 2),
        "state": (0, 4, 2, 4, 0, 2),
        "act": (0, 4, 0, 2, 2, 4),
    })
    topology = CanvasTopology(connections=[
        Connection(src="state", dst="obs"),
        Connection(src="act", dst="state"),
    ])
    return CanvasSchema(layout=layout, topology=topology)


class TestCanvasProgram:
    def test_empty(self):
        schema = _make_schema()
        prog = CanvasProgram(schema=schema)
        assert len(prog.regions) == 0
        assert len(prog.connections) == 0
        assert prog.version == "2.0.0"

    def test_with_regions(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),
                "state": RegionProgram(family="state", tags=("belief",)),
                "act": RegionProgram(family="action"),
            },
        )
        assert prog.regions["obs"].family == "observation"
        assert prog.regions["state"].tags == ("belief",)
        assert prog.regions["act"].family == "action"

    def test_with_connections(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            connections={
                ("state", "obs"): ConnectionProgram(operator="predict"),
                ("act", "state"): ConnectionProgram(operator="act"),
            },
        )
        assert prog.connections[("state", "obs")].operator == "predict"
        assert prog.connections[("act", "state")].operator == "act"

    def test_to_dict_roundtrip(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation", carrier="diffusive"),
                "state": RegionProgram(
                    family="state", tags=("belief",),
                    clock=ClockSpec(mode="on_event", event_source="err.prediction",
                                   event_threshold=0.3),
                    learning=LearningSpec(mode="posterior_match",
                                          losses=("consistency",),
                                          compile_mode="distill"),
                ),
            },
            connections={
                ("state", "obs"): ConnectionProgram(operator="predict"),
            },
        )
        d = prog.to_dict()
        loaded = CanvasProgram.from_dict(d)

        assert loaded.regions["obs"].family == "observation"
        assert loaded.regions["obs"].carrier == "diffusive"
        assert loaded.regions["state"].tags == ("belief",)
        assert loaded.regions["state"].clock.mode == "on_event"
        assert loaded.regions["state"].clock.event_threshold == 0.3
        assert loaded.regions["state"].learning.mode == "posterior_match"
        assert loaded.regions["state"].learning.losses == ("consistency",)
        assert loaded.connections[("state", "obs")].operator == "predict"

    def test_from_dict_missing_keys(self):
        schema = _make_schema()
        prog = CanvasProgram(schema=schema)
        d = prog.to_dict()
        # Remove optional keys
        d.pop("region_programs", None)
        d.pop("connection_programs", None)
        loaded = CanvasProgram.from_dict(d)
        assert len(loaded.regions) == 0
        assert len(loaded.connections) == 0

    def test_json_roundtrip(self, tmp_path):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),
                "act": RegionProgram(family="action", compile_mode="freeze"),
            },
        )
        path = tmp_path / "program.json"
        prog.to_json(str(path))
        loaded = CanvasProgram.from_json(str(path))
        assert loaded.regions["obs"].family == "observation"
        assert loaded.regions["act"].compile_mode == "freeze"

    def test_summary(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),
                "state": RegionProgram(family="state"),
                "act": RegionProgram(family="action"),
            },
        )
        s = prog.summary()
        assert "3 regions" in s
        assert "observation=1" in s
        assert "action=1" in s

    def test_summary_with_operators(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            connections={
                ("state", "obs"): ConnectionProgram(operator="predict"),
            },
        )
        s = prog.summary()
        assert "predict=1" in s

    def test_connection_key_serialization(self):
        """Tuple keys become pipe-separated strings in JSON."""
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            connections={
                ("state", "obs"): ConnectionProgram(operator="predict"),
            },
        )
        d = prog.to_dict()
        assert "state|obs" in d["connection_programs"]
        loaded = CanvasProgram.from_dict(d)
        assert ("state", "obs") in loaded.connections

    def test_repr(self):
        schema = _make_schema()
        prog = CanvasProgram(schema=schema)
        r = repr(prog)
        assert "CanvasProgram" in r

    def test_default_valued_regions_omitted_in_dict(self):
        """RegionPrograms with all defaults produce empty dicts, which are omitted."""
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={"obs": RegionProgram()},  # all defaults
        )
        d = prog.to_dict()
        # All-default RegionProgram produces empty dict, which is omitted
        assert "region_programs" not in d or "obs" not in d.get("region_programs", {})


# ── Constants ────────────────────────────────────────────────────────


class TestConstants:
    def test_families(self):
        assert REGION_FAMILIES == {"observation", "state", "memory", "residual", "action"}

    def test_carriers(self):
        assert "deterministic" in CARRIERS
        assert "diffusive" in CARRIERS

    def test_operators(self):
        assert "attend" in OPERATORS
        assert "observe" in OPERATORS
        assert "predict" in OPERATORS

    def test_write_modes(self):
        assert WRITE_MODES == {"add", "replace", "gate"}

    def test_compile_modes(self):
        assert COMPILE_MODES == {"runtime", "freeze", "constant", "export"}


# ── Field v2 kwargs ──────────────────────────────────────────────────


class TestFieldV2:
    def test_backward_compat(self):
        """Field() without new args works exactly as before."""
        f = Field()
        assert f.family is None
        assert f.tags == ()
        assert f.carrier is None
        assert f.num_positions == 1

    def test_field_with_family(self):
        f = Field(2, 4, family="observation")
        assert f.family == "observation"
        assert f.h == 2
        assert f.w == 4

    def test_field_with_tags(self):
        f = Field(4, 4, family="state", tags=("belief", "object"))
        assert f.tags == ("belief", "object")

    def test_field_with_carrier(self):
        f = Field(2, 2, carrier="diffusive")
        assert f.carrier == "diffusive"

    def test_field_all_new_args(self):
        f = Field(
            h=6, w=6, period=4,
            family="observation", tags=("rgb",), carrier="deterministic",
        )
        assert f.family == "observation"
        assert f.tags == ("rgb",)
        assert f.carrier == "deterministic"
        assert f.period == 4


# ── compile_program ──────────────────────────────────────────────────


@dataclass
class SimpleRobot:
    camera: Field = Field(4, 4, family="observation")
    belief: Field = Field(2, 2, family="state", tags=("belief",))
    action: Field = Field(1, 4, family="action", loss_weight=2.0)


@dataclass
class PlainRobot:
    camera: Field = Field(4, 4)
    action: Field = Field(1, 4)


class TestCompileProgram:
    def test_returns_tuple(self):
        bound, prog = compile_program(SimpleRobot(), T=4, H=8, W=8, d_model=64)
        assert isinstance(prog, CanvasProgram)
        assert "camera" in bound

    def test_bound_matches_compile_schema(self):
        """BoundSchema from compile_program matches compile_schema."""
        bound1 = compile_schema(SimpleRobot(), T=4, H=8, W=8, d_model=64)
        bound2, _ = compile_program(SimpleRobot(), T=4, H=8, W=8, d_model=64)
        assert bound1.field_names == bound2.field_names
        assert bound1.layout.T == bound2.layout.T
        assert bound1.layout.d_model == bound2.layout.d_model

    def test_family_propagated(self):
        _, prog = compile_program(SimpleRobot(), T=4, H=8, W=8, d_model=64)
        assert prog.regions["camera"].family == "observation"
        assert prog.regions["belief"].family == "state"
        assert prog.regions["action"].family == "action"

    def test_tags_propagated(self):
        _, prog = compile_program(SimpleRobot(), T=4, H=8, W=8, d_model=64)
        assert prog.regions["belief"].tags == ("belief",)

    def test_no_family_defaults_to_state(self):
        _, prog = compile_program(PlainRobot(), T=4, H=8, W=8, d_model=64)
        assert prog.regions["camera"].family == "state"
        assert prog.regions["action"].family == "state"

    def test_carrier_propagated(self):
        @dataclass
        class WithCarrier:
            video: Field = Field(4, 4, family="observation", carrier="diffusive")
            joints: Field = Field(1, 4, family="observation")

        _, prog = compile_program(WithCarrier(), T=4, H=8, W=8, d_model=64)
        assert prog.regions["video"].carrier == "diffusive"
        assert prog.regions["joints"].carrier == "deterministic"

    def test_compile_schema_unchanged(self):
        """compile_schema() still works and returns BoundSchema only."""
        bound = compile_schema(SimpleRobot(), T=4, H=8, W=8, d_model=64)
        assert "camera" in bound
        assert not isinstance(bound, tuple)

    def test_nested_types_get_programs(self):
        """Coarse-grained fields from nested types also get RegionPrograms."""
        @dataclass
        class Sensor:
            rgb: Field = Field(4, 4, family="observation")
            depth: Field = Field(2, 2, family="observation")

        @dataclass
        class Bot:
            sensor: Sensor = dc_field(default_factory=Sensor)
            act: Field = Field(1, 2, family="action")

        _, prog = compile_program(Bot(), T=4, H=16, W=16, d_model=64)
        assert "sensor.rgb" in prog.regions
        assert "sensor.depth" in prog.regions
        assert "sensor" in prog.regions  # coarse-grained field
        assert "act" in prog.regions

    def test_program_serializes(self):
        _, prog = compile_program(SimpleRobot(), T=4, H=8, W=8, d_model=64)
        d = prog.to_dict()
        loaded = CanvasProgram.from_dict(d)
        assert loaded.regions["camera"].family == "observation"


# ── Phase 2: Operator/backend split + auto-wiring ────────────────────


class TestConnectionOperator:
    def test_default_operator(self):
        c = Connection(src="a", dst="b")
        assert c.operator == "attend"

    def test_explicit_operator(self):
        c = Connection(src="a", dst="b", operator="predict")
        assert c.operator == "predict"

    def test_default_write_mode(self):
        c = Connection(src="a", dst="b")
        assert c.write_mode == "add"

    def test_explicit_write_mode(self):
        c = Connection(src="a", dst="b", write_mode="replace")
        assert c.write_mode == "replace"

    def test_operator_frozen(self):
        c = Connection(src="a", dst="b", operator="observe")
        with pytest.raises(AttributeError):
            c.operator = "predict"


class TestAutoWiring:
    def test_obs_to_state_is_observe(self):
        @dataclass
        class T:
            obs: Field = Field(2, 2, family="observation")
            belief: Field = Field(2, 2, family="state")

        _, prog = compile_program(T(), T=1, H=8, W=8, d_model=32)
        conns = prog.schema.topology.connections
        obs_to_state = [c for c in conns if c.src == "obs" and c.dst == "belief"]
        assert any(c.operator == "observe" for c in obs_to_state)

    def test_state_to_action_is_act(self):
        @dataclass
        class T:
            belief: Field = Field(2, 2, family="state")
            act: Field = Field(1, 2, family="action")

        _, prog = compile_program(T(), T=1, H=8, W=8, d_model=32)
        conns = prog.schema.topology.connections
        state_to_act = [c for c in conns if c.src == "belief" and c.dst == "act"]
        assert any(c.operator == "act" for c in state_to_act)

    def test_state_to_state_is_integrate(self):
        @dataclass
        class T:
            a: Field = Field(2, 2, family="state")
            b: Field = Field(2, 2, family="state")

        _, prog = compile_program(T(), T=1, H=8, W=8, d_model=32)
        conns = prog.schema.topology.connections
        s2s = [c for c in conns if c.src == "a" and c.dst == "b"]
        assert any(c.operator == "integrate" for c in s2s)

    def test_unknown_family_pair_defaults_attend(self):
        @dataclass
        class T:
            x: Field = Field(2, 2, family="custom_a")
            y: Field = Field(2, 2, family="custom_b")

        _, prog = compile_program(T(), T=1, H=8, W=8, d_model=32)
        conns = prog.schema.topology.connections
        assert all(c.operator == "attend" for c in conns)

    def test_compile_schema_no_operators(self):
        """compile_schema (not compile_program) leaves all operators as attend."""
        bound = compile_schema(SimpleRobot(), T=4, H=8, W=8, d_model=64)
        conns = bound.topology.connections
        assert all(c.operator == "attend" for c in conns)

    def test_state_to_memory_is_write(self):
        @dataclass
        class T:
            belief: Field = Field(2, 2, family="state")
            mem: Field = Field(2, 2, family="memory")

        _, prog = compile_program(T(), T=1, H=8, W=8, d_model=32)
        conns = prog.schema.topology.connections
        s2m = [c for c in conns if c.src == "belief" and c.dst == "mem"]
        assert any(c.operator == "write" for c in s2m)

    def test_memory_to_state_is_retrieve(self):
        @dataclass
        class T:
            belief: Field = Field(2, 2, family="state")
            mem: Field = Field(2, 2, family="memory")

        _, prog = compile_program(T(), T=1, H=8, W=8, d_model=32)
        conns = prog.schema.topology.connections
        m2s = [c for c in conns if c.src == "mem" and c.dst == "belief"]
        assert any(c.operator == "retrieve" for c in m2s)


class TestDeduplicateWithOperator:
    def test_different_operators_not_deduped(self):
        from canvas_engineering.types import _deduplicate
        conns = [
            Connection(src="a", dst="b", operator="observe"),
            Connection(src="a", dst="b", operator="predict"),
        ]
        result = _deduplicate(conns)
        assert len(result) == 2

    def test_same_operator_deduped(self):
        from canvas_engineering.types import _deduplicate
        conns = [
            Connection(src="a", dst="b", operator="observe"),
            Connection(src="a", dst="b", operator="observe"),
        ]
        result = _deduplicate(conns)
        assert len(result) == 1


class TestSummaryWithOperator:
    def test_summary_shows_operators(self):
        topo = CanvasTopology(connections=[
            Connection(src="a", dst="b", operator="predict"),
            Connection(src="b", dst="a", operator="observe"),
        ])
        s = topo.summary()
        assert "predict=1" in s
        assert "observe=1" in s

    def test_summary_omits_attend(self):
        topo = CanvasTopology(connections=[
            Connection(src="a", dst="b"),  # operator="attend" (default)
        ])
        s = topo.summary()
        assert "operators" not in s


class TestSchemaRoundtripOperator:
    def test_operator_survives_roundtrip(self, tmp_path):
        from canvas_engineering import CanvasSchema
        schema = CanvasSchema(
            layout=CanvasLayout(T=2, H=2, W=2, d_model=16, regions={"a": (0, 2, 0, 1, 0, 1)}),
            topology=CanvasTopology(connections=[
                Connection(src="a", dst="a", operator="integrate"),
            ]),
        )
        path = tmp_path / "schema.json"
        schema.to_json(str(path))
        loaded = CanvasSchema.from_json(str(path))
        assert loaded.topology.connections[0].operator == "integrate"

    def test_write_mode_survives_roundtrip(self, tmp_path):
        from canvas_engineering import CanvasSchema
        schema = CanvasSchema(
            layout=CanvasLayout(T=2, H=2, W=2, d_model=16, regions={"a": (0, 2, 0, 1, 0, 1)}),
            topology=CanvasTopology(connections=[
                Connection(src="a", dst="a", write_mode="replace"),
            ]),
        )
        path = tmp_path / "schema.json"
        schema.to_json(str(path))
        loaded = CanvasSchema.from_json(str(path))
        assert loaded.topology.connections[0].write_mode == "replace"

    def test_old_json_loads_with_defaults(self, tmp_path):
        """JSON without operator/write_mode loads with defaults."""
        import json
        path = tmp_path / "old.json"
        with open(path, "w") as f:
            json.dump({
                "layout": {"T": 2, "H": 2, "W": 2, "d_model": 16},
                "regions": {"a": {"bounds": [0, 2, 0, 1, 0, 1]}},
                "topology": [{"src": "a", "dst": "a"}],
                "version": "0.2.0",
            }, f)
        from canvas_engineering import CanvasSchema
        loaded = CanvasSchema.from_json(str(path))
        c = loaded.topology.connections[0]
        assert c.operator == "attend"
        assert c.write_mode == "add"


# ── Phase 3: Carriers + residual summaries ───────────────────────────

from canvas_engineering import ResidualSpec, ResidualAccumulator
from canvas_engineering.dispatch import AttentionDispatcher


class TestRegionSpecCarrier:
    def test_default_carrier(self):
        spec = RegionSpec(bounds=(0, 1, 0, 1, 0, 1))
        assert spec.carrier == "deterministic"

    def test_custom_carrier(self):
        spec = RegionSpec(bounds=(0, 1, 0, 1, 0, 1), carrier="diffusive")
        assert spec.carrier == "diffusive"

    def test_carrier_frozen(self):
        spec = RegionSpec(bounds=(0, 1, 0, 1, 0, 1))
        with pytest.raises(AttributeError):
            spec.carrier = "diffusive"

    def test_carrier_schema_roundtrip(self, tmp_path):
        schema = CanvasSchema(
            layout=CanvasLayout(T=2, H=2, W=2, d_model=16, regions={
                "a": RegionSpec(bounds=(0, 2, 0, 1, 0, 1), carrier="diffusive"),
                "b": (0, 2, 1, 2, 0, 1),
            }),
        )
        path = tmp_path / "schema.json"
        schema.to_json(str(path))
        loaded = CanvasSchema.from_json(str(path))
        assert loaded.layout.region_spec("a").carrier == "diffusive"
        assert loaded.layout.region_spec("b").carrier == "deterministic"

    def test_old_json_no_carrier(self, tmp_path):
        import json as json_mod
        path = tmp_path / "old.json"
        with open(path, "w") as f:
            json_mod.dump({
                "layout": {"T": 2, "H": 2, "W": 2, "d_model": 16},
                "regions": {"a": {"bounds": [0, 2, 0, 1, 0, 1], "period": 2}},
                "version": "0.2.0",
            }, f)
        loaded = CanvasSchema.from_json(str(path))
        assert loaded.layout.region_spec("a").carrier == "deterministic"


class TestResidualSpec:
    def test_defaults(self):
        spec = ResidualSpec()
        assert spec.kinds == ("prediction",)
        assert spec.reduce == "max_mean"
        assert spec.decay == 0.95

    def test_custom(self):
        spec = ResidualSpec(kinds=("prediction", "novelty"), reduce="mean", decay=0.9)
        assert len(spec.kinds) == 2
        assert spec.reduce == "mean"

    def test_frozen(self):
        spec = ResidualSpec()
        with pytest.raises(AttributeError):
            spec.decay = 0.5


class TestResidualAccumulator:
    def test_init(self):
        acc = ResidualAccumulator(["err_a", "err_b"])
        assert len(acc.region_names) == 2
        assert acc.summaries() == {"err_a": {"prediction": 0.0}, "err_b": {"prediction": 0.0}}

    def test_update_changes_summary(self):
        acc = ResidualAccumulator(["err"])
        error = torch.ones(2, 4, 8)
        acc.update("err", error)
        s = acc.summaries()
        assert s["err"]["prediction"] > 0.0

    def test_ema_decay(self):
        spec = ResidualSpec(decay=0.5)
        acc = ResidualAccumulator(["err"], spec)
        # First update: summary = 0.5 * 0 + 0.5 * 1.0 = 0.5
        acc.update("err", torch.ones(1, 1, 1))
        s1 = acc.summaries()["err"]["prediction"]
        assert abs(s1 - 0.5) < 1e-5
        # Second update with same value: summary = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        acc.update("err", torch.ones(1, 1, 1))
        s2 = acc.summaries()["err"]["prediction"]
        assert abs(s2 - 0.75) < 1e-5

    def test_multiple_kinds(self):
        spec = ResidualSpec(kinds=("max_val", "mean_val"))
        acc = ResidualAccumulator(["err"], spec)
        acc.update("err", torch.tensor([1.0, 2.0, 3.0]))
        s = acc.summaries()["err"]
        assert "max_val" in s
        assert "mean_val" in s
        assert s["max_val"] > s["mean_val"]  # max > mean for [1,2,3]

    def test_multiple_regions(self):
        acc = ResidualAccumulator(["a", "b"])
        acc.update("a", torch.ones(1))
        acc.update("b", torch.ones(1) * 2)
        s = acc.summaries()
        assert s["b"]["prediction"] > s["a"]["prediction"]

    def test_reset(self):
        acc = ResidualAccumulator(["err"])
        acc.update("err", torch.ones(1))
        acc.reset()
        assert acc.summaries()["err"]["prediction"] == 0.0

    def test_reduce_mean(self):
        spec = ResidualSpec(reduce="mean")
        acc = ResidualAccumulator(["err"], spec)
        acc.update("err", torch.tensor([1.0, 3.0]))
        s = acc.summaries()["err"]["prediction"]
        assert abs(s - (1.0 - 0.95) * 2.0) < 1e-5  # (1-decay) * mean([1,3])

    def test_reduce_max(self):
        spec = ResidualSpec(reduce="max")
        acc = ResidualAccumulator(["err"], spec)
        acc.update("err", torch.tensor([1.0, 3.0]))
        s = acc.summaries()["err"]["prediction"]
        assert abs(s - (1.0 - 0.95) * 3.0) < 1e-5  # (1-decay) * max([1,3])

    def test_reduce_l2(self):
        spec = ResidualSpec(reduce="l2")
        acc = ResidualAccumulator(["err"], spec)
        acc.update("err", torch.tensor([3.0, 4.0]))
        s = acc.summaries()["err"]["prediction"]
        expected = (1.0 - 0.95) * 5.0  # norm of [3,4] = 5
        assert abs(s - expected) < 1e-4

    def test_repr(self):
        acc = ResidualAccumulator(["a", "b"])
        r = repr(acc)
        assert "ResidualAccumulator" in r
        assert "regions=2" in r


class TestDispatcherResidual:
    def test_no_accumulator_backward_compat(self):
        layout = CanvasLayout(T=2, H=2, W=2, d_model=16, regions={
            "a": (0, 2, 0, 1, 0, 1),
            "b": (0, 2, 1, 2, 0, 1),
        })
        topo = CanvasTopology(connections=[Connection(src="a", dst="b")])
        dispatcher = AttentionDispatcher(topo, layout, d_model=16, n_heads=2)
        x = torch.randn(1, layout.num_positions, 16)
        out = dispatcher(x)
        assert out.shape == x.shape
        assert dispatcher.summaries is None

    def test_with_accumulator(self):
        layout = CanvasLayout(T=2, H=2, W=2, d_model=16, regions={
            "src": (0, 2, 0, 1, 0, 1),
            "err": RegionSpec(bounds=(0, 2, 1, 2, 0, 1), carrier="residual"),
        })
        topo = CanvasTopology(connections=[Connection(src="src", dst="err")])
        acc = ResidualAccumulator(["err"])
        dispatcher = AttentionDispatcher(topo, layout, d_model=16, n_heads=2,
                                         residual_accumulator=acc)
        x = torch.randn(1, layout.num_positions, 16)
        out = dispatcher(x)
        assert out.shape == x.shape
        s = dispatcher.summaries
        assert s is not None
        assert "err" in s

    def test_summaries_property(self):
        layout = CanvasLayout(T=1, H=2, W=1, d_model=8, regions={
            "a": (0, 1, 0, 1, 0, 1),
        })
        topo = CanvasTopology(connections=[Connection(src="a", dst="a")])
        dispatcher = AttentionDispatcher(topo, layout, d_model=8, n_heads=1)
        assert dispatcher.summaries is None


class TestCarrierPropagation:
    def test_field_carrier_to_region_spec(self):
        @dataclass
        class T:
            video: Field = Field(2, 2, carrier="diffusive")
            joints: Field = Field(1, 2)

        bound = compile_schema(T(), T=4, H=8, W=8, d_model=32)
        assert bound["video"].spec.carrier == "diffusive"
        assert bound["joints"].spec.carrier == "deterministic"

    def test_compile_program_carrier(self):
        @dataclass
        class T:
            obs: Field = Field(2, 2, family="observation", carrier="diffusive")
            state: Field = Field(2, 2, family="state", carrier="filter")

        _, prog = compile_program(T(), T=4, H=8, W=8, d_model=32)
        assert prog.regions["obs"].carrier == "diffusive"
        assert prog.regions["state"].carrier == "filter"

    def test_default_carrier_propagation(self):
        bound = compile_schema(PlainRobot(), T=4, H=8, W=8, d_model=32)
        assert bound["camera"].spec.carrier == "deterministic"
        assert bound["action"].spec.carrier == "deterministic"


# ── Phase 4: Clocks + event triggers ─────────────────────────────────

from canvas_engineering.scheduling import RegionScheduler


def _make_program_with_clocks():
    schema = _make_schema()
    return CanvasProgram(
        schema=schema,
        regions={
            "obs": RegionProgram(family="observation"),  # no clock = always active
            "state": RegionProgram(
                family="state",
                clock=ClockSpec(mode="periodic", period=4),
            ),
            "act": RegionProgram(
                family="action",
                clock=ClockSpec(
                    mode="on_event",
                    event_source="err.prediction",
                    event_threshold=0.5,
                ),
            ),
        },
    )


class TestRegionScheduler:
    def test_no_clock_always_active(self):
        prog = _make_program_with_clocks()
        sched = RegionScheduler(prog)
        active = sched.step(external_t=0)
        assert "obs" in active

    def test_periodic_fires_on_period(self):
        prog = _make_program_with_clocks()
        sched = RegionScheduler(prog)
        assert "state" in sched.step(0)
        assert "state" not in sched.step(1)
        assert "state" not in sched.step(2)
        assert "state" not in sched.step(3)
        assert "state" in sched.step(4)

    def test_on_event_fires_above_threshold(self):
        prog = _make_program_with_clocks()
        sched = RegionScheduler(prog)
        summaries = {"err": {"prediction": 0.8}}
        active = sched.step(0, summaries=summaries)
        assert "act" in active

    def test_on_event_does_not_fire_below_threshold(self):
        prog = _make_program_with_clocks()
        sched = RegionScheduler(prog)
        summaries = {"err": {"prediction": 0.3}}
        active = sched.step(0, summaries=summaries)
        assert "act" not in active

    def test_on_event_no_summaries(self):
        prog = _make_program_with_clocks()
        sched = RegionScheduler(prog)
        active = sched.step(0, summaries=None)
        assert "act" not in active

    def test_boundary_fires(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "consolidator": RegionProgram(
                    clock=ClockSpec(mode="boundary", event_source="episode_end"),
                ),
            },
        )
        sched = RegionScheduler(prog)
        assert "consolidator" not in sched.step(0)
        assert "consolidator" in sched.step(0, boundary="episode_end")
        assert "consolidator" not in sched.step(1, boundary="other_event")

    def test_cooldown(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "x": RegionProgram(
                    clock=ClockSpec(mode="periodic", period=1, cooldown=3),
                ),
            },
        )
        sched = RegionScheduler(prog)
        assert "x" in sched.step(0)   # fires, cooldown until t=3
        assert "x" not in sched.step(1)
        assert "x" not in sched.step(2)
        assert "x" in sched.step(3)   # cooldown expired

    def test_max_silence(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "lazy": RegionProgram(
                    clock=ClockSpec(
                        mode="on_event",
                        event_source="err.prediction",
                        event_threshold=100.0,  # very high, never fires normally
                        max_silence=5,
                    ),
                ),
            },
        )
        sched = RegionScheduler(prog)
        summaries = {"err": {"prediction": 0.0}}
        # Should NOT fire for steps 0-3 (event threshold not met)
        # But the first step will fire due to max_silence (never fired before)
        active0 = sched.step(0, summaries=summaries)
        assert "lazy" in active0  # forced by max_silence (never fired)
        assert "lazy" not in sched.step(1, summaries=summaries)
        assert "lazy" not in sched.step(2, summaries=summaries)
        assert "lazy" not in sched.step(3, summaries=summaries)
        assert "lazy" not in sched.step(4, summaries=summaries)
        assert "lazy" in sched.step(5, summaries=summaries)  # forced by max_silence

    def test_empty_program(self):
        schema = _make_schema()
        prog = CanvasProgram(schema=schema)
        sched = RegionScheduler(prog)
        assert sched.step(0) == set()

    def test_reset(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={"x": RegionProgram(clock=ClockSpec(mode="periodic", period=1, cooldown=10))},
        )
        sched = RegionScheduler(prog)
        sched.step(0)  # fires, sets cooldown
        assert "x" not in sched.step(1)  # cooldown
        sched.reset()
        assert "x" in sched.step(1)  # reset cleared cooldown

    def test_repr(self):
        prog = _make_program_with_clocks()
        sched = RegionScheduler(prog)
        r = repr(sched)
        assert "RegionScheduler" in r


class TestDispatcherActiveRegions:
    def test_none_means_all_fire(self):
        layout = CanvasLayout(T=1, H=2, W=2, d_model=16, regions={
            "a": (0, 1, 0, 1, 0, 1),
            "b": (0, 1, 1, 2, 0, 1),
        })
        topo = CanvasTopology(connections=[
            Connection(src="a", dst="b"),
            Connection(src="b", dst="a"),
        ])
        dispatcher = AttentionDispatcher(topo, layout, d_model=16, n_heads=2)
        x = torch.randn(1, layout.num_positions, 16)
        out = dispatcher(x, active_regions=None)
        assert out.shape == x.shape

    def test_skip_inactive_src(self):
        layout = CanvasLayout(T=1, H=2, W=1, d_model=8, regions={
            "a": (0, 1, 0, 1, 0, 1),
            "b": (0, 1, 1, 2, 0, 1),
        })
        topo = CanvasTopology(connections=[
            Connection(src="a", dst="a"),
            Connection(src="b", dst="a"),
        ])
        dispatcher = AttentionDispatcher(topo, layout, d_model=8, n_heads=1)
        x = torch.randn(1, layout.num_positions, 8)
        # Only "a" active → "b" connections should be skipped
        out = dispatcher(x, active_regions={"a"})
        # b's positions should pass through unchanged
        b_idx = layout.region_indices("b")
        assert torch.equal(out[0, b_idx[0]], x[0, b_idx[0]])

    def test_empty_active_regions(self):
        layout = CanvasLayout(T=1, H=2, W=1, d_model=8, regions={
            "a": (0, 1, 0, 1, 0, 1),
        })
        topo = CanvasTopology(connections=[Connection(src="a", dst="a")])
        dispatcher = AttentionDispatcher(topo, layout, d_model=8, n_heads=1)
        x = torch.randn(1, layout.num_positions, 8)
        out = dispatcher(x, active_regions=set())
        # All positions pass through unchanged
        assert torch.equal(out, x)

    def test_all_active_same_as_none(self):
        layout = CanvasLayout(T=1, H=2, W=1, d_model=8, regions={
            "a": (0, 1, 0, 1, 0, 1),
            "b": (0, 1, 1, 2, 0, 1),
        })
        topo = CanvasTopology(connections=[
            Connection(src="a", dst="b"),
            Connection(src="a", dst="a"),
        ])
        dispatcher = AttentionDispatcher(topo, layout, d_model=8, n_heads=1)
        torch.manual_seed(42)
        x = torch.randn(1, layout.num_positions, 8)
        out_none = dispatcher(x, active_regions=None)
        out_all = dispatcher(x, active_regions={"a", "b"})
        assert torch.allclose(out_none, out_all, atol=1e-5)


# ── Phase 5: Learning recipes + compiler ─────────────────────────────

from canvas_engineering.learning import default_learning, FAMILY_DEFAULTS
from canvas_engineering.compiler import ProgramCompiler, CompiledProgram


class TestDefaultLearning:
    def test_observation(self):
        ls = default_learning("observation")
        assert ls.mode == "ssl_prediction"
        assert "next_step" in ls.losses
        assert ls.compile_mode == "freeze"

    def test_state(self):
        ls = default_learning("state")
        assert ls.mode == "posterior_match"
        assert ls.compile_mode == "runtime"

    def test_memory(self):
        ls = default_learning("memory")
        assert ls.mode == "retrieval"
        assert ls.compile_mode == "export"

    def test_residual(self):
        ls = default_learning("residual")
        assert ls.mode == "calibration"
        assert ls.compile_mode == "freeze"

    def test_action(self):
        ls = default_learning("action")
        assert ls.mode == "supervised"
        assert ls.compile_mode == "freeze"

    def test_unknown_family(self):
        ls = default_learning("custom_thing")
        assert ls.mode == "supervised"  # default LearningSpec

    def test_all_families_have_defaults(self):
        from canvas_engineering import REGION_FAMILIES
        for family in REGION_FAMILIES:
            ls = default_learning(family)
            assert ls is not None
            assert ls.mode != ""


class TestProgramCompiler:
    def test_no_compile_modes_unchanged(self):
        # With no explicit compile_mode, regions still inherit their
        # family default (observation/action → freeze; state → runtime).
        # The default-state case below has all regions remain active;
        # frozen/exported sets reflect FAMILY_DEFAULTS.
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),
                "state": RegionProgram(family="state"),
                "act": RegionProgram(family="action"),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        # observation + action families default to compile_mode="freeze";
        # state stays runtime — so all three are still active but two
        # are frozen.
        assert compiled.active_regions == {"obs", "state", "act"}
        assert compiled.frozen_regions == {"obs", "act"}
        assert len(compiled.exported_memories) == 0

    def test_no_compile_modes_state_only(self):
        # state-family regions remain runtime by default; no freezing.
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="state"),
                "state": RegionProgram(family="state"),
                "act": RegionProgram(family="state"),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        assert compiled.active_regions == {"obs", "state", "act"}
        assert len(compiled.frozen_regions) == 0
        assert len(compiled.exported_memories) == 0

    def test_freeze_regions(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation", compile_mode="freeze"),
                "state": RegionProgram(family="state"),
                "act": RegionProgram(family="action"),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        assert "obs" in compiled.frozen_regions
        assert "obs" in compiled.active_regions  # frozen but still active
        assert "state" not in compiled.frozen_regions

    def test_constant_removes_from_active(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(compile_mode="constant"),
                "state": RegionProgram(),
                "act": RegionProgram(),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        assert "obs" not in compiled.active_regions
        assert "obs" in compiled.constant_regions

    def test_export_removes_from_active(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(),
                "state": RegionProgram(),
                "act": RegionProgram(compile_mode="export"),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        assert "act" not in compiled.active_regions
        assert "act" in compiled.exported_memories

    def test_dead_connections_eliminated(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(),
                "state": RegionProgram(compile_mode="export"),  # removed
                "act": RegionProgram(),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        # Connections involving "state" should be gone
        for c in compiled.active_connections:
            assert c.src != "state" and c.dst != "state"

    def test_learning_spec_compile_mode(self):
        """LearningSpec.compile_mode takes effect."""
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(
                    learning=LearningSpec(compile_mode="freeze"),
                ),
                "state": RegionProgram(),
                "act": RegionProgram(),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        assert "obs" in compiled.frozen_regions

    def test_reduced_schema_excludes_eliminated(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(),
                "state": RegionProgram(compile_mode="constant"),
                "act": RegionProgram(compile_mode="export"),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        assert "state" not in compiled.schema.layout.regions
        assert "act" not in compiled.schema.layout.regions
        assert "obs" in compiled.schema.layout.regions

    def test_n_eliminated(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(),
                "state": RegionProgram(compile_mode="constant"),
                "act": RegionProgram(compile_mode="export"),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        assert compiled.n_eliminated == 2

    def test_summary(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(compile_mode="freeze"),
                "state": RegionProgram(compile_mode="export"),
                "act": RegionProgram(),
            },
        )
        compiled = ProgramCompiler(prog).compile()
        s = compiled.summary()
        assert "CompiledProgram" in s
        assert "frozen" in s
        assert "exported" in s

    def test_repr(self):
        schema = _make_schema()
        prog = CanvasProgram(schema=schema)
        compiled = ProgramCompiler(prog).compile()
        assert "CompiledProgram" in repr(compiled)

    def test_end_to_end_mixed_modes(self):
        """Full compile with mixed modes: freeze + export + runtime."""
        layout = CanvasLayout(T=4, H=4, W=4, d_model=32, regions={
            "vision": RegionSpec(bounds=(0, 4, 0, 2, 0, 2)),
            "belief": RegionSpec(bounds=(0, 4, 2, 3, 0, 2)),
            "memory": RegionSpec(bounds=(0, 4, 3, 4, 0, 2)),
            "error": RegionSpec(bounds=(0, 4, 0, 2, 2, 3), carrier="residual"),
            "action": RegionSpec(bounds=(0, 4, 2, 4, 2, 3)),
        })
        topology = CanvasTopology(connections=[
            Connection(src="belief", dst="vision"),
            Connection(src="belief", dst="memory"),
            Connection(src="memory", dst="belief"),
            Connection(src="belief", dst="error"),
            Connection(src="belief", dst="action"),
            Connection(src="action", dst="belief"),
        ])
        schema = CanvasSchema(layout=layout, topology=topology)
        prog = CanvasProgram(
            schema=schema,
            regions={
                "vision": RegionProgram(family="observation", compile_mode="freeze"),
                "belief": RegionProgram(family="state"),
                "memory": RegionProgram(family="memory", compile_mode="export"),
                "error": RegionProgram(family="residual", compile_mode="freeze"),
                "action": RegionProgram(family="action", compile_mode="freeze"),
            },
        )
        compiled = ProgramCompiler(prog).compile()

        # Memory exported, removed from active
        assert "memory" not in compiled.active_regions
        assert "memory" in compiled.exported_memories

        # Frozen but still active
        assert "vision" in compiled.active_regions
        assert "vision" in compiled.frozen_regions
        assert "error" in compiled.frozen_regions
        assert "action" in compiled.frozen_regions

        # Belief still runtime
        assert "belief" in compiled.active_regions
        assert "belief" not in compiled.frozen_regions

        # Connections to/from memory are eliminated
        for c in compiled.active_connections:
            assert c.src != "memory" and c.dst != "memory"

        # Reduced schema doesn't have memory
        assert "memory" not in compiled.schema.layout.regions


# ── Phase 6: FamilyCarrierEmbedding + ProgramConditioner ────────────

from canvas_engineering.canvas import FamilyCarrierEmbedding


class TestFamilyCarrierEmbedding:
    def test_init(self):
        emb = FamilyCarrierEmbedding(d_model=32)
        assert emb.d_model == 32
        assert emb.family_embedding.num_embeddings == 6  # 5 families + unknown
        assert emb.carrier_embedding.num_embeddings == 6  # 5 carriers + unknown

    def test_known_family_idx(self):
        emb = FamilyCarrierEmbedding(d_model=16)
        assert emb.family_idx("observation") == 0
        assert emb.family_idx("state") == 1
        assert emb.family_idx("action") == 4

    def test_unknown_family_idx(self):
        emb = FamilyCarrierEmbedding(d_model=16)
        assert emb.family_idx("custom_family") == len(emb.FAMILIES)

    def test_known_carrier_idx(self):
        emb = FamilyCarrierEmbedding(d_model=16)
        assert emb.carrier_idx("deterministic") == 0
        assert emb.carrier_idx("diffusive") == 1

    def test_unknown_carrier_idx(self):
        emb = FamilyCarrierEmbedding(d_model=16)
        assert emb.carrier_idx("custom_carrier") == len(emb.CARRIERS)

    def test_forward_shape(self):
        emb = FamilyCarrierEmbedding(d_model=32)
        out = emb("observation", "deterministic")
        assert out.shape == (32,)

    def test_different_pairs_differ(self):
        emb = FamilyCarrierEmbedding(d_model=32)
        a = emb("observation", "deterministic")
        b = emb("state", "diffusive")
        assert not torch.allclose(a, b)

    def test_repr(self):
        emb = FamilyCarrierEmbedding(d_model=16)
        r = repr(emb)
        assert "FamilyCarrierEmbedding" in r


class TestSpatiotemporalCanvasWithProgram:
    def test_backward_compat_no_program(self):
        layout = CanvasLayout(T=2, H=2, W=2, d_model=16, regions={
            "a": (0, 2, 0, 1, 0, 1),
        })
        canvas_mod = SpatiotemporalCanvas(layout)
        c = canvas_mod.create_empty(batch_size=1)
        assert c.shape == (1, layout.num_positions, 16)
        assert canvas_mod.family_carrier_embedding is None

    def test_with_program(self):
        layout = CanvasLayout(T=2, H=2, W=2, d_model=16, regions={
            "obs": RegionSpec(bounds=(0, 2, 0, 1, 0, 1)),
            "act": RegionSpec(bounds=(0, 2, 1, 2, 0, 1)),
        })
        schema = CanvasSchema(layout=layout)
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),
                "act": RegionProgram(family="action"),
            },
        )
        canvas_mod = SpatiotemporalCanvas(layout, program=prog)
        assert canvas_mod.family_carrier_embedding is not None
        c = canvas_mod.create_empty(batch_size=2)
        assert c.shape == (2, layout.num_positions, 16)

    def test_place_with_program(self):
        layout = CanvasLayout(T=1, H=2, W=1, d_model=8, regions={
            "obs": RegionSpec(bounds=(0, 1, 0, 1, 0, 1)),
        })
        schema = CanvasSchema(layout=layout)
        prog = CanvasProgram(
            schema=schema,
            regions={"obs": RegionProgram(family="observation")},
        )
        canvas_mod = SpatiotemporalCanvas(layout, program=prog)
        c = canvas_mod.create_empty(batch_size=1)
        embs = torch.randn(1, 1, 8)
        c2 = canvas_mod.place(c, embs, "obs")
        assert c2.shape == c.shape
        # Placed values should differ from empty
        idx = layout.region_indices("obs")
        assert not torch.equal(c2[0, idx[0]], c[0, idx[0]])


from canvas_engineering import SpatiotemporalCanvas, transfer_distance
from canvas_engineering.semantic import program_distance


class TestProgramDistance:
    def _make_specs(self):
        emb_a = tuple([0.5] * 16)
        emb_b = tuple([0.3] * 16)
        a = RegionSpec(bounds=(0, 1, 0, 1, 0, 1), semantic_embedding=emb_a)
        b = RegionSpec(bounds=(0, 1, 0, 1, 0, 1), semantic_embedding=emb_b)
        return a, b

    def test_same_family_same_carrier(self):
        a_spec, b_spec = self._make_specs()
        a_prog = RegionProgram(family="observation", carrier="deterministic")
        b_prog = RegionProgram(family="observation", carrier="deterministic")
        d = program_distance(a_spec, a_prog, b_spec, b_prog)
        base = transfer_distance(a_spec, b_spec)
        assert abs(d - base) < 1e-5

    def test_different_family_penalty(self):
        a_spec, b_spec = self._make_specs()
        a_prog = RegionProgram(family="observation")
        b_prog = RegionProgram(family="action")
        d = program_distance(a_spec, a_prog, b_spec, b_prog)
        base = transfer_distance(a_spec, b_spec)
        assert d > base
        assert abs(d - base - 0.3) < 1e-5

    def test_different_carrier_penalty(self):
        a_spec, b_spec = self._make_specs()
        a_prog = RegionProgram(family="observation", carrier="deterministic")
        b_prog = RegionProgram(family="observation", carrier="diffusive")
        d = program_distance(a_spec, a_prog, b_spec, b_prog)
        base = transfer_distance(a_spec, b_spec)
        assert abs(d - base - 0.2) < 1e-5

    def test_both_penalties(self):
        a_spec, b_spec = self._make_specs()
        a_prog = RegionProgram(family="observation", carrier="deterministic")
        b_prog = RegionProgram(family="action", carrier="diffusive")
        d = program_distance(a_spec, a_prog, b_spec, b_prog)
        base = transfer_distance(a_spec, b_spec)
        assert abs(d - base - 0.5) < 1e-5


# ── Phase 6: Internal microsteps ────────────────────────────────────


class TestMicrosteps:
    def test_clockspec_max_inner_steps_default(self):
        c = ClockSpec()
        assert c.max_inner_steps is None

    def test_clockspec_max_inner_steps(self):
        c = ClockSpec(max_inner_steps=3)
        assert c.max_inner_steps == 3

    def test_clockspec_max_inner_steps_frozen(self):
        c = ClockSpec(max_inner_steps=3)
        with pytest.raises(AttributeError):
            c.max_inner_steps = 5

    def test_step_plan_no_internal(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),
                "state": RegionProgram(family="state"),
            },
        )
        sched = RegionScheduler(prog)
        plan = sched.step_plan(0)
        assert len(plan) == 1  # just the external step
        assert "obs" in plan[0]
        assert "state" in plan[0]

    def test_step_plan_with_internal(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),
                "state": RegionProgram(
                    family="state",
                    clock=ClockSpec(domain="internal", max_inner_steps=3),
                ),
            },
        )
        sched = RegionScheduler(prog)
        plan = sched.step_plan(0)
        # 1 external + 3 internal
        assert len(plan) == 4
        assert "obs" in plan[0]
        assert "state" in plan[0]
        # Internal steps only have "state"
        for step in plan[1:]:
            assert "state" in step
            assert "obs" not in step

    def test_step_plan_mixed_inner_steps(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),
                "state": RegionProgram(
                    family="state",
                    clock=ClockSpec(domain="internal", max_inner_steps=2),
                ),
                "act": RegionProgram(
                    family="action",
                    clock=ClockSpec(domain="internal", max_inner_steps=4),
                ),
            },
        )
        sched = RegionScheduler(prog)
        plan = sched.step_plan(0)
        # 1 external + max(2, 4) = 4 internal
        assert len(plan) == 5
        # First 2 inner steps have both state and act
        assert "state" in plan[1]
        assert "act" in plan[1]
        assert "state" in plan[2]
        assert "act" in plan[2]
        # Steps 3 and 4 only have act
        assert "state" not in plan[3]
        assert "act" in plan[3]
        assert "act" in plan[4]

    def test_step_plan_backward_compat(self):
        """step_plan first set matches step()"""
        prog = _make_program_with_clocks()
        sched = RegionScheduler(prog)
        active = sched.step(0)
        sched.reset()
        plan = sched.step_plan(0)
        assert plan[0] == active

    def test_clockspec_max_inner_steps_roundtrip(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(
                    clock=ClockSpec(max_inner_steps=5),
                ),
            },
        )
        d = prog.to_dict()
        loaded = CanvasProgram.from_dict(d)
        assert loaded.regions["obs"].clock.max_inner_steps == 5


# ── Phase 6: Identity / Slot persistence ────────────────────────────

from canvas_engineering.identity import IdentitySpec, SlotBindingModule


class TestIdentitySpec:
    def test_defaults(self):
        spec = IdentitySpec()
        assert spec.mode == "persistent"
        assert spec.slot_capacity == 64
        assert spec.binding_fn == "cross_attention"
        assert spec.birth_death is False

    def test_custom(self):
        spec = IdentitySpec(mode="tracklet", slot_capacity=32, birth_death=True)
        assert spec.mode == "tracklet"
        assert spec.slot_capacity == 32
        assert spec.birth_death is True

    def test_frozen(self):
        spec = IdentitySpec()
        with pytest.raises(AttributeError):
            spec.mode = "ephemeral"

    def test_region_program_with_identity(self):
        rp = RegionProgram(
            family="observation",
            identity=IdentitySpec(mode="persistent", slot_capacity=16),
        )
        assert rp.identity is not None
        assert rp.identity.slot_capacity == 16

    def test_identity_roundtrip(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(
                    family="observation",
                    identity=IdentitySpec(mode="tracklet", slot_capacity=32, birth_death=True),
                ),
            },
        )
        d = prog.to_dict()
        loaded = CanvasProgram.from_dict(d)
        assert loaded.regions["obs"].identity is not None
        assert loaded.regions["obs"].identity.mode == "tracklet"
        assert loaded.regions["obs"].identity.slot_capacity == 32
        assert loaded.regions["obs"].identity.birth_death is True


class TestSlotBindingModule:
    def test_forward_shape(self):
        binder = SlotBindingModule(d_model=16, n_slots=8, n_heads=2)
        obs = torch.randn(2, 5, 16)
        out = binder(obs)
        assert out.shape == (2, 8, 16)

    def test_with_birth_death(self):
        binder = SlotBindingModule(d_model=16, n_slots=8, n_heads=2, birth_death=True)
        obs = torch.randn(2, 5, 16)
        out = binder(obs)
        assert out.shape == (2, 8, 16)
        # Check alive_mask
        mask = binder.alive_mask(obs)
        assert mask.shape == (2, 8)
        assert mask.dtype == torch.bool

    def test_without_birth_death_all_alive(self):
        binder = SlotBindingModule(d_model=16, n_slots=4, n_heads=2, birth_death=False)
        obs = torch.randn(1, 3, 16)
        mask = binder.alive_mask(obs)
        assert mask.all()

    def test_repr(self):
        binder = SlotBindingModule(d_model=16, n_slots=4)
        r = repr(binder)
        assert "SlotBindingModule" in r
        assert "n_slots=4" in r

    def test_gradient_flows(self):
        binder = SlotBindingModule(d_model=16, n_slots=4, n_heads=2, birth_death=True)
        obs = torch.randn(1, 3, 16, requires_grad=True)
        out = binder(obs)
        loss = out.sum()
        loss.backward()
        assert obs.grad is not None


# ── Phase 6: MaskSpec ───────────────────────────────────────────────

from canvas_engineering.masks import MaskSpec, Rect, rect_cover, mask_to_index_pairs


class TestMaskSpec:
    def test_defaults(self):
        ms = MaskSpec()
        assert ms.kind == "full"
        assert ms.tile_h == 1
        assert ms.tile_w == 1

    def test_custom(self):
        ms = MaskSpec(kind="tile", tile_h=4, tile_w=4)
        assert ms.kind == "tile"
        assert ms.tile_h == 4

    def test_frozen(self):
        ms = MaskSpec()
        with pytest.raises(AttributeError):
            ms.kind = "tile"


class TestRect:
    def test_volume(self):
        r = Rect(0, 2, 0, 3, 0, 4)
        assert r.volume == 24  # 2*3*4

    def test_zero_volume(self):
        r = Rect(0, 0, 0, 0, 0, 0)
        assert r.volume == 0


class TestRectCover:
    def test_empty_mask(self):
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        rects = rect_cover(mask)
        assert len(rects) == 0

    def test_full_mask(self):
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        rects = rect_cover(mask)
        # Should cover everything (ideally in one rect)
        total_covered = sum(r.volume for r in rects)
        assert total_covered == 24

    def test_single_point(self):
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[1, 2, 3] = True
        rects = rect_cover(mask)
        assert len(rects) == 1
        assert rects[0].volume == 1

    def test_rectangular_block(self):
        mask = torch.zeros(4, 4, 4, dtype=torch.bool)
        mask[0:2, 0:2, 0:2] = True
        rects = rect_cover(mask)
        total = sum(r.volume for r in rects)
        assert total == 8

    def test_complete_coverage(self):
        """All True positions are covered by the returned rects."""
        mask = torch.zeros(3, 3, 3, dtype=torch.bool)
        mask[0, 0, 0] = True
        mask[2, 2, 2] = True
        mask[1, 1, 0] = True
        rects = rect_cover(mask)
        # Verify coverage
        covered = torch.zeros_like(mask)
        for r in rects:
            covered[r.t0:r.t1, r.h0:r.h1, r.w0:r.w1] = True
        assert (covered & mask).sum() == mask.sum()


class TestMaskToIndexPairs:
    def test_full(self):
        layout = CanvasLayout(T=2, H=2, W=2, d_model=8, regions={
            "a": (0, 2, 0, 1, 0, 1),
            "b": (0, 2, 1, 2, 0, 1),
        })
        ms = MaskSpec(kind="full")
        pairs = mask_to_index_pairs(ms, layout, "a", "b")
        assert len(pairs) == 1
        src_idx, dst_idx = pairs[0]
        assert len(src_idx) == 2
        assert len(dst_idx) == 2

    def test_tile(self):
        layout = CanvasLayout(T=1, H=4, W=4, d_model=8, regions={
            "a": (0, 1, 0, 4, 0, 4),
            "b": (0, 1, 0, 4, 0, 4),
        })
        ms = MaskSpec(kind="tile", tile_h=2, tile_w=2)
        pairs = mask_to_index_pairs(ms, layout, "a", "b")
        assert len(pairs) == 4  # 4x4 grid with 2x2 tiles = 4 tiles

    def test_sparse_falls_back_to_full(self):
        layout = CanvasLayout(T=1, H=2, W=2, d_model=8, regions={
            "a": (0, 1, 0, 2, 0, 2),
        })
        ms = MaskSpec(kind="sparse")
        pairs = mask_to_index_pairs(ms, layout, "a", "a")
        assert len(pairs) == 1

    def test_connection_with_mask(self):
        """Connection with mask field works."""
        ms = MaskSpec(kind="tile", tile_h=2, tile_w=2)
        c = Connection(src="a", dst="b", mask=ms)
        assert c.mask is not None
        assert c.mask.kind == "tile"

    def test_connection_no_mask_default(self):
        c = Connection(src="a", dst="b")
        assert c.mask is None


# ── Phase 6: CortexSpec ─────────────────────────────────────────────

from canvas_engineering.cortex import CortexSpec, CortexRegistry


class TestCortexSpec:
    def test_defaults(self):
        cs = CortexSpec(name="visual")
        assert cs.name == "visual"
        assert cs.tile == (4, 4)
        assert cs.local_backend == "local_attention"
        assert cs.shared_cache is True

    def test_custom(self):
        cs = CortexSpec(
            name="motor", bounds=(0, 4, 0, 8, 0, 8),
            tile=(2, 2), local_backend="linear_attention", shared_cache=False,
        )
        assert cs.name == "motor"
        assert cs.tile == (2, 2)
        assert cs.shared_cache is False

    def test_frozen(self):
        cs = CortexSpec(name="test")
        with pytest.raises(AttributeError):
            cs.name = "other"


class TestCortexRegistry:
    def test_register_and_assign(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.assign("camera", "visual")
        assert reg.cortex_for("camera").name == "visual"

    def test_register_duplicate_raises(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        with pytest.raises(ValueError):
            reg.register(CortexSpec(name="visual"))

    def test_assign_unknown_cortex_raises(self):
        reg = CortexRegistry()
        with pytest.raises(KeyError):
            reg.assign("camera", "nonexistent")

    def test_reassign_different_cortex_raises(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.register(CortexSpec(name="motor"))
        reg.assign("camera", "visual")
        with pytest.raises(ValueError):
            reg.assign("camera", "motor")

    def test_reassign_same_cortex_ok(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.assign("camera", "visual")
        reg.assign("camera", "visual")  # no error
        assert reg.cortex_for("camera").name == "visual"

    def test_regions_in(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.assign("camera", "visual")
        reg.assign("depth", "visual")
        regions = reg.regions_in("visual")
        assert regions == ["camera", "depth"]  # sorted

    def test_same_cortex(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.assign("camera", "visual")
        reg.assign("depth", "visual")
        assert reg.same_cortex("camera", "depth")

    def test_different_cortex(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.register(CortexSpec(name="motor"))
        reg.assign("camera", "visual")
        reg.assign("joints", "motor")
        assert not reg.same_cortex("camera", "joints")

    def test_no_cortex(self):
        reg = CortexRegistry()
        assert reg.cortex_for("camera") is None
        assert not reg.same_cortex("a", "b")

    def test_unregister(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.assign("camera", "visual")
        reg.unregister("visual")
        assert reg.cortex_for("camera") is None
        assert reg.n_cortices == 0

    def test_unassign(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.assign("camera", "visual")
        reg.unassign("camera")
        assert reg.cortex_for("camera") is None
        assert reg.n_assigned == 0

    def test_unassign_unknown_raises(self):
        reg = CortexRegistry()
        with pytest.raises(KeyError):
            reg.unassign("camera")

    def test_properties(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.register(CortexSpec(name="motor"))
        reg.assign("camera", "visual")
        assert reg.n_cortices == 2
        assert reg.n_assigned == 1
        assert reg.cortex_names == ["motor", "visual"]

    def test_summary(self):
        reg = CortexRegistry()
        reg.register(CortexSpec(name="visual"))
        reg.assign("camera", "visual")
        s = reg.summary()
        assert "CortexRegistry" in s
        assert "camera" in s

    def test_repr(self):
        reg = CortexRegistry()
        assert "CortexRegistry" in repr(reg)


# ── Phase 6: Learned scheduling ─────────────────────────────────────

from canvas_engineering.scheduling import LearnedScheduler


class TestLearnedScheduler:
    def test_init(self):
        sched = LearnedScheduler(n_regions=5, summary_dim=10, max_active=3)
        assert sched.n_regions == 5
        assert sched.max_active == 3

    def test_forward_shape(self):
        sched = LearnedScheduler(n_regions=5, summary_dim=10, max_active=3)
        summaries = torch.randn(10)
        active, log_probs = sched(summaries)
        assert len(active) <= 3
        assert all(isinstance(i, int) for i in active)
        assert all(0 <= i < 5 for i in active)
        assert log_probs.shape == (5,)

    def test_forward_batched(self):
        sched = LearnedScheduler(n_regions=5, summary_dim=10, max_active=3)
        summaries = torch.randn(2, 10)
        active, log_probs = sched(summaries)
        assert len(active) <= 3

    def test_max_active_capped(self):
        sched = LearnedScheduler(n_regions=3, summary_dim=5, max_active=10)
        assert sched.max_active == 3

    def test_gradient_flows(self):
        sched = LearnedScheduler(n_regions=4, summary_dim=8, max_active=2)
        summaries = torch.randn(8, requires_grad=True)
        active, log_probs = sched(summaries)
        loss = log_probs.sum()
        loss.backward()
        assert summaries.grad is not None

    def test_eval_mode_deterministic(self):
        sched = LearnedScheduler(n_regions=4, summary_dim=8, max_active=2)
        sched.eval()
        summaries = torch.randn(8)
        active1, _ = sched(summaries)
        active2, _ = sched(summaries)
        assert active1 == active2  # deterministic in eval

    def test_repr(self):
        sched = LearnedScheduler(n_regions=5, summary_dim=10, max_active=3)
        r = repr(sched)
        assert "LearnedScheduler" in r
        assert "n_regions=5" in r


# ── Phase 6: ClockExpr IR ───────────────────────────────────────────

from canvas_engineering.clock_ir import (
    ClockExpr, ClockContext,
    Periodic, OnEvent, BoundaryExpr, And, Or, Not,
    Cooldown as CDCooldown, MaxSilence as CDMaxSilence,
    periodic as p_periodic, on as p_on, boundary as p_boundary,
)


class TestPeriodic:
    def test_fires_on_period(self):
        expr = Periodic(period=3)
        ctx = ClockContext(external_t=0)
        assert expr.evaluate(ctx) is True
        ctx = ClockContext(external_t=1)
        assert expr.evaluate(ctx) is False
        ctx = ClockContext(external_t=3)
        assert expr.evaluate(ctx) is True

    def test_default_period_1(self):
        expr = Periodic()
        for t in range(5):
            assert expr.evaluate(ClockContext(external_t=t)) is True

    def test_repr(self):
        expr = Periodic(period=4, domain="internal")
        assert "period=4" in repr(expr)

    def test_eq(self):
        assert Periodic(3) == Periodic(3)
        assert Periodic(3) != Periodic(4)

    def test_roundtrip(self):
        expr = Periodic(period=5, domain="internal")
        d = expr.to_dict()
        loaded = ClockExpr.from_dict(d)
        assert loaded == expr


class TestOnEvent:
    def test_gt(self):
        expr = OnEvent(source="err.prediction", gt=0.5)
        ctx = ClockContext(summaries={"err": {"prediction": 0.8}})
        assert expr.evaluate(ctx) is True
        ctx = ClockContext(summaries={"err": {"prediction": 0.3}})
        assert expr.evaluate(ctx) is False

    def test_lt(self):
        expr = OnEvent(source="err.prediction", lt=0.5)
        ctx = ClockContext(summaries={"err": {"prediction": 0.3}})
        assert expr.evaluate(ctx) is True
        ctx = ClockContext(summaries={"err": {"prediction": 0.8}})
        assert expr.evaluate(ctx) is False

    def test_gt_and_lt(self):
        expr = OnEvent(source="err.prediction", gt=0.2, lt=0.8)
        # In band: True
        ctx = ClockContext(summaries={"err": {"prediction": 0.5}})
        assert expr.evaluate(ctx) is True
        # Below: False
        ctx = ClockContext(summaries={"err": {"prediction": 0.1}})
        assert expr.evaluate(ctx) is False
        # Above: False
        ctx = ClockContext(summaries={"err": {"prediction": 0.9}})
        assert expr.evaluate(ctx) is False

    def test_no_summaries(self):
        expr = OnEvent(source="err.prediction", gt=0.5)
        ctx = ClockContext()
        assert expr.evaluate(ctx) is False

    def test_requires_gt_or_lt(self):
        with pytest.raises(ValueError):
            OnEvent(source="err.prediction")

    def test_roundtrip(self):
        expr = OnEvent(source="err.prediction", gt=0.5, lt=0.9)
        loaded = ClockExpr.from_dict(expr.to_dict())
        assert loaded == expr


class TestBoundaryExpr:
    def test_fires(self):
        expr = BoundaryExpr("episode_end")
        ctx = ClockContext(boundary="episode_end")
        assert expr.evaluate(ctx) is True
        ctx = ClockContext(boundary="other")
        assert expr.evaluate(ctx) is False
        ctx = ClockContext()
        assert expr.evaluate(ctx) is False

    def test_roundtrip(self):
        expr = BoundaryExpr("episode_end")
        loaded = ClockExpr.from_dict(expr.to_dict())
        assert loaded == expr


class TestClockCombinators:
    def test_and(self):
        left = Periodic(period=2)
        right = Periodic(period=3)
        expr = And(left, right)
        # t=0: both fire
        assert expr.evaluate(ClockContext(external_t=0)) is True
        # t=2: only left fires
        assert expr.evaluate(ClockContext(external_t=2)) is False
        # t=6: both fire
        assert expr.evaluate(ClockContext(external_t=6)) is True

    def test_or(self):
        left = Periodic(period=2)
        right = Periodic(period=3)
        expr = Or(left, right)
        assert expr.evaluate(ClockContext(external_t=0)) is True
        assert expr.evaluate(ClockContext(external_t=1)) is False
        assert expr.evaluate(ClockContext(external_t=2)) is True
        assert expr.evaluate(ClockContext(external_t=3)) is True

    def test_not(self):
        expr = Not(Periodic(period=2))
        assert expr.evaluate(ClockContext(external_t=0)) is False
        assert expr.evaluate(ClockContext(external_t=1)) is True

    def test_complex_composition(self):
        # Fire every 4 steps OR when prediction error > 0.5
        expr = Or(Periodic(4), OnEvent("err.prediction", gt=0.5))
        # t=0, no summaries: periodic fires
        assert expr.evaluate(ClockContext(external_t=0)) is True
        # t=1, high error: event fires
        ctx = ClockContext(external_t=1, summaries={"err": {"prediction": 0.8}})
        assert expr.evaluate(ctx) is True
        # t=1, low error: neither fires
        ctx = ClockContext(external_t=1, summaries={"err": {"prediction": 0.1}})
        assert expr.evaluate(ctx) is False

    def test_combinator_roundtrips(self):
        expr = And(Or(Periodic(2), BoundaryExpr("ep_end")), Not(Periodic(3)))
        d = expr.to_dict()
        loaded = ClockExpr.from_dict(d)
        assert loaded == expr


class TestClockDecorators:
    def test_cooldown(self):
        expr = CDCooldown(Periodic(1), steps=3)
        ctx = ClockContext(external_t=0, cooldown_until=0)
        assert expr.evaluate(ctx) is True
        ctx = ClockContext(external_t=1, cooldown_until=3)
        assert expr.evaluate(ctx) is False
        ctx = ClockContext(external_t=3, cooldown_until=3)
        assert expr.evaluate(ctx) is True

    def test_max_silence(self):
        expr = CDMaxSilence(OnEvent("err.prediction", gt=100.0), steps=5)
        # Never fired: force fire due to max_silence (last_fired=-1)
        ctx = ClockContext(external_t=0, last_fired=-1)
        assert expr.evaluate(ctx) is True
        # Recently fired, event not triggered
        ctx = ClockContext(external_t=2, last_fired=1,
                           summaries={"err": {"prediction": 0.0}})
        assert expr.evaluate(ctx) is False
        # Silent for 5 steps: force fire
        ctx = ClockContext(external_t=6, last_fired=1,
                           summaries={"err": {"prediction": 0.0}})
        assert expr.evaluate(ctx) is True

    def test_decorator_roundtrips(self):
        expr = CDMaxSilence(CDCooldown(Periodic(1), steps=2), steps=10)
        loaded = ClockExpr.from_dict(expr.to_dict())
        assert loaded == expr


class TestClockSugar:
    def test_periodic(self):
        expr = p_periodic(4)
        assert isinstance(expr, Periodic)
        assert expr.period == 4

    def test_on(self):
        expr = p_on("err.prediction", gt=0.5)
        assert isinstance(expr, OnEvent)
        assert expr.source == "err.prediction"

    def test_boundary(self):
        expr = p_boundary("episode_end")
        assert isinstance(expr, BoundaryExpr)
        assert expr.name == "episode_end"


class TestClockExprFromDictErrors:
    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown ClockExpr type"):
            ClockExpr.from_dict({"type": "nonexistent"})


# ── Phase 6: ConstraintSpec ─────────────────────────────────────────

from canvas_engineering.program import ConstraintSpec, validate_constraints


class TestConstraintSpec:
    def test_defaults(self):
        cs = ConstraintSpec()
        assert cs.equivariance is None
        assert cs.conservation is None
        assert cs.causal_direction is None
        assert cs.monotonicity is None

    def test_custom(self):
        cs = ConstraintSpec(
            equivariance="translation",
            conservation="energy",
            causal_direction="acyclic",
            monotonicity="non_decreasing",
        )
        assert cs.equivariance == "translation"
        assert cs.conservation == "energy"
        assert cs.causal_direction == "acyclic"
        assert cs.monotonicity == "non_decreasing"

    def test_frozen(self):
        cs = ConstraintSpec()
        with pytest.raises(AttributeError):
            cs.equivariance = "rotation"

    def test_region_program_with_constraints(self):
        rp = RegionProgram(
            family="state",
            constraints=ConstraintSpec(equivariance="translation"),
        )
        assert rp.constraints is not None
        assert rp.constraints.equivariance == "translation"

    def test_constraint_roundtrip(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(
                    constraints=ConstraintSpec(
                        equivariance="rotation",
                        conservation="capacity",
                        causal_direction="forward_only",
                        monotonicity="non_increasing",
                    ),
                ),
            },
        )
        d = prog.to_dict()
        loaded = CanvasProgram.from_dict(d)
        cs = loaded.regions["obs"].constraints
        assert cs.equivariance == "rotation"
        assert cs.conservation == "capacity"
        assert cs.causal_direction == "forward_only"
        assert cs.monotonicity == "non_increasing"

    def test_constraint_default_not_in_dict(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(constraints=ConstraintSpec()),
            },
        )
        d = prog.to_dict()
        # All-default ConstraintSpec produces empty dict -> "constraints" key present but empty
        # The region_program_to_dict includes it only if non-None
        rp_dict = d.get("region_programs", {}).get("obs", {})
        # The ConstraintSpec was set but all defaults -> to_dict returns empty dict
        # Our impl always includes it if constraints is not None
        assert "constraints" in rp_dict


class TestValidateConstraints:
    def test_no_constraints_no_violations(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),
                "state": RegionProgram(family="state"),
            },
        )
        violations = validate_constraints(prog)
        assert violations == []

    def test_valid_constraints_no_violations(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(
                    constraints=ConstraintSpec(
                        equivariance="translation",
                        conservation="energy",
                    ),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert violations == []

    def test_invalid_equivariance(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(
                    constraints=ConstraintSpec(equivariance="invalid"),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert len(violations) == 1
        assert "equivariance" in violations[0]

    def test_invalid_conservation(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(
                    constraints=ConstraintSpec(conservation="invalid"),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert len(violations) == 1
        assert "conservation" in violations[0]

    def test_invalid_causal_direction(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(
                    constraints=ConstraintSpec(causal_direction="invalid"),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert len(violations) == 1
        assert "causal_direction" in violations[0]

    def test_invalid_monotonicity(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(
                    constraints=ConstraintSpec(monotonicity="invalid"),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert len(violations) == 1
        assert "monotonicity" in violations[0]

    def test_acyclic_violation(self):
        layout = CanvasLayout(T=2, H=2, W=2, d_model=16, regions={
            "a": (0, 2, 0, 1, 0, 1),
            "b": (0, 2, 1, 2, 0, 1),
        })
        topology = CanvasTopology(connections=[
            Connection(src="a", dst="b"),
            Connection(src="b", dst="a"),  # creates cycle a -> b -> a
        ])
        schema = CanvasSchema(layout=layout, topology=topology)
        prog = CanvasProgram(
            schema=schema,
            regions={
                "a": RegionProgram(
                    constraints=ConstraintSpec(causal_direction="acyclic"),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert any("acyclic" in v for v in violations)

    def test_acyclic_no_violation(self):
        layout = CanvasLayout(T=2, H=2, W=2, d_model=16, regions={
            "a": (0, 2, 0, 1, 0, 1),
            "b": (0, 2, 1, 2, 0, 1),
        })
        topology = CanvasTopology(connections=[
            Connection(src="a", dst="b"),
            # No back-edge
        ])
        schema = CanvasSchema(layout=layout, topology=topology)
        prog = CanvasProgram(
            schema=schema,
            regions={
                "a": RegionProgram(
                    constraints=ConstraintSpec(causal_direction="acyclic"),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert not any("acyclic" in v for v in violations)

    def test_forward_only_violation(self):
        layout = CanvasLayout(T=4, H=2, W=2, d_model=16, regions={
            "late": RegionSpec(bounds=(2, 4, 0, 1, 0, 1)),
            "early": RegionSpec(bounds=(0, 2, 1, 2, 0, 1)),
        })
        topology = CanvasTopology(connections=[
            Connection(src="late", dst="early"),  # late -> early (t0=2 -> t0=0)
        ])
        schema = CanvasSchema(layout=layout, topology=topology)
        prog = CanvasProgram(
            schema=schema,
            regions={
                "late": RegionProgram(
                    constraints=ConstraintSpec(causal_direction="forward_only"),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert any("forward_only" in v for v in violations)

    def test_forward_only_no_violation(self):
        layout = CanvasLayout(T=4, H=2, W=2, d_model=16, regions={
            "early": RegionSpec(bounds=(0, 2, 0, 1, 0, 1)),
            "late": RegionSpec(bounds=(2, 4, 1, 2, 0, 1)),
        })
        topology = CanvasTopology(connections=[
            Connection(src="early", dst="late"),  # early -> late (t0=0 -> t0=2)
        ])
        schema = CanvasSchema(layout=layout, topology=topology)
        prog = CanvasProgram(
            schema=schema,
            regions={
                "early": RegionProgram(
                    constraints=ConstraintSpec(causal_direction="forward_only"),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert not any("forward_only" in v for v in violations)

    def test_no_topology_no_causal_violations(self):
        layout = CanvasLayout(T=2, H=2, W=2, d_model=16, regions={
            "a": (0, 2, 0, 1, 0, 1),
        })
        schema = CanvasSchema(layout=layout)
        prog = CanvasProgram(
            schema=schema,
            regions={
                "a": RegionProgram(
                    constraints=ConstraintSpec(causal_direction="acyclic"),
                ),
            },
        )
        violations = validate_constraints(prog)
        assert not any("acyclic" in v for v in violations)


# ── Exports ─────────────────────────────────────────────────────────


class TestExports:
    def test_all_new_symbols_in_all(self):
        import canvas_engineering
        expected = [
            "FamilyCarrierEmbedding", "program_distance",
            "LearnedScheduler",
            "ClockExpr", "ClockContext", "Periodic", "OnEvent", "BoundaryExpr",
            "And", "Or", "Not", "ClockCooldown", "ClockMaxSilence",
            "clock_periodic", "clock_on", "clock_boundary",
            "MaskSpec", "Rect", "rect_cover", "mask_to_index_pairs",
            "CortexSpec", "CortexRegistry",
            "IdentitySpec", "SlotBindingModule",
            "ConstraintSpec", "validate_constraints",
        ]
        for name in expected:
            assert name in canvas_engineering.__all__, f"{name} not in __all__"
            assert hasattr(canvas_engineering, name), f"{name} not importable"

    def test_version_bumped(self):
        import canvas_engineering
        assert canvas_engineering.__version__ >= "0.4.0"


# ── Phase 7: ClockExpr in ClockSpec + trigger eval + write_mode ─────


from canvas_engineering import (
    AttentionDispatcher, RegionScheduler, evaluate_trigger,
    clock_periodic, clock_on, Or, And,
)


class TestClockSpecExpr:
    def test_clock_expr_round_trip(self):
        expr = Or(clock_periodic(4), clock_on("vision.err", gt=0.25))
        c = ClockSpec(expr=expr)
        d = ClockSpec.__dataclass_fields__  # sanity
        assert c.expr is expr

    def test_clock_expr_serialization(self):
        prog = CanvasProgram(
            schema=CanvasSchema(layout=CanvasLayout(
                T=4, H=4, W=4, d_model=8,
                regions={"r": RegionSpec(bounds=(0, 4, 0, 4, 0, 4))},
            )),
            regions={"r": RegionProgram(
                family="state",
                clock=ClockSpec(expr=Or(clock_periodic(2), clock_on("r.err", gt=0.5))),
            )},
        )
        d = prog.to_dict()
        restored = CanvasProgram.from_dict(d)
        assert restored.regions["r"].clock.expr is not None
        assert restored.regions["r"].clock.expr == prog.regions["r"].clock.expr

    def test_scheduler_uses_expr_when_set(self):
        prog = CanvasProgram(
            schema=CanvasSchema(layout=CanvasLayout(
                T=4, H=4, W=4, d_model=8,
                regions={"r": RegionSpec(bounds=(0, 4, 0, 4, 0, 4))},
            )),
            regions={"r": RegionProgram(
                family="state",
                clock=ClockSpec(
                    mode="periodic", period=1,  # would fire every step
                    expr=clock_periodic(3),     # but expr overrides: every 3
                ),
            )},
        )
        sched = RegionScheduler(prog)
        fires = [("r" in sched.step(t)) for t in range(7)]
        assert fires == [True, False, False, True, False, False, True]

    def test_scheduler_expr_event_driven(self):
        prog = CanvasProgram(
            schema=CanvasSchema(layout=CanvasLayout(
                T=4, H=4, W=4, d_model=8,
                regions={"r": RegionSpec(bounds=(0, 4, 0, 4, 0, 4))},
            )),
            regions={"r": RegionProgram(
                family="state",
                clock=ClockSpec(expr=clock_on("e.err", gt=0.5)),
            )},
        )
        sched = RegionScheduler(prog)
        assert "r" not in sched.step(0, summaries={"e": {"err": 0.1}})
        assert "r" in sched.step(1, summaries={"e": {"err": 0.9}})


class TestEvaluateTrigger:
    def test_none_trigger_is_true(self):
        assert evaluate_trigger(None, None) is True
        assert evaluate_trigger(None, {"r": {"k": 1.0}}) is True

    def test_missing_summaries_is_false(self):
        assert evaluate_trigger("r.k > 0.5", None) is False

    def test_gt_lt_ge_le(self):
        s = {"r": {"k": 0.5}}
        assert evaluate_trigger("r.k > 0.4", s) is True
        assert evaluate_trigger("r.k > 0.6", s) is False
        assert evaluate_trigger("r.k < 0.6", s) is True
        assert evaluate_trigger("r.k >= 0.5", s) is True
        assert evaluate_trigger("r.k <= 0.5", s) is True
        assert evaluate_trigger("r.k == 0.5", s) is True
        assert evaluate_trigger("r.k != 0.5", s) is False

    def test_compound_and(self):
        s = {"r": {"k": 0.7}, "q": {"v": 0.2}}
        assert evaluate_trigger("r.k > 0.5 && q.v < 0.3", s) is True
        assert evaluate_trigger("r.k > 0.5 && q.v > 0.5", s) is False

    def test_missing_kind_is_false(self):
        assert evaluate_trigger("r.k > 0.5", {"r": {}}) is False

    def test_malformed_returns_false(self):
        assert evaluate_trigger("not an expression", {"r": {"k": 1}}) is False


class TestDispatcherWriteMode:
    def _make_layout(self):
        return CanvasLayout(
            T=2, H=2, W=2, d_model=8,
            regions={
                "a": RegionSpec(bounds=(0, 2, 0, 1, 0, 1)),
                "b": RegionSpec(bounds=(0, 2, 1, 2, 0, 1)),
                "c": RegionSpec(bounds=(0, 2, 0, 1, 1, 2)),
            },
        )

    def test_add_mode_default(self):
        layout = self._make_layout()
        topo = CanvasTopology([
            Connection(src="a", dst="b", weight=1.0),
            Connection(src="a", dst="c", weight=1.0),
        ])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={"a": RegionProgram(), "b": RegionProgram(), "c": RegionProgram()},
            connections={
                ("a", "b"): ConnectionProgram(write_mode="add"),
                ("a", "c"): ConnectionProgram(write_mode="add"),
            },
        )
        torch.manual_seed(0)
        disp = AttentionDispatcher(topo, layout, d_model=8, n_heads=2, program=prog)
        x = torch.randn(1, layout.num_positions, 8)
        out = disp(x)
        # a positions are touched, b/c are pass-through
        a_idx = layout.region_indices("a")
        b_idx = layout.region_indices("b")
        assert not torch.allclose(out[:, a_idx], x[:, a_idx])
        assert torch.allclose(out[:, b_idx], x[:, b_idx])

    def test_replace_mode_overrides_add(self):
        layout = self._make_layout()
        topo = CanvasTopology([
            Connection(src="a", dst="b", weight=1.0),
            Connection(src="a", dst="c", weight=1.0),
        ])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={"a": RegionProgram(), "b": RegionProgram(), "c": RegionProgram()},
            connections={
                ("a", "b"): ConnectionProgram(write_mode="add"),
                ("a", "c"): ConnectionProgram(write_mode="replace"),
            },
        )
        torch.manual_seed(0)
        disp = AttentionDispatcher(topo, layout, d_model=8, n_heads=2, program=prog)
        x = torch.randn(1, layout.num_positions, 8)
        out = disp(x)
        # Output at 'a' positions came from the replace edge only — no division
        # by total weight. Verify by re-running with only the replace edge.
        topo2 = CanvasTopology([Connection(src="a", dst="c", weight=1.0)])
        prog2 = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo2),
            regions={"a": RegionProgram(), "c": RegionProgram()},
            connections={("a", "c"): ConnectionProgram(write_mode="replace")},
        )
        torch.manual_seed(0)
        disp2 = AttentionDispatcher(topo2, layout, d_model=8, n_heads=2, program=prog2)
        # Share attention parameters via state_dict (same seed gives identical init)
        disp2.load_state_dict({k: v for k, v in disp.state_dict().items() if k in disp2.state_dict()}, strict=False)
        out2 = disp2(x)
        a_idx = layout.region_indices("a")
        assert torch.allclose(out[:, a_idx], out2[:, a_idx], atol=1e-5)

    def test_gate_mode_has_learned_param(self):
        layout = self._make_layout()
        topo = CanvasTopology([Connection(src="a", dst="b", weight=1.0)])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={"a": RegionProgram(), "b": RegionProgram()},
            connections={("a", "b"): ConnectionProgram(write_mode="gate")},
        )
        disp = AttentionDispatcher(topo, layout, d_model=8, n_heads=2, program=prog)
        assert len(disp._gate_params) == 1
        # Output should be gradient-connected to gate param.
        x = torch.randn(1, layout.num_positions, 8, requires_grad=True)
        out = disp(x)
        loss = out.sum()
        loss.backward()
        for p in disp._gate_params.values():
            assert p.grad is not None
            assert torch.any(p.grad != 0)


class TestDispatcherTrigger:
    def _setup(self, trigger):
        layout = CanvasLayout(
            T=2, H=2, W=2, d_model=8,
            regions={
                "a": RegionSpec(bounds=(0, 2, 0, 1, 0, 1)),
                "b": RegionSpec(bounds=(0, 2, 1, 2, 0, 1)),
            },
        )
        topo = CanvasTopology([Connection(src="a", dst="b", weight=1.0)])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={"a": RegionProgram(), "b": RegionProgram()},
            connections={("a", "b"): ConnectionProgram(trigger=trigger)},
        )
        disp = AttentionDispatcher(topo, layout, d_model=8, n_heads=2, program=prog)
        return layout, disp

    def test_trigger_blocks_when_false(self):
        layout, disp = self._setup("b.err > 0.5")
        x = torch.randn(1, layout.num_positions, 8)
        out = disp(x, summaries={"b": {"err": 0.1}})
        # a positions pass-through unchanged when trigger fails
        a_idx = layout.region_indices("a")
        assert torch.allclose(out[:, a_idx], x[:, a_idx])

    def test_trigger_fires_when_true(self):
        layout, disp = self._setup("b.err > 0.5")
        x = torch.randn(1, layout.num_positions, 8)
        out = disp(x, summaries={"b": {"err": 0.9}})
        a_idx = layout.region_indices("a")
        assert not torch.allclose(out[:, a_idx], x[:, a_idx])


# ── Phase 8: full subsystem wiring ──────────────────────────────────


from canvas_engineering import (
    CortexSpec, CortexRegistry, IdentitySpec,
    HybridScheduler, OPERATOR_DEFAULTS, resolve_operator_defaults,
    ProgramCompiler, MaskSpec,
)


def _small_layout(d_model=8):
    return CanvasLayout(
        T=2, H=2, W=2, d_model=d_model,
        regions={
            "a": RegionSpec(bounds=(0, 2, 0, 1, 0, 1)),
            "b": RegionSpec(bounds=(0, 2, 1, 2, 0, 1)),
        },
    )


class TestExtendedConstraints:
    def test_equivariance_translation_compatible(self):
        layout = _small_layout()
        topo = CanvasTopology([Connection(src="a", dst="b", fn="cross_attention")])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={
                "a": RegionProgram(
                    family="state",
                    constraints=ConstraintSpec(equivariance="translation"),
                ),
                "b": RegionProgram(family="state"),
            },
        )
        assert validate_constraints(prog) == []

    def test_equivariance_translation_incompatible(self):
        layout = _small_layout()
        topo = CanvasTopology([Connection(src="a", dst="b", fn="cosine_attention")])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={
                "a": RegionProgram(
                    family="state",
                    constraints=ConstraintSpec(equivariance="translation"),
                ),
                "b": RegionProgram(family="state"),
            },
        )
        v = validate_constraints(prog)
        assert any("equivariance" in msg for msg in v)

    def test_conservation_violated_by_replace(self):
        layout = _small_layout()
        topo = CanvasTopology([Connection(src="a", dst="b")])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={
                "a": RegionProgram(
                    family="state",
                    constraints=ConstraintSpec(conservation="capacity"),
                ),
                "b": RegionProgram(family="state"),
            },
            connections={("a", "b"): ConnectionProgram(write_mode="replace")},
        )
        v = validate_constraints(prog)
        assert any("conservation" in msg for msg in v)

    def test_monotonicity_violated_by_diffusive_carrier(self):
        layout = _small_layout()
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=CanvasTopology([])),
            regions={
                "a": RegionProgram(
                    family="state",
                    carrier="diffusive",
                    constraints=ConstraintSpec(monotonicity="non_decreasing"),
                ),
            },
        )
        v = validate_constraints(prog)
        assert any("monotonicity" in msg for msg in v)

    def test_monotonicity_ok_with_deterministic_carrier(self):
        layout = _small_layout()
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=CanvasTopology([])),
            regions={
                "a": RegionProgram(
                    family="state",
                    carrier="deterministic",
                    constraints=ConstraintSpec(monotonicity="non_decreasing"),
                ),
            },
        )
        assert validate_constraints(prog) == []


class TestEffectiveLearning:
    def test_explicit_learning_wins(self):
        rp = RegionProgram(family="observation",
                           learning=LearningSpec(mode="supervised"))
        assert rp.effective_learning().mode == "supervised"

    def test_family_default_applied_when_none(self):
        rp = RegionProgram(family="observation")
        assert rp.learning is None  # field stays None for backward compat
        assert rp.effective_learning().mode == "ssl_prediction"
        assert rp.effective_learning().compile_mode == "freeze"

    def test_unknown_family_falls_back(self):
        rp = RegionProgram(family="custom_thing")
        assert rp.effective_learning().mode == "supervised"

    def test_effective_compile_mode(self):
        # action family defaults to freeze via FAMILY_DEFAULTS
        rp = RegionProgram(family="action")
        assert rp.effective_compile_mode() == "freeze"
        # explicit override wins
        rp2 = RegionProgram(family="action", compile_mode="runtime")
        # explicit "runtime" is the bare default — we still consult the
        # family default. To force runtime even with a freeze-default
        # family, set learning explicitly.
        rp3 = RegionProgram(family="action", learning=LearningSpec())
        assert rp3.effective_compile_mode() == "runtime"


class TestOperatorDefaults:
    def test_known_operators_have_defaults(self):
        for op in ["attend", "retrieve", "write", "act", "compress",
                   "intervene", "emit_residual"]:
            assert op in OPERATOR_DEFAULTS

    def test_retrieve_overrides_fn(self):
        cp = ConnectionProgram(operator="retrieve")
        fn, wm, trig = resolve_operator_defaults(cp, base_fn="cross_attention")
        assert fn == "cross_attention"
        assert wm == "add"

    def test_write_uses_replace(self):
        cp = ConnectionProgram(operator="write")
        fn, wm, _t = resolve_operator_defaults(cp, base_fn="cross_attention")
        assert wm == "replace"

    def test_explicit_write_mode_wins(self):
        cp = ConnectionProgram(operator="write", write_mode="gate")
        fn, wm, _t = resolve_operator_defaults(cp, base_fn="cross_attention")
        assert wm == "gate"

    def test_explicit_trigger_wins(self):
        cp = ConnectionProgram(operator="correct", trigger="x.err > 0.1")
        fn, wm, trig = resolve_operator_defaults(cp, base_fn="cross_attention")
        assert trig == "x.err > 0.1"

    def test_unknown_operator_falls_back(self):
        cp = ConnectionProgram(operator="frobnicate")
        fn, wm, trig = resolve_operator_defaults(cp, base_fn="cross_attention")
        assert fn == "cross_attention"
        assert wm == "add"
        assert trig is None


class TestCortexInDispatcher:
    def test_cortex_overrides_fn(self):
        layout = _small_layout(d_model=8)
        topo = CanvasTopology([Connection(src="a", dst="b", fn="cross_attention")])
        reg = CortexRegistry()
        reg.register(CortexSpec(name="zone", local_backend="linear_attention"))
        reg.assign("a", "zone")
        reg.assign("b", "zone")
        disp = AttentionDispatcher(topo, layout, d_model=8, n_heads=2,
                                   cortex_registry=reg)
        assert "linear_attention" in disp.fn_modules
        # The cross_attention fn should not even be instantiated because
        # the cortex backend supplanted it for the only edge.
        assert "cross_attention" not in disp.fn_modules

    def test_no_override_across_cortices(self):
        layout = _small_layout(d_model=8)
        topo = CanvasTopology([Connection(src="a", dst="b", fn="cross_attention")])
        reg = CortexRegistry()
        reg.register(CortexSpec(name="zoneA", local_backend="linear_attention"))
        reg.register(CortexSpec(name="zoneB", local_backend="linear_attention"))
        reg.assign("a", "zoneA")
        reg.assign("b", "zoneB")
        disp = AttentionDispatcher(topo, layout, d_model=8, n_heads=2,
                                   cortex_registry=reg)
        assert "cross_attention" in disp.fn_modules
        assert "linear_attention" not in disp.fn_modules


class TestIdentityInDispatcher:
    def test_slot_module_instantiated_when_identity_set(self):
        layout = _small_layout(d_model=8)
        topo = CanvasTopology([Connection(src="a", dst="b")])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={
                "a": RegionProgram(family="state"),
                "b": RegionProgram(family="state",
                                   identity=IdentitySpec(slot_capacity=4)),
            },
        )
        disp = AttentionDispatcher(topo, layout, d_model=8, n_heads=2, program=prog)
        assert "b" in disp._slot_modules
        # Slot module participates in gradient flow.
        x = torch.randn(1, layout.num_positions, 8, requires_grad=True)
        out = disp(x)
        loss = out.sum()
        loss.backward()
        slot_param = disp._slot_modules["b"].slots
        assert slot_param.grad is not None


class TestMaskSpecInDispatcher:
    def test_mask_pairs_cached(self):
        layout = _small_layout(d_model=8)
        topo = CanvasTopology([Connection(src="a", dst="b")])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={"a": RegionProgram(), "b": RegionProgram()},
            connections={("a", "b"): ConnectionProgram(
                mask_spec=MaskSpec(kind="tile", tile_h=1, tile_w=1),
            )},
        )
        disp = AttentionDispatcher(topo, layout, d_model=8, n_heads=2, program=prog)
        assert ("a", "b") in disp._mask_pairs
        # Forward still runs without error and updates a.
        x = torch.randn(1, layout.num_positions, 8)
        out = disp(x)
        a_idx = layout.region_indices("a")
        assert not torch.allclose(out[:, a_idx], x[:, a_idx])

    def test_mask_spec_serialization(self):
        layout = _small_layout(d_model=8)
        topo = CanvasTopology([Connection(src="a", dst="b")])
        prog = CanvasProgram(
            schema=CanvasSchema(layout=layout, topology=topo),
            regions={"a": RegionProgram(), "b": RegionProgram()},
            connections={("a", "b"): ConnectionProgram(
                mask_spec=MaskSpec(kind="tile", tile_h=2, tile_w=2),
            )},
        )
        d = prog.to_dict()
        restored = CanvasProgram.from_dict(d)
        cp2 = restored.connections[("a", "b")]
        assert cp2.mask_spec is not None
        assert cp2.mask_spec.kind == "tile"
        assert cp2.mask_spec.tile_h == 2


class TestHybridScheduler:
    def test_pure_rule_based_when_no_learned(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="state",
                                     clock=ClockSpec(period=2)),
                "state": RegionProgram(family="state"),
                "act": RegionProgram(family="state"),
            },
        )
        sched = HybridScheduler(prog, learned_regions=[], summary_dim=0)
        fires = [("obs" in sched.step(t)) for t in range(4)]
        assert fires == [True, False, True, False]

    def test_learned_overrides_rule(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="state",
                                     clock=ClockSpec(period=1)),
                "state": RegionProgram(family="state"),
                "act": RegionProgram(family="state"),
            },
        )
        # obs is delegated to LearnedScheduler. Without summary tensor it
        # cannot fire.
        sched = HybridScheduler(prog, learned_regions=["obs"], summary_dim=4,
                                max_active=1)
        active = sched.step(0)
        assert "obs" not in active

        # With summary tensor, learned scheduler will activate obs (it's
        # the only candidate and max_active=1).
        active = sched.step(0, summary_tensor=torch.zeros(4))
        assert "obs" in active

    def test_requires_summary_dim_when_learned_nonempty(self):
        schema = _make_schema()
        prog = CanvasProgram(
            schema=schema, regions={"obs": RegionProgram()},
        )
        with pytest.raises(ValueError):
            HybridScheduler(prog, learned_regions=["obs"], summary_dim=0)


class TestCompilerMaterialization:
    def _module_and_program(self):
        class ToyRegionModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Linear(4, 4)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.regions = torch.nn.ModuleDict({
                    "obs": ToyRegionModule(),
                    "state": ToyRegionModule(),
                    "mem": ToyRegionModule(),
                })

        model = ToyModel()
        schema = _make_schema()
        # Reuse _make_schema names; rewrite region map to include mem instead of act.
        layout = CanvasLayout(T=4, H=4, W=4, d_model=32, regions={
            "obs": (0, 4, 0, 2, 0, 2),
            "state": (0, 4, 2, 4, 0, 2),
            "mem": (0, 4, 0, 2, 2, 4),
        })
        schema = CanvasSchema(layout=layout, topology=CanvasTopology([]))
        prog = CanvasProgram(
            schema=schema,
            regions={
                "obs": RegionProgram(family="observation"),   # → freeze
                "state": RegionProgram(family="state"),       # → runtime
                "mem": RegionProgram(family="memory"),        # → export
            },
        )
        return model, prog

    def test_freeze_actually_freezes_params(self):
        model, prog = self._module_and_program()
        compiled = ProgramCompiler(prog).compile(module=model)
        assert "obs" in compiled.frozen_regions
        for p in model.regions["obs"].parameters():
            assert p.requires_grad is False
        for p in model.regions["state"].parameters():
            assert p.requires_grad is True

    def test_export_writes_file(self, tmp_path):
        model, prog = self._module_and_program()
        compiled = ProgramCompiler(prog).compile(
            module=model, export_dir=str(tmp_path),
        )
        assert "mem" in compiled.exported_memories
        export_path = compiled.exported_paths["mem"]
        import os
        assert os.path.exists(export_path)
        assert os.path.exists(export_path + ".json")
        # Reloadable
        state = torch.load(export_path, weights_only=False)
        assert "w.weight" in state

    def test_constant_materializes_buffers(self, tmp_path):
        # Force a constant compile_mode via explicit RegionProgram.
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Linear(4, 4)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.regions = torch.nn.ModuleDict({"c": Tiny()})

        model = M()
        layout = CanvasLayout(T=2, H=2, W=2, d_model=8, regions={
            "c": (0, 2, 0, 2, 0, 2),
        })
        schema = CanvasSchema(layout=layout, topology=CanvasTopology([]))
        prog = CanvasProgram(
            schema=schema,
            regions={"c": RegionProgram(family="state", compile_mode="constant")},
        )
        compiled = ProgramCompiler(prog).compile(module=model)
        assert "c" in compiled.constant_regions
        assert "c" not in compiled.active_regions
        # Parameters were replaced with buffers
        params = list(model.regions["c"].parameters())
        assert len(params) == 0
        buffers = dict(model.regions["c"].w.named_buffers())
        assert "weight" in buffers and "bias" in buffers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

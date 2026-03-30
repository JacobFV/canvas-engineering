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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

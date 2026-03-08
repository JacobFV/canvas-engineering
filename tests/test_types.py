"""Tests for canvas_engineering.types — compositional type system."""

import pytest
import torch
from dataclasses import dataclass, field as dc_field

from canvas_engineering.types import (
    Field, LayoutStrategy, ConnectivityPolicy, BoundSchema, BoundField,
    compile_schema,
    _walk, _flatten_fields, _pack_strip, _pack_interleaved,
    _intra_connections, _parent_child_connections, _array_element_connections,
    _generate_connections, _apply_temporal, _deduplicate,
    _insert_gateways, _auto_canvas_size,
)
from canvas_engineering.canvas import SpatiotemporalCanvas
from canvas_engineering.connectivity import Connection
from canvas_engineering.schema import CanvasSchema


# ── Test fixtures ────────────────────────────────────────────────────

@dataclass
class SimpleType:
    x: Field = Field(2, 4)
    y: Field = Field(1, 4)


@dataclass
class Agent:
    thought: Field = Field(4, 4)
    goal: Field = Field(2, 4)


@dataclass
class Employee(Agent):
    role: Field = Field(1, 4)


@dataclass
class Product:
    description: Field = Field(3, 4)


@dataclass
class Sensor:
    camera: Field = Field(6, 6)
    depth: Field = Field(6, 6)


@dataclass
class Robot:
    sensor: Sensor = dc_field(default_factory=Sensor)
    plan: Field = Field(2, 4)
    action: Field = Field(1, 4, loss_weight=2.0)


@dataclass
class Company(Agent):
    employees: list = dc_field(default_factory=list)
    products: list = dc_field(default_factory=list)


# ── Field tests ──────────────────────────────────────────────────────

class TestField:
    def test_defaults(self):
        f = Field()
        assert f.h == 1
        assert f.w == 1
        assert f.period == 1
        assert f.is_output is True
        assert f.loss_weight == 1.0
        assert f.attn == "cross_attention"
        assert f.semantic_type is None
        assert f.temporal_extent is None
        assert f.num_positions == 1

    def test_custom(self):
        f = Field(12, 12, period=2, loss_weight=3.0, attn="mamba")
        assert f.h == 12
        assert f.w == 12
        assert f.num_positions == 144
        assert f.period == 2
        assert f.loss_weight == 3.0
        assert f.attn == "mamba"

    def test_scalar_default(self):
        """Field() is a scalar — 1 position."""
        f = Field()
        assert f.num_positions == 1


# ── Tree walking tests ───────────────────────────────────────────────

class TestTreeWalking:
    def test_simple(self):
        node = _walk(SimpleType())
        assert len(node.fields) == 2
        assert node.fields[0].path == "x"
        assert node.fields[0].local_name == "x"
        assert node.fields[1].path == "y"

    def test_inherited(self):
        node = _walk(Employee())
        assert len(node.fields) == 3
        names = [f.local_name for f in node.fields]
        assert "thought" in names
        assert "goal" in names
        assert "role" in names

    def test_nested(self):
        node = _walk(Robot())
        assert len(node.fields) == 2  # plan, action
        assert len(node.children) == 1  # sensor
        sensor_node = node.children[0]
        assert len(sensor_node.fields) == 2  # camera, depth
        assert sensor_node.fields[0].path == "sensor.camera"
        assert sensor_node.fields[1].path == "sensor.depth"

    def test_arrays(self):
        company = Company(
            employees=[Employee(), Employee(), Employee()],
            products=[Product()],
        )
        node = _walk(company)
        assert len(node.fields) == 2  # thought, goal (inherited)
        assert "employees" in node.arrays
        assert len(node.arrays["employees"]) == 3
        assert "products" in node.arrays
        assert len(node.arrays["products"]) == 1

        # Check paths
        emp0 = node.arrays["employees"][0]
        assert emp0.fields[0].path == "employees[0].thought"

    def test_empty_list(self):
        company = Company()
        node = _walk(company)
        assert len(node.arrays) == 0

    def test_flatten(self):
        company = Company(employees=[Employee(), Employee()])
        node = _walk(company)
        flat = _flatten_fields(node)
        paths = [p for p, _, _ in flat]
        assert "thought" in paths
        assert "goal" in paths
        assert "employees[0].thought" in paths
        assert "employees[0].goal" in paths
        assert "employees[0].role" in paths
        assert "employees[1].thought" in paths
        assert len(flat) == 2 + 2 * 3  # 2 company + 2 * 3 employee fields

    def test_nested_flatten(self):
        robot = Robot()
        node = _walk(robot)
        flat = _flatten_fields(node)
        paths = [p for p, _, _ in flat]
        assert "plan" in paths
        assert "action" in paths
        assert "sensor.camera" in paths
        assert "sensor.depth" in paths
        assert len(flat) == 4


# ── Packing tests ────────────────────────────────────────────────────

class TestPacking:
    def test_strip_basic(self):
        fields = [("a", 2, 4), ("b", 2, 4)]
        result = _pack_strip(fields, H=8, W=8)
        assert result["a"] == (0, 2, 0, 4)
        assert result["b"] == (0, 2, 4, 8)

    def test_strip_new_row(self):
        fields = [("a", 2, 4), ("b", 2, 4), ("c", 2, 4)]
        result = _pack_strip(fields, H=8, W=8)
        assert result["a"] == (0, 2, 0, 4)
        assert result["b"] == (0, 2, 4, 8)
        # c doesn't fit in row (4+4=8, next 4 would be 12>8), starts new row
        assert result["c"] == (2, 4, 0, 4)

    def test_strip_mixed_heights(self):
        fields = [("a", 3, 4), ("b", 1, 4)]
        result = _pack_strip(fields, H=8, W=8)
        assert result["a"] == (0, 3, 0, 4)
        assert result["b"] == (0, 1, 4, 8)

    def test_strip_overflow_error(self):
        # Two 5x8 fields need H=10, but grid is only H=8
        fields = [("a", 5, 8), ("b", 5, 8)]
        with pytest.raises(ValueError, match="exceeded"):
            _pack_strip(fields, H=8, W=8)

    def test_strip_width_overflow(self):
        fields = [("a", 2, 10)]
        with pytest.raises(ValueError, match="exceeds"):
            _pack_strip(fields, H=8, W=8)

    def test_interleaved_groups(self):
        fields = [
            ("parent.x", 2, 4, "x"),
            ("child.x", 2, 4, "x"),
            ("parent.y", 1, 4, "y"),
            ("child.y", 1, 4, "y"),
        ]
        result = _pack_interleaved(fields, H=8, W=8)
        # x fields should be in the same row
        assert result["parent.x"][0] == result["child.x"][0]  # same row start
        # y fields should be in the same row (after x)
        assert result["parent.y"][0] == result["child.y"][0]
        # y row starts after x row
        assert result["parent.y"][0] >= result["parent.x"][1]


# ── Connectivity tests ───────────────────────────────────────────────

class TestConnectivity:
    def test_intra_dense(self):
        node = _walk(SimpleType())
        policy = ConnectivityPolicy(intra="dense")
        conns = _intra_connections(node, policy)
        pairs = {(c.src, c.dst) for c in conns}
        assert ("x", "x") in pairs
        assert ("x", "y") in pairs
        assert ("y", "x") in pairs
        assert ("y", "y") in pairs
        assert len(conns) == 4

    def test_intra_isolated(self):
        node = _walk(SimpleType())
        policy = ConnectivityPolicy(intra="isolated")
        conns = _intra_connections(node, policy)
        pairs = {(c.src, c.dst) for c in conns}
        assert ("x", "x") in pairs
        assert ("y", "y") in pairs
        assert len(conns) == 2

    def test_intra_causal_chain(self):
        node = _walk(SimpleType())
        policy = ConnectivityPolicy(intra="causal_chain")
        conns = _intra_connections(node, policy)
        pairs = {(c.src, c.dst) for c in conns}
        assert ("x", "x") in pairs
        assert ("y", "y") in pairs
        assert ("y", "x") in pairs  # y queries x (causal: later reads earlier)
        assert ("x", "y") not in pairs
        assert len(conns) == 3

    def test_intra_star(self):
        @dataclass
        class ThreeFields:
            hub: Field = Field(2, 2)
            a: Field = Field(1, 1)
            b: Field = Field(1, 1)

        node = _walk(ThreeFields())
        policy = ConnectivityPolicy(intra="star")
        conns = _intra_connections(node, policy)
        pairs = {(c.src, c.dst) for c in conns}
        # Hub self-attends
        assert ("hub", "hub") in pairs
        # Spokes self-attend
        assert ("a", "a") in pairs
        assert ("b", "b") in pairs
        # Bidirectional hub-spoke
        assert ("hub", "a") in pairs
        assert ("a", "hub") in pairs
        assert ("hub", "b") in pairs
        assert ("b", "hub") in pairs
        # No spoke-spoke
        assert ("a", "b") not in pairs

    def test_parent_child_matched_fields(self):
        company = Company(employees=[Employee()])
        node = _walk(company)
        emp_node = node.arrays["employees"][0]
        policy = ConnectivityPolicy(parent_child="matched_fields")
        conns = _parent_child_connections(node, emp_node, policy)
        pairs = {(c.src, c.dst) for c in conns}
        # thought matches thought
        assert ("thought", "employees[0].thought") in pairs
        assert ("employees[0].thought", "thought") in pairs
        # goal matches goal
        assert ("goal", "employees[0].goal") in pairs
        assert ("employees[0].goal", "goal") in pairs
        # role has no match in parent
        assert not any(c.src == "employees[0].role" or c.dst == "employees[0].role"
                       for c in conns if "role" in c.src or "role" in c.dst)

    def test_parent_child_hub_spoke(self):
        robot = Robot()
        node = _walk(robot)
        sensor_node = node.children[0]
        policy = ConnectivityPolicy(parent_child="hub_spoke")
        conns = _parent_child_connections(node, sensor_node, policy)
        # All parent fields connect to all child fields bidirectionally
        assert len(conns) == 2 * 2 * 2  # 2 parent fields * 2 child fields * 2 directions

    def test_parent_child_none(self):
        robot = Robot()
        node = _walk(robot)
        sensor_node = node.children[0]
        policy = ConnectivityPolicy(parent_child="none")
        conns = _parent_child_connections(node, sensor_node, policy)
        assert len(conns) == 0

    def test_array_element_isolated(self):
        company = Company(employees=[Employee(), Employee()])
        node = _walk(company)
        elements = node.arrays["employees"]
        policy = ConnectivityPolicy(array_element="isolated")
        conns = _array_element_connections(elements, policy)
        assert len(conns) == 0

    def test_array_element_dense(self):
        company = Company(employees=[Employee(), Employee()])
        node = _walk(company)
        elements = node.arrays["employees"]
        policy = ConnectivityPolicy(array_element="dense")
        conns = _array_element_connections(elements, policy)
        # Each element has 3 fields, cross-connects to other element's 3 fields
        # 2 elements, 3*3 per direction, 2 directions (i→j and j→i)
        assert len(conns) == 2 * 3 * 3  # 18

    def test_array_element_matched_fields(self):
        company = Company(employees=[Employee(), Employee()])
        node = _walk(company)
        elements = node.arrays["employees"]
        policy = ConnectivityPolicy(array_element="matched_fields")
        conns = _array_element_connections(elements, policy)
        pairs = {(c.src, c.dst) for c in conns}
        # thought↔thought, goal↔goal, role↔role across elements
        assert ("employees[0].thought", "employees[1].thought") in pairs
        assert ("employees[1].thought", "employees[0].thought") in pairs
        # 3 matching fields * 2 directions
        assert len(conns) == 6

    def test_array_element_ring(self):
        company = Company(employees=[Employee(), Employee(), Employee()])
        node = _walk(company)
        elements = node.arrays["employees"]
        policy = ConnectivityPolicy(array_element="ring")
        conns = _array_element_connections(elements, policy)
        src_dst = {(c.src.split(".")[0], c.dst.split(".")[0]) for c in conns}
        # Ring: 0↔1, 1↔2, 2↔0
        assert ("employees[0]", "employees[1]") in src_dst
        assert ("employees[1]", "employees[0]") in src_dst
        assert ("employees[1]", "employees[2]") in src_dst
        assert ("employees[2]", "employees[1]") in src_dst
        assert ("employees[2]", "employees[0]") in src_dst
        assert ("employees[0]", "employees[2]") in src_dst


# ── Temporal policy tests ────────────────────────────────────────────

class TestTemporal:
    def test_dense_passthrough(self):
        conns = [Connection(src="a", dst="b")]
        result = _apply_temporal(conns, ConnectivityPolicy(temporal="dense"))
        assert len(result) == 1
        assert result[0].t_src is None
        assert result[0].t_dst is None

    def test_same_frame(self):
        conns = [Connection(src="a", dst="b")]
        result = _apply_temporal(conns, ConnectivityPolicy(temporal="same_frame"))
        assert len(result) == 1
        assert result[0].t_src == 0
        assert result[0].t_dst == 0

    def test_causal_self(self):
        conns = [Connection(src="a", dst="a")]
        result = _apply_temporal(conns, ConnectivityPolicy(temporal="causal"))
        assert len(result) == 2
        # Same-frame self
        assert any(c.t_src == 0 and c.t_dst == 0 for c in result)
        # Prev-frame self
        assert any(c.t_src == 0 and c.t_dst == -1 for c in result)

    def test_causal_cross(self):
        conns = [Connection(src="a", dst="b")]
        result = _apply_temporal(conns, ConnectivityPolicy(temporal="causal"))
        assert len(result) == 1
        assert result[0].t_src == 0
        assert result[0].t_dst == -1


# ── Deduplication tests ──────────────────────────────────────────────

class TestDeduplicate:
    def test_removes_duplicates(self):
        conns = [
            Connection(src="a", dst="b"),
            Connection(src="a", dst="b"),
            Connection(src="b", dst="a"),
        ]
        result = _deduplicate(conns)
        assert len(result) == 2

    def test_preserves_different_weights(self):
        conns = [
            Connection(src="a", dst="b", weight=1.0),
            Connection(src="a", dst="b", weight=0.5),
        ]
        result = _deduplicate(conns)
        assert len(result) == 2


# ── Full compilation tests ───────────────────────────────────────────

class TestCompileSchema:
    def test_simple(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        assert "x" in bound
        assert "y" in bound
        assert len(bound.field_names) == 2

    def test_regions_correct(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        x_spec = bound["x"].spec
        assert x_spec.bounds[1] - x_spec.bounds[0] == 4  # full T
        h = x_spec.bounds[3] - x_spec.bounds[2]
        w = x_spec.bounds[5] - x_spec.bounds[4]
        assert h == 2
        assert w == 4

    def test_loss_weight_propagated(self):
        robot = Robot()
        bound = compile_schema(robot, T=4, H=16, W=16, d_model=128)
        assert bound["action"].spec.loss_weight == 2.0

    def test_temporal_extent(self):
        @dataclass
        class Mixed:
            full: Field = Field(2, 4)
            static: Field = Field(2, 4, temporal_extent=1)

        bound = compile_schema(Mixed(), T=8, H=8, W=8, d_model=128)
        assert bound["full"].spec.bounds[0] == 0
        assert bound["full"].spec.bounds[1] == 8   # full T
        assert bound["static"].spec.bounds[0] == 0
        assert bound["static"].spec.bounds[1] == 1  # temporal_extent=1

    def test_nested(self):
        robot = Robot()
        bound = compile_schema(robot, T=4, H=16, W=16, d_model=128)
        assert "plan" in bound
        assert "action" in bound
        assert "sensor.camera" in bound
        assert "sensor.depth" in bound
        assert len(bound.field_names) == 4

    def test_arrays(self):
        company = Company(
            employees=[Employee(), Employee()],
            products=[Product()],
        )
        bound = compile_schema(company, T=4, H=32, W=32, d_model=128)
        assert "thought" in bound
        assert "goal" in bound
        assert "employees[0].thought" in bound
        assert "employees[0].role" in bound
        assert "employees[1].thought" in bound
        assert "products[0].description" in bound
        # 2 company + 2*3 employee + 1*1 product = 9
        assert len(bound.field_names) == 9

    def test_connectivity_generated(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        assert bound.topology is not None
        conns = bound.topology.connections
        # Dense intra: x↔x, x↔y, y↔x, y↔y = 4
        assert len(conns) == 4

    def test_matched_fields_connectivity(self):
        company = Company(employees=[Employee()])
        bound = compile_schema(company, T=4, H=16, W=16, d_model=128)
        conns = bound.topology.connections
        pairs = {(c.src, c.dst) for c in conns}
        # Parent-child matched: thought↔thought, goal↔goal
        assert ("thought", "employees[0].thought") in pairs
        assert ("employees[0].thought", "thought") in pairs

    def test_isolated_array_elements(self):
        company = Company(employees=[Employee(), Employee()])
        bound = compile_schema(company, T=4, H=32, W=32, d_model=128)
        conns = bound.topology.connections
        pairs = {(c.src, c.dst) for c in conns}
        # Elements should NOT connect to each other (isolated default)
        assert ("employees[0].thought", "employees[1].thought") not in pairs

    def test_dense_array_elements(self):
        company = Company(employees=[Employee(), Employee()])
        policy = ConnectivityPolicy(array_element="dense")
        bound = compile_schema(
            company, T=4, H=32, W=32, d_model=128, connectivity=policy)
        conns = bound.topology.connections
        pairs = {(c.src, c.dst) for c in conns}
        assert ("employees[0].thought", "employees[1].thought") in pairs

    def test_causal_temporal(self):
        policy = ConnectivityPolicy(temporal="causal")
        bound = compile_schema(
            SimpleType(), T=4, H=8, W=8, d_model=128, connectivity=policy)
        conns = bound.topology.connections
        # Self-connections get split into same-frame + prev-frame
        self_conns = [c for c in conns if c.src == "x" and c.dst == "x"]
        assert any(c.t_src == 0 and c.t_dst == 0 for c in self_conns)
        assert any(c.t_src == 0 and c.t_dst == -1 for c in self_conns)

    def test_no_connectivity(self):
        policy = ConnectivityPolicy(intra="isolated", parent_child="none")
        bound = compile_schema(
            SimpleType(), T=4, H=8, W=8, d_model=128, connectivity=policy)
        conns = bound.topology.connections
        # Only self-loops
        assert all(c.src == c.dst for c in conns)

    def test_interleaved_layout(self):
        company = Company(employees=[Employee()])
        bound = compile_schema(
            company, T=4, H=32, W=32, d_model=128,
            layout_strategy=LayoutStrategy.INTERLEAVED)
        # Thought fields should be in the same row
        co_thought = bound["thought"].spec.bounds
        emp_thought = bound["employees[0].thought"].spec.bounds
        assert co_thought[2] == emp_thought[2]  # same h0 (same row)

    def test_layout_string(self):
        bound = compile_schema(
            SimpleType(), T=4, H=8, W=8, d_model=128,
            layout_strategy="packed")
        assert len(bound.field_names) == 2

    def test_no_fields_error(self):
        @dataclass
        class Empty:
            x: int = 5

        with pytest.raises(ValueError, match="No Field"):
            compile_schema(Empty(), T=4, H=8, W=8, d_model=128)

    def test_grid_too_small(self):
        @dataclass
        class Big:
            a: Field = Field(10, 10)

        with pytest.raises(ValueError):
            compile_schema(Big(), T=4, H=8, W=8, d_model=128)

    def test_attn_propagated(self):
        @dataclass
        class WithAttn:
            fast: Field = Field(2, 4, attn="linear_attention")
            slow: Field = Field(2, 4, attn="mamba")

        bound = compile_schema(WithAttn(), T=4, H=8, W=8, d_model=128)
        assert bound["fast"].spec.default_attn == "linear_attention"
        assert bound["slow"].spec.default_attn == "mamba"

    def test_is_output_propagated(self):
        @dataclass
        class WithInput:
            obs: Field = Field(4, 4)
            prompt: Field = Field(2, 4, is_output=False)

        bound = compile_schema(WithInput(), T=4, H=8, W=8, d_model=128)
        assert bound["obs"].spec.is_output is True
        assert bound["prompt"].spec.is_output is False

    def test_period_propagated(self):
        @dataclass
        class MultiFreq:
            fast: Field = Field(2, 4, period=1)
            slow: Field = Field(2, 4, period=4)

        bound = compile_schema(MultiFreq(), T=8, H=8, W=8, d_model=128)
        assert bound["fast"].spec.period == 1
        assert bound["slow"].spec.period == 4


# ── BoundSchema / BoundField tests ──────────────────────────────────

class TestBoundSchema:
    def test_getitem(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        bf = bound["x"]
        assert isinstance(bf, BoundField)
        assert bf.region_name == "x"

    def test_contains(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        assert "x" in bound
        assert "z" not in bound

    def test_layout_shortcut(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        assert bound.layout.T == 4
        assert bound.layout.d_model == 128

    def test_topology_shortcut(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        assert bound.topology is not None

    def test_build_canvas(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=64)
        canvas = bound.build_canvas()
        assert isinstance(canvas, SpatiotemporalCanvas)

    def test_create_batch(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=64)
        batch = bound.create_batch(2)
        assert batch.shape == (2, 4 * 8 * 8, 64)

    def test_place_extract(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=64)
        canvas = bound.build_canvas()
        batch = bound.create_batch(2)
        n_pos = bound["x"].num_positions
        embs = torch.randn(2, n_pos, 64)
        batch = bound["x"].place(batch, embs)
        extracted = bound["x"].extract(batch)
        assert extracted.shape == (2, n_pos, 64)

    def test_place_without_canvas_errors(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=64)
        batch = torch.zeros(2, 4 * 8 * 8, 64)
        with pytest.raises(RuntimeError, match="No canvas"):
            bound["x"].place(batch, torch.zeros(2, 10, 64))

    def test_place_with_explicit_canvas(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=64)
        canvas = SpatiotemporalCanvas(bound.layout)
        batch = canvas.create_empty(2)
        n_pos = bound["x"].num_positions
        embs = torch.randn(2, n_pos, 64)
        batch = bound["x"].place(batch, embs, canvas=canvas)
        assert batch.shape == (2, 4 * 8 * 8, 64)

    def test_summary(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        s = bound.summary()
        assert "BoundSchema" in s
        assert "x" in s
        assert "y" in s

    def test_repr(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        r = repr(bound)
        assert "BoundSchema" in r

    def test_bound_field_repr(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        r = repr(bound["x"])
        assert "BoundField" in r
        assert "x" in r

    def test_bound_field_indices(self):
        bound = compile_schema(SimpleType(), T=4, H=8, W=8, d_model=128)
        indices = bound["x"].indices()
        assert len(indices) == bound["x"].num_positions


# ── Schema round-trip test ───────────────────────────────────────────

class TestSchemaRoundTrip:
    def test_to_json_from_json(self, tmp_path):
        company = Company(
            employees=[Employee(), Employee()],
            products=[Product()],
        )
        bound = compile_schema(company, T=4, H=32, W=32, d_model=128)
        path = tmp_path / "schema.json"
        bound.schema.to_json(str(path))
        loaded = CanvasSchema.from_json(str(path))
        assert set(loaded.layout.regions.keys()) == set(bound.layout.regions.keys())
        if loaded.topology and bound.topology:
            assert len(loaded.topology.connections) == len(bound.topology.connections)


# ── Full integration: Company example ────────────────────────────────

class TestCompanyIntegration:
    def test_full_company(self):
        """The motivating example from the spec."""
        company = Company(
            employees=[
                Employee(thought=Field(8, 8)),   # CEO: bigger thought
                Employee(),                       # default employee
                Employee(),
            ],
            products=[Product(), Product()],
        )
        bound = compile_schema(
            company, T=4, H=64, W=64, d_model=256,
            connectivity=ConnectivityPolicy(
                intra="dense",
                parent_child="matched_fields",
                array_element="isolated",
            ),
        )

        # Correct number of fields
        # Company: thought, goal (2)
        # 3 employees × 3 fields (thought, goal, role) = 9
        # 2 products × 1 field (description) = 2
        assert len(bound.field_names) == 13

        # CEO got bigger thought
        ceo_thought = bound["employees[0].thought"]
        default_thought = bound["employees[1].thought"]
        assert ceo_thought.spec.bounds != default_thought.spec.bounds
        ceo_h = ceo_thought.spec.bounds[3] - ceo_thought.spec.bounds[2]
        ceo_w = ceo_thought.spec.bounds[5] - ceo_thought.spec.bounds[4]
        assert ceo_h == 8 and ceo_w == 8

        # Matched fields connectivity exists
        conns = bound.topology.connections
        pairs = {(c.src, c.dst) for c in conns}
        assert ("thought", "employees[0].thought") in pairs
        assert ("thought", "employees[1].thought") in pairs
        assert ("goal", "employees[0].goal") in pairs

        # Products have no matched fields with Company
        assert not any(
            "products" in c.src and c.dst in ("thought", "goal")
            for c in conns
        )

        # Can build canvas and create batch
        canvas = bound.build_canvas()
        batch = bound.create_batch(2)
        assert batch.shape[0] == 2
        assert batch.shape[2] == 256

    def test_company_interleaved(self):
        """Interleaved layout: thoughts are spatially adjacent."""
        company = Company(employees=[Employee(), Employee()])
        bound = compile_schema(
            company, T=4, H=32, W=32, d_model=128,
            layout_strategy=LayoutStrategy.INTERLEAVED,
        )
        # All thought fields should be in the same row
        co_h0 = bound["thought"].spec.bounds[2]
        e0_h0 = bound["employees[0].thought"].spec.bounds[2]
        e1_h0 = bound["employees[1].thought"].spec.bounds[2]
        assert co_h0 == e0_h0 == e1_h0


# ── Plain class support ──────────────────────────────────────────────

class TestPlainClass:
    def test_plain_object(self):
        class MyType:
            def __init__(self):
                self.x = Field(2, 4)
                self.y = Field(1, 4)

        bound = compile_schema(MyType(), T=4, H=8, W=8, d_model=128)
        assert "x" in bound
        assert "y" in bound

    def test_nested_plain(self):
        class Inner:
            def __init__(self):
                self.a = Field(2, 2)

        class Outer:
            def __init__(self):
                self.inner = Inner()
                self.b = Field(1, 2)

        bound = compile_schema(Outer(), T=4, H=8, W=8, d_model=128)
        assert "inner.a" in bound
        assert "b" in bound


# ── Auto-sizing tests ───────────────────────────────────────────────

class TestAutoSizing:
    def test_auto_size_simple(self):
        """Auto-sizing should produce a canvas that fits all fields."""
        bound = compile_schema(SimpleType(), T=4, d_model=64)
        # Should succeed (H, W auto-computed)
        assert "x" in bound
        assert "y" in bound

    def test_auto_size_nested(self):
        """Auto-sizing with nested types."""
        bound = compile_schema(Robot(), T=4, d_model=64)
        assert "sensor.camera" in bound
        assert "plan" in bound
        assert "action" in bound

    def test_auto_size_company(self):
        """Auto-sizing with arrays."""
        company = Company(
            employees=[Employee(), Employee()],
            products=[Product()],
        )
        bound = compile_schema(company, T=1, d_model=64)
        assert "employees[0].thought" in bound
        assert "products[0].description" in bound

    def test_auto_size_matches_manual(self):
        """Auto-sized H, W should be >= what the fields need."""
        fields = [(p, f.h, f.w) for p, f, _ in _flatten_fields(_walk(SimpleType()))]
        H, W = _auto_canvas_size(fields)
        # Should be able to pack with the computed size
        _pack_strip(fields, H, W)  # should not raise

    def test_auto_size_only_h(self):
        """Can auto-size H while specifying W."""
        bound = compile_schema(SimpleType(), T=4, W=8, d_model=64)
        assert bound.layout.W == 8

    def test_auto_size_only_w(self):
        """Can auto-size W while specifying H."""
        bound = compile_schema(SimpleType(), T=4, H=8, d_model=64)
        assert bound.layout.H == 8

    def test_auto_size_empty_returns_minimum(self):
        """Empty field list should return minimum canvas."""
        H, W = _auto_canvas_size([])
        assert H >= 4 and W >= 4


# ── Bottleneck / gateway tests ──────────────────────────────────────

class TestBottleneck:
    def test_gateway_created_for_nested_type(self):
        """Bottleneck should create a gateway field at the child's path."""
        bound = compile_schema(
            Robot(), T=4, H=16, W=16, d_model=64,
            connectivity=ConnectivityPolicy(parent_child="bottleneck"),
        )
        # Robot has nested Sensor — should get a gateway at "sensor"
        assert "sensor" in bound
        # Original fields should still exist
        assert "sensor.camera" in bound
        assert "sensor.depth" in bound
        assert "plan" in bound
        assert "action" in bound

    def test_gateway_is_1x1(self):
        """Gateway fields should be 1×1."""
        bound = compile_schema(
            Robot(), T=4, H=16, W=16, d_model=64,
            connectivity=ConnectivityPolicy(parent_child="bottleneck"),
        )
        gw = bound["sensor"]
        t0, t1, h0, h1, w0, w1 = gw.spec.bounds
        assert (h1 - h0) == 1 and (w1 - w0) == 1

    def test_gateway_connectivity(self):
        """Gateway should connect to both parent and child fields."""
        bound = compile_schema(
            Robot(), T=1, H=16, W=16, d_model=64,
            connectivity=ConnectivityPolicy(parent_child="bottleneck"),
        )
        conns = bound.topology.connections
        # Gateway "sensor" should connect to parent fields ("plan", "action")
        gw_to_parent = [c for c in conns
                        if (c.src == "sensor" and c.dst in ("plan", "action"))
                        or (c.dst == "sensor" and c.src in ("plan", "action"))]
        assert len(gw_to_parent) > 0

        # Gateway "sensor" should connect to child fields ("sensor.camera", "sensor.depth")
        gw_to_child = [c for c in conns
                       if (c.src == "sensor" and c.dst in ("sensor.camera", "sensor.depth"))
                       or (c.dst == "sensor" and c.src in ("sensor.camera", "sensor.depth"))]
        assert len(gw_to_child) > 0

    def test_no_direct_parent_child_connections(self):
        """With bottleneck, parent fields should NOT connect directly to child fields."""
        bound = compile_schema(
            Robot(), T=1, H=16, W=16, d_model=64,
            connectivity=ConnectivityPolicy(parent_child="bottleneck"),
        )
        conns = bound.topology.connections
        # "plan" should NOT connect directly to "sensor.camera"
        direct = [c for c in conns
                  if (c.src == "plan" and c.dst == "sensor.camera")
                  or (c.src == "sensor.camera" and c.dst == "plan")]
        assert len(direct) == 0

    def test_gateway_for_array_elements(self):
        """Each array element should get its own gateway."""
        company = Company(
            employees=[Employee(), Employee(), Employee()],
        )
        bound = compile_schema(
            company, T=1, H=32, W=32, d_model=64,
            connectivity=ConnectivityPolicy(parent_child="bottleneck"),
        )
        # Each employee should have a gateway at "employees[i]"
        assert "employees[0]" in bound
        assert "employees[1]" in bound
        assert "employees[2]" in bound
        # Original fields should still exist
        assert "employees[0].thought" in bound
        assert "employees[1].goal" in bound

    def test_no_cross_element_direct_connections(self):
        """Array elements should interact through gateways, not directly."""
        company = Company(
            employees=[Employee(), Employee()],
        )
        bound = compile_schema(
            company, T=1, H=32, W=32, d_model=64,
            connectivity=ConnectivityPolicy(
                parent_child="bottleneck",
                array_element="isolated",
            ),
        )
        conns = bound.topology.connections
        # employees[0].thought should NOT connect to employees[1].thought
        cross = [c for c in conns
                 if "employees[0]." in c.src and "employees[1]." in c.dst]
        assert len(cross) == 0

    def test_hierarchical_gateways(self):
        """Deep nesting should create gateways at each level."""
        @dataclass
        class Inner:
            x: Field = Field(2, 2)

        @dataclass
        class Middle:
            inner: Inner = dc_field(default_factory=Inner)
            y: Field = Field(1, 2)

        @dataclass
        class Outer:
            middle: Middle = dc_field(default_factory=Middle)
            z: Field = Field(1, 2)

        bound = compile_schema(
            Outer(), T=1, H=16, W=16, d_model=64,
            connectivity=ConnectivityPolicy(parent_child="bottleneck"),
        )
        # Gateway for Middle
        assert "middle" in bound
        # Gateway for Inner (inside Middle's subtree)
        assert "middle.inner" in bound
        # Original fields
        assert "middle.inner.x" in bound
        assert "middle.y" in bound
        assert "z" in bound

    def test_bottleneck_with_auto_sizing(self):
        """Bottleneck + auto-sizing should work together."""
        company = Company(
            employees=[Employee(), Employee()],
            products=[Product()],
        )
        bound = compile_schema(
            company, T=1, d_model=64,
            connectivity=ConnectivityPolicy(parent_child="bottleneck"),
        )
        # Should have gateways and be auto-sized
        assert "employees[0]" in bound
        assert "products[0]" in bound
        assert bound.layout.H > 0
        assert bound.layout.W > 0

    def test_gateway_semantic_type(self):
        """Gateway fields should have descriptive semantic types."""
        bound = compile_schema(
            Robot(), T=1, H=16, W=16, d_model=64,
            connectivity=ConnectivityPolicy(parent_child="bottleneck"),
        )
        gw = bound["sensor"]
        assert "gateway" in gw.spec.semantic_type
        assert "sensor" in gw.spec.semantic_type

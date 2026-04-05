"""Tests for N-dimensional spatial support.

Verifies that 1D, 2D, and 3D spatial canvases work correctly and that
the new spatial_shape API produces identical results to the old H/W API.
"""
import math
import torch
import pytest
from dataclasses import dataclass

from canvas_engineering.canvas import (
    CanvasLayout, RegionSpec, SpatiotemporalCanvas,
    SinusoidalPositionalEncodingND, SinusoidalPositionalEncoding3D,
    _parse_bounds,
)
from canvas_engineering.types import Field, compile_schema
from canvas_engineering.masks import Rect, rect_cover, MaskSpec, mask_to_index_pairs
from canvas_engineering.schema import CanvasSchema
from canvas_engineering.connectivity import CanvasTopology


# =====================================================================
# Phase 1: CanvasLayout — core data model
# =====================================================================

class TestCanvasLayout1D:
    def test_basic(self):
        layout = CanvasLayout(T=4, spatial_shape=(16,), d_model=32)
        assert layout.num_positions == 64
        assert layout.n_spatial_dims == 1
        assert layout.spatial_volume == 16
        assert layout.spatial_shape == (16,)

    def test_region_indices(self):
        layout = CanvasLayout(
            T=2, spatial_shape=(8,), d_model=32,
            regions={"a": (0, 1, 2, 5)},  # t=0, x=2..5
        )
        indices = layout.region_indices("a")
        assert indices == [2, 3, 4]

    def test_region_indices_at_t(self):
        layout = CanvasLayout(
            T=3, spatial_shape=(4,), d_model=32,
            regions={"a": (1, 3, 0, 2)},  # t=1..3, x=0..2
        )
        assert layout.region_indices_at_t("a", 0) == []
        assert layout.region_indices_at_t("a", 1) == [4, 5]  # t=1: base=4
        assert layout.region_indices_at_t("a", 2) == [8, 9]  # t=2: base=8

    def test_output_mask(self):
        layout = CanvasLayout(T=3, spatial_shape=(4,), d_model=32, t_current=2)
        mask = layout.output_mask()
        assert len(mask) == 4  # 1 future timestep x 4 positions

    def test_flat_index(self):
        layout = CanvasLayout(T=2, spatial_shape=(8,), d_model=32)
        assert layout.flat_index(0, 0) == 0
        assert layout.flat_index(0, 3) == 3
        assert layout.flat_index(1, 0) == 8
        assert layout.flat_index(1, 5) == 13

    def test_region_size(self):
        layout = CanvasLayout(
            T=4, spatial_shape=(10,), d_model=32,
            regions={"a": (1, 3, 2, 7)},
        )
        assert layout.region_size("a") == (2, 5)
        assert layout.region_numel("a") == 10

    def test_strides(self):
        layout = CanvasLayout(T=2, spatial_shape=(8,), d_model=32)
        assert layout._strides == (8, 1)


class TestCanvasLayout3D:
    def test_basic(self):
        layout = CanvasLayout(T=2, spatial_shape=(4, 4, 4), d_model=64)
        assert layout.num_positions == 128  # 2 * 64
        assert layout.n_spatial_dims == 3
        assert layout.spatial_volume == 64

    def test_region_indices(self):
        layout = CanvasLayout(
            T=1, spatial_shape=(4, 4, 4), d_model=32,
            regions={"a": (0, 1, 0, 2, 0, 2, 0, 2)},
        )
        indices = layout.region_indices("a")
        assert len(indices) == 8  # 1*2*2*2

    def test_flat_index(self):
        layout = CanvasLayout(T=1, spatial_shape=(3, 4, 5), d_model=32)
        # strides = (60, 20, 5, 1)
        assert layout.flat_index(0, 0, 0, 0) == 0
        assert layout.flat_index(0, 0, 0, 1) == 1
        assert layout.flat_index(0, 0, 1, 0) == 5
        assert layout.flat_index(0, 1, 0, 0) == 20

    def test_strides(self):
        layout = CanvasLayout(T=2, spatial_shape=(3, 4, 5), d_model=32)
        assert layout._strides == (60, 20, 5, 1)

    def test_output_mask(self):
        layout = CanvasLayout(
            T=2, spatial_shape=(2, 2, 2), d_model=32, t_current=1
        )
        mask = layout.output_mask()
        assert len(mask) == 8  # 1 future timestep x 2*2*2


class TestCanvasLayoutBackwardCompat:
    def test_hw_still_works(self):
        layout = CanvasLayout(T=5, H=8, W=8, d_model=64)
        assert layout.spatial_shape == (8, 8)
        assert layout.H == 8
        assert layout.W == 8
        assert layout.num_positions == 320

    def test_spatial_shape_sets_hw(self):
        layout = CanvasLayout(T=5, spatial_shape=(8, 8), d_model=64)
        assert layout.H == 8
        assert layout.W == 8

    def test_1d_has_no_w(self):
        layout = CanvasLayout(T=4, spatial_shape=(16,), d_model=32)
        assert layout.H == 16
        assert layout.W is None

    def test_cannot_specify_both(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            CanvasLayout(T=5, spatial_shape=(8, 8), d_model=64, H=8, W=8)


# =====================================================================
# Phase 1: Field
# =====================================================================

class TestFieldND:
    def test_spatial_shape_1d(self):
        f = Field(spatial_shape=(8,))
        assert f.spatial_shape == (8,)
        assert f.num_positions == 8
        assert f.h == 8  # backfilled

    def test_spatial_shape_3d(self):
        f = Field(spatial_shape=(2, 3, 4))
        assert f.spatial_shape == (2, 3, 4)
        assert f.num_positions == 24
        assert f.h == 2
        assert f.w == 3

    def test_hw_backward_compat(self):
        f = Field(4, 8)
        assert f.spatial_shape == (4, 8)
        assert f.h == 4
        assert f.w == 8
        assert f.num_positions == 32

    def test_default(self):
        f = Field()
        assert f.spatial_shape == (1, 1)
        assert f.num_positions == 1


# =====================================================================
# Phase 2: Packing and compile_schema
# =====================================================================

class TestCompileSchema1D:
    def test_auto_sized(self):
        @dataclass
        class Simple1D:
            features: Field = Field(spatial_shape=(8,))
            output: Field = Field(spatial_shape=(4,))

        bound = compile_schema(Simple1D(), T=2, d_model=32)
        layout = bound.layout
        assert layout.n_spatial_dims == 1
        assert layout.T == 2
        # Both fields should be packed into 1D spatial shape
        assert "features" in bound
        assert "output" in bound
        # Total positions = T * spatial_volume
        assert layout.num_positions == layout.T * layout.spatial_volume
        # Verify indices are valid
        for name in bound.field_names:
            indices = layout.region_indices(name)
            assert all(0 <= i < layout.num_positions for i in indices)

    def test_explicit_spatial(self):
        @dataclass
        class Simple1D:
            x: Field = Field(spatial_shape=(4,))

        bound = compile_schema(Simple1D(), T=3, spatial_shape=(16,), d_model=32)
        assert bound.layout.spatial_shape == (16,)

    def test_2d_via_hw(self):
        """Backward compat: H/W kwargs still work."""
        @dataclass
        class Robot:
            camera: Field = Field(4, 4)
            action: Field = Field(1, 2)

        bound = compile_schema(Robot(), T=2, H=8, W=8, d_model=32)
        assert bound.layout.spatial_shape == (8, 8)
        assert bound.layout.H == 8


class TestCompileSchema3D:
    def test_basic(self):
        @dataclass
        class Volumetric:
            voxels: Field = Field(spatial_shape=(2, 2, 2))
            scalar: Field = Field(spatial_shape=(1, 1, 1))

        bound = compile_schema(
            Volumetric(), T=2, spatial_shape=(4, 4, 4), d_model=32
        )
        layout = bound.layout
        assert layout.n_spatial_dims == 3
        assert layout.spatial_shape == (4, 4, 4)
        # Verify indices
        for name in bound.field_names:
            indices = layout.region_indices(name)
            assert all(0 <= i < layout.num_positions for i in indices)


# =====================================================================
# Phase 3: Positional encoding
# =====================================================================

class TestPEND:
    def test_1d(self):
        pe = SinusoidalPositionalEncodingND(d_model=32, max_T=4, max_spatial=(8,))
        out = pe(T=2, spatial_shape=(4,))
        assert out.shape == (2, 4, 32)

    def test_2d(self):
        pe = SinusoidalPositionalEncodingND(d_model=64, max_T=4, max_spatial=(8, 8))
        out = pe(T=2, spatial_shape=(4, 4))
        assert out.shape == (2, 4, 4, 64)

    def test_3d(self):
        pe = SinusoidalPositionalEncodingND(d_model=64, max_T=4, max_spatial=(4, 4, 4))
        out = pe(T=2, spatial_shape=(2, 2, 2))
        assert out.shape == (2, 2, 2, 2, 64)

    def test_backward_compat_3d_class(self):
        pe = SinusoidalPositionalEncoding3D(d_model=63, max_T=4, max_H=8, max_W=8)
        out = pe(T=2, H=4, W=4)
        assert out.shape == (2, 4, 4, 63)


# =====================================================================
# Phase 4: Masks and Rect
# =====================================================================

class TestRectND:
    def test_compat_construction(self):
        r = Rect(0, 2, 0, 3, 0, 4)
        assert r.volume == 24
        assert r.t0 == 0
        assert r.t1 == 2
        assert r.h0 == 0
        assert r.h1 == 3

    def test_ranges_construction(self):
        r = Rect(ranges=((0, 2), (0, 3), (0, 4)))
        assert r.volume == 24

    def test_1d_rect(self):
        r = Rect(ranges=((0, 3), (2, 8)))
        assert r.volume == 18
        assert r.t0 == 0
        assert r.t1 == 3


class TestRectCoverND:
    def test_2d_mask(self):
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[0, 0:3] = True
        mask[1, 1:4] = True
        rects = rect_cover(mask)
        # Should cover all True positions
        covered = set()
        for r in rects:
            for t in range(r.ranges[0][0], r.ranges[0][1]):
                for x in range(r.ranges[1][0], r.ranges[1][1]):
                    covered.add((t, x))
        expected = {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3)}
        assert covered == expected

    def test_3d_mask(self):
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        rects = rect_cover(mask)
        total = sum(r.volume for r in rects)
        assert total == 24  # covers all positions


class TestMaskToIndexPairsND:
    def test_tile_1d(self):
        layout = CanvasLayout(
            T=1, spatial_shape=(8,), d_model=32,
            regions={
                "src": (0, 1, 0, 8),
                "dst": (0, 1, 0, 8),
            },
        )
        spec = MaskSpec(kind="tile", tile_shape=(4,))
        pairs = mask_to_index_pairs(spec, layout, "src", "dst")
        assert len(pairs) == 2  # 8 / 4 = 2 tiles


# =====================================================================
# Phase 5: Serialization
# =====================================================================

class TestSchemaNDSerialization:
    def test_roundtrip_1d(self):
        layout = CanvasLayout(
            T=4, spatial_shape=(16,), d_model=32,
            regions={"x": RegionSpec(bounds=(0, 4, 2, 10))},
        )
        schema = CanvasSchema(layout=layout)
        d = schema.to_dict()
        assert d["layout"]["spatial_shape"] == [16]
        loaded = CanvasSchema.from_dict(d)
        assert loaded.layout.spatial_shape == (16,)
        assert loaded.layout.T == 4
        assert loaded.layout.region_indices("x") == layout.region_indices("x")

    def test_roundtrip_3d(self):
        layout = CanvasLayout(
            T=2, spatial_shape=(4, 4, 4), d_model=64,
            regions={"v": RegionSpec(bounds=(0, 2, 0, 2, 0, 2, 0, 2))},
        )
        schema = CanvasSchema(layout=layout)
        d = schema.to_dict()
        assert d["layout"]["spatial_shape"] == [4, 4, 4]
        loaded = CanvasSchema.from_dict(d)
        assert loaded.layout.spatial_shape == (4, 4, 4)

    def test_from_old_format(self):
        """Load old-format dict with H and W keys."""
        d = {
            "schema_version": "0.2.0",
            "layout": {"T": 4, "H": 8, "W": 8, "d_model": 64, "t_current": 0},
            "regions": {"a": {"bounds": [0, 4, 0, 4, 0, 4]}},
        }
        schema = CanvasSchema.from_dict(d)
        assert schema.layout.spatial_shape == (8, 8)
        assert schema.layout.H == 8


# =====================================================================
# Golden test: new API == old API for 2D
# =====================================================================

class TestGolden2DEquivalence:
    """Verify that spatial_shape=(H, W) produces identical results to H=, W=."""

    def _make_layouts(self):
        regions = {
            "vis": RegionSpec(bounds=(0, 3, 0, 4, 0, 4)),
            "act": RegionSpec(bounds=(0, 3, 4, 5, 0, 2)),
        }
        old = CanvasLayout(T=3, H=6, W=6, d_model=64, regions=regions, t_current=1)
        new = CanvasLayout(T=3, spatial_shape=(6, 6), d_model=64, regions=regions, t_current=1)
        return old, new

    def test_num_positions(self):
        old, new = self._make_layouts()
        assert old.num_positions == new.num_positions

    def test_region_indices(self):
        old, new = self._make_layouts()
        for name in ("vis", "act"):
            assert old.region_indices(name) == new.region_indices(name)

    def test_region_indices_at_t(self):
        old, new = self._make_layouts()
        for name in ("vis", "act"):
            for t in range(3):
                assert old.region_indices_at_t(name, t) == new.region_indices_at_t(name, t)

    def test_output_mask(self):
        old, new = self._make_layouts()
        assert old.output_mask() == new.output_mask()

    def test_loss_weight_mask(self):
        old, new = self._make_layouts()
        assert torch.equal(old.loss_weight_mask(), new.loss_weight_mask())

    def test_flat_index(self):
        old, new = self._make_layouts()
        for t in range(3):
            for h in range(6):
                for w in range(6):
                    assert old.flat_index(t, h, w) == new.flat_index(t, h, w)

    def test_compile_schema_equivalence(self):
        @dataclass
        class Robot:
            camera: Field = Field(4, 4)
            joints: Field = Field(1, 4)

        bound_old = compile_schema(Robot(), T=2, H=8, W=8, d_model=32)
        bound_new = compile_schema(Robot(), T=2, spatial_shape=(8, 8), d_model=32)

        for name in bound_old.field_names:
            assert (
                bound_old.layout.region_indices(name)
                == bound_new.layout.region_indices(name)
            )


# =====================================================================
# SpatiotemporalCanvas with different dims
# =====================================================================

class TestSpatiotemporalCanvasND:
    def test_1d_canvas(self):
        layout = CanvasLayout(
            T=2, spatial_shape=(8,), d_model=32,
            regions={"x": (0, 2, 0, 4)},
        )
        canvas = SpatiotemporalCanvas(layout)
        batch = canvas.create_empty(2)
        assert batch.shape == (2, 16, 32)  # 2 * 8 = 16 positions

    def test_3d_canvas(self):
        layout = CanvasLayout(
            T=1, spatial_shape=(2, 2, 2), d_model=16,
            regions={"x": (0, 1, 0, 2, 0, 2, 0, 2)},
        )
        canvas = SpatiotemporalCanvas(layout)
        batch = canvas.create_empty(1)
        assert batch.shape == (1, 8, 16)  # 1 * 2*2*2 = 8 positions

    def test_1d_place_extract(self):
        layout = CanvasLayout(
            T=1, spatial_shape=(8,), d_model=16,
            regions={"a": (0, 1, 0, 4)},
        )
        canvas = SpatiotemporalCanvas(layout)
        batch = canvas.create_empty(1)
        embs = torch.ones(1, 4, 16) * 99.0
        batch = canvas.place(batch, embs, "a")
        out = canvas.extract(batch, "a")
        assert out.shape == (1, 4, 16)
        assert out.mean().item() > 90  # close to 99 + small embeddings


# =====================================================================
# Parse bounds utility
# =====================================================================

class TestParseBounds:
    def test_1d(self):
        (t0, t1), spatial = _parse_bounds((0, 3, 5, 10), 1)
        assert (t0, t1) == (0, 3)
        assert spatial == ((5, 10),)

    def test_2d(self):
        (t0, t1), spatial = _parse_bounds((0, 3, 2, 6, 1, 5), 2)
        assert (t0, t1) == (0, 3)
        assert spatial == ((2, 6), (1, 5))

    def test_3d(self):
        (t0, t1), spatial = _parse_bounds((0, 2, 0, 4, 0, 4, 0, 4), 3)
        assert (t0, t1) == (0, 2)
        assert spatial == ((0, 4), (0, 4), (0, 4))

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="Bounds length"):
            _parse_bounds((0, 2, 0, 4), 2)  # needs 6, got 4

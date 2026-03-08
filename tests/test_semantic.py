"""Tests for semantic conditioning."""

import torch
import pytest
from dataclasses import dataclass

from canvas_engineering import (
    Field, compile_schema, ConnectivityPolicy,
    SemanticConditioner, auto_semantic_type, compute_semantic_embeddings,
)
from canvas_engineering.canvas import CanvasLayout, RegionSpec


# ── auto_semantic_type ──────────────────────────────────────────────────

class TestAutoSemanticType:
    def test_simple_path(self):
        result = auto_semantic_type("financial.yield_curves.ten_year")
        assert "financial" in result
        assert "yield curves" in result
        assert "ten year" in result

    def test_country_path(self):
        result = auto_semantic_type("country_us.macro.output.gdp_nowcast")
        assert "country us" in result
        assert "macro" in result
        assert "gdp nowcast" in result

    def test_array_path(self):
        result = auto_semantic_type("firms[0].financials.revenue")
        assert "firms" in result
        assert "financials" in result
        assert "revenue" in result


# ── SemanticConditioner ─────────────────────────────────────────────────

EMBED_DIM = 16  # small for testing


def _make_embeddings(names):
    """Create random embeddings for testing."""
    return {name: tuple(torch.randn(EMBED_DIM).tolist()) for name in names}


class TestSemanticConditioner:
    def test_init(self):
        embs = _make_embeddings(["a", "b", "c"])
        cond = SemanticConditioner(d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs)
        assert cond.n_regions == 3
        assert set(cond.region_names) == {"a", "b", "c"}

    def test_get_conditioning(self):
        embs = _make_embeddings(["field_a", "field_b"])
        cond = SemanticConditioner(d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs)
        vec = cond.get_conditioning("field_a")
        assert vec.shape == (32,)

    def test_get_all_conditioning(self):
        embs = _make_embeddings(["x", "y", "z"])
        cond = SemanticConditioner(d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs)
        all_cond = cond.get_all_conditioning()
        assert all_cond.shape == (3, 32)

    def test_frozen_embeddings(self):
        embs = _make_embeddings(["a"])
        cond = SemanticConditioner(
            d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs,
            freeze_embeddings=True,
        )
        # _embeddings should be a buffer, not a parameter
        assert "_embeddings" not in dict(cond.named_parameters())
        # But projector and residuals should be parameters
        param_names = [n for n, _ in cond.named_parameters()]
        assert any("projector" in n for n in param_names)
        assert any("residuals" in n for n in param_names)

    def test_unfrozen_embeddings(self):
        embs = _make_embeddings(["a"])
        cond = SemanticConditioner(
            d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs,
            freeze_embeddings=False,
        )
        # _embeddings should be a parameter
        assert "_embeddings" in dict(cond.named_parameters())

    def test_no_residuals(self):
        embs = _make_embeddings(["a"])
        cond = SemanticConditioner(
            d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs,
            learn_residuals=False,
        )
        assert cond.residuals is None

    def test_zero_init_residuals(self):
        embs = _make_embeddings(["a", "b"])
        cond = SemanticConditioner(d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs)
        assert torch.all(cond.residuals == 0)

    def test_condition_canvas(self):
        layout = CanvasLayout(
            T=1, H=4, W=4, d_model=32,
            regions={
                "field_a": RegionSpec(bounds=(0, 1, 0, 2, 0, 2)),
                "field_b": RegionSpec(bounds=(0, 1, 2, 4, 0, 2)),
            },
        )
        embs = _make_embeddings(["field_a", "field_b"])
        cond = SemanticConditioner(d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs)

        canvas = torch.zeros(2, 16, 32)  # (B=2, N=16, d=32)
        result = cond.condition_canvas(canvas, layout)

        assert result.shape == canvas.shape
        # Positions in field_a should be non-zero
        a_indices = layout.region_indices("field_a")
        assert result[0, a_indices[0]].abs().sum() > 0
        # Original canvas unchanged
        assert canvas.abs().sum() == 0

    def test_condition_canvas_partial(self):
        """Conditioner with only some of the layout's regions."""
        layout = CanvasLayout(
            T=1, H=4, W=4, d_model=32,
            regions={
                "has_emb": RegionSpec(bounds=(0, 1, 0, 2, 0, 2)),
                "no_emb": RegionSpec(bounds=(0, 1, 2, 4, 0, 2)),
            },
        )
        embs = _make_embeddings(["has_emb"])
        cond = SemanticConditioner(d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs)

        canvas = torch.zeros(1, 16, 32)
        result = cond.condition_canvas(canvas, layout)

        # has_emb positions should be conditioned
        has_indices = layout.region_indices("has_emb")
        assert result[0, has_indices[0]].abs().sum() > 0
        # no_emb positions should remain zero
        no_indices = layout.region_indices("no_emb")
        assert result[0, no_indices[0]].abs().sum() == 0

    def test_gradient_flows(self):
        embs = _make_embeddings(["a", "b"])
        cond = SemanticConditioner(d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs)

        layout = CanvasLayout(
            T=1, H=4, W=4, d_model=32,
            regions={
                "a": RegionSpec(bounds=(0, 1, 0, 2, 0, 2)),
                "b": RegionSpec(bounds=(0, 1, 2, 4, 0, 2)),
            },
        )

        canvas = torch.zeros(1, 16, 32)
        result = cond.condition_canvas(canvas, layout)
        loss = result.sum()
        loss.backward()

        # Projector should have gradients
        assert cond.projector.weight.grad is not None
        # Residuals should have gradients
        assert cond.residuals.grad is not None


# ── Integration with compile_schema ─────────────────────────────────────

class TestCompileSchemaSemanticType:
    def test_auto_semantic_type_on_compile(self):
        @dataclass
        class Simple:
            temperature: Field = Field(1, 1)
            pressure: Field = Field(1, 1)

        bound = compile_schema(Simple(), T=1, H=4, W=4, d_model=32)

        # Check that semantic_type was auto-generated
        for name in bound.field_names:
            spec = bound[name].spec
            assert spec.semantic_type is not None
            assert len(spec.semantic_type) > 0

    def test_explicit_semantic_type_preserved(self):
        @dataclass
        class WithType:
            obs: Field = Field(1, 1, semantic_type="RGB observation from camera")

        bound = compile_schema(WithType(), T=1, H=4, W=4, d_model=32)
        assert bound["obs"].spec.semantic_type == "RGB observation from camera"


# ── SpatiotemporalCanvas with SemanticConditioner ────────────────────────

class TestCanvasWithConditioner:
    def test_canvas_with_conditioner(self):
        layout = CanvasLayout(
            T=1, H=4, W=4, d_model=32,
            regions={
                "a": RegionSpec(bounds=(0, 1, 0, 2, 0, 2)),
                "b": RegionSpec(bounds=(0, 1, 2, 4, 0, 2)),
            },
        )
        embs = _make_embeddings(["a", "b"])
        cond = SemanticConditioner(d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs)

        from canvas_engineering.canvas import SpatiotemporalCanvas
        canvas_mod = SpatiotemporalCanvas(layout, semantic_conditioner=cond)

        # Should not have modality_embeddings
        assert canvas_mod.modality_embeddings is None

        # create_empty should include semantic conditioning
        batch = canvas_mod.create_empty(2)
        assert batch.shape == (2, 16, 32)

    def test_canvas_without_conditioner(self):
        """Backward compatibility: no conditioner = learned modality embeddings."""
        layout = CanvasLayout(
            T=1, H=4, W=4, d_model=32,
            regions={
                "a": RegionSpec(bounds=(0, 1, 0, 2, 0, 2)),
            },
        )
        from canvas_engineering.canvas import SpatiotemporalCanvas
        canvas_mod = SpatiotemporalCanvas(layout)
        assert canvas_mod.modality_embeddings is not None
        batch = canvas_mod.create_empty(1)
        assert batch.shape == (1, 16, 32)

    def test_place_with_conditioner(self):
        layout = CanvasLayout(
            T=1, H=4, W=4, d_model=32,
            regions={
                "a": RegionSpec(bounds=(0, 1, 0, 2, 0, 2)),
            },
        )
        embs = _make_embeddings(["a"])
        cond = SemanticConditioner(d_model=32, embed_dim=EMBED_DIM, region_embeddings=embs)

        from canvas_engineering.canvas import SpatiotemporalCanvas
        canvas_mod = SpatiotemporalCanvas(layout, semantic_conditioner=cond)

        batch = canvas_mod.create_empty(1)
        data = torch.randn(1, 4, 32)
        result = canvas_mod.place(batch, data, "a")
        assert result.shape == batch.shape


# ── compute_semantic_embeddings ──────────────────────────────────────────

class TestComputeEmbeddings:
    def test_basic(self):
        def mock_embed(texts):
            return [[float(i)] * EMBED_DIM for i, _ in enumerate(texts)]

        result = compute_semantic_embeddings(
            ["field.a", "field.b", "field.c"],
            mock_embed,
        )

        assert len(result) == 3
        assert "field.a" in result
        assert len(result["field.a"]) == EMBED_DIM

    def test_custom_semantic_types(self):
        called_with = []

        def mock_embed(texts):
            called_with.extend(texts)
            return [[0.0] * EMBED_DIM for _ in texts]

        compute_semantic_embeddings(
            ["field.a", "field.b"],
            mock_embed,
            semantic_types={"field.a": "Custom description for field A"},
        )

        assert "Custom description for field A" in called_with
        # field.b should get auto-generated
        assert any("field" in t and "b" in t for t in called_with)


# ── BoundSchema.build_semantic_conditioner ───────────────────────────────

class TestBoundSchemaSemantic:
    def test_build_conditioner(self):
        @dataclass
        class Model:
            obs: Field = Field(2, 2)
            act: Field = Field(1, 1)

        bound = compile_schema(Model(), T=1, H=4, W=4, d_model=32)

        def mock_embed(texts):
            return [[0.1] * EMBED_DIM for _ in texts]

        cond = bound.build_semantic_conditioner(mock_embed, embed_dim=EMBED_DIM)
        assert cond.n_regions == 2

        # Build canvas with conditioner
        canvas = bound.build_canvas(semantic_conditioner=cond)
        batch = canvas.create_empty(1)
        assert batch.shape[1] == bound.layout.num_positions

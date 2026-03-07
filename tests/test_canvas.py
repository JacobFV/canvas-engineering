"""Tests for canvas layout and spatiotemporal canvas."""
import torch
import pytest
from canvas_engine import CanvasLayout, RegionSpec, SpatiotemporalCanvas


def test_layout_basic():
    layout = CanvasLayout(T=5, H=8, W=8, d_model=64, regions={"visual": (0, 5, 0, 6, 0, 6)})
    assert layout.num_positions == 320
    assert layout.region_size("visual") == (5, 6, 6)
    assert layout.region_numel("visual") == 180


def test_layout_indices():
    layout = CanvasLayout(T=2, H=3, W=4, d_model=32, regions={"a": (0, 1, 0, 1, 0, 2)})
    indices = layout.region_indices("a")
    assert indices == [0, 1]


def test_output_mask():
    layout = CanvasLayout(T=4, H=2, W=2, d_model=32, t_current=2)
    mask = layout.output_mask()
    assert len(mask) == 2 * 2 * 2  # 2 future timesteps x 2 x 2


def test_canvas_create_empty():
    layout = CanvasLayout(T=3, H=4, W=4, d_model=64, regions={"vis": (0, 3, 0, 4, 0, 4)})
    canvas = SpatiotemporalCanvas(layout)
    batch = canvas.create_empty(2)
    assert batch.shape == (2, 48, 64)


def test_canvas_place_and_extract():
    layout = CanvasLayout(T=2, H=4, W=4, d_model=32, regions={"action": (0, 2, 3, 4, 0, 1)})
    canvas = SpatiotemporalCanvas(layout)
    batch = canvas.create_empty(1)

    embs = torch.ones(1, 2, 32) * 42.0
    batch = canvas.place(batch, embs, "action")
    out = canvas.extract(batch, "action")
    assert out.shape == (1, 2, 32)
    # The placed values should be 42 + modality_embedding (close to 42)
    assert out.mean().item() > 40


def test_canvas_place_truncates():
    layout = CanvasLayout(T=2, H=2, W=2, d_model=16, regions={"x": (0, 1, 0, 1, 0, 1)})
    canvas = SpatiotemporalCanvas(layout)
    batch = canvas.create_empty(1)
    embs = torch.randn(1, 10, 16)  # 10 embeddings but region only has 1 slot
    batch = canvas.place(batch, embs, "x")  # should truncate without error


# --- RegionSpec tests ---


def test_region_spec_defaults():
    spec = RegionSpec(bounds=(0, 4, 0, 8, 0, 8))
    assert spec.period == 1
    assert spec.is_output is True
    assert spec.loss_weight == 1.0


def test_region_spec_frozen():
    spec = RegionSpec(bounds=(0, 1, 0, 1, 0, 1))
    with pytest.raises(AttributeError):
        spec.period = 2


def test_region_spec_method_wraps_tuple():
    layout = CanvasLayout(T=4, H=4, W=4, d_model=32, regions={"a": (0, 2, 0, 2, 0, 2)})
    spec = layout.region_spec("a")
    assert isinstance(spec, RegionSpec)
    assert spec.bounds == (0, 2, 0, 2, 0, 2)
    assert spec.period == 1
    assert spec.is_output is True


def test_region_spec_method_passthrough():
    s = RegionSpec(bounds=(0, 4, 0, 4, 0, 4), period=4, loss_weight=2.0)
    layout = CanvasLayout(T=4, H=4, W=4, d_model=32, regions={"a": s})
    assert layout.region_spec("a") is s


def test_mixed_tuple_and_spec_layout():
    """Existing methods work with a mix of raw tuples and RegionSpec."""
    layout = CanvasLayout(
        T=4, H=8, W=8, d_model=32,
        regions={
            "video": (0, 4, 0, 6, 0, 6),
            "thought": RegionSpec(bounds=(0, 2, 6, 8, 0, 4), period=2),
        },
    )
    assert layout.region_size("video") == (4, 6, 6)
    assert layout.region_size("thought") == (2, 2, 4)
    assert layout.region_numel("video") == 144
    assert layout.region_numel("thought") == 16
    assert len(layout.region_indices("thought")) == 16
    assert layout.region_timesteps("thought") == [0, 1]
    assert layout.region_indices_at_t("thought", 0) == layout.region_indices_at_t("thought", 0)
    assert layout.region_indices_at_t("thought", 5) == []


def test_loss_weight_mask_basic():
    layout = CanvasLayout(
        T=2, H=2, W=2, d_model=16,
        regions={"a": (0, 1, 0, 1, 0, 1)},  # 1 position at index 0
    )
    weights = layout.loss_weight_mask()
    assert weights.shape == (8,)
    assert weights[0].item() == 1.0
    assert weights[1:].sum().item() == 0.0


def test_loss_weight_mask_custom_weight():
    layout = CanvasLayout(
        T=2, H=2, W=2, d_model=16,
        regions={"a": RegionSpec(bounds=(0, 1, 0, 1, 0, 1), loss_weight=3.0)},
    )
    weights = layout.loss_weight_mask()
    assert weights[0].item() == 3.0


def test_loss_weight_mask_is_output_false():
    layout = CanvasLayout(
        T=2, H=2, W=2, d_model=16,
        regions={"prompt": RegionSpec(bounds=(0, 2, 0, 2, 0, 2), is_output=False)},
    )
    weights = layout.loss_weight_mask()
    assert weights.sum().item() == 0.0


def test_loss_weight_mask_mixed():
    layout = CanvasLayout(
        T=2, H=4, W=4, d_model=32,
        regions={
            "screen": (0, 2, 0, 3, 0, 3),                                    # 18 positions, weight=1
            "mouse": RegionSpec(bounds=(0, 2, 3, 4, 0, 1), loss_weight=2.0),  # 2 positions, weight=2
            "task": RegionSpec(bounds=(0, 1, 3, 4, 1, 2), is_output=False),   # 1 position, weight=0
        },
    )
    weights = layout.loss_weight_mask()
    assert weights.shape == (32,)
    # screen indices get 1.0
    screen_idx = layout.region_indices("screen")
    for i in screen_idx:
        assert weights[i].item() == 1.0
    # mouse indices get 2.0
    mouse_idx = layout.region_indices("mouse")
    for i in mouse_idx:
        assert weights[i].item() == 2.0
    # task indices get 0.0
    task_idx = layout.region_indices("task")
    for i in task_idx:
        assert weights[i].item() == 0.0


def test_loss_weight_mask_device():
    layout = CanvasLayout(T=1, H=1, W=1, d_model=8, regions={"a": (0, 1, 0, 1, 0, 1)})
    weights = layout.loss_weight_mask("cpu")
    assert weights.device == torch.device("cpu")


def test_real_frame():
    layout = CanvasLayout(
        T=4, H=4, W=4, d_model=32,
        regions={"thought": RegionSpec(bounds=(0, 4, 0, 4, 0, 4), period=4)},
    )
    assert layout.real_frame("thought", 0) == 0
    assert layout.real_frame("thought", 1) == 4
    assert layout.real_frame("thought", 2) == 8
    assert layout.real_frame("thought", 3) == 12


def test_real_frame_period_1():
    layout = CanvasLayout(T=4, H=2, W=2, d_model=16, regions={"v": (0, 4, 0, 2, 0, 2)})
    assert layout.real_frame("v", 0) == 0
    assert layout.real_frame("v", 3) == 3


def test_canvas_frame():
    layout = CanvasLayout(
        T=4, H=4, W=4, d_model=32,
        regions={"thought": RegionSpec(bounds=(0, 4, 0, 4, 0, 4), period=4)},
    )
    assert layout.canvas_frame("thought", 0) == 0
    assert layout.canvas_frame("thought", 4) == 1
    assert layout.canvas_frame("thought", 8) == 2
    assert layout.canvas_frame("thought", 12) == 3


def test_canvas_frame_unaligned():
    layout = CanvasLayout(
        T=4, H=4, W=4, d_model=32,
        regions={"thought": RegionSpec(bounds=(0, 4, 0, 4, 0, 4), period=4)},
    )
    assert layout.canvas_frame("thought", 1) is None
    assert layout.canvas_frame("thought", 7) is None


def test_canvas_frame_out_of_range():
    layout = CanvasLayout(
        T=4, H=4, W=4, d_model=32,
        regions={"thought": RegionSpec(bounds=(0, 4, 0, 4, 0, 4), period=4)},
    )
    assert layout.canvas_frame("thought", 16) is None  # beyond region extent
    assert layout.canvas_frame("thought", -4) is None   # negative


def test_canvas_frame_period_1():
    layout = CanvasLayout(T=4, H=2, W=2, d_model=16, regions={"v": (0, 4, 0, 2, 0, 2)})
    assert layout.canvas_frame("v", 0) == 0
    assert layout.canvas_frame("v", 3) == 3
    assert layout.canvas_frame("v", 4) is None  # out of range


def test_full_example():
    """End-to-end test matching the plan's example usage."""
    layout = CanvasLayout(
        T=16, H=32, W=32, d_model=768,
        regions={
            "screen": (0, 16, 0, 24, 0, 24),
            "mouse": RegionSpec(bounds=(0, 16, 24, 26, 0, 4), period=1, loss_weight=2.0),
            "thought": RegionSpec(bounds=(0, 4, 28, 32, 0, 8), period=4, loss_weight=1.0),
            "task_prompt": RegionSpec(bounds=(0, 1, 26, 28, 0, 4), is_output=False),
        },
    )
    weights = layout.loss_weight_mask()
    assert weights.shape == (16 * 32 * 32,)
    # task_prompt positions should be 0
    for i in layout.region_indices("task_prompt"):
        assert weights[i].item() == 0.0
    # mouse positions should be 2.0
    for i in layout.region_indices("mouse"):
        assert weights[i].item() == 2.0
    # frame mapping
    assert layout.real_frame("thought", 2) == 8
    assert layout.canvas_frame("thought", 8) == 2
    assert layout.canvas_frame("thought", 7) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

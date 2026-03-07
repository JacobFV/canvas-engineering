# The Canvas

A **canvas** is a 3D grid `(T, H, W)` of `d_model`-dimensional vectors. Each modality occupies a named region. The diffusion process operates on "output" regions; "input" regions serve as conditioning context.

## CanvasLayout

```python
layout = CanvasLayout(
    T=16, H=32, W=32, d_model=768,
    regions={
        "screen": (0, 16, 0, 24, 0, 24),         # raw tuple — defaults
        "mouse":  RegionSpec(bounds=(0, 16, 24, 26, 0, 4), loss_weight=2.0),
        "thought": RegionSpec(bounds=(0, 4, 28, 32, 0, 8), period=4),
        "prompt": RegionSpec(bounds=(0, 1, 26, 28, 0, 4), is_output=False),
    },
)
```

Raw 6-tuples auto-wrap as `RegionSpec(bounds=tuple)` — full backward compatibility.

## RegionSpec fields

| Field | Default | Meaning |
|-------|---------|---------|
| `bounds` | *(required)* | `(t0, t1, h0, h1, w0, w1)` spatial-temporal extent |
| `period` | `1` | Canvas frames per real-world update |
| `is_output` | `True` | Participates in diffusion loss? |
| `loss_weight` | `1.0` | Relative loss weight |
| `semantic_type` | `None` | Human-readable modality description |
| `semantic_embedding` | `None` | Frozen vector for transfer distance |
| `embedding_model` | `"openai/text-embedding-3-small"` | Which model produced the embedding |
| `default_attn` | `"cross_attention"` | Default attention fn for outgoing connections |

## Temporal frequency

A region with `period=4` spanning `t=0..3` means its 4 canvas slots map to real-world frames 0, 4, 8, 12.

```python
layout.real_frame("thought", canvas_t=2)    # → 8
layout.canvas_frame("thought", real_t=8)    # → 2
layout.canvas_frame("thought", real_t=7)    # → None (not aligned)
```

## Loss weight mask

```python
weights = layout.loss_weight_mask("cuda")   # (N,) tensor
loss = (per_position_loss * weights).sum() / weights.sum()
```

Positions in `is_output=True` regions get their `loss_weight`; `is_output=False` or uncovered positions get 0. Overlapping regions accumulate additively.

## SpatiotemporalCanvas

The `SpatiotemporalCanvas` module manages the tensor with positional + modality embeddings:

- **Positional encoding**: 3D sinusoidal, d_model split into thirds for (t, h, w)
- **Empty token**: Learned parameter for unoccupied positions
- **Modality embeddings**: Learned per-region embedding added during `place()`

```python
canvas_mod = SpatiotemporalCanvas(layout)
batch = canvas_mod.create_empty(4)           # (4, T*H*W, d_model)
batch = canvas_mod.place(batch, embs, "visual")
out = canvas_mod.extract(batch, "action")
```

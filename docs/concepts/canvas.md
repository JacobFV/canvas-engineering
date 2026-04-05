# The Canvas

A **canvas** is an N-dimensional grid `(T, *spatial_shape)` of `d_model`-dimensional vectors. `T` is the temporal axis; `spatial_shape` is an arbitrary tuple of spatial dimensions — `(W,)` for 1D, `(H, W)` for 2D, `(H, W, D)` for 3D, etc. Each modality occupies a named region. The diffusion process operates on "output" regions; "input" regions serve as conditioning context.

## CanvasLayout

```python
# 2D spatial (the classic form):
layout = CanvasLayout(
    T=16, spatial_shape=(32, 32), d_model=768,
    regions={
        "screen": (0, 16, 0, 24, 0, 24),         # raw tuple — defaults
        "mouse":  RegionSpec(bounds=(0, 16, 24, 26, 0, 4), loss_weight=2.0),
        "thought": RegionSpec(bounds=(0, 4, 28, 32, 0, 8), period=4),
        "prompt": RegionSpec(bounds=(0, 1, 26, 28, 0, 4), is_output=False),
    },
)

# 1D spatial (time-series):
layout_1d = CanvasLayout(T=8, spatial_shape=(128,), d_model=64, ...)

# 3D spatial (volumetric):
layout_3d = CanvasLayout(T=4, spatial_shape=(8, 8, 8), d_model=128, ...)

# Backward-compatible H/W form (still works):
layout = CanvasLayout(T=16, H=32, W=32, d_model=768, ...)
```

Region bounds are flat tuples of length `2 + 2 * n_spatial_dims`: `(t0, t1, s0_lo, s0_hi, ...)`. Raw tuples auto-wrap as `RegionSpec(bounds=tuple)`.

## RegionSpec fields

| Field | Default | Meaning |
|-------|---------|---------|
| `bounds` | *(required)* | `(t0, t1, s0_lo, s0_hi, ...)` spatial-temporal extent |
| `period` | `1` | Canvas frames per real-world update |
| `is_output` | `True` | Participates in diffusion loss? |
| `loss_weight` | `1.0` | Relative loss weight |
| `semantic_type` | `None` | Human-readable modality description |
| `semantic_embedding` | `None` | Frozen vector for transfer distance |
| `embedding_model` | `"openai/text-embedding-3-small"` | Which model produced the embedding |
| `default_attn` | `"cross_attention"` | Default attention fn for outgoing connections |
| `carrier` | `"deterministic"` | Dynamics carrier: deterministic, diffusive, filter, memory, residual |

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

## Temporal frequency and fill modes

A region with `period=4` updates every 4 real-world frames. When a fast region (period=1) attends to a slow region (period=4) at a timestep where the slow region hasn't updated, the connection's `temporal_fill` mode determines behavior. See [Topology — Temporal fill modes](topology.md#temporal-fill-modes) for details.

Fill resolution operates in **real-time space**: a slow region's canvas frames are mapped to real times via `canvas_t * period`, creating natural gaps that INTERPOLATE can exploit. This is fully transparent — period=1 regions behave identically to the original canvas-frame-based resolution.

## SpatiotemporalCanvas

The `SpatiotemporalCanvas` module manages the tensor with positional + modality + period embeddings:

- **Positional encoding**: N-D sinusoidal, d_model split evenly across all dimensions (t, s0, s1, ...)
- **Empty token**: Learned parameter for unoccupied positions
- **Modality embeddings**: Learned per-region embedding added during `place()`
- **Period embedding**: Learned embedding indexed by log-bucketed temporal period, summed into each position so the model knows its native update rate
- **Carrier field**: Each `RegionSpec` declares a `carrier` (default `"deterministic"`) that describes the region's dynamics. See [Carriers](carriers.md) for the full breakdown

```python
canvas_mod = SpatiotemporalCanvas(layout)
batch = canvas_mod.create_empty(4)           # (4, T*prod(spatial_shape), d_model)
batch = canvas_mod.place(batch, embs, "visual")
out = canvas_mod.extract(batch, "action")
```

### PeriodEmbedding

Each position's representation includes a `PeriodEmbedding` — a learned vector indexed by the region's temporal period. Period values are mapped to buckets via log scaling (period=1 → bucket 0, period=576 → bucket 10). This lets the model infer staleness when reading held values from slower regions via temporal fill connections.

```python
from canvas_engineering import PeriodEmbedding

pe = PeriodEmbedding(d_model=256, n_buckets=16)
pe.bucket(1)    # → 0
pe.bucket(576)  # → 10
emb = pe(4)     # → (256,) learned vector for period=4
```

# Topology

Canvas regions don't interact via Euclidean adjacency. A `CanvasTopology` declares the full attention compute DAG as data.

## Connections

Each `Connection` is one cross-attention operation: src tokens (queries) attend to dst tokens (keys/values).

```python
Connection(src="action", dst="visual")           # action reads from visual
Connection(src="visual", dst="visual")           # self-attention
Connection(src="action", dst="obs", t_src=0, t_dst=-1)  # prev-frame cross
Connection(src="thought", dst="visual", fn="perceiver")  # compressed read
```

## Convenience constructors

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/topology_constructors.png" alt="Topology constructors" width="100%">
</p>

```python
CanvasTopology.dense(["a", "b", "c"])          # fully connected
CanvasTopology.isolated(["a", "b", "c"])       # block-diagonal
CanvasTopology.hub_spoke("hub", ["a", "b"])    # star topology
CanvasTopology.causal_chain(["obs", "plan", "act"])  # A → B → C
CanvasTopology.causal_temporal(["obs", "act"]) # same-frame self + prev-frame cross
```

## Temporal connectivity

| `t_src` | `t_dst` | Behavior |
|---------|---------|----------|
| `None` | `None` | All src ↔ all dst (dense in time) |
| `0` | `0` | Same-frame only |
| `0` | `-1` | Src at current frame queries dst at previous frame |
| `None` | `0` | All src timesteps query dst at each reference frame |

## Temporal fill modes

When regions update at different frequencies (e.g., fast at period=1, slow at period=4), a fast region may query a slow region at a timestep where the slow region has no value. The `temporal_fill` parameter on `Connection` controls what happens:

| Mode | Behavior | Use case |
|------|----------|----------|
| `TemporalFill.DROP` | No connection | Event signals — alerts, anomaly flags, one-shot observations |
| `TemporalFill.HOLD` | Use most recent past value (default) | State signals — sensor readings, positions, embeddings |
| `TemporalFill.INTERPOLATE` | Weighted blend of surrounding values | Smooth signals with known update schedule |

```python
from canvas_engineering import Connection, TemporalFill

# State signal: hold the most recent GDP until the next quarterly release
Connection(src="daily_market", dst="quarterly_gdp", t_src=0, t_dst=0,
           temporal_fill=TemporalFill.HOLD)

# Event signal: an alert fires once, don't hold it forever
Connection(src="monitor", dst="alerts", t_src=0, t_dst=0,
           temporal_fill=TemporalFill.DROP)

# Smooth signal: interpolate temperature between hourly readings
Connection(src="fast_control", dst="temperature", t_src=0, t_dst=0,
           temporal_fill=TemporalFill.INTERPOLATE)
```

Fill resolution operates in **real-time space**: a slow region with `period=4` and 2 canvas frames maps to real times {0, 4}. A fast region at real time 2 sees the natural gap and can interpolate. For period=1 regions, real time equals canvas time — fully backward compatible.

### Interpolation order

`TemporalFill.INTERPOLATE` supports higher-order interpolation via `interpolation_order` on `Connection`:

| Order | Method | Anchors used | Best for |
|-------|--------|-------------|----------|
| 1 (default) | Linear lerp | 2 (nearest past + future) | General purpose |
| 2 | Inverse-distance weighting, 1/d² | 3 nearest | Smooth signals with curvature |
| 3 | IDW, 1/d³ | 4 nearest | Very smooth signals |

Weights are always non-negative and normalized. No learned parameters.

```python
Connection(src="fast", dst="slow", t_src=0, t_dst=0,
           temporal_fill=TemporalFill.INTERPOLATE,
           interpolation_order=2)  # quadratic IDW
```

## Attention mask compilation

```python
mask = topology.to_attention_mask(layout)  # (N, N) float tensor
ops = topology.attention_ops(layout)       # [(src, dst, weight, fn), ...]
```

The mask is compiled by iterating over connections, resolving temporal constraints and fill modes in real-time space, and setting `mask[i, j] = weight` for each (query_position, key_position) pair. Fill weights may be fractional for INTERPOLATE connections.

## Operators

`Connection.operator` declares the semantic intent of an edge, separate from the backend attention function (`fn`). When `compile_program()` has family information for both endpoints, it auto-sets the operator from the `DEFAULT_WIRING` table:

| (src family, dst family) | Operator |
|--------------------------|----------|
| (observation, state) | `observe` |
| (state, observation) | `predict` |
| (state, state) | `integrate` |
| (state, memory) | `write` |
| (memory, state) | `retrieve` |
| (state, action) | `act` |
| (action, state) | `intervene` |
| (state, residual) | `emit_residual` |
| (observation, residual) | `emit_residual` |
| (observation, observation) | `attend` |

Connections without family information, or family pairs not in the table, keep the default `"attend"` operator. The operator is metadata for the program layer -- it does not change which attention function runs. `fn` controls the backend; `operator` controls the semantics.

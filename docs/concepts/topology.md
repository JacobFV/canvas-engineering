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

## Attention mask compilation

```python
mask = topology.to_attention_mask(layout)  # (N, N) float tensor
ops = topology.attention_ops(layout)       # [(src, dst, weight, fn), ...]
```

The mask is compiled by iterating over connections, resolving temporal constraints, and setting `mask[i, j] = weight` for each (query_position, key_position) pair.

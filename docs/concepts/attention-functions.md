# Attention Function Types

Not all connections should use the same attention mechanism. Canvas-engineering lets you declare the *type of function* used for each edge in the compute graph.

## Declaration

```python
# Per-region default (applies to all outgoing connections)
RegionSpec(bounds=..., default_attn="linear_attention")

# Per-connection override
Connection(src="thought", dst="visual", fn="perceiver")
```

**Resolution order:** `connection.fn` → `region.default_attn` → `"cross_attention"`.

## The full lineup

### Dot-product family

| Type | Complexity | Description |
|------|-----------|-------------|
| **`cross_attention`** | O(NM) | Standard scaled dot-product QKV with softmax. The default. |
| **`linear_attention`** | O(N+M) | No softmax — kernel trick with elu+1 or ReLU features. Good for low-dimensional streams where full quadratic attention is overkill. |
| **`cosine_attention`** | O(NM) | Cosine similarity instead of scaled dot-product. No temperature parameter. Stable gradients. |
| **`sigmoid_attention`** | O(NM) | Sigmoid instead of softmax — each position independently gates each key. Non-exclusive attention for multi-label patterns. |

### Gating family

| Type | Complexity | Description |
|------|-----------|-------------|
| **`gated`** | O(NM) | Gated cross-attention (Flamingo-style). A learned sigmoid gate controls whether to incorporate context. Best for optional conditioning — goals, instructions, memory retrieval. |

### Compression family

| Type | Complexity | Description |
|------|-----------|-------------|
| **`perceiver`** | O(NK) | Cross-attend through a learned latent bottleneck (K << M). Compresses a large dst region into a fixed-size representation. Good for reading from large visual fields. |
| **`pooling`** | O(N+M) | Mean-pool dst into a single vector, broadcast to all src positions. Cheapest possible information transfer. Good for scalar conditioning. |
| **`copy`** | O(N) | Direct tensor transfer — no learned parameters. For broadcast regions, multi-agent latent sharing, or identity connections. |

### State-space / recurrence family

| Type | Complexity | Description |
|------|-----------|-------------|
| **`mamba`** | O(N) | Selective state-space model (S6). Input-dependent gating over compressed state. Hardware-efficient for long temporal sequences. |
| **`rwkv`** | O(N) | Linear attention with learned exponential decay. Recurrent formulation with position-dependent forgetting. Good for causal temporal connections. |
| **`hyena`** | O(N log N) | Long convolution with data-dependent gating via FFT. Sub-quadratic alternative for very long sequences. |

### Sparse / structured family

| Type | Complexity | Description |
|------|-----------|-------------|
| **`sparse_attention`** | O(NK) | Top-k attention — only the k highest logits survive softmax. Sparse gradient flow for selective binding. |
| **`local_attention`** | O(NW) | Windowed attention — each position only attends within a local spatial/temporal window W. For spatially local interactions. |

### Meta / experimental

| Type | Complexity | Description |
|------|-----------|-------------|
| **`none`** | O(0) | Edge exists in schema but is disabled. For ablation studies. |
| **`random_fixed`** | O(NK) | Random sparse attention pattern, frozen at init. Baseline for measuring whether learned patterns matter. |
| **`mixture`** | O(NK) | MoE-style learned routing. Each src position is routed to a subset of dst by a learned router. |

## Design recipes

### Robot manipulation
```python
"visual":  default_attn="cross_attention"   # spatial reasoning needs full attention
"proprio": default_attn="linear_attention"  # 12D vector, O(N²) is wasteful
"action":  default_attn="cross_attention"   # content-based visual selection
# proprio → action: fn="pooling"            # just inject the state vector
```

### Embodied agent with memory
```python
"perception": default_attn="cross_attention"
"memory":     default_attn="mamba"           # O(N) over long episode history
"policy":     default_attn="cross_attention"
# memory → perception: fn="gated"           # selective memory retrieval
# perception → memory: fn="perceiver"       # compress into fixed-size buffer
```

### Multi-agent coordination
```python
"agent_a.thought": default_attn="rwkv"      # causal temporal within agent
"agent_b.thought": default_attn="rwkv"
# agent_a → agent_b: fn="copy"              # direct latent relay
# both → shared_task: fn="cross_attention"   # selective broadcast
```

### Vision transformer
```python
"patches":  default_attn="local_attention"   # each patch attends locally
"cls":      default_attn="cross_attention"   # global token aggregates
# cls → patches: fn="cross_attention"        # global readout
# patches → cls: fn="pooling"               # compress to single vector
```

## Dispatch: from declaration to execution

All 17 attention types are fully implemented as `nn.Module` classes in `canvas_engineering.attention`. The `AttentionDispatcher` routes each topology connection to its resolved function:

```python
from canvas_engineering import AttentionDispatcher

dispatcher = AttentionDispatcher(
    topology=topology,
    layout=layout,
    d_model=256,
    n_heads=4,
)
output = dispatcher(hidden_states)  # per-connection dispatch
```

A frozen CogVideoX backbone runs all positions through the same blocks (full attention), so it can only honor `weight` modulation. A custom or scratch backbone can use `AttentionDispatcher` for true per-connection dispatch.

Custom attention types can be registered at runtime:

```python
from canvas_engineering import register_attention

register_attention("my_custom_attn", MyCustomAttentionModule)
```

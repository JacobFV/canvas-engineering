# Recipe: Custom Architectures

Canvas engineering is backbone-agnostic. Here are schemas for non-standard architectures.

## Vision Transformer with structured attention

Replace ViT's dense self-attention with a topology-aware pattern:

```python
layout = CanvasLayout(
    T=1, H=14, W=14, d_model=768,
    regions={
        "patches": RegionSpec(bounds=(0,1, 0,14, 0,14),
                              default_attn="local_attention"),
        "cls": RegionSpec(bounds=(0,1, 0,1, 0,1),   # reuse corner
                          default_attn="cross_attention"),
    },
)

topology = CanvasTopology(connections=[
    Connection(src="patches", dst="patches"),          # local self-attn
    Connection(src="cls", dst="patches"),               # global aggregation
    Connection(src="patches", dst="cls", fn="pooling"), # compress to cls
])
```

## Mamba-Transformer hybrid

Temporal connections via Mamba, spatial via attention:

```python
layout = CanvasLayout(
    T=64, H=8, W=8, d_model=512,
    regions={
        "spatial": RegionSpec(bounds=(0,64, 0,8, 0,8),
                              default_attn="cross_attention"),
        "temporal": RegionSpec(bounds=(0,64, 0,8, 0,8),
                               default_attn="mamba"),
    },
)

# Spatial: same-frame self-attention
# Temporal: sequential state-space across frames
topology = CanvasTopology(connections=[
    Connection(src="spatial", dst="spatial", t_src=0, t_dst=0),
    Connection(src="temporal", dst="temporal"),  # mamba processes full sequence
    Connection(src="spatial", dst="temporal", fn="copy"),  # share features
])
```

## Perceiver-style bottleneck

Large input compressed through a small latent:

```python
layout = CanvasLayout(
    T=1, H=32, W=32, d_model=768,
    regions={
        "input": RegionSpec(bounds=(0,1, 0,32, 0,32), is_output=False),
        "latent": RegionSpec(bounds=(0,1, 0,4, 0,4),    # 16 positions
                             default_attn="cross_attention"),
        "output": RegionSpec(bounds=(0,1, 0,8, 0,8)),
    },
)

topology = CanvasTopology(connections=[
    Connection(src="latent", dst="input", fn="perceiver"),  # compress 1024→16
    Connection(src="latent", dst="latent"),                  # process in latent
    Connection(src="output", dst="latent"),                  # decode from latent
])
```

## RWKV-style linear recurrence

For very long sequences where O(N²) attention is infeasible:

```python
layout = CanvasLayout(
    T=1024, H=1, W=1, d_model=512,
    regions={
        "sequence": RegionSpec(bounds=(0,1024, 0,1, 0,1),
                               default_attn="rwkv"),
    },
)

topology = CanvasTopology(connections=[
    Connection(src="sequence", dst="sequence"),  # RWKV processes full sequence
])
# O(N) instead of O(N²) — feasible for 1024+ timesteps
```

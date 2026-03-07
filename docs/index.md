# canvas-engineering

### Prompt engineering, but for latent space.

**canvas-engineering** gives video diffusion models structured latent space. You declare which regions carry video, actions, proprioception, reward, or thought — their geometry, temporal frequency, connectivity, attention function types, and loss participation — and the canvas compiles that declaration into attention masks, loss weights, and frame mappings.

The layout is the schema. The topology is the compute graph. Together they form a **type system for multimodal latent computation**.

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/canvas_layouts_combined.png" alt="Canvas allocations" width="100%">
</p>

## Two orthogonal ideas

### 1. The canvas — structured multimodal latent space

Large video diffusion models generate video. The **spatiotemporal canvas** extends them to predict actions, estimate rewards, and process proprioception by placing heterogeneous modalities on a shared 3D grid. You design the schema, the model attends over everything.

### 2. Looped attention — weight-sharing regularization

**Looped attention** iterates transformer blocks with learned iteration embeddings. Result: **1.73x parameter efficiency** (p<0.001). A frozen CogVideoX-2B backbone + **350K trainable loop parameters** outperforms **11.5M unfrozen parameters**. 3 loops is optimal.

## Key features

- **`RegionSpec`** — Declare geometry, temporal frequency, loss weight, semantic type, and default attention function per region
- **`CanvasTopology`** — Directed graph of attention operations with temporal constraints and per-edge function types
- **`CanvasSchema`** — Portable JSON-serializable bundle (layout + topology + metadata)
- **`transfer_distance()`** — Cosine distance between semantic type embeddings estimates adapter cost
- **16 attention function types** — From standard cross-attention to Mamba, Perceiver, RWKV, and more
- **`graft_looped_blocks()`** — One-line grafting onto CogVideoX with 350K trainable params

## Install

```bash
pip install canvas-engineering
```

## Next steps

- [Quick Start](getting-started/quickstart.md) — 30-line graft-and-train
- [The Canvas](concepts/canvas.md) — How regions, frequency, and loss weighting work
- [Attention Functions](concepts/attention-functions.md) — The full lineup of 16 function types
- [Design Recipes](recipes/robot.md) — Real-world schema patterns

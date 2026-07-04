# Canvas Engineering: Declared Causal Macrostructure for Reverse-Diffusion Latent Dynamics

**Jacob Valdez** · jacobfv123@gmail.com · July 2026
*Denseform paper — every sentence load-bearing. Code: [github.com/JacobFV/canvas-engineering](https://github.com/JacobFV/canvas-engineering)*

## Abstract

Prompt engineering structures what a language model *sees*; canvas engineering structures what a diffusion model *thinks in*. The practitioner declares the macrostructure of a diffusion transformer's latent space — which regions carry which modalities, their geometry, temporal update frequencies, loss participation, and, critically, a directed graph of permitted block-to-block attention operations — and a compiler lowers that declaration into attention masks, loss weights, and frame mappings. Because reverse diffusion is an iterated denoising map over the whole canvas, hard connectivity constraints on attention induce an explicit, human-legible causal interaction graph *inside* the generative dynamics, while gradient descent remains free to shape all fine structure within and along the declared edges. This is a neurosymbolic architecture in the precise sense: transparent symbolic macrostructure (a type system — offsets, signatures, calling conventions, an ABI) hosted directly within learnable neural mechanics, with no discrete/continuous interface to cross. We describe the abstraction stack (regions → topology → typed programs → compiled deployment), the compilation of nested entity schemas into hierarchically coarse-grained canvases, and the empirical record — 26 experiments and 236 training runs on CogVideoX-2B — including a 1.73× parameter-efficiency result for looped attention (p<0.001), a frozen 350K-parameter configuration that beats 11.7M unfrozen parameters on action prediction, and two honestly falsified hypotheses. We argue this style of *declared* latent structure is the production-ready on-ramp to intuition-guided symbolic world modeling: the symbolic model is not synthesized by the network, it is authored by the engineer, and the network learns everything else.

## 1. Position

Chollet's convergence thesis — that AI trends toward *intuition-guided symbolic world modeling*, i.e., deep-learning-guided program synthesis — identifies the destination but underdetermines the route. The dominant proposed route is **synthesis**: a neural system emits symbolic programs, gaining compactness and generalization at the cost of a brittle discrete search loop and a hard neural/symbolic boundary. Canvas engineering takes the dual route, **declaration**: the human authors the symbolic model up front as the *interaction topology of latent space itself*, and the neural substrate fills it in. Both routes buy the same asset — a compact, reusable, generalizable macrostructure acquired without data — but declaration ships today, because it compiles to nothing more exotic than attention masks and loss weights on an off-the-shelf pretrained diffusion transformer.

The economic framing matters. Neurosymbolic methods will remain research curiosities as long as the marginal dollar buys more from data than from structure. Declared structure changes that calculus in exactly the regimes where data is expensive: robotics, control, multi-agent coordination, world modeling. A schema is free; a demonstration is not. The question this paper answers is *how to make structure declarable* at all inside a modern generative architecture.

## 2. Mechanism

### 2.1 The canvas

A **canvas** is a 3D spatiotemporal grid (T, H, W) of d-dimensional latent positions, flattened into the token sequence of a diffusion transformer (DiT). A `CanvasLayout` partitions it into named regions:

```python
layout = CanvasLayout(
    T=5, H=8, W=8, d_model=256,
    regions={
        "visual": (0,5, 0,6, 0,6),   # 180 positions — video patches
        "action": (0,5, 6,7, 0,1),   #   5 positions — per-frame actions
        "reward": (2,3, 7,8, 0,1),   #   1 position  — scalar reward
    },
    t_current=2,                      # t ≥ 2 is future (diffusion output)
)
```

Each region carries a `RegionSpec`: bounds, a temporal `period` mapping canvas frames to real-world frames (a `thought` region at period=4 and a `perception` region at period=1 coexist on one canvas), `is_output` and `loss_weight` (loss participation is type-directed codegen — `loss_weight_mask()` compiles the declaration into a per-position weight tensor), an optional `semantic_type` with a frozen embedding, and a `default_attn` function family. Heterogeneous modalities enter and exit through per-region encoders/decoders; the pretrained video backbone attends over everything.

### 2.2 Topology: the causal graph is the attention mask

A `CanvasTopology` is a directed multigraph of discrete cross-attention operations. Each `Connection(src, dst)` licenses src tokens to query dst keys/values; absence of an edge is a *hard* prohibition, not a soft prior:

```python
CanvasTopology(connections=[
    Connection(src="action", dst="visual"),                # obs → act (diffusion policy)
    Connection(src="action", dst="obs", t_src=0, t_dst=-1),# act at t reads obs at t−1
    Connection(src="r1_cam", dst="r2_cam", weight=0.5),    # cross-agent coordination
])
```

Temporal offsets (`t_src`, `t_dst`) restrict which frame pairs participate — same-frame-only, previous-frame causal, sliding-window — and `TemporalFill` (HOLD / DROP / INTERPOLATE) resolves cross-frequency queries in real-time space. Convenience constructors (`dense`, `isolated`, `hub_spoke`, `causal_chain`, `causal_temporal`) name the standard patterns; a standard transformer is the degenerate `dense` case.

The consequence is the paper's central claim. Reverse diffusion applies the denoiser to the full canvas at every step, so *which regions may influence which* under iteration is precisely the topology's transitive structure. Declaring the topology therefore declares a causal interaction graph over the generative dynamics — attention-based rather than convolutional, so the graph is over *semantic regions*, not spatial neighborhoods, and non-Euclidean by construction. Diffusion policy (observation → action) is the two-node base case. Multi-agent perceptual diffusion — each agent self-attending over its own observations and actions, coordinating only through declared cross-edges — is the same primitive composed.

What is fixed and what is learned splits cleanly: the engineer fixes *whether* an edge exists, its direction, temporal extent, and function family; gradients — from data or intrinsic losses — determine *what flows* along it and everything within regions. Macro is symbolic; micro is neural; there is no interface, because the symbolic layer *is* the mask.

### 2.3 A type system, literally

| Type-system concept | Canvas equivalent | Implementation |
|---|---|---|
| Struct field (offset + size) | Region bounds | `region_indices()` — an offset calculation |
| Field type annotation | period, loss_weight, semantic_type, default_attn | `RegionSpec` |
| Pointer / reference | Connection | `CanvasTopology` |
| Function signature | Topology pattern + fn type | `attention_ops()` |
| Type-directed codegen | Loss mask from declarations | `loss_weight_mask()` |
| ABI compatibility | Schema compatibility | `compatible_regions()` |
| Coercion cost | Transfer distance | `transfer_distance()` |

This is not analogy-as-decoration. `region_indices()` computes memory offsets; the topology is a calling convention; a serialized `CanvasSchema` (layout + topology + metadata, human-readable JSON) is a complete type signature. Two models sharing a schema can exchange latent state directly — no tokenization, no re-encoding — because the schema fixes what every position means. Across differing schemas, frozen semantic-type embeddings make modality compatibility computable: `transfer_distance(cam, depth) ≈ 0.15` (bridgeable in 1–2 adapter layers) vs. `transfer_distance(cam, joints) ≈ 0.65` (full adapter), contingent on the representation-stability hypothesis of §5.

### 2.4 Attention function types

Edges declare *how* information flows, not just whether: 17 registered function families spanning dot-product (`cross_attention`, `linear_attention`, `sigmoid_attention`), gating (`gated`), compression (`perceiver`, `pooling`), transfer (`copy`), state-space (`mamba`, `rwkv`), convolutional (`hyena`), sparse (`local_attention`, `sparse_attention`), and meta (`none`, `random_fixed`, `mixture`). Resolution order: `connection.fn` → `region.default_attn` → global `cross_attention`. Each choice encodes a theory of the edge — `pooling` for a 12-D proprioception summary, `perceiver` to bottleneck 864 visual tokens into a thought region, `copy` for direct latent relay between agents. The schema declares intent; the executor chooses implementation.

## 3. Compilation: from entity schema to canvas

Hand-placing rectangles does not scale, so the compiler accepts typed entity declarations — nested dataclasses (a pydantic surface is the same move) — and flattens the entity/relationship graph they denote into layout + topology:

```python
@dataclass
class Vehicle:
    __coarse__ = Field(4, 4)          # each vehicle → 4×4 summary seen by parent
    camera: Field = Field(8, 8)
    plan:   Field = Field(2, 4)
    action: Field = Field(1, 4, loss_weight=2.0)

@dataclass
class Fleet:
    dispatch: Field = Field(4, 4)
    vehicles: list = dc_field(default_factory=list)

bound = compile_schema(Fleet(vehicles=[Vehicle() for _ in range(50)]), d_model=256)
```

Every nested type automatically receives a **coarse-grained field** — a compressed representation that bottlenecks all cross-level attention. Fifty vehicles under dense cross-attention would cost O(50² × fields²) connections; under coarse-graining, each interacts through its 4×4 summary — O(50 × 16). Deep nesting chains the bottlenecks: in a world-model schema, the path from `us.macro.gdp` to `cn.macro.inflation` is forced through `us.macro (coarse) → us (coarse) → regime → cn (coarse) → cn.macro (coarse)`. Each hop compresses, so hierarchical abstraction is not an emergent hope but a topological consequence. The declared schema is, in effect, a graphical model over latent factors, compiled into the attention structure of the denoiser rather than into a message-passing runtime.

The **program layer** (v2) adds process semantics per region — a *family* (observation / state / memory / residual / action), a *carrier* (deterministic / diffusive / filter / memory / residual), a *clock* (periodic, event-triggered — composable firing rules like `Or(periodic(4), on("err.prediction", gt=0.5))`), and a *compile mode* (runtime / freeze / constant / export). Connection operators are auto-derived from family pairs (observation→state = "observe", state→action = "act"); edges can carry triggers (skipped when a predicate over region statistics is false) and learned sigmoid gates. `ProgramCompiler` materializes deploy-time semantics: frozen regions get `requires_grad=False`, constants become buffers, exports write state dicts to disk. The stack is thus: **entity types → canvas schema → wired dispatch → compiled deployment**, each layer inspectable, serializable, and diffable.

## 4. Empirical record

All results: CogVideoX-2B backbone, Bridge V2 robot video, 26 experiments, 236 training runs.

**Looped attention.** Orthogonal to the canvas: iterate frozen DiT blocks k times with zero-initialized learned iteration embeddings (at init, exactly the pretrained model — no distribution shift). From a 12-condition grid (loops × freeze level, 36 runs):

| Action loss ↓ | Frozen (350K) | Half (3.7M) | Unfrozen (11.7M) |
|---|---|---|---|
| 1 loop | 0.121 | 0.115 | 0.108 |
| **3 loops** | **0.073** | 0.107 | 0.088 |
| 4 loops | 0.104 | 0.137 | 0.124 |

Three loops wins at every freeze level; the frozen 3-loop condition — **350K trainable parameters** — beats every unfrozen 11.7M-parameter condition. Recurrence over depth: **1.73× parameter efficiency** (p<0.001). Freeze level does not affect action loss at all (p=0.72); it only affects video-generation quality (8–9× diffusion-loss gap). Loop representations converge toward fixed points (cosine similarity to loop 1: 0.926 → 0.996; token velocities decay exponentially, action tokens fastest).

**What was falsified.** (i) *Looping enables iterative reasoning*: three independent nulls (p=0.97, p>0.05, p>0.05). The benefit is weight-sharing regularization, not reasoning depth — at this scale. (ii) *A shared canvas creates multi-modal binding for free*: joint prediction was 19% *worse* (p<0.0001). Omnimodal capability comes from the multi-encoder/decoder canvas architecture, not from co-residence in one tensor. (iii) *Token allocation follows power laws*: borderline (R²=0.902 but α=0.011 — doubling a region's tokens moves loss 0.8%).

These negatives are load-bearing. They say the current wins come from *structure as regularization and interface*, not from emergent reasoning inside the loop — which is precisely the honest version of the neurosymbolic pitch: you get what you declare, plus parameter efficiency; you do not get free cognition.

## 5. Open problems

**Representation stability (the linchpin).** Everything interoperable — schema-mediated latent exchange, transfer-distance calibration, plug-and-play modalities, swap-the-policy-keep-the-perception composition — assumes a platonic-representation-style claim: that identical declared structure induces predictably aligned latent geometry across seeds, datasets, and backbones. Plausible, unproven. The test battery is specified: seed-stability CKA across corresponding regions; topology→specialization ablations on fixed data; cross-model grafting cost; regression of measured transfer cost against semantic-embedding distance.

**Binding.** The 19% joint-prediction penalty means naive co-residence hurts; whether *declared topology* (rather than a flat shared canvas) recovers or exceeds independent-model performance is the next ablation the falsification demands.

**Scale.** All evidence is at 2B parameters and robot-video scale. Whether declared macrostructure keeps paying as data grows — or is eventually washed out by scale, the bitter-lesson failure mode — is open. The asymmetric bet: schemas cost nothing to author, compile onto pretrained backbones, and can be relaxed edge-by-edge (`fn="none"` → dense) if they bind.

**Learned topology.** Today the graph is authored. The natural continuation — propose-edges/prune-edges under sparsity pressure, with the declared schema as prior rather than constraint — would close the loop back to Chollet's synthesis route, with the canvas as the substrate both routes share.

## 6. Related work

Diffusion policy establishes observation-conditioned action diffusion (the two-node canvas). Unified multimodal DiTs and any-to-any models share one latent sequence but leave structure implicit, to be discovered from surface gradients — the exact move canvas engineering replaces with declaration. Masked/structured attention (sparse patterns, Perceiver bottlenecks, graph attention) supplies the mechanisms; the contribution here is not a mechanism but an *authorable, compilable, serializable schema layer* over them. Probabilistic graphical models and structural causal models are the intellectual ancestors: the canvas topology is a causal graph whose message-passing is implemented by a pretrained denoiser's attention rather than by belief propagation. Looped/universal-transformer recurrence is well studied as adaptive compute; the finding here is that its practical value on frozen video backbones is regularization, not iteration depth.

## 7. Conclusion

The neurosymbolic program stalls in production because it usually demands a new runtime at the neural/symbolic boundary. Canvas engineering dissolves the boundary instead: the symbolic artifact is a schema — a type system for latent computation — and its execution is nothing but masked attention inside a stock diffusion transformer. You declare the causal macrostructure of thought; SGD fills in the rest. The empirical record is early, partly negative, and stated plainly — 1.73× parameter efficiency and 350K-beats-11.7M on the one hand; no iterative reasoning and no free binding on the other. But the abstraction is the point: once latent structure is *declarable*, it is versionable, diffable, compilable, and shareable — and structure you can ship is the only structure that will ever out-compete paying for more data.

---

*Apache 2.0. `pip install canvas-engineering`. Full experiment archive and the companion empirical paper ("Looped Attention in Video Diffusion Transformers: 26 Experiments on What Works, What Doesn't, and Why," Valdez & Claude Opus 4.6) at [github.com/JacobFV/recursive-omnimodal-video-action-model](https://github.com/JacobFV/recursive-omnimodal-video-action-model).*

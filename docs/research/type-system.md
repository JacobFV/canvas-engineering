# The Type System Thesis

Canvas engineering is a type system for latent dynamics. This page explains the analogy and the open research questions.

## The analogy

| Type system concept | Canvas equivalent | Implementation |
|---|---|---|
| Struct field (offset + size) | RegionSpec bounds | `region_indices()` |
| Field type annotation | period, loss_weight, semantic_type, default_attn | `RegionSpec` fields |
| Pointer / reference | Connection | `CanvasTopology` |
| Function signature | Topology pattern + fn type | `attention_ops()` |
| Memory layout | Flattened tensor positions | `num_positions` |
| ABI compatibility | Schema compatibility | `compatible_regions()` |
| Type-directed codegen | Loss mask from declarations | `loss_weight_mask()` |
| Type coercion cost | Transfer distance | `transfer_distance()` |

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/canvas_type_system.png" alt="Type system analogy" width="80%">
</p>

This isn't metaphor. `region_indices()` is literally a memory offset calculation. `loss_weight_mask()` is type-directed codegen for the loss function. The topology is a calling convention between regions.

## What the schema declares

A `CanvasSchema` = layout + topology = complete type signature:

1. **Geometry** — Where each modality lives in the (T, H, W) grid
2. **Frequency** — How fast each region updates (period)
3. **Loss participation** — Which regions are outputs, with what weight
4. **Connectivity** — Who attends to whom, with what temporal constraints
5. **Function types** — What kind of attention each connection uses
6. **Semantic types** — What each modality *means*, as a machine-comparable embedding

Two agents with the same schema can share latent state directly. Two agents with different schemas can estimate compatibility via `transfer_distance()`.

## The open question: representation stability

Everything above assumes the **platonic representation hypothesis** applies: that declared canvas structure produces predictable, stable latent geometry. This is plausible but unproven.

If canvas schemas produce stable representations:
- **Plug-and-play modality**: Retrain only encoder/decoder surface layers
- **Cross-model interop**: Two models on the same schema can read each other's latents
- **Compositional agents**: Swap policy canvas while keeping perception frozen

### Experiments that would test this (31+)

1. **Seed stability**: Train identical schemas with different seeds. CKA between corresponding regions.
2. **Topology → specialization**: Different topologies, same data. Does connectivity predict specialization?
3. **Cross-model grafting**: Freeze trained perception, graft fresh policy. Transfer cost?
4. **Embedding distance calibration**: Train N modality pairs. Plot actual transfer cost vs. semantic embedding distance.

## Design notes

- [Canvas as type system (gist)](https://gist.github.com/JacobFV/86d3ed47bede11da088428bdafb38825)
- [Agent state architecture (gist)](https://gist.github.com/JacobFV/c0dd11498ed715f98094fe84dd2513bb)

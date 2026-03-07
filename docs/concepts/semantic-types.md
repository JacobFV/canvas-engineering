# Semantic Types & Transfer Distance

Each canvas region represents a modality. `RegionSpec` lets you declare the modality's **semantic type** as a human-readable string and a frozen embedding vector from a fixed model.

## Why semantic types?

Two regions can have identical bounds and period but represent completely different signals. Semantic types make modality compatibility a computable quantity instead of a human judgment call.

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/transfer_distance.png" alt="Transfer distance" width="65%">
</p>

## Declaring semantic types

```python
cam = RegionSpec(
    bounds=(0, 8, 0, 12, 0, 12),
    semantic_type="RGB video 224x224 30fps from front-facing monocular camera",
    semantic_embedding=embed("RGB video 224x224 30fps from front-facing monocular camera"),
    embedding_model="openai/text-embedding-3-small",  # fixed, declared
)
```

- **`semantic_type`** — The human-readable source of truth
- **`semantic_embedding`** — Frozen vector from the declared embedding model
- **`embedding_model`** — Must stay constant within an ecosystem. Declared so it can always be re-derived.

## Transfer distance

```python
from canvas_engineering import transfer_distance

d = transfer_distance(cam, depth)  # cosine distance in [0, 2]
# 0.0 = identical modality
# ~0.15 = cheap to bridge (1-2 adapter layers)
# ~0.65 = expensive (full MLP adapter)
# 1.0 = orthogonal
```

## Cross-schema compatibility

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/schema_alignment.png" alt="Schema alignment" width="90%">
</p>

```python
pairs = schema_a.compatible_regions(schema_b, threshold=0.3)
# → [("visual", "camera", 0.04), ("action", "gripper_cmd", 0.12)]
```

Two agents with compatible schemas can potentially share latent state directly — no tokenization, no re-encoding.

# Canvas Types

Canvas Types is a **compiler** for latent space structure. You declare Python types whose fields are latent regions. The compiler flattens, packs, and wires them into a `CanvasSchema` — concrete attention masks, loss weights, and region assignments ready for training.

## The problem

Today, building a canvas schema means manually specifying region bounds, hand-wiring connections, and hand-tuning loss weights. This is like writing assembly: full control, zero abstraction.

Real-world systems have **compositional structure** — agents contain subsystems, organizations contain agents, robots contain sensors and actuators. This structure implies natural connectivity patterns and allocation proportions. **Canvas Types** lets you declare the structure and compiles the rest.

## From types to schemas

```python
from dataclasses import dataclass
from canvas_engineering import Field, compile_schema

@dataclass
class Robot:
    camera: Field = Field(12, 12)              # (1) 144 canvas positions
    joints: Field = Field(1, 8)                # 8 positions
    action: Field = Field(1, 8, loss_weight=2.0)

bound = compile_schema(Robot(), T=8, H=16, W=16, d_model=256)  # (2)

canvas = bound.build_canvas()                  # (3)
batch = bound.create_batch(4)
bound["camera"].place(batch, camera_embs)      # (4)
actions = bound["action"].extract(batch)
```

1. **`Field(h, w)`** declares a region of h&times;w positions per timestep. Default `(1, 1)` = scalar.
2. **`compile_schema()`** walks the object tree, packs fields onto the grid, auto-wires connectivity.
3. **`build_canvas()`** creates a `SpatiotemporalCanvas` with positional and modality embeddings.
4. **`bound["name"]`** returns a `BoundField` for direct place/extract access.

## Field

`Field` is a frozen dataclass. It declares a region's spatial footprint and training semantics:

```python
Field(
    h=1, w=1,              # spatial extent on the grid
    period=1,              # temporal frequency (frames per real-world update)
    is_output=True,        # participates in diffusion loss?
    loss_weight=1.0,       # relative gradient weight
    attn="cross_attention", # default attention function type
    semantic_type=None,    # human-readable modality description
    temporal_extent=None,  # timesteps this field spans (None = full T)
)
```

Fields default to `(1, 1)` — a single canvas position (scalar). This means `Field()` is the simplest possible declaration.

## Composition: nesting and arrays

Types compose naturally through Python:

```python
@dataclass
class Sensor:
    camera: Field = Field(8, 8)
    depth: Field = Field(8, 8)

@dataclass
class Robot:
    sensor: Sensor = field(default_factory=Sensor)
    plan: Field = Field(4, 4)
    action: Field = Field(1, 4)
```

Compiled region names use dotted paths: `"sensor.camera"`, `"sensor.depth"`, `"plan"`, `"action"`.

Arrays are just Python lists:

```python
@dataclass
class Fleet:
    coordinator: Field = Field(4, 4)
    robots: list = field(default_factory=list)

fleet = Fleet(robots=[Robot(), Robot(), Robot()])
bound = compile_schema(fleet, T=8, H=32, W=32, d_model=256)
# Regions: "coordinator", "robots[0].sensor.camera", "robots[0].plan", ...
```

Array sizes are determined at instantiation, not declaration. Different instances can have different sizes.

## Per-instance configuration

Different instances of the same type can have different field sizes:

```python
ceo = Employee(thought=Field(8, 8))    # big thinker
intern = Employee(thought=Field(2, 2)) # smaller capacity
```

## Connectivity policies

The type hierarchy implies natural wiring. `ConnectivityPolicy` controls four axes:

| Policy | Options | Default |
|--------|---------|---------|
| **intra** | `dense`, `isolated`, `causal_chain`, `star` | `dense` |
| **parent_child** | `matched_fields`, `hub_spoke`, `broadcast`, `aggregate`, `none` | `matched_fields` |
| **array_element** | `isolated`, `dense`, `matched_fields`, `ring` | `isolated` |
| **temporal** | `dense`, `causal`, `same_frame` | `dense` |

**`matched_fields`** is the key insight: if a parent type and a child type share a field name (e.g., both inherit `thought` from `Agent`), the compiler auto-connects them. The shared name is the semantic link.

## Layout strategies

| Strategy | Description |
|----------|-------------|
| **`PACKED`** (default) | Each type instance gets a contiguous sub-grid |
| **`INTERLEAVED`** | Same-named fields from parent and children are spatially adjacent |

Interleaved layout is powerful when you want a pretrained model's spatial attention priors to bridge parent-child fields.

## Works with anything

`compile_schema()` walks any Python object. It supports:

- **dataclasses** — the recommended approach
- **Pydantic models** — if you prefer Pydantic
- **plain classes** — any object with `Field` attributes

No special base class required. No metaclass magic.

## Examples

See the [Examples](../examples/index.md) section for runnable demonstrations:

- [Hello Canvas Types](../examples/01-hello-types.md) — 15-line declare-compile-train
- [Multi-Frequency Fusion](../examples/02-multi-frequency.md) — structured vs flat comparison
- [CartPole Control](../examples/03-cartpole.md) — real gym env with self-consistency loss

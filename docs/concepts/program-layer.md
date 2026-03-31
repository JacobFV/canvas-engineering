# Program Layer

v1 gave canvas-engineering a **typed layout**: where regions live, who talks to whom, what attention function to use. v2 adds a **typed process layer**: what each region *is*, how it *learns*, when it *runs*, and what gets *compiled away* at deploy time.

The layout is the schema. The program is the execution plan.

## From layout to process

| | v1 (`compile_schema`) | v2 (`compile_program`) |
|---|---|---|
| **Declares** | Geometry, connectivity, loss weights | + family, carrier, clock, learning, compile mode |
| **Output** | `BoundSchema` | `BoundSchema` + `CanvasProgram` |
| **Operators** | All edges are "attend" | Family-derived: observe, predict, correct, act, ... |
| **Scheduling** | All regions fire every step | Clock-driven: periodic, event-triggered, boundary |
| **Deploy** | Keep everything | Freeze / materialize / export per region |

## The 5 families

Every region belongs to a **family** that declares what kind of state it holds:

| Family | Role | Default learning | Default compile mode |
|--------|------|-----------------|---------------------|
| `observation` | Sensory evidence from the world | SSL prediction | freeze |
| `state` | Internal beliefs, estimates, goals | Posterior matching | runtime |
| `memory` | Persistent lookup (episodic, semantic) | Retrieval accuracy | export |
| `residual` | Error signals for scheduling/diagnostics | Calibration | freeze |
| `action` | Motor commands, outputs to the world | Supervised | freeze |

Everything beyond the 5 families is expressed via **tags** -- semantic sub-labels within a family. A state region might have `tags=("belief", "object")` or `tags=("goal",)`. Tags are free-form strings; families are structural.

## The 5 carriers

Each region also declares a **carrier** -- what dynamics govern it. See [Carriers](carriers.md) for the full breakdown.

| Carrier | Dynamics | Example |
|---------|----------|---------|
| `deterministic` | Standard forward updates | Joint angles, reward scalars |
| `diffusive` | Noise/denoise | Video, images |
| `filter` | Predict/correct | Kalman-style belief state |
| `memory` | Persistent lookup | Episodic memory bank |
| `residual` | Error traces | Prediction error signals |

Family and carrier are orthogonal. A "future video" region is `family="observation", carrier="diffusive"`. A "current video" region is `family="observation", carrier="deterministic"`.

## Operators

When `compile_program()` wires connections, it replaces the default `"attend"` operator with a family-derived operator from `DEFAULT_WIRING`. This is the semantic intent of the edge -- separate from the backend attention function (`fn`).

See [Topology -- Operators](topology.md#operators) for the wiring table.

## compile_program() vs compile_schema()

`compile_schema()` produces a `BoundSchema` -- geometry + topology. `compile_program()` calls `compile_schema()` internally, then reads `family`, `tags`, and `carrier` from each `Field` to build a `CanvasProgram` alongside the schema.

```python
from dataclasses import dataclass
from canvas_engineering import Field, compile_program

@dataclass
class Robot:
    camera: Field = Field(12, 12, family="observation")
    joints: Field = Field(1, 8, family="observation", carrier="deterministic")
    belief: Field = Field(4, 4, family="state", tags=("belief",))
    action: Field = Field(1, 8, family="action", loss_weight=2.0)

bound, program = compile_program(Robot(), T=8, d_model=256)

print(program.summary())
# CanvasProgram (v2.0.0, 4 regions programmed):
#   families: action=1, observation=2, state=1
```

Both `BoundSchema` and `CanvasProgram` are JSON-serializable. The program layers on top of the schema -- existing `compile_schema()` code continues to work unchanged.

## Further reading

- [Carriers](carriers.md) -- Not everything is diffusive
- [Canvas Types](canvas-types.md) -- The compositional type system
- [Topology](topology.md) -- Connectivity and operators
- [API: program](../api/program.md) -- ClockSpec, LearningSpec, RegionProgram, CanvasProgram
- [API: compile_program](../api/types.md) -- The compiler entry point

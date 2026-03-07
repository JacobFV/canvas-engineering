# Example 04: Autonomous Vehicle Fleet

Multi-agent cooperative trajectory prediction. Ring topology between vehicles and interleaved layout create measurable advantages over isolated or fully-dense connectivity.

**Source**: [`examples/04_vehicle_fleet.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/04_vehicle_fleet.py) *(coming soon)*

## What it demonstrates

- **Ring topology** between N vehicles — each vehicle attends only to its neighbors, not the full fleet
- **Interleaved layout** groups matching fields (all `position` regions adjacent, all `intent` regions adjacent) vs packed layout where each vehicle occupies a contiguous block
- **Contrastive loss on intent** — vehicles with similar trajectories should have similar intent embeddings
- **Social force data** — physically grounded synthetic trajectories with repulsion + lane attraction

## Type hierarchy

```python
@dataclass
class Vehicle:
    position: Field = Field(1, 4)              # x, y, vx, vy
    heading: Field = Field(1, 2)               # sin(theta), cos(theta)
    lane_context: Field = Field(1, 4, is_output=False)  # input-only
    intent: Field = Field(2, 4)                # latent driving intent

@dataclass
class Fleet:
    vehicles: list[Vehicle]                    # ring-connected
```

## Connectivity

```python
bound = compile_schema(
    Fleet(vehicles=[Vehicle() for _ in range(N)]),
    T=1, H=8, W=8, d_model=64,
    connectivity=ConnectivityPolicy(
        intra="dense",
        array_element="ring",      # each vehicle sees only neighbors
        parent_child="hub_spoke",
    ),
    layout=LayoutStrategy.INTERLEAVED,
)
```

`array_element="ring"` creates ring connections between Fleet.vehicles[i] and Fleet.vehicles[i±1]. This is the key condition being ablated.

## Conditions compared

| Condition | Connectivity | Layout |
|-----------|-------------|--------|
| Baseline | isolated | packed |
| Ring | ring | packed |
| Ring + interleaved | ring | interleaved |
| Dense | dense | packed |

!!! note "Task spec"
    Full implementation details in [`examples/tasks/04_vehicle_fleet.md`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/tasks/04_vehicle_fleet.md).

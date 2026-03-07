# Example 06: Air Traffic Conflict Detection

Safety-critical multi-agent prediction. `loss_weight=10` on the conflict detection head demonstrably catches more conflicts — and dense inter-aircraft connectivity is essential here, unlike the vehicle fleet where ring topology sufficed.

**Source**: [`examples/06_air_traffic.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/06_air_traffic.py) *(coming soon)*

## What it demonstrates

- **Loss weight drives safety recall** — 10× weight on conflict detection measurably shifts precision/recall tradeoff
- **Dense vs ring connectivity** — air traffic requires all-pairs awareness; ring is insufficient
- **Conflict field** — explicit latent region per aircraft pair for collision geometry
- **Synthetic TRACON** — realistic sector geometry with crossing traffic flows

## Type hierarchy

```python
@dataclass
class Aircraft:
    state: Field = Field(1, 6)               # x, y, z, vx, vy, vz
    intent: Field = Field(2, 4)              # flight plan / trajectory intent
    separation: Field = Field(1, 2, loss_weight=10.0)  # conflict prediction

@dataclass
class Sector:
    aircraft: list[Aircraft]                 # dense all-pairs connectivity
    weather: Field = Field(2, 4, is_output=False)  # wind + turbulence
```

## Why dense, not ring

In vehicle fleets, a driver primarily needs to know about adjacent vehicles. In air traffic, a separation conflict can develop between any aircraft pair in the sector — missed by ring topology but caught by dense connectivity.

## Conditions compared

| Condition | Connectivity | Conflict weight |
|-----------|-------------|-----------------|
| Baseline | isolated | 1.0 |
| Dense, equal weight | dense | 1.0 |
| Dense, safety weight | dense | 10.0 |
| Ring (hypothesis-falsifying) | ring | 10.0 |

!!! note "Task spec"
    Full implementation details in [`examples/tasks/06_air_traffic.md`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/tasks/06_air_traffic.md).

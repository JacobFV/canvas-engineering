# Example 11: Mars Colony — Multi-System Autonomous Control

The capstone. 70+ fields across 6 subsystems with hub-spoke connectivity. Cascading failure prediction through cross-system reasoning — something a per-subsystem model structurally cannot do.

**Source**: [`examples/11_mars_colony.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/11_mars_colony.py) *(coming soon)*

## What it demonstrates

- **Six-subsystem hierarchy** — life support, power, thermal, communications, ISRU, crew
- **Cross-system cascading failures** — a power anomaly propagates to thermal, then life support; the model must route through these connections
- **Hub-spoke colony-level coordination** — colony-level fields see all subsystems; subsystems see each other only through the hub
- **Crew psychological state** — stress, fatigue, morale as first-class fields that affect mission-critical decisions

## Type hierarchy (partial)

```python
@dataclass
class LifeSupport:
    atmosphere: Field = Field(2, 4, period=1)    # O2, CO2, pressure, humidity
    water_loop: Field = Field(1, 4, period=2)
    food_stores: Field = Field(1, 2, period=60)
    lss_health: Field = Field(1, 1, loss_weight=5.0)

@dataclass
class PowerSystem:
    solar_output: Field = Field(2, 4, period=1)
    battery_state: Field = Field(1, 4, period=1)
    load_distribution: Field = Field(2, 4, period=1)
    power_health: Field = Field(1, 1, loss_weight=4.0)

@dataclass
class CrewState:
    physical_health: Field = Field(2, 4, period=4)
    psychological: Field = Field(2, 4, period=8)
    workload: Field = Field(1, 2, period=2)
    morale: Field = Field(1, 1, period=8)

@dataclass
class MarsColony:
    life_support: LifeSupport = field(default_factory=LifeSupport)
    power: PowerSystem = field(default_factory=PowerSystem)
    thermal: ThermalControl = field(default_factory=ThermalControl)
    comms: Communications = field(default_factory=Communications)
    isru: ISRU = field(default_factory=ISRU)
    crew: CrewState = field(default_factory=CrewState)
    mission_health: Field = Field(1, 1, loss_weight=3.0)   # colony-level
    abort_risk: Field = Field(1, 1, loss_weight=10.0)      # evacuation signal
```

## Hub-spoke cross-system reasoning

```python
bound = compile_schema(
    MarsColony(), T=4, H=16, W=16, d_model=128,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",  # mission_health sees everything
    ),
)
```

Each subsystem's `_health` field attends to all fields within its subsystem. The colony-level `mission_health` and `abort_risk` attend to all `_health` fields. This two-level hub-spoke forces cascading failure information to propagate upward through the hierarchy — not shortcut through a flat dense connection.

## Ablation: can a flat model predict cascading failures?

A flat model trained on colony data learns statistical correlations between subsystems but has no mechanism to trace the causal path. The canvas model, forced by topology, learns the pathway. The ablation plots the failure prediction recall broken down by failure origin — single-system vs cascading — where the topology advantage is largest for cascades.

!!! note "Task spec"
    Full implementation details in [`examples/tasks/11_mars_colony.md`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/tasks/11_mars_colony.md).

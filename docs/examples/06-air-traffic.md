# Example 06: Air Traffic Control

Safety-critical multi-agent prediction with 12 aircraft and 3 weather cells in a TRACON sector. `loss_weight=10` on the conflict detection head demonstrably affects recall — and dense inter-aircraft connectivity is compared against isolated baselines.

**Source**: [`examples/06_air_traffic_control.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/06_air_traffic_control.py)

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/06_atc.png" alt="Example 06 results" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/06_atc.gif" alt="Example 06 ATC radar animation" width="100%">
</p>

<video width="100%" controls>
  <source src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/06_atc.mp4" type="video/mp4">
</video>

**4×4 panel figure**: TRACON sector geometry, aircraft trajectories with wake turbulence categories, conflict detection heatmaps, topology comparison, separation distance analysis, and training curves — all in ATC green-on-black aesthetic.

**Animation**: Radar sweep display with 12 aircraft, data blocks (callsign, flight level, speed, wake category), conflict alert lines, weather cells, and HUD overlays.

## Type hierarchy

```python
@dataclass
class Aircraft:
    state: Field = Field(1, 3)                           # x, y, z
    flight_plan: Field = Field(1, 4, is_output=False)    # route input
    trajectory: Field = Field(1, 4, loss_weight=3.0)     # predicted path
    conflict: Field = Field(1, 2, loss_weight=10.0)      # separation prediction
    wake_category: Field = Field(1, 1, is_output=False)  # H/M/L wake class
    intent: Field = Field(1, 2)                          # latent intent

@dataclass
class WeatherCell:
    position: Field = Field(1, 2, is_output=False)
    intensity: Field = Field(1, 1, is_output=False)
    movement: Field = Field(1, 2)

@dataclass
class TRACON:
    weather: Field = Field(1, 4, is_output=False)
    sector_load: Field = Field(1, 2)
    runway_state: Field = Field(1, 2)
    weather_cells: list  # 3 cells
    aircraft: list       # 12 aircraft
```

## Connectivity

Three conditions compared:

```python
# Dense, conflict weight=10 — full all-pairs + safety emphasis
bound_dense_w10 = make_schema(array_element="dense", conflict_weight=10.0)

# Dense, conflict weight=1 — full all-pairs, equal weighting
bound_dense_w1 = make_schema(array_element="dense", conflict_weight=1.0)

# Isolated, conflict weight=10 — no inter-aircraft attention
bound_isolated = make_schema(array_element="isolated", conflict_weight=10.0)
```

Dense topologies: 5,760 connections. Isolated: 954 connections.

## Why dense, not ring

In vehicle fleets, a driver primarily needs to know about adjacent vehicles. In air traffic, a separation conflict can develop between any aircraft pair in the sector — missed by ring topology but caught by dense connectivity.

## Run it

```bash
python examples/06_air_traffic_control.py
# Generates: assets/examples/06_atc.{png,gif,mp4}
```

# Example 04: Autonomous Vehicle Fleet

Multi-agent cooperative trajectory prediction on complex road networks. 64 vehicles across 4 traffic zones on highways, roundabouts, intersections, and ramps — with isolated, ring, and dense topologies compared.

**Source**: [`examples/04_autonomous_vehicle_fleet.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/04_autonomous_vehicle_fleet.py)

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/04_fleet.png" alt="Example 04 results" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/04_fleet.gif" alt="Example 04 fleet animation" width="100%">
</p>

<video width="100%" controls>
  <source src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/04_fleet.mp4" type="video/mp4">
</video>

**16-panel figure**: Road network geometry, per-zone trajectory plots, speed heatmaps, topology comparison metrics (ADE, FDE, collision rate), attention masks, canvas layout visualization, and training curves.

**Animation**: 64 vehicles moving through the road network with glowing trails and speed-mapped colors.

## Type hierarchy

```python
@dataclass
class Vehicle:
    position: Field = Field(1, 1)
    velocity: Field = Field(1, 1)
    heading: Field = Field(1, 1)
    road_context: Field = Field(1, 2, is_output=False)
    intent: Field = Field(1, 2)
    trajectory: Field = Field(1, 2, loss_weight=4.0)

@dataclass
class TrafficZone:
    signal_state: Field = Field(1, 1, is_output=False)
    congestion: Field = Field(1, 1, loss_weight=2.0)
    vehicles: list  # 16 vehicles per zone

@dataclass
class RoadNetwork:
    global_flow: Field = Field(1, 2)
    zones: list     # 4 zones
```

## Connectivity

Three topology variants compared on the same data:

```python
# Isolated — no inter-vehicle attention
bound_isolated = make_schema(ConnectivityPolicy(
    intra="dense", parent_child="hub_spoke",
    array_element="isolated", temporal="dense"))

# Ring — each vehicle attends to neighbors only
bound_ring = make_schema(ConnectivityPolicy(
    intra="dense", parent_child="hub_spoke",
    array_element="ring", temporal="dense"))

# Dense — full all-pairs attention
bound_dense = make_schema(ConnectivityPolicy(
    intra="dense", parent_child="hub_spoke",
    array_element="dense", temporal="dense"))
```

Canvas: 25×24 = 600 positions. Isolated: 3,873 connections. Ring: 8,513. Dense: 38,481.

## Key metrics

| Topology | ADE | FDE | Collision Rate |
|----------|-----|-----|----------------|
| Isolated | 1.32 | 2.75 | 23.8% |
| Ring | 2.72 | 4.82 | 32.3% |
| Dense | 4.57 | 7.73 | 92.0% |

## Run it

```bash
python examples/04_autonomous_vehicle_fleet.py
# Generates: assets/examples/04_fleet.{png,gif,mp4}
```

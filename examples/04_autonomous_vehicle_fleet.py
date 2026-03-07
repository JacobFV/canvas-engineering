"""Autonomous vehicle fleet: V2X cooperative perception and planning.

A fleet of autonomous vehicles that share perception through a latent
communication channel. Each vehicle has:
- Multi-camera surround view (6 cameras -> BEV)
- LiDAR point cloud features
- HD map context (input-only conditioning)
- Predicted trajectory (the output we care about)
- A "broadcast" field for V2X latent communication

The fleet coordinator aggregates all vehicle broadcasts into a shared
traffic state, which feeds back into each vehicle's planner.

This is exactly the architecture papers like V2X-ViT and CoBEVT describe,
but declared as a type hierarchy instead of hand-wired modules.

Key design choices:
- Interleaved layout: all vehicle "broadcast" fields are spatially adjacent
  so the pretrained spatial attention naturally fuses them
- Ring connectivity between vehicles (geographic neighbors share perception)
- Map context is input-only (no loss, just conditioning)
- Trajectory output gets 4x loss weight (safety of life)
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy, LayoutStrategy


@dataclass
class Camera:
    """One camera in the surround-view rig."""
    features: Field = Field(4, 4)      # 16 positions per camera


@dataclass
class Vehicle:
    """One autonomous vehicle with full sensor suite and planner."""
    # Perception
    front_cam: Camera = dc_field(default_factory=Camera)
    rear_cam: Camera = dc_field(default_factory=Camera)
    left_cam: Camera = dc_field(default_factory=Camera)
    right_cam: Camera = dc_field(default_factory=Camera)
    front_left_cam: Camera = dc_field(default_factory=Camera)
    front_right_cam: Camera = dc_field(default_factory=Camera)

    lidar_bev: Field = Field(8, 8)     # 64 positions: bird's-eye LiDAR features

    # Context
    hd_map: Field = Field(4, 4,        # 16 positions: HD map around vehicle
                          is_output=False,
                          semantic_type="HD map lanes+signs 50m radius")

    # Planning
    plan: Field = Field(4, 8,          # 32 positions: internal planning state
                        attn="mamba",   # sequential temporal reasoning
                        semantic_type="ego vehicle planning state")

    trajectory: Field = Field(2, 8,    # 16 positions: predicted future waypoints
                              loss_weight=4.0,  # safety of life
                              semantic_type="predicted trajectory 3s horizon")

    # V2X communication
    broadcast: Field = Field(2, 4,     # 8 positions: what this vehicle shares
                             semantic_type="V2X cooperative perception broadcast")


@dataclass
class FleetCoordinator:
    """Central coordinator that fuses all vehicle broadcasts."""
    traffic_state: Field = Field(8, 8,  # 64 positions: global traffic picture
                                 semantic_type="fused fleet traffic state")
    vehicles: list = dc_field(default_factory=list)


# --- 4-vehicle fleet ---

fleet = FleetCoordinator(
    vehicles=[Vehicle() for _ in range(4)],
)

bound = compile_schema(
    fleet, T=8, H=64, W=64, d_model=512,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="matched_fields",
        array_element="ring",          # geographic neighbors share perception
        temporal="causal",             # strictly causal for safety
    ),
    layout_strategy=LayoutStrategy.INTERLEAVED,  # broadcasts spatially adjacent
)

print("=== Autonomous Vehicle Fleet ===")
print(f"Vehicles: 4")
print(f"Fields per vehicle: {sum(1 for n in bound.field_names if 'vehicles[0]' in n)}")
print(f"Total fields: {len(bound.field_names)}")
print(f"Total positions: {bound.layout.num_positions:,}")
print(f"Used positions: {sum(bf.num_positions for bf in bound.fields.values()):,}")
print()

# Show that broadcasts are spatially co-located (interleaved)
print("Broadcast field positions (should be adjacent rows):")
for i in range(4):
    bf = bound[f"vehicles[{i}].broadcast"]
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    print(f"  vehicle[{i}].broadcast: h=[{h0},{h1}), w=[{w0},{w1})")

# Show ring connectivity between vehicles
print("\nRing connections (vehicle-to-vehicle):")
ring_conns = [c for c in bound.topology.connections
              if "vehicles[" in c.src and "vehicles[" in c.dst
              and c.src.split("]")[0] != c.dst.split("]")[0]]  # cross-vehicle only
# Deduplicate by vehicle pair
pairs_seen = set()
for c in ring_conns:
    v_src = c.src.split("]")[0] + "]"
    v_dst = c.dst.split("]")[0] + "]"
    pair = tuple(sorted([v_src, v_dst]))
    if pair not in pairs_seen:
        pairs_seen.add(pair)
        print(f"  {v_src} <-> {v_dst}")

# Trajectory gets dominant loss share
weights = bound.layout.loss_weight_mask("cpu")
total_weight = weights.sum().item()
for i in range(4):
    traj = bound[f"vehicles[{i}].trajectory"]
    traj_indices = traj.indices()
    traj_weight = sum(weights[idx].item() for idx in traj_indices)
    print(f"\n  Vehicle {i} trajectory: {traj_weight/total_weight*100:.1f}% of total loss"
          f" ({traj.spec.loss_weight}x weight)")

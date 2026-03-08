"""Autonomous Vehicle Fleet: 64-vehicle cooperative trajectory prediction.

Complex road network (highway + roundabout + intersections) with social-force
dynamics. Three topology comparisons across four traffic zones:
  1. Isolated: vehicles can't see each other
  2. Ring: geographic neighbors share perception
  3. Dense: every vehicle sees every other vehicle

Plus a contrastive loss on the intent field — vehicles with similar future
trajectories should have similar intents.

Outputs:
  assets/examples/04_fleet.png  — multi-panel analysis figure
  assets/examples/04_fleet.gif  — animated aerial simulation

Run:  python examples/04_autonomous_vehicle_fleet.py
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field as dc_field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch, Circle
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from canvas_engineering import Field, compile_schema, ConnectivityPolicy, LayoutStrategy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

N_VEHICLES = 64
N_ZONES = 4
VEHICLES_PER_ZONE = N_VEHICLES // N_ZONES  # 16
PRED_HORIZON = 8   # future steps to predict
DT = 0.15           # seconds per step
LANE_WIDTH = 3.2


# ── 1. Road Network: parametric curves ──────────────────────────────

def bezier_cubic(p0, p1, p2, p3, n=80):
    """Cubic Bezier curve returning (n, 2) array."""
    t = np.linspace(0, 1, n).reshape(-1, 1)
    pts = ((1-t)**3 * p0 + 3*(1-t)**2*t * p1
           + 3*(1-t)*t**2 * p2 + t**3 * p3)
    return pts.astype(np.float32)


def arc_segment(cx, cy, r, a0, a1, n=60):
    """Circular arc from angle a0 to a1, returning (n, 2) array."""
    angles = np.linspace(a0, a1, n)
    pts = np.stack([cx + r * np.cos(angles),
                    cy + r * np.sin(angles)], axis=-1)
    return pts.astype(np.float32)


def line_segment(p0, p1, n=40):
    """Straight line from p0 to p1."""
    t = np.linspace(0, 1, n).reshape(-1, 1)
    return (p0 * (1 - t) + p1 * t).astype(np.float32)


def offset_curve(center_pts, offset):
    """Offset a polyline by `offset` perpendicular to its tangent."""
    tangents = np.diff(center_pts, axis=0)
    tangents = np.vstack([tangents, tangents[-1:]])
    norms = np.sqrt((tangents**2).sum(axis=1, keepdims=True)).clip(min=1e-6)
    tangents = tangents / norms
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)
    return (center_pts + offset * normals).astype(np.float32)


# Build the road network
# Segment 0: Main highway (3-lane, curved left-to-right)
HWY_CENTER = bezier_cubic(
    np.array([-60, 0]), np.array([-20, 12]),
    np.array([20, -8]), np.array([65, 5]), n=120)

# Segment 1: Off-ramp from highway to roundabout
RAMP_START = HWY_CENTER[45]  # approx midpoint
RAMP = bezier_cubic(
    RAMP_START, RAMP_START + np.array([3, -8]),
    np.array([10, -22]), np.array([18, -28]), n=60)

# Segment 2: Roundabout (traffic circle)
RBT_CENTER = np.array([22.0, -32.0])
RBT_RADIUS = 7.0
ROUNDABOUT = arc_segment(RBT_CENTER[0], RBT_CENTER[1], RBT_RADIUS,
                         0, 2*np.pi, n=100)

# Segment 3: Road entering roundabout from south
SOUTH_ROAD = line_segment(
    np.array([22, -55]), RBT_CENTER + np.array([0, -RBT_RADIUS]), n=50)

# Segment 4: Road exiting roundabout to east
EAST_ROAD = bezier_cubic(
    RBT_CENTER + np.array([RBT_RADIUS, 0]),
    np.array([38, -30]), np.array([48, -25]),
    np.array([60, -20]), n=60)

# Segment 5: T-intersection road off the south road
T_JUNCTION_PT = np.array([22, -48])
T_ROAD_WEST = line_segment(T_JUNCTION_PT, T_JUNCTION_PT + np.array([-25, 0]), n=40)
T_ROAD_EAST = line_segment(T_JUNCTION_PT, T_JUNCTION_PT + np.array([25, 0]), n=40)

# Segment 6: Curved connector (east road curves back north)
CURVE_CONNECTOR = bezier_cubic(
    np.array([60, -20]), np.array([65, -10]),
    np.array([62, 8]), np.array([65, 5]), n=50)

# Collect all road segments with their lane counts
ROAD_SEGMENTS = [
    ('highway',    HWY_CENTER, 3),
    ('ramp',       RAMP, 1),
    ('roundabout', ROUNDABOUT, 2),
    ('south_road', SOUTH_ROAD, 2),
    ('east_road',  EAST_ROAD, 2),
    ('t_west',     T_ROAD_WEST, 1),
    ('t_east',     T_ROAD_EAST, 1),
    ('connector',  CURVE_CONNECTOR, 1),
]

# Traffic signals at intersections (positions, initial states)
TRAFFIC_SIGNALS = [
    {'pos': RAMP_START, 'state': 'green', 'period': 40},       # highway off-ramp
    {'pos': RBT_CENTER + np.array([0, RBT_RADIUS]), 'state': 'red', 'period': 30},
    {'pos': T_JUNCTION_PT, 'state': 'green', 'period': 35},
    {'pos': RBT_CENTER + np.array([RBT_RADIUS, 0]), 'state': 'red', 'period': 25},
]


def get_nearest_road_point(pos, segment_pts):
    """Find the nearest point on a road segment to pos."""
    dists = np.sqrt(((segment_pts - pos)**2).sum(axis=1))
    idx = np.argmin(dists)
    return idx, dists[idx], segment_pts[idx]


def road_tangent_at(segment_pts, idx):
    """Get tangent direction at index along a road segment."""
    if idx < len(segment_pts) - 1:
        t = segment_pts[idx + 1] - segment_pts[idx]
    else:
        t = segment_pts[idx] - segment_pts[idx - 1]
    norm = np.sqrt((t**2).sum()).clip(min=1e-6)
    return t / norm


# ── 2. Type declarations ─────────────────────────────────────────────

@dataclass
class Vehicle:
    position: Field = Field(1, 1)                          # x, y packed
    velocity: Field = Field(1, 1)                          # vx, vy packed
    heading: Field = Field(1, 1)                           # heading angle
    road_context: Field = Field(1, 2, is_output=False)     # lane info, signal state
    intent: Field = Field(1, 2)                            # latent driving intent
    trajectory: Field = Field(1, 2, loss_weight=4.0)       # predicted future xy


@dataclass
class TrafficZone:
    signal_state: Field = Field(1, 1, is_output=False)     # traffic light state
    congestion: Field = Field(1, 1, loss_weight=2.0)       # congestion level
    vehicles: list = dc_field(default_factory=list)


@dataclass
class RoadNetwork:
    global_flow: Field = Field(1, 2)                       # global traffic flow
    zones: list = dc_field(default_factory=list)


CANVAS_H, CANVAS_W = 25, 24  # fits 586 positions


def make_schema(connectivity_policy):
    zones = []
    for z in range(N_ZONES):
        zone = TrafficZone(
            vehicles=[Vehicle() for _ in range(VEHICLES_PER_ZONE)]
        )
        zones.append(zone)
    network = RoadNetwork(zones=zones)
    return compile_schema(
        network, T=1, H=CANVAS_H, W=CANVAS_W, d_model=32,
        connectivity=connectivity_policy,
    )


# Three topologies to compare
print("Compiling schemas...")
bound_isolated = make_schema(ConnectivityPolicy(
    intra="dense", parent_child="hub_spoke",
    array_element="isolated", temporal="dense"))
bound_ring = make_schema(ConnectivityPolicy(
    intra="dense", parent_child="hub_spoke",
    array_element="ring", temporal="dense"))
bound_dense = make_schema(ConnectivityPolicy(
    intra="dense", parent_child="hub_spoke",
    array_element="dense", temporal="dense"))

for name, b in [("isolated", bound_isolated), ("ring", bound_ring),
                ("dense", bound_dense)]:
    n_conn = len(b.topology.connections) if b.topology else 0
    print(f"  {name}: {b.layout.num_positions} positions, {n_conn} connections")


# ── 3. Synthetic data: social-force on road network ─────────────────

def assign_vehicles_to_roads(n_vehicles):
    """Place vehicles on random road segments with lane offsets."""
    positions = np.zeros((n_vehicles, 2), dtype=np.float32)
    velocities = np.zeros((n_vehicles, 2), dtype=np.float32)
    headings = np.zeros(n_vehicles, dtype=np.float32)
    road_ids = np.zeros(n_vehicles, dtype=np.int32)
    lane_offsets = np.zeros(n_vehicles, dtype=np.float32)
    road_indices = np.zeros(n_vehicles, dtype=np.int32)

    # Weight roads by length (highway gets more vehicles)
    seg_lengths = []
    for name, pts, n_lanes in ROAD_SEGMENTS:
        diffs = np.diff(pts, axis=0)
        length = np.sqrt((diffs**2).sum(axis=1)).sum()
        seg_lengths.append(length * n_lanes)
    seg_probs = np.array(seg_lengths)
    seg_probs = seg_probs / seg_probs.sum()

    for i in range(n_vehicles):
        seg_id = np.random.choice(len(ROAD_SEGMENTS), p=seg_probs)
        name, pts, n_lanes = ROAD_SEGMENTS[seg_id]
        # Random position along segment
        idx = np.random.randint(5, len(pts) - 5)
        # Random lane offset
        lane = np.random.randint(0, n_lanes)
        lane_off = (lane - (n_lanes - 1) / 2.0) * LANE_WIDTH

        center_pt = pts[idx]
        tangent = road_tangent_at(pts, idx)
        normal = np.array([-tangent[1], tangent[0]])

        pos = center_pt + lane_off * normal
        # Speed along tangent (8-18 m/s), roundabout slower
        base_speed = 6.0 if name == 'roundabout' else np.random.uniform(8, 18)
        vel = tangent * base_speed

        positions[i] = pos
        velocities[i] = vel
        headings[i] = np.arctan2(tangent[1], tangent[0])
        road_ids[i] = seg_id
        lane_offsets[i] = lane_off
        road_indices[i] = idx

    return positions, velocities, headings, road_ids, lane_offsets, road_indices


def social_force_trajectory(n_scenes=1024, n_vehicles=N_VEHICLES,
                            n_history=4, n_future=PRED_HORIZON):
    """Generate multi-vehicle trajectories with road-following social force model."""
    total_steps = n_history + n_future
    all_history = []
    all_future = []
    all_road_ctx = []
    all_zone_ids = []

    for scene_i in range(n_scenes):
        pos, vel, hdg, road_ids, lane_offs, road_idx = assign_vehicles_to_roads(n_vehicles)
        x, y = pos[:, 0].copy(), pos[:, 1].copy()
        vx, vy = vel[:, 0].copy(), vel[:, 1].copy()

        # Assign vehicles to zones by road region
        zone_ids = np.zeros(n_vehicles, dtype=np.int32)
        for i in range(n_vehicles):
            if road_ids[i] <= 1:
                zone_ids[i] = 0   # highway + ramp
            elif road_ids[i] == 2:
                zone_ids[i] = 1   # roundabout
            elif road_ids[i] in [3, 4]:
                zone_ids[i] = 2   # south + east roads
            else:
                zone_ids[i] = 3   # T-intersection + connector

        states = []
        for step in range(total_steps):
            states.append(np.stack([x.copy(), y.copy(), vx.copy(), vy.copy()], axis=-1))

            fx = np.zeros(n_vehicles, dtype=np.float32)
            fy = np.zeros(n_vehicles, dtype=np.float32)

            for i in range(n_vehicles):
                # Road-following force: attract to nearest road center
                seg_name, seg_pts, seg_lanes = ROAD_SEGMENTS[road_ids[i]]
                _, dist_to_road, nearest_pt = get_nearest_road_point(
                    np.array([x[i], y[i]]), seg_pts)
                nearest_idx, _, _ = get_nearest_road_point(
                    np.array([x[i], y[i]]), seg_pts)
                tang = road_tangent_at(seg_pts, nearest_idx)

                # Lane-keeping force
                normal = np.array([-tang[1], tang[0]])
                target_pt = nearest_pt + lane_offs[i] * normal
                dx_road = target_pt[0] - x[i]
                dy_road = target_pt[1] - y[i]
                fx[i] += 3.0 * dx_road
                fy[i] += 3.0 * dy_road

                # Speed regulation along tangent
                speed = np.sqrt(vx[i]**2 + vy[i]**2).clip(min=0.5)
                target_speed = 6.0 if seg_name == 'roundabout' else 12.0
                speed_err = target_speed - speed
                fx[i] += 0.8 * speed_err * tang[0]
                fy[i] += 0.8 * speed_err * tang[1]

                # Collision repulsion from nearby vehicles
                for j in range(n_vehicles):
                    if i == j:
                        continue
                    ddx = x[i] - x[j]
                    ddy = y[i] - y[j]
                    dist = max(np.sqrt(ddx**2 + ddy**2), 0.3)
                    if dist < 10.0:
                        force = 20.0 / (dist ** 2)
                        fx[i] += force * ddx / dist
                        fy[i] += force * ddy / dist

            # Random perturbation
            fx += np.random.randn(n_vehicles).astype(np.float32) * 0.4
            fy += np.random.randn(n_vehicles).astype(np.float32) * 0.4

            vx += fx * DT
            vy += fy * DT
            speed = np.sqrt(vx**2 + vy**2)
            max_speed = 22.0
            too_fast = speed > max_speed
            if too_fast.any():
                vx[too_fast] *= max_speed / speed[too_fast]
                vy[too_fast] *= max_speed / speed[too_fast]
            x += vx * DT
            y += vy * DT

        states = np.array(states)  # (total_steps, n_vehicles, 4)
        current = states[n_history - 1]  # last history frame
        fut_xy = states[n_history:, :, :2] - current[None, :, :2]

        # Road context per vehicle: (lane_offset, road_id_norm, signal_state, speed_limit_norm)
        road_ctx = np.zeros((n_vehicles, 4), dtype=np.float32)
        for i in range(n_vehicles):
            road_ctx[i, 0] = lane_offs[i] / LANE_WIDTH
            road_ctx[i, 1] = road_ids[i] / len(ROAD_SEGMENTS)
            # Signal state for nearest signal
            min_sig_dist = 999
            sig_state = 0.0
            for sig in TRAFFIC_SIGNALS:
                sd = np.sqrt(((sig['pos'] - np.array([x[i], y[i]]))**2).sum())
                if sd < min_sig_dist:
                    min_sig_dist = sd
                    step_in_cycle = (scene_i * 7 + step) % sig['period']
                    sig_state = 1.0 if step_in_cycle < sig['period'] // 2 else 0.0
            road_ctx[i, 2] = sig_state
            road_ctx[i, 3] = 1.0 if ROAD_SEGMENTS[road_ids[i]][0] == 'highway' else 0.6

        all_history.append(current)
        all_future.append(fut_xy.transpose(1, 0, 2))
        all_road_ctx.append(road_ctx)
        all_zone_ids.append(zone_ids)

    history = torch.tensor(np.array(all_history), dtype=torch.float32)
    future = torch.tensor(np.array(all_future), dtype=torch.float32)
    road_ctx = torch.tensor(np.array(all_road_ctx), dtype=torch.float32)
    zone_ids = np.array(all_zone_ids)

    return history, future, road_ctx, zone_ids


print("Generating social-force trajectories on road network...")
hist_tr, fut_tr, ctx_tr, zones_tr = social_force_trajectory(1024)
hist_val, fut_val, ctx_val, zones_val = social_force_trajectory(256)
print(f"  Train: {hist_tr.shape[0]} scenes, {N_VEHICLES} vehicles each")
print(f"  Future shape: {fut_tr.shape} (scenes, vehicles, steps, xy)")


# ── 4. Model ────────────────────────────────────────────────────────

class FleetModel(nn.Module):
    """Canvas-structured fleet trajectory predictor with zone hierarchy."""

    def __init__(self, bound, d=32, nhead=4):
        super().__init__()
        self.bound = bound
        self.d = d
        N = bound.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, N, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=128,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        mask = bound.topology.to_additive_mask(bound.layout)
        self.register_buffer('mask', mask)

        # Field sizes (from zone 0, vehicle 0)
        prefix = 'zones[0].vehicles[0]'
        self.fs = {}
        for fname in ['position', 'velocity', 'heading', 'road_context',
                       'intent', 'trajectory']:
            self.fs[fname] = len(bound.layout.region_indices(f'{prefix}.{fname}'))

        # Global flow field
        self.fs['global_flow'] = len(bound.layout.region_indices('global_flow'))
        self.fs['signal_state'] = len(bound.layout.region_indices('zones[0].signal_state'))
        self.fs['congestion'] = len(bound.layout.region_indices('zones[0].congestion'))

        # Input projections
        self.pos_proj = nn.Linear(2, self.fs['position'] * d)
        self.vel_proj = nn.Linear(2, self.fs['velocity'] * d)
        self.hdg_proj = nn.Linear(2, self.fs['heading'] * d)  # sin/cos
        self.ctx_proj = nn.Linear(4, self.fs['road_context'] * d)
        self.flow_proj = nn.Linear(2, self.fs['global_flow'] * d)
        self.sig_proj = nn.Linear(1, self.fs['signal_state'] * d)

        # Output heads
        self.traj_head = nn.Linear(self.fs['trajectory'] * d, PRED_HORIZON * 2)
        self.intent_head = nn.Linear(self.fs['intent'] * d, 8)
        self.cong_head = nn.Linear(self.fs['congestion'] * d, 1)

    def forward(self, states, road_ctx, zone_ids_batch=None):
        """states: (B, N_VEH, 4), road_ctx: (B, N_VEH, 4)."""
        B = states.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        # Place global flow
        gf_idx = self.bound.layout.region_indices('global_flow')
        # Compute mean flow from all vehicle velocities
        mean_vel = states[:, :, 2:4].mean(dim=1)  # (B, 2)
        gf_emb = self.flow_proj(mean_vel).reshape(B, self.fs['global_flow'], self.d)
        canvas[:, gf_idx] = canvas[:, gf_idx] + gf_emb

        # Place per-zone signals
        for z in range(N_ZONES):
            sig_idx = self.bound.layout.region_indices(f'zones[{z}].signal_state')
            sig_input = torch.zeros(B, 1)
            if z < len(TRAFFIC_SIGNALS):
                sig_input[:, 0] = 1.0
            sig_emb = self.sig_proj(sig_input).reshape(B, self.fs['signal_state'], self.d)
            canvas[:, sig_idx] = canvas[:, sig_idx] + sig_emb

        # Place per-vehicle data
        for z in range(N_ZONES):
            for v in range(VEHICLES_PER_ZONE):
                vi = z * VEHICLES_PER_ZONE + v  # global vehicle index
                prefix = f'zones[{z}].vehicles[{v}]'

                p_idx = self.bound.layout.region_indices(f'{prefix}.position')
                v_idx = self.bound.layout.region_indices(f'{prefix}.velocity')
                h_idx = self.bound.layout.region_indices(f'{prefix}.heading')
                c_idx = self.bound.layout.region_indices(f'{prefix}.road_context')

                # Position
                p_emb = self.pos_proj(states[:, vi, :2]).reshape(
                    B, self.fs['position'], self.d)
                canvas[:, p_idx] = canvas[:, p_idx] + p_emb

                # Velocity
                v_emb = self.vel_proj(states[:, vi, 2:4]).reshape(
                    B, self.fs['velocity'], self.d)
                canvas[:, v_idx] = canvas[:, v_idx] + v_emb

                # Heading (sin/cos encoding)
                speed = torch.sqrt(states[:, vi, 2]**2 + states[:, vi, 3]**2).clamp(min=0.1)
                hdg_vec = torch.stack([states[:, vi, 2] / speed,
                                       states[:, vi, 3] / speed], dim=-1)
                h_emb = self.hdg_proj(hdg_vec).reshape(B, self.fs['heading'], self.d)
                canvas[:, h_idx] = canvas[:, h_idx] + h_emb

                # Road context
                c_emb = self.ctx_proj(road_ctx[:, vi]).reshape(
                    B, self.fs['road_context'], self.d)
                canvas[:, c_idx] = canvas[:, c_idx] + c_emb

        canvas = self.encoder(canvas, mask=self.mask)

        # Read outputs
        trajs = []
        intents = []
        congestions = []
        for z in range(N_ZONES):
            for v in range(VEHICLES_PER_ZONE):
                prefix = f'zones[{z}].vehicles[{v}]'
                t_idx = self.bound.layout.region_indices(f'{prefix}.trajectory')
                i_idx = self.bound.layout.region_indices(f'{prefix}.intent')

                traj = self.traj_head(canvas[:, t_idx].reshape(B, -1))
                trajs.append(traj.reshape(B, PRED_HORIZON, 2))

                intent = self.intent_head(canvas[:, i_idx].reshape(B, -1))
                intents.append(intent)

            cg_idx = self.bound.layout.region_indices(f'zones[{z}].congestion')
            cg = self.cong_head(canvas[:, cg_idx].reshape(B, -1))
            congestions.append(cg)

        trajs = torch.stack(trajs, dim=1)       # (B, N_VEH, horizon, 2)
        intents = torch.stack(intents, dim=1)   # (B, N_VEH, 8)
        congestions = torch.stack(congestions, dim=1)  # (B, N_ZONES, 2)
        return trajs, intents, congestions


# ── 5. Training ──────────────────────────────────────────────────────

def contrastive_intent_loss(intents, future_trajs, temperature=0.1):
    """Vehicles with similar futures should have similar intents."""
    B, V, D = intents.shape
    fut_flat = future_trajs.reshape(B, V, -1)
    loss = torch.tensor(0.0)
    count = 0
    for b in range(min(B, 16)):
        intent_norm = F.normalize(intents[b], dim=-1)
        intent_sim = intent_norm @ intent_norm.T
        fut_norm = F.normalize(fut_flat[b], dim=-1)
        traj_sim = fut_norm @ fut_norm.T
        loss = loss + F.mse_loss(intent_sim, traj_sim.detach())
        count += 1
    return loss / max(count, 1)


def train_model(bound, label, n_epochs=250, bs=64):
    model = FleetModel(bound)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []

    for ep in range(n_epochs):
        idx = torch.randint(0, len(hist_tr), (bs,))
        pred_traj, pred_intent, pred_cong = model(hist_tr[idx], ctx_tr[idx])

        traj_loss = F.mse_loss(pred_traj, fut_tr[idx])
        c_loss = contrastive_intent_loss(pred_intent, fut_tr[idx])
        loss = traj_loss + 0.3 * c_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if ep % 100 == 0:
            print(f"  [{label}] ep {ep:3d}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        vp, vi, vc = model(hist_val, ctx_val)
        vl = F.mse_loss(vp, fut_val).item()
    print(f"  [{label}] val_traj_mse={vl:.4f}")
    return model, losses, vl


print("\nTraining isolated topology...")
iso_model, iso_losses, iso_vl = train_model(bound_isolated, "isolated")
print("Training ring topology...")
ring_model, ring_losses, ring_vl = train_model(bound_ring, "ring")
print("Training dense topology...")
dense_model, dense_losses, dense_vl = train_model(bound_dense, "dense")


# ── 6. Evaluation ───────────────────────────────────────────────────

def compute_metrics(model, hist, fut, ctx):
    """Compute ADE, FDE, and collision rate."""
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(hist, ctx)

    ade = torch.sqrt(((pred - fut) ** 2).sum(dim=-1)).mean().item()
    fde = torch.sqrt(((pred[:, :, -1] - fut[:, :, -1]) ** 2).sum(dim=-1)).mean().item()

    # Collision rate: predicted positions within 1.5m
    collisions = 0
    total_pairs = 0
    n_check = min(len(pred), 100)
    for b in range(n_check):
        for t in range(PRED_HORIZON):
            pos = pred[b, :, t]  # (N_VEH, 2)
            dists = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0))[0]
            n_v = pos.shape[0]
            mask = torch.triu(torch.ones(n_v, n_v, dtype=torch.bool), diagonal=1)
            close = (dists[mask] < 1.5).sum().item()
            collisions += close
            total_pairs += mask.sum().item()

    collision_rate = collisions / max(total_pairs, 1)
    return ade, fde, collision_rate


print("\nMetrics:")
metrics = {}
for name, model in [("isolated", iso_model), ("ring", ring_model), ("dense", dense_model)]:
    ade, fde, cr = compute_metrics(model, hist_val, fut_val, ctx_val)
    metrics[name] = (ade, fde, cr)
    print(f"  {name:10s}: ADE={ade:.3f}  FDE={fde:.3f}  CollRate={cr:.4f}")


# ── 7. Animation data: full simulation for GIF ──────────────────────

print("\nGenerating animation simulation...")
np.random.seed(7)
N_ANIM_STEPS = 80

anim_pos, anim_vel, anim_hdg, anim_roads, anim_lanes, anim_ridx = \
    assign_vehicles_to_roads(N_VEHICLES)

anim_x, anim_y = anim_pos[:, 0].copy(), anim_pos[:, 1].copy()
anim_vx, anim_vy = anim_vel[:, 0].copy(), anim_vel[:, 1].copy()

anim_traj_x = [anim_x.copy()]
anim_traj_y = [anim_y.copy()]
anim_speeds = [np.sqrt(anim_vx**2 + anim_vy**2).copy()]

for step in range(N_ANIM_STEPS):
    fx = np.zeros(N_VEHICLES, dtype=np.float32)
    fy = np.zeros(N_VEHICLES, dtype=np.float32)

    for i in range(N_VEHICLES):
        seg_name, seg_pts, seg_lanes = ROAD_SEGMENTS[anim_roads[i]]
        _, _, nearest_pt = get_nearest_road_point(
            np.array([anim_x[i], anim_y[i]]), seg_pts)
        nearest_idx, _, _ = get_nearest_road_point(
            np.array([anim_x[i], anim_y[i]]), seg_pts)
        tang = road_tangent_at(seg_pts, nearest_idx)
        normal = np.array([-tang[1], tang[0]])
        target_pt = nearest_pt + anim_lanes[i] * normal

        fx[i] += 3.0 * (target_pt[0] - anim_x[i])
        fy[i] += 3.0 * (target_pt[1] - anim_y[i])

        speed = np.sqrt(anim_vx[i]**2 + anim_vy[i]**2).clip(min=0.5)
        target_speed = 6.0 if seg_name == 'roundabout' else 12.0
        fx[i] += 0.8 * (target_speed - speed) * tang[0]
        fy[i] += 0.8 * (target_speed - speed) * tang[1]

        for j in range(N_VEHICLES):
            if i == j:
                continue
            ddx = anim_x[i] - anim_x[j]
            ddy = anim_y[i] - anim_y[j]
            dist = max(np.sqrt(ddx**2 + ddy**2), 0.3)
            if dist < 10.0:
                force = 20.0 / (dist**2)
                fx[i] += force * ddx / dist
                fy[i] += force * ddy / dist

    fx += np.random.randn(N_VEHICLES).astype(np.float32) * 0.3
    fy += np.random.randn(N_VEHICLES).astype(np.float32) * 0.3

    anim_vx += fx * DT
    anim_vy += fy * DT
    speed = np.sqrt(anim_vx**2 + anim_vy**2)
    too_fast = speed > 22.0
    if too_fast.any():
        anim_vx[too_fast] *= 22.0 / speed[too_fast]
        anim_vy[too_fast] *= 22.0 / speed[too_fast]
    anim_x += anim_vx * DT
    anim_y += anim_vy * DT

    anim_traj_x.append(anim_x.copy())
    anim_traj_y.append(anim_y.copy())
    anim_speeds.append(np.sqrt(anim_vx**2 + anim_vy**2).copy())

anim_traj_x = np.array(anim_traj_x)
anim_traj_y = np.array(anim_traj_y)
anim_speeds = np.array(anim_speeds)


# ── 8. Visualization: multi-panel figure ─────────────────────────────

print("Generating multi-panel figure...")

BG_DARK = '#0a0a1a'
BG_PANEL = '#0e0e24'
ACCENT_CYAN = '#00e5ff'
ACCENT_MAGENTA = '#ff00e5'
ACCENT_LIME = '#76ff03'
ACCENT_AMBER = '#ffab00'
ACCENT_RED = '#ff1744'
ACCENT_BLUE = '#2979ff'
TOPO_COLORS = {
    'isolated': '#ff6e40',
    'ring': '#448aff',
    'dense': '#69f0ae',
}

fig = plt.figure(figsize=(22, 17), dpi=150)
fig.patch.set_facecolor(BG_DARK)
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.30,
                      left=0.04, right=0.97, top=0.94, bottom=0.03)

fig.suptitle('AUTONOMOUS VEHICLE FLEET — 64-VEHICLE COOPERATIVE PREDICTION',
             fontsize=18, fontweight='bold', color='white',
             fontfamily='monospace', y=0.98)
fig.text(0.5, 0.955,
         'Canvas topology comparison on complex road network  |  '
         'Social-force dynamics  |  Contrastive intent learning',
         ha='center', fontsize=9, color='#666688', fontfamily='monospace')


def setup_dark_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG_PANEL)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', color='white',
                     fontfamily='monospace', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7, color='#888899', fontfamily='monospace')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7, color='#888899', fontfamily='monospace')
    ax.tick_params(colors='#555566', labelsize=6)
    for spine in ax.spines.values():
        spine.set_color('#1a1a3a')


# ── Panel (0,0)+(0,1): Full aerial bird's-eye ────────────────────────
ax = fig.add_subplot(gs[0, 0:2])
setup_dark_ax(ax, 'AERIAL VIEW — 64 VEHICLES ON ROAD NETWORK')

# Draw road segments
for name, pts, n_lanes in ROAD_SEGMENTS:
    total_w = n_lanes * LANE_WIDTH
    for lane_i in range(n_lanes):
        off = (lane_i - (n_lanes - 1) / 2.0) * LANE_WIDTH
        lane_pts = offset_curve(pts, off)
        ax.plot(lane_pts[:, 0], lane_pts[:, 1], '-', color='#1a2a3a', lw=3.0)
    # Center dashes
    if n_lanes > 1:
        for lane_i in range(1, n_lanes):
            off = (lane_i - (n_lanes - 1) / 2.0) * LANE_WIDTH - LANE_WIDTH / 2
            dash_pts = offset_curve(pts, off)
            ax.plot(dash_pts[:, 0], dash_pts[:, 1], '--', color='#2a3a4a',
                    lw=0.5, alpha=0.6)
    # Road edges
    edge_l = offset_curve(pts, -total_w / 2)
    edge_r = offset_curve(pts, total_w / 2)
    ax.plot(edge_l[:, 0], edge_l[:, 1], '-', color='#3a4a5a', lw=0.8)
    ax.plot(edge_r[:, 0], edge_r[:, 1], '-', color='#3a4a5a', lw=0.8)

# Traffic signals
for sig in TRAFFIC_SIGNALS:
    color = '#69f0ae' if sig['state'] == 'green' else ACCENT_RED
    ax.plot(sig['pos'][0], sig['pos'][1], 's', color=color, markersize=6,
            zorder=10, markeredgecolor='white', markeredgewidth=0.3)

# Plot vehicles (scene 0 from validation data), color by speed
scene_idx = 0
veh_speeds = torch.sqrt(hist_val[scene_idx, :, 2]**2 +
                         hist_val[scene_idx, :, 3]**2).numpy()
speed_norm = (veh_speeds - veh_speeds.min()) / (veh_speeds.max() - veh_speeds.min() + 1e-6)

speed_cmap = plt.cm.plasma
for v in range(N_VEHICLES):
    cx = hist_val[scene_idx, v, 0].item()
    cy = hist_val[scene_idx, v, 1].item()
    color = speed_cmap(speed_norm[v])
    ax.plot(cx, cy, 'o', color=color, markersize=4.5, zorder=5,
            markeredgecolor='white', markeredgewidth=0.2)

# Speed colorbar
sm = plt.cm.ScalarMappable(cmap=speed_cmap,
                           norm=plt.Normalize(vmin=veh_speeds.min(),
                                              vmax=veh_speeds.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01, aspect=20)
cbar.set_label('Speed (m/s)', fontsize=6, color='#888899', fontfamily='monospace')
cbar.ax.tick_params(colors='#555566', labelsize=5)

# Road labels
ax.text(-55, 4, 'HIGHWAY (3-LANE)', fontsize=5, color='#4a5a6a',
        fontfamily='monospace', fontstyle='italic')
ax.text(15, -23, 'ROUNDABOUT', fontsize=5, color='#4a5a6a',
        fontfamily='monospace', fontstyle='italic')
ax.text(22, -52, 'T-INTERSECTION', fontsize=5, color='#4a5a6a',
        fontfamily='monospace', fontstyle='italic')
ax.text(50, -18, 'EAST ROAD', fontsize=5, color='#4a5a6a',
        fontfamily='monospace', fontstyle='italic')

ax.set_xlim(-65, 70)
ax.set_ylim(-60, 20)
ax.set_aspect('equal')
ax.text(0.01, 0.97, f'N={N_VEHICLES}  |  t=0.0s', transform=ax.transAxes,
        fontsize=7, color=ACCENT_CYAN, fontfamily='monospace', va='top')


# ── Panel (0,2): Close-up intersection tracking ─────────────────────
ax = fig.add_subplot(gs[0, 2])
setup_dark_ax(ax, 'INTERSECTION TRACKING')

# Zoom into roundabout area
rbt_cx, rbt_cy = RBT_CENTER
zoom_r = 18

for name, pts, n_lanes in ROAD_SEGMENTS:
    total_w = n_lanes * LANE_WIDTH
    for lane_i in range(n_lanes):
        off = (lane_i - (n_lanes - 1) / 2.0) * LANE_WIDTH
        lane_pts = offset_curve(pts, off)
        ax.plot(lane_pts[:, 0], lane_pts[:, 1], '-', color='#1a2a3a', lw=4)
    edge_l = offset_curve(pts, -total_w / 2)
    edge_r = offset_curve(pts, total_w / 2)
    ax.plot(edge_l[:, 0], edge_l[:, 1], '-', color='#3a4a5a', lw=1)
    ax.plot(edge_r[:, 0], edge_r[:, 1], '-', color='#3a4a5a', lw=1)

# Vehicles near roundabout
ring_model.eval()
with torch.no_grad():
    pred_ring, intents_ring, _ = ring_model(
        hist_val[scene_idx:scene_idx+1], ctx_val[scene_idx:scene_idx+1])

near_rbt = []
for v in range(N_VEHICLES):
    cx = hist_val[scene_idx, v, 0].item()
    cy = hist_val[scene_idx, v, 1].item()
    if abs(cx - rbt_cx) < zoom_r and abs(cy - rbt_cy) < zoom_r:
        near_rbt.append(v)

cluster_colors = ['#ff4081', '#00e5ff', '#76ff03', '#ffab00', '#e040fb',
                  '#00bcd4', '#cddc39', '#ff6e40', '#7c4dff', '#64ffda']
for ci, v in enumerate(near_rbt[:10]):
    c = cluster_colors[ci % len(cluster_colors)]
    cx = hist_val[scene_idx, v, 0].item()
    cy = hist_val[scene_idx, v, 1].item()
    ax.plot(cx, cy, 'o', color=c, markersize=8, zorder=5,
            markeredgecolor='white', markeredgewidth=0.5)
    ax.text(cx + 0.8, cy + 0.8, f'V{v}', fontsize=5, color=c,
            fontfamily='monospace', fontweight='bold')

    # True future
    true_traj = fut_val[scene_idx, v].numpy()
    ax.plot(cx + true_traj[:, 0], cy + true_traj[:, 1], '--',
            color=c, lw=1, alpha=0.4)

    # Predicted future
    pred_traj = pred_ring[0, v].numpy()
    ax.plot(cx + pred_traj[:, 0], cy + pred_traj[:, 1], '-',
            color=c, lw=2, alpha=0.8)

ax.set_xlim(rbt_cx - zoom_r, rbt_cx + zoom_r)
ax.set_ylim(rbt_cy - zoom_r, rbt_cy + zoom_r)
ax.set_aspect('equal')
n_shown = min(len(near_rbt), 10)
ax.text(0.01, 0.97, f'{n_shown} vehicles tracked', transform=ax.transAxes,
        fontsize=6, color=ACCENT_CYAN, fontfamily='monospace', va='top')
ax.text(0.01, 0.90, 'solid=predicted  dash=true', transform=ax.transAxes,
        fontsize=5, color='#666688', fontfamily='monospace', va='top')


# ── Panel (0,3): Speed heatmap on road ───────────────────────────────
ax = fig.add_subplot(gs[0, 3])
setup_dark_ax(ax, 'SPEED HEATMAP OVERLAY')

# Draw road network faintly
for name, pts, n_lanes in ROAD_SEGMENTS:
    ax.plot(pts[:, 0], pts[:, 1], '-', color='#151530', lw=5)

# Scatter vehicles with speed-based glow
for v in range(N_VEHICLES):
    cx = hist_val[scene_idx, v, 0].item()
    cy = hist_val[scene_idx, v, 1].item()
    spd = veh_speeds[v]
    color = speed_cmap(speed_norm[v])
    # Glow effect
    for r, a in [(12, 0.05), (8, 0.1), (5, 0.2)]:
        ax.plot(cx, cy, 'o', color=color, markersize=r, alpha=a)
    ax.plot(cx, cy, 'o', color=color, markersize=3, alpha=0.9)

ax.set_xlim(-65, 70)
ax.set_ylim(-60, 20)
ax.set_aspect('equal')


# ── Panel (1,0): Vehicle density flow ────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
setup_dark_ax(ax, 'DENSITY FLOW MAP', 'X (m)', 'Y (m)')

# 2D histogram of vehicle positions across multiple scenes
all_x = hist_val[:50, :, 0].numpy().flatten()
all_y = hist_val[:50, :, 1].numpy().flatten()
h, xedges, yedges = np.histogram2d(all_x, all_y, bins=40)
h = np.log1p(h.T)
im = ax.imshow(h, origin='lower', cmap='inferno',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               aspect='auto', alpha=0.9)
# Road overlay
for name, pts, n_lanes in ROAD_SEGMENTS:
    ax.plot(pts[:, 0], pts[:, 1], '-', color='#ffffff', lw=0.3, alpha=0.3)

cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('log(density+1)', fontsize=6, color='#888899', fontfamily='monospace')
cbar.ax.tick_params(colors='#555566', labelsize=5)


# ── Panel (1,1): Intent embedding space (PCA) ───────────────────────
ax = fig.add_subplot(gs[1, 1])
setup_dark_ax(ax, 'INTENT EMBEDDING SPACE (PCA)', 'PC1', 'PC2')

ring_model.eval()
with torch.no_grad():
    _, all_intents, _ = ring_model(hist_val[:32], ctx_val[:32])

# PCA on intent vectors
intent_flat = all_intents.reshape(-1, 8).numpy()  # (32*64, 8)
intent_centered = intent_flat - intent_flat.mean(axis=0)
U, S, Vt = np.linalg.svd(intent_centered, full_matrices=False)
pc = intent_centered @ Vt[:2].T  # (N, 2)

# Color by heading
headings_flat = np.arctan2(
    hist_val[:32, :, 3].numpy().flatten(),
    hist_val[:32, :, 2].numpy().flatten()
)
headings_norm = (headings_flat + np.pi) / (2 * np.pi)

sc = ax.scatter(pc[:, 0], pc[:, 1], c=headings_norm, cmap='hsv',
                s=3, alpha=0.5, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Heading (rad)', fontsize=6, color='#888899', fontfamily='monospace')
cbar.ax.tick_params(colors='#555566', labelsize=5)
ax.text(0.01, 0.97, f'var explained: {(S[0]**2+S[1]**2)/((S**2).sum()):.1%}',
        transform=ax.transAxes, fontsize=6, color=ACCENT_CYAN,
        fontfamily='monospace', va='top')


# ── Panel (1,2): Topology comparison bar chart ───────────────────────
ax = fig.add_subplot(gs[1, 2])
setup_dark_ax(ax, 'TOPOLOGY COMPARISON', '', 'Error / Rate')

x_pos = np.arange(3)
width = 0.25
names = ['isolated', 'ring', 'dense']
ade_vals = [metrics[n][0] for n in names]
fde_vals = [metrics[n][1] for n in names]
cr_vals = [metrics[n][2] * 100 for n in names]  # scale for visibility
colors_t = [TOPO_COLORS[n] for n in names]

bars1 = ax.bar(x_pos - width, ade_vals, width, label='ADE (m)',
               color=colors_t, alpha=0.9, edgecolor='white', linewidth=0.3)
bars2 = ax.bar(x_pos, fde_vals, width, label='FDE (m)',
               color=colors_t, alpha=0.6, edgecolor='white', linewidth=0.3)
bars3 = ax.bar(x_pos + width, cr_vals, width, label='Coll% x100',
               color=colors_t, alpha=0.3, edgecolor=colors_t, linewidth=1.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=7, color='#aaaacc', fontfamily='monospace')
ax.legend(fontsize=6, facecolor=BG_PANEL, edgecolor='#333355',
          labelcolor='#aaaacc')
ax.grid(True, alpha=0.1, axis='y', color='#333355')

# Value annotations
for i, (a, f, c) in enumerate(zip(ade_vals, fde_vals, cr_vals)):
    ax.text(i - width, a + 0.02, f'{a:.2f}', ha='center', fontsize=5,
            color=colors_t[i], fontfamily='monospace', fontweight='bold')
    ax.text(i, f + 0.02, f'{f:.2f}', ha='center', fontsize=5,
            color=colors_t[i], fontfamily='monospace')


# ── Panel (1,3): Training loss curves ────────────────────────────────
ax = fig.add_subplot(gs[1, 3])
setup_dark_ax(ax, 'TRAINING LOSS (LOG SCALE)', 'Epoch', 'Loss')

w = 20
def smooth(a, w=w):
    return np.convolve(a, np.ones(w)/w, mode='valid')

for name, losses, color in [
    ('isolated', iso_losses, TOPO_COLORS['isolated']),
    ('ring', ring_losses, TOPO_COLORS['ring']),
    ('dense', dense_losses, TOPO_COLORS['dense']),
]:
    smoothed = smooth(losses)
    ax.plot(smoothed, color=color, lw=1.5, label=name, alpha=0.9)
    # Endpoint marker
    ax.plot(len(smoothed)-1, smoothed[-1], 'o', color=color, markersize=4)
    ax.text(len(smoothed)+2, smoothed[-1], f'{smoothed[-1]:.3f}',
            fontsize=5, color=color, fontfamily='monospace', va='center')

ax.legend(fontsize=6, facecolor=BG_PANEL, edgecolor='#333355',
          labelcolor='#aaaacc')
ax.set_yscale('log')
ax.grid(True, alpha=0.1, color='#333355')


# ── Panel (2,0): Canvas layout grid ─────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
setup_dark_ax(ax, f'CANVAS LAYOUT (ring, {bound_ring.layout.H}x{bound_ring.layout.W})')

H, W = bound_ring.layout.H, bound_ring.layout.W
grid = np.zeros((H, W, 3))  # black base

field_colors_map = {
    'global_flow': '#ff6e40',
    'signal_state': '#ffab00',
    'congestion': '#ff4081',
    'position': '#00e5ff',
    'velocity': '#2979ff',
    'heading': '#7c4dff',
    'road_context': '#546e7a',
    'intent': '#e040fb',
    'trajectory': '#69f0ae',
}

for field_name, bf in bound_ring.fields.items():
    key = field_name.split('.')[-1]
    color_hex = field_colors_map.get(key, '#333355')
    r = int(color_hex[1:3], 16) / 255.0
    g = int(color_hex[3:5], 16) / 255.0
    b = int(color_hex[5:7], 16) / 255.0
    t0, t1, h0, h1, w0, w1 = bf.spec.bounds
    grid[h0:h1, w0:w1] = [r * 0.7, g * 0.7, b * 0.7]

ax.imshow(grid, aspect='equal', interpolation='nearest')
legend_elements = [Patch(facecolor=c, label=n, edgecolor='#333355')
                   for n, c in field_colors_map.items()]
ax.legend(handles=legend_elements, fontsize=4, loc='lower right', ncol=2,
          facecolor=BG_PANEL, edgecolor='#333355', labelcolor='#aaaacc')
ax.text(0.01, 0.97, f'{bound_ring.layout.num_positions} positions',
        transform=ax.transAxes, fontsize=6, color=ACCENT_CYAN,
        fontfamily='monospace', va='top')


# ── Panel (2,1): Separation distance matrix ─────────────────────────
ax = fig.add_subplot(gs[2, 1])
setup_dark_ax(ax, 'VEHICLE SEPARATION MATRIX (scene 0)')

# Compute pairwise distances for scene 0
pos_scene = hist_val[scene_idx, :, :2].numpy()  # (64, 2)
n_show = min(32, N_VEHICLES)  # show subset for readability
dists = np.sqrt(((pos_scene[:n_show, None, :] - pos_scene[None, :n_show, :])**2).sum(axis=-1))
np.fill_diagonal(dists, np.nan)

im = ax.imshow(dists, cmap='magma_r', aspect='equal', interpolation='nearest')
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Distance (m)', fontsize=6, color='#888899', fontfamily='monospace')
cbar.ax.tick_params(colors='#555566', labelsize=5)
ax.set_xlabel('Vehicle ID', fontsize=6, color='#888899', fontfamily='monospace')
ax.set_ylabel('Vehicle ID', fontsize=6, color='#888899', fontfamily='monospace')

# Mark close pairs
close_mask = dists < 5.0
close_pairs = np.argwhere(close_mask & np.triu(np.ones_like(close_mask), k=1).astype(bool))
ax.text(0.01, 0.97, f'{len(close_pairs)} close pairs (<5m)',
        transform=ax.transAxes, fontsize=6, color=ACCENT_RED,
        fontfamily='monospace', va='top')


# ── Panel (2,2): Throughput over time at key segments ────────────────
ax = fig.add_subplot(gs[2, 2])
setup_dark_ax(ax, 'SEGMENT THROUGHPUT', 'Time step', 'Vehicle count')

# Count vehicles near key segments over animation steps
segments_to_track = {
    'Highway': (HWY_CENTER[30], HWY_CENTER[80]),
    'Roundabout': (RBT_CENTER - 10, RBT_CENTER + 10),
    'T-junction': (T_JUNCTION_PT - 8, T_JUNCTION_PT + 8),
}

seg_track_colors = [ACCENT_CYAN, ACCENT_MAGENTA, ACCENT_LIME]

for (seg_name, (lo, hi)), color in zip(segments_to_track.items(), seg_track_colors):
    counts = []
    for t in range(len(anim_traj_x)):
        in_box = ((anim_traj_x[t] >= lo[0]) & (anim_traj_x[t] <= hi[0]) &
                  (anim_traj_y[t] >= lo[1]) & (anim_traj_y[t] <= hi[1])).sum()
        counts.append(in_box)
    ax.plot(counts, color=color, lw=1.5, label=seg_name, alpha=0.8)

ax.legend(fontsize=6, facecolor=BG_PANEL, edgecolor='#333355',
          labelcolor='#aaaacc')
ax.grid(True, alpha=0.1, color='#333355')


# ── Panel (2,3): Zone congestion comparison ──────────────────────────
ax = fig.add_subplot(gs[2, 3])
setup_dark_ax(ax, 'ZONE VEHICLE DISTRIBUTION')

zone_names = ['Highway\n+Ramp', 'Round-\nabout', 'South\n+East', 'T-junc\n+Conn']
zone_counts = [0, 0, 0, 0]
for v in range(N_VEHICLES):
    # Use animation road assignments
    rid = anim_roads[v]
    if rid <= 1:
        zone_counts[0] += 1
    elif rid == 2:
        zone_counts[1] += 1
    elif rid in [3, 4]:
        zone_counts[2] += 1
    else:
        zone_counts[3] += 1

zone_colors = [ACCENT_CYAN, ACCENT_MAGENTA, ACCENT_LIME, ACCENT_AMBER]
bars = ax.bar(range(4), zone_counts, color=zone_colors, alpha=0.7,
              edgecolor='white', linewidth=0.5, width=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(zone_names, fontsize=6, color='#aaaacc', fontfamily='monospace')
ax.grid(True, alpha=0.1, axis='y', color='#333355')

for i, (cnt, c) in enumerate(zip(zone_counts, zone_colors)):
    ax.text(i, cnt + 0.3, str(cnt), ha='center', fontsize=8,
            color=c, fontfamily='monospace', fontweight='bold')


# ── Panel (3,0)+(3,1): Predicted vs true trajectories (dense model) ──
ax = fig.add_subplot(gs[3, 0:2])
setup_dark_ax(ax, 'TRAJECTORY PREDICTION: DENSE TOPOLOGY (scene 0)')

dense_model.eval()
with torch.no_grad():
    pred_dense, _, _ = dense_model(
        hist_val[scene_idx:scene_idx+1], ctx_val[scene_idx:scene_idx+1])

# Draw road network faintly
for name, pts, n_lanes in ROAD_SEGMENTS:
    ax.plot(pts[:, 0], pts[:, 1], '-', color='#151530', lw=4)

# Plot vehicles with true vs predicted trajectories
veh_cmap = plt.cm.twilight
for v in range(N_VEHICLES):
    cx = hist_val[scene_idx, v, 0].item()
    cy = hist_val[scene_idx, v, 1].item()
    c = veh_cmap(v / N_VEHICLES)

    # Current position
    ax.plot(cx, cy, 'o', color=c, markersize=3.5, zorder=5,
            markeredgecolor='white', markeredgewidth=0.2)

    # True future (dashed)
    true_t = fut_val[scene_idx, v].numpy()
    ax.plot(cx + true_t[:, 0], cy + true_t[:, 1], '--',
            color=c, lw=0.7, alpha=0.4)

    # Predicted future (solid)
    pred_t = pred_dense[0, v].numpy()
    ax.plot(cx + pred_t[:, 0], cy + pred_t[:, 1], '-',
            color=c, lw=1.2, alpha=0.8)

ax.set_xlim(-65, 70)
ax.set_ylim(-60, 20)
ax.set_aspect('equal')
ax.text(0.01, 0.97, 'solid=PREDICTED  dashed=GROUND TRUTH',
        transform=ax.transAxes, fontsize=6, color='#666688',
        fontfamily='monospace', va='top')
ax.text(0.01, 0.90,
        f'ADE={metrics["dense"][0]:.3f}m  FDE={metrics["dense"][1]:.3f}m',
        transform=ax.transAxes, fontsize=6, color=ACCENT_LIME,
        fontfamily='monospace', va='top')


# ── Panel (3,2): Intent similarity matrix ────────────────────────────
ax = fig.add_subplot(gs[3, 2])
setup_dark_ax(ax, 'INTENT SIMILARITY (ring, 16 vehicles)')

ring_model.eval()
with torch.no_grad():
    _, intents_r, _ = ring_model(hist_val[:16], ctx_val[:16])
    avg_intent = intents_r.mean(dim=0)  # (N_VEH, 8)

# Show 16x16 subset for readability
n_show_intent = 16
intent_sub = avg_intent[:n_show_intent]
intent_norm = F.normalize(intent_sub, dim=-1)
sim_matrix = (intent_norm @ intent_norm.T).numpy()

im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal',
               interpolation='nearest')
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Cosine sim', fontsize=6, color='#888899', fontfamily='monospace')
cbar.ax.tick_params(colors='#555566', labelsize=5)
ax.set_xticks(range(0, n_show_intent, 4))
ax.set_yticks(range(0, n_show_intent, 4))
ax.set_xticklabels([f'V{i}' for i in range(0, n_show_intent, 4)],
                    fontsize=5, color='#aaaacc', fontfamily='monospace')
ax.set_yticklabels([f'V{i}' for i in range(0, n_show_intent, 4)],
                    fontsize=5, color='#aaaacc', fontfamily='monospace')


# ── Panel (3,3): Connectivity statistics ─────────────────────────────
ax = fig.add_subplot(gs[3, 3])
setup_dark_ax(ax, 'SCHEMA STATISTICS')

# Summary table
stats_data = []
for name, b in [("isolated", bound_isolated), ("ring", bound_ring),
                ("dense", bound_dense)]:
    n_conn = len(b.topology.connections) if b.topology else 0
    n_fields = len(b.fields)
    n_pos = b.layout.num_positions
    ade_v, fde_v, cr_v = metrics[name]
    stats_data.append([name, n_pos, n_conn, n_fields, f'{ade_v:.3f}', f'{fde_v:.3f}'])

col_labels = ['Topology', 'Positions', 'Connections', 'Fields', 'ADE', 'FDE']
ax.axis('off')

table = ax.table(
    cellText=stats_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.6)

for key, cell in table.get_celld().items():
    cell.set_edgecolor('#333355')
    cell.set_text_props(fontfamily='monospace')
    if key[0] == 0:
        cell.set_facecolor('#1a1a3a')
        cell.set_text_props(color='white', fontweight='bold', fontfamily='monospace')
    else:
        cell.set_facecolor(BG_PANEL)
        row_name = stats_data[key[0]-1][0]
        cell.set_text_props(color=TOPO_COLORS.get(row_name, '#aaaacc'),
                            fontfamily='monospace')

# Additional text info
ax.text(0.5, 0.08, f'Grid: {bound_ring.layout.H}x{bound_ring.layout.W} | '
        f'd_model=32 | T=1 | N_VEH={N_VEHICLES}',
        transform=ax.transAxes, ha='center', fontsize=6,
        color='#555577', fontfamily='monospace')
ax.text(0.5, 0.02, f'Zone hierarchy: RoadNetwork > TrafficZone[{N_ZONES}] > '
        f'Vehicle[{VEHICLES_PER_ZONE}]',
        transform=ax.transAxes, ha='center', fontsize=5,
        color='#444466', fontfamily='monospace')

# Save figure
path = os.path.join(ASSETS, "04_fleet.png")
fig.savefig(path, bbox_inches='tight', facecolor=BG_DARK, dpi=150)
plt.close()
print(f"\nSaved {path}")


# ── 9. Animation: aerial view GIF ───────────────────────────────────

print("Generating fleet animation...")

fig_anim, ax_anim = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
fig_anim.patch.set_facecolor(BG_DARK)

speed_cmap_anim = plt.cm.plasma


def draw_roads(ax):
    """Draw the road network on a dark background."""
    for name, pts, n_lanes in ROAD_SEGMENTS:
        total_w = n_lanes * LANE_WIDTH
        # Road surface
        for lane_i in range(n_lanes):
            off = (lane_i - (n_lanes - 1) / 2.0) * LANE_WIDTH
            lane_pts = offset_curve(pts, off)
            ax.plot(lane_pts[:, 0], lane_pts[:, 1], '-',
                    color='#12122a', lw=max(3, n_lanes * 1.5))
        # Lane dashes
        if n_lanes > 1:
            for lane_i in range(1, n_lanes):
                off = (lane_i - (n_lanes - 1) / 2.0) * LANE_WIDTH - LANE_WIDTH / 2
                dash_pts = offset_curve(pts, off)
                ax.plot(dash_pts[:, 0], dash_pts[:, 1], '--',
                        color='#2a2a4a', lw=0.4, alpha=0.5)
        # Edges
        edge_l = offset_curve(pts, -total_w / 2)
        edge_r = offset_curve(pts, total_w / 2)
        ax.plot(edge_l[:, 0], edge_l[:, 1], '-', color='#2a3a4a', lw=0.5)
        ax.plot(edge_r[:, 0], edge_r[:, 1], '-', color='#2a3a4a', lw=0.5)


def animate_fleet(frame):
    ax_anim.clear()
    ax_anim.set_facecolor(BG_DARK)

    draw_roads(ax_anim)

    # Traffic signals (cycle with time)
    for sig in TRAFFIC_SIGNALS:
        step_in_cycle = frame % sig['period']
        is_green = step_in_cycle < sig['period'] // 2
        color = '#69f0ae' if is_green else ACCENT_RED
        ax_anim.plot(sig['pos'][0], sig['pos'][1], 's', color=color,
                     markersize=5, zorder=10, markeredgecolor='white',
                     markeredgewidth=0.3, alpha=0.8)

    # Vehicles with glowing trails
    spds = anim_speeds[frame]
    spd_norm_frame = (spds - 2) / 20.0
    spd_norm_frame = np.clip(spd_norm_frame, 0, 1)

    for v in range(N_VEHICLES):
        x = anim_traj_x[frame, v]
        y = anim_traj_y[frame, v]
        color = speed_cmap_anim(spd_norm_frame[v])

        # Glowing trail (last 10 frames)
        trail_start = max(0, frame - 10)
        if frame > trail_start:
            trail_x = anim_traj_x[trail_start:frame+1, v]
            trail_y = anim_traj_y[trail_start:frame+1, v]
            n_trail = len(trail_x)
            for ti in range(n_trail - 1):
                alpha = 0.03 + 0.25 * (ti / n_trail)
                ax_anim.plot(trail_x[ti:ti+2], trail_y[ti:ti+2], '-',
                             color=color, lw=1.5, alpha=alpha)

        # Vehicle dot with glow
        ax_anim.plot(x, y, 'o', color=color, markersize=6, alpha=0.15)
        ax_anim.plot(x, y, 'o', color=color, markersize=4, alpha=0.3)
        ax_anim.plot(x, y, 'o', color=color, markersize=2.5,
                     zorder=5, markeredgecolor='white', markeredgewidth=0.15)

    ax_anim.set_xlim(-65, 70)
    ax_anim.set_ylim(-60, 20)
    ax_anim.set_aspect('equal')

    # HUD overlay
    t_sec = frame * DT
    ax_anim.text(0.02, 0.97,
                 f'T={t_sec:.1f}s  |  N={N_VEHICLES}  |  '
                 f'avg spd={spds.mean():.1f} m/s',
                 transform=ax_anim.transAxes, fontsize=9,
                 color=ACCENT_CYAN, fontfamily='monospace', va='top',
                 fontweight='bold')
    ax_anim.text(0.98, 0.97, 'FLEET SIMULATION',
                 transform=ax_anim.transAxes, fontsize=8,
                 color='#333355', fontfamily='monospace', va='top',
                 ha='right')
    ax_anim.text(0.02, 0.03, 'canvas-engineering | social-force dynamics',
                 transform=ax_anim.transAxes, fontsize=6,
                 color='#222244', fontfamily='monospace')

    ax_anim.tick_params(colors='#222233', labelsize=5)
    for spine in ax_anim.spines.values():
        spine.set_color('#1a1a2a')


anim = animation.FuncAnimation(fig_anim, animate_fleet,
                                frames=range(0, N_ANIM_STEPS + 1, 2),
                                interval=120)
gif_path = os.path.join(ASSETS, "04_fleet.gif")
anim.save(gif_path, writer='pillow', fps=10)
plt.close()
print(f"Saved {gif_path}")
print("\nDone.")

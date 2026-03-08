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

import os, math, time
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
HWY_CENTER = bezier_cubic(
    np.array([-60, 0]), np.array([-20, 12]),
    np.array([20, -8]), np.array([65, 5]), n=120)

RAMP_START = HWY_CENTER[45]
RAMP = bezier_cubic(
    RAMP_START, RAMP_START + np.array([3, -8]),
    np.array([10, -22]), np.array([18, -28]), n=60)

RBT_CENTER = np.array([22.0, -32.0])
RBT_RADIUS = 7.0
ROUNDABOUT = arc_segment(RBT_CENTER[0], RBT_CENTER[1], RBT_RADIUS,
                         0, 2*np.pi, n=100)

SOUTH_ROAD = line_segment(
    np.array([22, -55]), RBT_CENTER + np.array([0, -RBT_RADIUS]), n=50)

EAST_ROAD = bezier_cubic(
    RBT_CENTER + np.array([RBT_RADIUS, 0]),
    np.array([38, -30]), np.array([48, -25]),
    np.array([60, -20]), n=60)

T_JUNCTION_PT = np.array([22, -48])
T_ROAD_WEST = line_segment(T_JUNCTION_PT, T_JUNCTION_PT + np.array([-25, 0]), n=40)
T_ROAD_EAST = line_segment(T_JUNCTION_PT, T_JUNCTION_PT + np.array([25, 0]), n=40)

CURVE_CONNECTOR = bezier_cubic(
    np.array([60, -20]), np.array([65, -10]),
    np.array([62, 8]), np.array([65, 5]), n=50)

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

TRAFFIC_SIGNALS = [
    {'pos': RAMP_START.copy(), 'state': 'green', 'period': 40},
    {'pos': (RBT_CENTER + np.array([0, RBT_RADIUS])).copy(), 'state': 'red', 'period': 30},
    {'pos': T_JUNCTION_PT.copy(), 'state': 'green', 'period': 35},
    {'pos': (RBT_CENTER + np.array([RBT_RADIUS, 0])).copy(), 'state': 'red', 'period': 25},
]

# Pre-compute road tangents for fast lookup
ROAD_TANGENTS = []
for name, pts, n_lanes in ROAD_SEGMENTS:
    tangents = np.diff(pts, axis=0)
    tangents = np.vstack([tangents, tangents[-1:]])
    norms = np.sqrt((tangents**2).sum(axis=1, keepdims=True)).clip(min=1e-6)
    ROAD_TANGENTS.append(tangents / norms)


# ── 2. Type declarations ─────────────────────────────────────────────

@dataclass
class Vehicle:
    position: Field = Field(1, 1)                          # x, y packed
    velocity: Field = Field(1, 1)                          # vx, vy packed
    heading: Field = Field(1, 1)                           # heading angle
    road_context: Field = Field(1, 2, is_output=False)     # lane info, signal
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


CANVAS_H, CANVAS_W = 25, 24


def make_schema(connectivity_policy):
    zones = [TrafficZone(vehicles=[Vehicle() for _ in range(VEHICLES_PER_ZONE)])
             for _ in range(N_ZONES)]
    network = RoadNetwork(zones=zones)
    return compile_schema(
        network, T=1, H=CANVAS_H, W=CANVAS_W, d_model=32,
        connectivity=connectivity_policy,
    )


print("Compiling schemas...")
t0 = time.time()
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
print(f"  Schema compilation: {time.time()-t0:.1f}s")


# ── 3. Synthetic data: vectorized social-force on road network ──────

def assign_vehicles_to_roads(n_vehicles):
    """Place vehicles on random road segments with lane offsets (vectorized)."""
    positions = np.zeros((n_vehicles, 2), dtype=np.float32)
    velocities = np.zeros((n_vehicles, 2), dtype=np.float32)
    road_ids = np.zeros(n_vehicles, dtype=np.int32)
    lane_offsets = np.zeros(n_vehicles, dtype=np.float32)

    # Weight roads by length * lanes
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
        idx = np.random.randint(5, len(pts) - 5)
        lane = np.random.randint(0, n_lanes)
        lane_off = (lane - (n_lanes - 1) / 2.0) * LANE_WIDTH

        tang = ROAD_TANGENTS[seg_id][idx]
        normal = np.array([-tang[1], tang[0]])

        positions[i] = pts[idx] + lane_off * normal
        base_speed = 6.0 if name == 'roundabout' else np.random.uniform(8, 18)
        velocities[i] = tang * base_speed
        road_ids[i] = seg_id
        lane_offsets[i] = lane_off

    return positions, velocities, road_ids, lane_offsets


def simulate_social_force(x, y, vx, vy, road_ids, lane_offs, n_steps,
                           noise_scale=0.4):
    """Vectorized social-force simulation. Returns trajectory arrays."""
    n = len(x)
    traj_x = np.zeros((n_steps + 1, n), dtype=np.float32)
    traj_y = np.zeros((n_steps + 1, n), dtype=np.float32)
    traj_vx = np.zeros((n_steps + 1, n), dtype=np.float32)
    traj_vy = np.zeros((n_steps + 1, n), dtype=np.float32)

    traj_x[0] = x.copy()
    traj_y[0] = y.copy()
    traj_vx[0] = vx.copy()
    traj_vy[0] = vy.copy()

    for step in range(n_steps):
        fx = np.zeros(n, dtype=np.float32)
        fy = np.zeros(n, dtype=np.float32)

        # Road-following forces (vectorized per segment)
        for seg_id in range(len(ROAD_SEGMENTS)):
            mask = road_ids == seg_id
            if not mask.any():
                continue
            seg_name, seg_pts, seg_lanes = ROAD_SEGMENTS[seg_id]
            tangs = ROAD_TANGENTS[seg_id]

            # Find nearest road point for each vehicle on this segment
            vx_m, vy_m = x[mask], y[mask]
            # Broadcast: (n_veh_on_seg, 1, 2) - (1, n_pts, 2)
            pos_v = np.stack([vx_m, vy_m], axis=-1)[:, None, :]
            diffs = pos_v - seg_pts[None, :, :]
            dists_sq = (diffs**2).sum(axis=-1)  # (n_veh, n_pts)
            nearest_idx = dists_sq.argmin(axis=1)

            nearest_pts = seg_pts[nearest_idx]
            nearest_tangs = tangs[nearest_idx]
            normals = np.stack([-nearest_tangs[:, 1], nearest_tangs[:, 0]], axis=-1)

            target_pts = nearest_pts + lane_offs[mask, None] * normals
            dx_road = target_pts[:, 0] - vx_m
            dy_road = target_pts[:, 1] - vy_m
            fx[mask] += 3.0 * dx_road
            fy[mask] += 3.0 * dy_road

            # Speed regulation
            speed = np.sqrt(vx[mask]**2 + vy[mask]**2).clip(min=0.5)
            target_speed = 6.0 if seg_name == 'roundabout' else 12.0
            speed_err = target_speed - speed
            fx[mask] += 0.8 * speed_err * nearest_tangs[:, 0]
            fy[mask] += 0.8 * speed_err * nearest_tangs[:, 1]

        # Collision repulsion (fully vectorized)
        dx_mat = x[:, None] - x[None, :]  # (n, n)
        dy_mat = y[:, None] - y[None, :]
        dist_mat = np.sqrt(dx_mat**2 + dy_mat**2).clip(min=0.3)
        interact = dist_mat < 10.0
        np.fill_diagonal(interact, False)
        force_mag = np.where(interact, 20.0 / (dist_mat**2), 0.0)
        fx += (force_mag * dx_mat / dist_mat).sum(axis=1)
        fy += (force_mag * dy_mat / dist_mat).sum(axis=1)

        # Noise
        fx += np.random.randn(n).astype(np.float32) * noise_scale
        fy += np.random.randn(n).astype(np.float32) * noise_scale

        vx = vx + fx * DT
        vy = vy + fy * DT
        speed = np.sqrt(vx**2 + vy**2)
        too_fast = speed > 22.0
        if too_fast.any():
            vx[too_fast] *= 22.0 / speed[too_fast]
            vy[too_fast] *= 22.0 / speed[too_fast]
        x = x + vx * DT
        y = y + vy * DT

        traj_x[step + 1] = x.copy()
        traj_y[step + 1] = y.copy()
        traj_vx[step + 1] = vx.copy()
        traj_vy[step + 1] = vy.copy()

    return traj_x, traj_y, traj_vx, traj_vy


def generate_dataset(n_scenes=512, n_vehicles=N_VEHICLES,
                     n_history=4, n_future=PRED_HORIZON):
    """Generate trajectory dataset."""
    total_steps = n_history + n_future
    all_history = []
    all_future = []
    all_road_ctx = []

    for scene_i in range(n_scenes):
        pos, vel, road_ids, lane_offs = assign_vehicles_to_roads(n_vehicles)
        x, y = pos[:, 0], pos[:, 1]
        vx, vy = vel[:, 0], vel[:, 1]

        tx, ty, tvx, tvy = simulate_social_force(
            x, y, vx, vy, road_ids, lane_offs, total_steps, noise_scale=0.4)

        # Current state = last history frame
        hi = n_history - 1
        current = np.stack([tx[hi], ty[hi], tvx[hi], tvy[hi]], axis=-1)  # (n_veh, 4)

        # Future trajectory (relative)
        fut_x = tx[n_history:n_history+n_future] - tx[hi:hi+1]
        fut_y = ty[n_history:n_history+n_future] - ty[hi:hi+1]
        fut = np.stack([fut_x, fut_y], axis=-1).transpose(1, 0, 2)  # (n_veh, n_future, 2)

        # Road context
        road_ctx = np.zeros((n_vehicles, 4), dtype=np.float32)
        road_ctx[:, 0] = lane_offs / LANE_WIDTH
        road_ctx[:, 1] = road_ids / len(ROAD_SEGMENTS)
        for i in range(n_vehicles):
            min_d = 999.0
            sig_val = 0.0
            for sig in TRAFFIC_SIGNALS:
                d = np.sqrt(((sig['pos'] - np.array([tx[hi, i], ty[hi, i]]))**2).sum())
                if d < min_d:
                    min_d = d
                    step_in_cycle = (scene_i * 7) % sig['period']
                    sig_val = 1.0 if step_in_cycle < sig['period'] // 2 else 0.0
            road_ctx[i, 2] = sig_val
            road_ctx[i, 3] = 1.0 if ROAD_SEGMENTS[road_ids[i]][0] == 'highway' else 0.6

        all_history.append(current)
        all_future.append(fut)
        all_road_ctx.append(road_ctx)

    history = torch.tensor(np.array(all_history), dtype=torch.float32)
    future = torch.tensor(np.array(all_future), dtype=torch.float32)
    road_ctx = torch.tensor(np.array(all_road_ctx), dtype=torch.float32)
    return history, future, road_ctx


print("Generating social-force trajectories on road network...")
t0 = time.time()
hist_tr, fut_tr, ctx_tr = generate_dataset(512)
hist_val, fut_val, ctx_val = generate_dataset(128)
print(f"  Train: {hist_tr.shape[0]} scenes, {N_VEHICLES} vehicles each")
print(f"  Future shape: {fut_tr.shape}")
print(f"  Data generation: {time.time()-t0:.1f}s")


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

        self.fs['global_flow'] = len(bound.layout.region_indices('global_flow'))
        self.fs['signal_state'] = len(bound.layout.region_indices('zones[0].signal_state'))
        self.fs['congestion'] = len(bound.layout.region_indices('zones[0].congestion'))

        # Input projections
        self.pos_proj = nn.Linear(2, self.fs['position'] * d)
        self.vel_proj = nn.Linear(2, self.fs['velocity'] * d)
        self.hdg_proj = nn.Linear(2, self.fs['heading'] * d)
        self.ctx_proj = nn.Linear(4, self.fs['road_context'] * d)
        self.flow_proj = nn.Linear(2, self.fs['global_flow'] * d)
        self.sig_proj = nn.Linear(1, self.fs['signal_state'] * d)

        # Output heads
        self.traj_head = nn.Linear(self.fs['trajectory'] * d, PRED_HORIZON * 2)
        self.intent_head = nn.Linear(self.fs['intent'] * d, 8)
        self.cong_head = nn.Linear(self.fs['congestion'] * d, 1)

        # Cache region indices for speed
        self._cache_indices()

    def _cache_indices(self):
        """Pre-compute all region indices to avoid repeated lookups."""
        self._gf_idx = self.bound.layout.region_indices('global_flow')
        self._sig_idx = [self.bound.layout.region_indices(f'zones[{z}].signal_state')
                         for z in range(N_ZONES)]
        self._cg_idx = [self.bound.layout.region_indices(f'zones[{z}].congestion')
                        for z in range(N_ZONES)]
        self._veh_idx = {}
        for z in range(N_ZONES):
            for v in range(VEHICLES_PER_ZONE):
                prefix = f'zones[{z}].vehicles[{v}]'
                self._veh_idx[(z, v)] = {
                    'p': self.bound.layout.region_indices(f'{prefix}.position'),
                    'v': self.bound.layout.region_indices(f'{prefix}.velocity'),
                    'h': self.bound.layout.region_indices(f'{prefix}.heading'),
                    'c': self.bound.layout.region_indices(f'{prefix}.road_context'),
                    't': self.bound.layout.region_indices(f'{prefix}.trajectory'),
                    'i': self.bound.layout.region_indices(f'{prefix}.intent'),
                }

    def forward(self, states, road_ctx):
        """states: (B, N_VEH, 4), road_ctx: (B, N_VEH, 4)."""
        B = states.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        # Global flow
        mean_vel = states[:, :, 2:4].mean(dim=1)
        gf_emb = self.flow_proj(mean_vel).reshape(B, self.fs['global_flow'], self.d)
        canvas[:, self._gf_idx] = canvas[:, self._gf_idx] + gf_emb

        # Per-zone signals
        for z in range(N_ZONES):
            sig_input = torch.ones(B, 1) if z < len(TRAFFIC_SIGNALS) else torch.zeros(B, 1)
            sig_emb = self.sig_proj(sig_input).reshape(B, self.fs['signal_state'], self.d)
            canvas[:, self._sig_idx[z]] = canvas[:, self._sig_idx[z]] + sig_emb

        # Per-vehicle
        for z in range(N_ZONES):
            for v in range(VEHICLES_PER_ZONE):
                vi = z * VEHICLES_PER_ZONE + v
                idx = self._veh_idx[(z, v)]

                p_emb = self.pos_proj(states[:, vi, :2]).reshape(
                    B, self.fs['position'], self.d)
                canvas[:, idx['p']] = canvas[:, idx['p']] + p_emb

                v_emb = self.vel_proj(states[:, vi, 2:4]).reshape(
                    B, self.fs['velocity'], self.d)
                canvas[:, idx['v']] = canvas[:, idx['v']] + v_emb

                speed = torch.sqrt(states[:, vi, 2]**2 + states[:, vi, 3]**2).clamp(min=0.1)
                hdg_vec = torch.stack([states[:, vi, 2] / speed,
                                       states[:, vi, 3] / speed], dim=-1)
                h_emb = self.hdg_proj(hdg_vec).reshape(B, self.fs['heading'], self.d)
                canvas[:, idx['h']] = canvas[:, idx['h']] + h_emb

                c_emb = self.ctx_proj(road_ctx[:, vi]).reshape(
                    B, self.fs['road_context'], self.d)
                canvas[:, idx['c']] = canvas[:, idx['c']] + c_emb

        canvas = self.encoder(canvas, mask=self.mask)

        # Read outputs
        trajs = []
        intents = []
        for z in range(N_ZONES):
            for v in range(VEHICLES_PER_ZONE):
                idx = self._veh_idx[(z, v)]
                traj = self.traj_head(canvas[:, idx['t']].reshape(B, -1))
                trajs.append(traj.reshape(B, PRED_HORIZON, 2))
                intent = self.intent_head(canvas[:, idx['i']].reshape(B, -1))
                intents.append(intent)

        trajs = torch.stack(trajs, dim=1)       # (B, N_VEH, horizon, 2)
        intents = torch.stack(intents, dim=1)   # (B, N_VEH, 8)
        return trajs, intents


# ── 5. Training ──────────────────────────────────────────────────────

def contrastive_intent_loss(intents, future_trajs):
    """Vehicles with similar futures should have similar intents."""
    B, V, D = intents.shape
    fut_flat = future_trajs.reshape(B, V, -1)
    loss = torch.tensor(0.0)
    count = 0
    for b in range(min(B, 8)):
        intent_norm = F.normalize(intents[b], dim=-1)
        intent_sim = intent_norm @ intent_norm.T
        fut_norm = F.normalize(fut_flat[b], dim=-1)
        traj_sim = fut_norm @ fut_norm.T
        loss = loss + F.mse_loss(intent_sim, traj_sim.detach())
        count += 1
    return loss / max(count, 1)


def train_model(bound, label, n_epochs=200, bs=32):
    model = FleetModel(bound)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []
    t0 = time.time()

    for ep in range(n_epochs):
        idx = torch.randint(0, len(hist_tr), (bs,))
        pred_traj, pred_intent = model(hist_tr[idx], ctx_tr[idx])

        traj_loss = F.mse_loss(pred_traj, fut_tr[idx])
        c_loss = contrastive_intent_loss(pred_intent, fut_tr[idx])
        loss = traj_loss + 0.3 * c_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if ep % 100 == 0:
            print(f"  [{label}] ep {ep:3d}: loss={loss.item():.4f}  "
                  f"({time.time()-t0:.0f}s)")

    model.eval()
    with torch.no_grad():
        vp, vi = model(hist_val, ctx_val)
        vl = F.mse_loss(vp, fut_val).item()
    print(f"  [{label}] val_traj_mse={vl:.4f}  total={time.time()-t0:.0f}s")
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
        pred, _ = model(hist, ctx)

    ade = torch.sqrt(((pred - fut) ** 2).sum(dim=-1)).mean().item()
    fde = torch.sqrt(((pred[:, :, -1] - fut[:, :, -1]) ** 2).sum(dim=-1)).mean().item()

    # Collision rate (vectorized)
    n_check = min(len(pred), 50)
    collisions = 0
    total_pairs = 0
    for b in range(n_check):
        for t in range(PRED_HORIZON):
            pos = pred[b, :, t]
            dists = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0))[0]
            mask = torch.triu(torch.ones(N_VEHICLES, N_VEHICLES, dtype=torch.bool), diagonal=1)
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


# ── 7. Animation data ──────────────────────────────────────────────

print("\nGenerating animation simulation...")
np.random.seed(7)
N_ANIM_STEPS = 80

anim_pos, anim_vel, anim_roads, anim_lanes = assign_vehicles_to_roads(N_VEHICLES)
anim_tx, anim_ty, anim_tvx, anim_tvy = simulate_social_force(
    anim_pos[:, 0], anim_pos[:, 1],
    anim_vel[:, 0], anim_vel[:, 1],
    anim_roads, anim_lanes, N_ANIM_STEPS, noise_scale=0.3)
anim_speeds = np.sqrt(anim_tvx**2 + anim_tvy**2)


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

fig.suptitle('AUTONOMOUS VEHICLE FLEET  —  64-VEHICLE COOPERATIVE PREDICTION',
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


def draw_roads_on_ax(ax, lw_scale=1.0, alpha=1.0):
    """Draw road network on an axes."""
    for name, pts, n_lanes in ROAD_SEGMENTS:
        total_w = n_lanes * LANE_WIDTH
        for lane_i in range(n_lanes):
            off = (lane_i - (n_lanes - 1) / 2.0) * LANE_WIDTH
            lane_pts = offset_curve(pts, off)
            ax.plot(lane_pts[:, 0], lane_pts[:, 1], '-',
                    color='#1a2a3a', lw=3.0 * lw_scale, alpha=alpha)
        if n_lanes > 1:
            for lane_i in range(1, n_lanes):
                off = (lane_i - (n_lanes - 1) / 2.0) * LANE_WIDTH - LANE_WIDTH / 2
                dash_pts = offset_curve(pts, off)
                ax.plot(dash_pts[:, 0], dash_pts[:, 1], '--',
                        color='#2a3a4a', lw=0.5 * lw_scale, alpha=0.6 * alpha)
        edge_l = offset_curve(pts, -total_w / 2)
        edge_r = offset_curve(pts, total_w / 2)
        ax.plot(edge_l[:, 0], edge_l[:, 1], '-',
                color='#3a4a5a', lw=0.8 * lw_scale, alpha=alpha)
        ax.plot(edge_r[:, 0], edge_r[:, 1], '-',
                color='#3a4a5a', lw=0.8 * lw_scale, alpha=alpha)


# ── Panel (0,0)+(0,1): Full aerial view ──────────────────────────────
ax = fig.add_subplot(gs[0, 0:2])
setup_dark_ax(ax, 'AERIAL VIEW  —  64 VEHICLES ON ROAD NETWORK')
draw_roads_on_ax(ax)

# Traffic signals
for sig in TRAFFIC_SIGNALS:
    color = '#69f0ae' if sig['state'] == 'green' else ACCENT_RED
    ax.plot(sig['pos'][0], sig['pos'][1], 's', color=color, markersize=6,
            zorder=10, markeredgecolor='white', markeredgewidth=0.3)

# Vehicles (scene 0), color by speed
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

sm = plt.cm.ScalarMappable(cmap=speed_cmap,
                           norm=plt.Normalize(vmin=veh_speeds.min(),
                                              vmax=veh_speeds.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01, aspect=20)
cbar.set_label('Speed (m/s)', fontsize=6, color='#888899', fontfamily='monospace')
cbar.ax.tick_params(colors='#555566', labelsize=5)

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


# ── Panel (0,2): Close-up intersection ───────────────────────────────
ax = fig.add_subplot(gs[0, 2])
setup_dark_ax(ax, 'INTERSECTION TRACKING')

rbt_cx, rbt_cy = RBT_CENTER
zoom_r = 18
draw_roads_on_ax(ax, lw_scale=1.3)

ring_model.eval()
with torch.no_grad():
    pred_ring, intents_ring = ring_model(
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
    true_traj = fut_val[scene_idx, v].numpy()
    ax.plot(cx + true_traj[:, 0], cy + true_traj[:, 1], '--',
            color=c, lw=1, alpha=0.4)
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


# ── Panel (0,3): Speed heatmap overlay ───────────────────────────────
ax = fig.add_subplot(gs[0, 3])
setup_dark_ax(ax, 'SPEED HEATMAP OVERLAY')

for name, pts, n_lanes in ROAD_SEGMENTS:
    ax.plot(pts[:, 0], pts[:, 1], '-', color='#151530', lw=5)

for v in range(N_VEHICLES):
    cx = hist_val[scene_idx, v, 0].item()
    cy = hist_val[scene_idx, v, 1].item()
    color = speed_cmap(speed_norm[v])
    for r, a in [(12, 0.05), (8, 0.1), (5, 0.2)]:
        ax.plot(cx, cy, 'o', color=color, markersize=r, alpha=a)
    ax.plot(cx, cy, 'o', color=color, markersize=3, alpha=0.9)

ax.set_xlim(-65, 70)
ax.set_ylim(-60, 20)
ax.set_aspect('equal')


# ── Panel (1,0): Density flow map ────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
setup_dark_ax(ax, 'DENSITY FLOW MAP', 'X (m)', 'Y (m)')

all_x = hist_val[:50, :, 0].numpy().flatten()
all_y = hist_val[:50, :, 1].numpy().flatten()
h, xedges, yedges = np.histogram2d(all_x, all_y, bins=40)
h = np.log1p(h.T)
im = ax.imshow(h, origin='lower', cmap='inferno',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               aspect='auto', alpha=0.9)
for name, pts, n_lanes in ROAD_SEGMENTS:
    ax.plot(pts[:, 0], pts[:, 1], '-', color='#ffffff', lw=0.3, alpha=0.3)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('log(density+1)', fontsize=6, color='#888899', fontfamily='monospace')
cbar.ax.tick_params(colors='#555566', labelsize=5)


# ── Panel (1,1): Intent PCA ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
setup_dark_ax(ax, 'INTENT EMBEDDING SPACE (PCA)', 'PC1', 'PC2')

ring_model.eval()
with torch.no_grad():
    _, all_intents = ring_model(hist_val[:32], ctx_val[:32])

# Compute heading colors first
headings_flat = np.arctan2(
    hist_val[:32, :, 3].numpy().flatten(),
    hist_val[:32, :, 2].numpy().flatten())
headings_norm_pca = (headings_flat + np.pi) / (2 * np.pi)

# PCA on intent vectors (robust to numerical issues from transformer masking)
import warnings
_orig_filters = warnings.filters[:]
warnings.filterwarnings("ignore", category=RuntimeWarning)

intent_flat = all_intents.reshape(-1, 8).float().numpy().copy()
intent_flat = np.clip(intent_flat, -100, 100)
intent_flat = np.nan_to_num(intent_flat, nan=0.0, posinf=0.0, neginf=0.0)
intent_centered = intent_flat - intent_flat.mean(axis=0)
rng_pca = np.random.RandomState(42)
intent_centered = intent_centered + rng_pca.randn(*intent_centered.shape).astype(np.float32) * 0.01
intent_centered = np.nan_to_num(intent_centered, nan=0.0, posinf=0.0, neginf=0.0)
cov = (intent_centered.T @ intent_centered) / len(intent_centered)
cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0) + np.eye(8) * 1e-6
eigvals, eigvecs = np.linalg.eigh(cov)
pc = intent_centered @ eigvecs[:, -2:]
pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)
valid_mask = np.isfinite(pc).all(axis=1)
pc_valid = pc[valid_mask]
hdg_valid = headings_norm_pca[valid_mask]
if len(pc_valid) > 10:
    lo, hi = np.percentile(pc_valid, 2, axis=0), np.percentile(pc_valid, 98, axis=0)
    pc_valid = np.clip(pc_valid, lo, hi)
S_pca = np.sqrt(np.abs(eigvals[::-1]) + 1e-12)

warnings.filters[:] = _orig_filters

sc = ax.scatter(pc_valid[:, 0], pc_valid[:, 1], c=hdg_valid, cmap='hsv',
                s=3, alpha=0.5, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Heading (rad)', fontsize=6, color='#888899', fontfamily='monospace')
cbar.ax.tick_params(colors='#555566', labelsize=5)
var_exp = (S_pca[0]**2 + S_pca[1]**2) / ((S_pca**2).sum() + 1e-12)
ax.text(0.01, 0.97, f'var explained: {var_exp:.1%}',
        transform=ax.transAxes, fontsize=6, color=ACCENT_CYAN,
        fontfamily='monospace', va='top')


# ── Panel (1,2): Topology bar chart ─────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
setup_dark_ax(ax, 'TOPOLOGY COMPARISON', '', 'Error / Rate')

x_pos = np.arange(3)
width = 0.25
names = ['isolated', 'ring', 'dense']
ade_vals = [metrics[n][0] for n in names]
fde_vals = [metrics[n][1] for n in names]
cr_vals = [metrics[n][2] * 100 for n in names]
colors_t = [TOPO_COLORS[n] for n in names]

ax.bar(x_pos - width, ade_vals, width, label='ADE (m)',
       color=colors_t, alpha=0.9, edgecolor='white', linewidth=0.3)
ax.bar(x_pos, fde_vals, width, label='FDE (m)',
       color=colors_t, alpha=0.6, edgecolor='white', linewidth=0.3)
ax.bar(x_pos + width, cr_vals, width, label='Coll% x100',
       color=colors_t, alpha=0.3, edgecolor=colors_t, linewidth=1.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=7, color='#aaaacc', fontfamily='monospace')
ax.legend(fontsize=6, facecolor=BG_PANEL, edgecolor='#333355',
          labelcolor='#aaaacc')
ax.grid(True, alpha=0.1, axis='y', color='#333355')

for i, (a, f) in enumerate(zip(ade_vals, fde_vals)):
    ax.text(i - width, a + 0.02, f'{a:.2f}', ha='center', fontsize=5,
            color=colors_t[i], fontfamily='monospace', fontweight='bold')
    ax.text(i, f + 0.02, f'{f:.2f}', ha='center', fontsize=5,
            color=colors_t[i], fontfamily='monospace')


# ── Panel (1,3): Training loss curves ────────────────────────────────
ax = fig.add_subplot(gs[1, 3])
setup_dark_ax(ax, 'TRAINING LOSS (LOG SCALE)', 'Epoch', 'Loss')

w = 15
def smooth(a, w=w):
    return np.convolve(a, np.ones(w)/w, mode='valid')

for name, losses, color in [
    ('isolated', iso_losses, TOPO_COLORS['isolated']),
    ('ring', ring_losses, TOPO_COLORS['ring']),
    ('dense', dense_losses, TOPO_COLORS['dense']),
]:
    smoothed = smooth(losses)
    ax.plot(smoothed, color=color, lw=1.5, label=name, alpha=0.9)
    ax.plot(len(smoothed)-1, smoothed[-1], 'o', color=color, markersize=4)
    ax.text(len(smoothed)+2, smoothed[-1], f'{smoothed[-1]:.3f}',
            fontsize=5, color=color, fontfamily='monospace', va='center')

ax.legend(fontsize=6, facecolor=BG_PANEL, edgecolor='#333355',
          labelcolor='#aaaacc')
ax.set_yscale('log')
ax.grid(True, alpha=0.1, color='#333355')


# ── Panel (2,0): Canvas layout grid ─────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
setup_dark_ax(ax, f'CANVAS LAYOUT (ring, {CANVAS_H}x{CANVAS_W})')

H_c, W_c = bound_ring.layout.H, bound_ring.layout.W
grid = np.zeros((H_c, W_c, 3))

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
    t0_, t1_, h0, h1, w0, w1 = bf.spec.bounds
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

pos_scene = hist_val[scene_idx, :, :2].numpy()
n_show = min(32, N_VEHICLES)
dists = np.sqrt(((pos_scene[:n_show, None, :] - pos_scene[None, :n_show, :])**2).sum(axis=-1))
np.fill_diagonal(dists, np.nan)

im = ax.imshow(dists, cmap='magma_r', aspect='equal', interpolation='nearest')
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Distance (m)', fontsize=6, color='#888899', fontfamily='monospace')
cbar.ax.tick_params(colors='#555566', labelsize=5)
ax.set_xlabel('Vehicle ID', fontsize=6, color='#888899', fontfamily='monospace')
ax.set_ylabel('Vehicle ID', fontsize=6, color='#888899', fontfamily='monospace')

close_mask = np.nan_to_num(dists, nan=999) < 5.0
close_pairs = np.argwhere(close_mask & np.triu(np.ones_like(close_mask), k=1).astype(bool))
ax.text(0.01, 0.97, f'{len(close_pairs)} close pairs (<5m)',
        transform=ax.transAxes, fontsize=6, color=ACCENT_RED,
        fontfamily='monospace', va='top')


# ── Panel (2,2): Throughput over time ────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
setup_dark_ax(ax, 'SEGMENT THROUGHPUT', 'Time step', 'Vehicle count')

# Define bounding boxes for key regions
seg_regions = {
    'Highway mid': ((-10, -5), (10, 15)),
    'Roundabout': ((RBT_CENTER[0]-10, RBT_CENTER[1]-10),
                   (RBT_CENTER[0]+10, RBT_CENTER[1]+10)),
    'T-junction': ((T_JUNCTION_PT[0]-10, T_JUNCTION_PT[1]-10),
                   (T_JUNCTION_PT[0]+10, T_JUNCTION_PT[1]+5)),
}
seg_colors = [ACCENT_CYAN, ACCENT_MAGENTA, ACCENT_LIME]

for (seg_name, (lo, hi)), color in zip(seg_regions.items(), seg_colors):
    counts = []
    for t in range(len(anim_tx)):
        in_box = ((anim_tx[t] >= lo[0]) & (anim_tx[t] <= hi[0]) &
                  (anim_ty[t] >= lo[1]) & (anim_ty[t] <= hi[1])).sum()
        counts.append(in_box)
    ax.plot(counts, color=color, lw=1.5, label=seg_name, alpha=0.8)

ax.legend(fontsize=6, facecolor=BG_PANEL, edgecolor='#333355',
          labelcolor='#aaaacc')
ax.grid(True, alpha=0.1, color='#333355')


# ── Panel (2,3): Zone vehicle distribution ───────────────────────────
ax = fig.add_subplot(gs[2, 3])
setup_dark_ax(ax, 'ZONE VEHICLE DISTRIBUTION')

zone_names = ['Highway\n+Ramp', 'Round-\nabout', 'South\n+East', 'T-junc\n+Conn']
zone_counts = [0, 0, 0, 0]
for v in range(N_VEHICLES):
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


# ── Panel (3,0)+(3,1): Dense model predictions ──────────────────────
ax = fig.add_subplot(gs[3, 0:2])
setup_dark_ax(ax, 'TRAJECTORY PREDICTION: DENSE TOPOLOGY (scene 0)')

dense_model.eval()
with torch.no_grad():
    pred_dense, _ = dense_model(
        hist_val[scene_idx:scene_idx+1], ctx_val[scene_idx:scene_idx+1])

for name, pts, n_lanes in ROAD_SEGMENTS:
    ax.plot(pts[:, 0], pts[:, 1], '-', color='#151530', lw=4)

veh_cmap = plt.cm.twilight
for v in range(N_VEHICLES):
    cx = hist_val[scene_idx, v, 0].item()
    cy = hist_val[scene_idx, v, 1].item()
    c = veh_cmap(v / N_VEHICLES)

    ax.plot(cx, cy, 'o', color=c, markersize=3.5, zorder=5,
            markeredgecolor='white', markeredgewidth=0.2)

    true_t = fut_val[scene_idx, v].numpy()
    ax.plot(cx + true_t[:, 0], cy + true_t[:, 1], '--',
            color=c, lw=0.7, alpha=0.4)

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
    _, intents_r = ring_model(hist_val[:16], ctx_val[:16])
    avg_intent = intents_r.mean(dim=0)

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


# ── Panel (3,3): Schema statistics ───────────────────────────────────
ax = fig.add_subplot(gs[3, 3])
setup_dark_ax(ax, 'SCHEMA STATISTICS')

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

table = ax.table(cellText=stats_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
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

ax.text(0.5, 0.08, f'Grid: {CANVAS_H}x{CANVAS_W} | '
        f'd_model=32 | T=1 | N_VEH={N_VEHICLES}',
        transform=ax.transAxes, ha='center', fontsize=6,
        color='#555577', fontfamily='monospace')
ax.text(0.5, 0.02, f'Zone hierarchy: RoadNetwork > TrafficZone[{N_ZONES}] > '
        f'Vehicle[{VEHICLES_PER_ZONE}]',
        transform=ax.transAxes, ha='center', fontsize=5,
        color='#444466', fontfamily='monospace')


# Save
path = os.path.join(ASSETS, "04_fleet.png")
fig.savefig(path, bbox_inches='tight', facecolor=BG_DARK, dpi=150)
plt.close()
print(f"\nSaved {path}")


# ── 9. Animation GIF ────────────────────────────────────────────────

print("Generating fleet animation...")

fig_anim, ax_anim = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
fig_anim.patch.set_facecolor(BG_DARK)

speed_cmap_anim = plt.cm.plasma


def animate_fleet(frame):
    ax_anim.clear()
    ax_anim.set_facecolor(BG_DARK)

    draw_roads_on_ax(ax_anim, lw_scale=0.8)

    # Traffic signals
    for sig in TRAFFIC_SIGNALS:
        step_in_cycle = frame % sig['period']
        is_green = step_in_cycle < sig['period'] // 2
        color = '#69f0ae' if is_green else ACCENT_RED
        ax_anim.plot(sig['pos'][0], sig['pos'][1], 's', color=color,
                     markersize=5, zorder=10, markeredgecolor='white',
                     markeredgewidth=0.3, alpha=0.8)

    # Vehicles
    spds = anim_speeds[frame]
    spd_norm_f = np.clip((spds - 2) / 20.0, 0, 1)

    for v in range(N_VEHICLES):
        x = anim_tx[frame, v]
        y = anim_ty[frame, v]
        color = speed_cmap_anim(spd_norm_f[v])

        # Glowing trail
        trail_start = max(0, frame - 10)
        if frame > trail_start:
            n_trail = frame - trail_start + 1
            for ti in range(n_trail - 1):
                alpha = 0.03 + 0.25 * (ti / n_trail)
                ax_anim.plot(anim_tx[trail_start+ti:trail_start+ti+2, v],
                             anim_ty[trail_start+ti:trail_start+ti+2, v],
                             '-', color=color, lw=1.5, alpha=alpha)

        # Vehicle with glow
        ax_anim.plot(x, y, 'o', color=color, markersize=6, alpha=0.15)
        ax_anim.plot(x, y, 'o', color=color, markersize=4, alpha=0.3)
        ax_anim.plot(x, y, 'o', color=color, markersize=2.5,
                     zorder=5, markeredgecolor='white', markeredgewidth=0.15)

    ax_anim.set_xlim(-65, 70)
    ax_anim.set_ylim(-60, 20)
    ax_anim.set_aspect('equal')

    # HUD
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
print(f"Saved {gif_path}")

mp4_path = os.path.join(ASSETS, "04_fleet.mp4")
writer_mp4 = animation.FFMpegWriter(fps=24, bitrate=4000,
                                     codec='libx264',
                                     extra_args=['-pix_fmt', 'yuv420p'])
anim.save(mp4_path, writer=writer_mp4)
plt.close()
print(f"Saved {mp4_path}")
print("\nDone.")

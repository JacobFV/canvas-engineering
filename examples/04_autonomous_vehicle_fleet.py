"""Autonomous Vehicle Fleet: cooperative trajectory prediction with canvas types.

Social-force synthetic data. Three topology comparisons:
  1. Isolated: vehicles can't see each other
  2. Ring: geographic neighbors share perception
  3. Dense: every vehicle sees every other vehicle

Plus a contrastive loss on the intent field — vehicles with similar future
trajectories should have similar intents.

Run:  python examples/04_autonomous_vehicle_fleet.py
Out:  assets/examples/04_fleet.png
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
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

from canvas_engineering import Field, compile_schema, ConnectivityPolicy, LayoutStrategy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

N_VEHICLES = 4
PRED_HORIZON = 8  # future steps to predict
DT = 0.2          # seconds per step
LANE_WIDTH = 3.5


# ── 1. Type declarations ─────────────────────────────────────────────

@dataclass
class Vehicle:
    position: Field = Field(1, 2)              # x/y + vx/vy packed
    heading: Field = Field(1, 1)               # heading scalar
    lane_context: Field = Field(1, 2, is_output=False)  # nearest lane info
    intent: Field = Field(1, 4)                # latent driving intent
    trajectory: Field = Field(2, 2, loss_weight=4.0)    # predicted future

@dataclass
class FleetScene:
    traffic_state: Field = Field(2, 4)         # global scene summary
    vehicles: list = dc_field(default_factory=list)


def make_schema(connectivity_policy, n_vehicles=N_VEHICLES):
    scene = FleetScene(vehicles=[Vehicle() for _ in range(n_vehicles)])
    return compile_schema(
        scene, T=1, H=16, W=16, d_model=32,
        connectivity=connectivity_policy,
    )


# Three topologies to compare
bound_isolated = make_schema(ConnectivityPolicy(
    intra="dense", parent_child="hub_spoke",
    array_element="isolated", temporal="dense"))
bound_ring = make_schema(ConnectivityPolicy(
    intra="dense", parent_child="hub_spoke",
    array_element="ring", temporal="dense"))
bound_dense = make_schema(ConnectivityPolicy(
    intra="dense", parent_child="hub_spoke",
    array_element="dense", temporal="dense"))


# ── 2. Synthetic data: social force model ────────────────────────────

def generate_lane_centers(n_lanes=3):
    """Parallel lanes along x-axis."""
    return np.array([i * LANE_WIDTH for i in range(n_lanes)])

LANES = generate_lane_centers()


def social_force_trajectory(n_scenes=2048, n_vehicles=N_VEHICLES,
                            n_history=4, n_future=PRED_HORIZON):
    """Generate multi-vehicle trajectories using social force model.

    Each vehicle has: lane attraction + collision repulsion + random perturbation.
    Returns history states, future trajectories, lane contexts.
    """
    total_steps = n_history + n_future
    all_history = []
    all_future = []
    all_lane_ctx = []

    for _ in range(n_scenes):
        # Initialize vehicles in random lanes with random x positions
        lane_assignments = np.random.randint(0, len(LANES), n_vehicles)
        x = np.random.uniform(0, 40, n_vehicles).astype(np.float32)
        y = LANES[lane_assignments].astype(np.float32)
        vx = np.random.uniform(5, 15, n_vehicles).astype(np.float32)
        vy = np.zeros(n_vehicles, dtype=np.float32)

        states = []  # (n_steps, n_vehicles, 4) — x, y, vx, vy
        for step in range(total_steps):
            states.append(np.stack([x.copy(), y.copy(), vx.copy(), vy.copy()], axis=-1))

            # Forces
            fx = np.zeros(n_vehicles, dtype=np.float32)
            fy = np.zeros(n_vehicles, dtype=np.float32)

            for i in range(n_vehicles):
                # Lane attraction: pull toward assigned lane center
                lane_y = LANES[lane_assignments[i]]
                fy[i] += 2.0 * (lane_y - y[i])

                # Speed regulation: target ~10 m/s
                fx[i] += 0.5 * (10.0 - vx[i])

                # Collision repulsion from other vehicles
                for j in range(n_vehicles):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = max(np.sqrt(dx**2 + dy**2), 0.5)
                    if dist < 8.0:  # interaction range
                        force = 15.0 / (dist ** 2)
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist

            # Random perturbation (lane changes, acceleration noise)
            fx += np.random.randn(n_vehicles).astype(np.float32) * 0.5
            fy += np.random.randn(n_vehicles).astype(np.float32) * 0.3

            # Integrate
            vx += fx * DT
            vy += fy * DT
            vx = np.clip(vx, 2, 20)
            vy = np.clip(vy, -3, 3)
            x += vx * DT
            y += vy * DT

        states = np.array(states)  # (total_steps, n_vehicles, 4)

        # Split into history and future
        hist = states[:n_history]   # (n_history, n_vehicles, 4)
        fut = states[n_history:]    # (n_future, n_vehicles, 4)

        # Last history state = current observation
        current = hist[-1]  # (n_vehicles, 4)

        # Future trajectory: relative to current position (x,y offsets)
        fut_xy = fut[:, :, :2] - current[None, :, :2]  # (n_future, n_vehicles, 2)

        # Lane context: distance to each lane center for each vehicle
        lane_ctx = np.zeros((n_vehicles, 4), dtype=np.float32)
        for i in range(n_vehicles):
            for j, lc in enumerate(LANES):
                lane_ctx[i, j] = current[i, 1] - lc  # signed distance
            lane_ctx[i, 3] = lane_assignments[i]  # assigned lane

        all_history.append(current)
        all_future.append(fut_xy.transpose(1, 0, 2))  # (n_vehicles, n_future, 2)
        all_lane_ctx.append(lane_ctx)

    history = torch.tensor(np.array(all_history), dtype=torch.float32)      # (N, n_veh, 4)
    future = torch.tensor(np.array(all_future), dtype=torch.float32)        # (N, n_veh, n_future, 2)
    lane_ctx = torch.tensor(np.array(all_lane_ctx), dtype=torch.float32)    # (N, n_veh, 4)

    return history, future, lane_ctx


print("Generating social-force trajectories...")
hist_tr, fut_tr, lane_tr = social_force_trajectory(2048)
hist_val, fut_val, lane_val = social_force_trajectory(512)
print(f"  Train: {hist_tr.shape[0]} scenes, {N_VEHICLES} vehicles each")
print(f"  Future shape: {fut_tr.shape} (scenes, vehicles, steps, xy)")


# ── 3. Model ────────────────────────────────────────────────────────

class FleetModel(nn.Module):
    """Canvas-structured fleet trajectory predictor."""

    def __init__(self, bound, d=32, nhead=4, n_vehicles=N_VEHICLES):
        super().__init__()
        self.bound = bound
        self.d = d
        self.n_vehicles = n_vehicles
        N = bound.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, N, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=128,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        mask = bound.topology.to_additive_mask(bound.layout)
        self.register_buffer('mask', mask)

        # Per-vehicle projections
        pos_n = len(bound.layout.region_indices('vehicles[0].position'))
        hdg_n = len(bound.layout.region_indices('vehicles[0].heading'))
        lane_n = len(bound.layout.region_indices('vehicles[0].lane_context'))
        traj_n = len(bound.layout.region_indices('vehicles[0].trajectory'))
        intent_n = len(bound.layout.region_indices('vehicles[0].intent'))

        self.pos_proj = nn.Linear(4, pos_n * d)
        self.hdg_proj = nn.Linear(2, hdg_n * d)
        self.lane_proj = nn.Linear(4, lane_n * d)
        self.traj_head = nn.Linear(traj_n * d, PRED_HORIZON * 2)
        self.intent_proj_out = nn.Linear(intent_n * d, 8)  # for contrastive

        self.pos_n = pos_n
        self.hdg_n = hdg_n
        self.lane_n = lane_n
        self.traj_n = traj_n
        self.intent_n = intent_n

    def forward(self, states, lane_ctx):
        """states: (B, n_veh, 4), lane_ctx: (B, n_veh, 4)."""
        B = states.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        for i in range(self.n_vehicles):
            p_idx = self.bound.layout.region_indices(f'vehicles[{i}].position')
            h_idx = self.bound.layout.region_indices(f'vehicles[{i}].heading')
            l_idx = self.bound.layout.region_indices(f'vehicles[{i}].lane_context')

            pos_emb = self.pos_proj(states[:, i]).reshape(B, self.pos_n, self.d)
            canvas[:, p_idx] = canvas[:, p_idx] + pos_emb

            # Heading from velocity
            speed = torch.sqrt(states[:, i, 2]**2 + states[:, i, 3]**2).clamp(min=0.1)
            heading = torch.stack([states[:, i, 3] / speed, states[:, i, 2] / speed], dim=-1)
            hdg_emb = self.hdg_proj(heading).reshape(B, self.hdg_n, self.d)
            canvas[:, h_idx] = canvas[:, h_idx] + hdg_emb

            lane_emb = self.lane_proj(lane_ctx[:, i]).reshape(B, self.lane_n, self.d)
            canvas[:, l_idx] = canvas[:, l_idx] + lane_emb

        canvas = self.encoder(canvas, mask=self.mask)

        # Extract trajectories and intents
        trajs = []
        intents = []
        for i in range(self.n_vehicles):
            t_idx = self.bound.layout.region_indices(f'vehicles[{i}].trajectory')
            traj = self.traj_head(canvas[:, t_idx].reshape(B, -1))
            trajs.append(traj.reshape(B, PRED_HORIZON, 2))

            i_idx = self.bound.layout.region_indices(f'vehicles[{i}].intent')
            intent = self.intent_proj_out(canvas[:, i_idx].reshape(B, -1))
            intents.append(intent)

        trajs = torch.stack(trajs, dim=1)       # (B, n_veh, horizon, 2)
        intents = torch.stack(intents, dim=1)   # (B, n_veh, 8)
        return trajs, intents


class FlatBaseline(nn.Module):
    """MLP baseline: concatenate all vehicle states -> predict all trajectories."""

    def __init__(self, n_vehicles=N_VEHICLES, hidden=192):
        super().__init__()
        in_dim = n_vehicles * (4 + 4)  # state + lane_ctx per vehicle
        out_dim = n_vehicles * PRED_HORIZON * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.n_vehicles = n_vehicles

    def forward(self, states, lane_ctx):
        B = states.shape[0]
        x = torch.cat([states.reshape(B, -1), lane_ctx.reshape(B, -1)], dim=-1)
        out = self.net(x).reshape(B, self.n_vehicles, PRED_HORIZON, 2)
        return out, None  # no intents


# ── 4. Training ──────────────────────────────────────────────────────

def contrastive_intent_loss(intents, future_trajs, temperature=0.1):
    """Vehicles with similar futures should have similar intents."""
    B, V, D = intents.shape
    # Flatten future trajectories for similarity
    fut_flat = future_trajs.reshape(B, V, -1)  # (B, V, horizon*2)

    loss = torch.tensor(0.0)
    count = 0
    for b in range(min(B, 32)):  # subsample for speed
        # Cosine similarity of intents
        intent_norm = F.normalize(intents[b], dim=-1)  # (V, D)
        intent_sim = intent_norm @ intent_norm.T        # (V, V)

        # Trajectory similarity (normalized MSE -> similarity)
        fut_norm = F.normalize(fut_flat[b], dim=-1)
        traj_sim = fut_norm @ fut_norm.T                # (V, V)

        # Pull intent similarity toward trajectory similarity
        loss = loss + F.mse_loss(intent_sim, traj_sim.detach())
        count += 1

    return loss / max(count, 1)


def train_model(bound, label, use_contrastive=True, n_epochs=300, bs=128):
    if bound is None:
        model = FlatBaseline()
    else:
        model = FleetModel(bound)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []

    for ep in range(n_epochs):
        idx = torch.randint(0, len(hist_tr), (bs,))
        pred_traj, pred_intent = model(hist_tr[idx], lane_tr[idx])

        # Trajectory loss (main)
        traj_loss = F.mse_loss(pred_traj, fut_tr[idx])

        loss = traj_loss
        if use_contrastive and pred_intent is not None:
            c_loss = contrastive_intent_loss(pred_intent, fut_tr[idx])
            loss = loss + 0.3 * c_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if ep % 200 == 0:
            print(f"  [{label}] ep {ep:3d}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        vp, vi = model(hist_val, lane_val)
        vl = F.mse_loss(vp, fut_val).item()
    print(f"  [{label}] val_traj_mse={vl:.4f}")
    return model, losses, vl


print("\nTraining flat baseline...")
flat_model, flat_losses, flat_vl = train_model(None, "flat", use_contrastive=False)
print("Training isolated topology...")
iso_model, iso_losses, iso_vl = train_model(bound_isolated, "isolated")
print("Training ring topology...")
ring_model, ring_losses, ring_vl = train_model(bound_ring, "ring")
print("Training dense topology...")
dense_model, dense_losses, dense_vl = train_model(bound_dense, "dense")


# ── 5. Evaluate ──────────────────────────────────────────────────────

def compute_metrics(model, hist, fut, lane_ctx):
    """Compute ADE, FDE, and collision rate."""
    model.eval()
    with torch.no_grad():
        pred, _ = model(hist, lane_ctx)

    # Average Displacement Error
    ade = torch.sqrt(((pred - fut) ** 2).sum(dim=-1)).mean().item()

    # Final Displacement Error
    fde = torch.sqrt(((pred[:, :, -1] - fut[:, :, -1]) ** 2).sum(dim=-1)).mean().item()

    # Collision rate: predicted trajectories within 1.5m of each other
    collisions = 0
    total_pairs = 0
    for b in range(min(len(pred), 200)):
        for t in range(PRED_HORIZON):
            for i in range(N_VEHICLES):
                for j in range(i + 1, N_VEHICLES):
                    dist = torch.sqrt(((pred[b, i, t] - pred[b, j, t]) ** 2).sum()).item()
                    if dist < 1.5:
                        collisions += 1
                    total_pairs += 1

    collision_rate = collisions / max(total_pairs, 1)
    return ade, fde, collision_rate


print("\nMetrics:")
for name, model in [("Flat", flat_model), ("Isolated", iso_model),
                     ("Ring", ring_model), ("Dense", dense_model)]:
    ade, fde, cr = compute_metrics(model, hist_val, fut_val, lane_val)
    print(f"  {name:10s}: ADE={ade:.3f}  FDE={fde:.3f}  CollRate={cr:.4f}")


# ── 6. Visualization ────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Autonomous Vehicle Fleet: Canvas Topology Comparison',
             fontsize=16, fontweight='bold', y=0.99)

COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
TOPO_COLORS = {'flat': '#95A5A6', 'isolated': '#E8734A', 'ring': '#4A90D9', 'dense': '#2ECC71'}

# (a) Bird's-eye view: one scene with predicted vs true trajectories
ax = axes[0, 0]
ax.set_title("Bird's-Eye View (ring model, scene 0)", fontsize=11, fontweight='bold')

# Draw lanes
for lc in LANES:
    ax.axhline(y=lc, color='#BDC3C7', ls='--', lw=1, alpha=0.5)
    ax.axhspan(lc - LANE_WIDTH / 2, lc + LANE_WIDTH / 2, color='#ECF0F1', alpha=0.3)

scene_idx = 0
ring_model.eval()
with torch.no_grad():
    pred_ring, _ = ring_model(hist_val[scene_idx:scene_idx+1], lane_val[scene_idx:scene_idx+1])

for v in range(N_VEHICLES):
    c = COLORS[v % len(COLORS)]
    # Current position
    cx, cy = hist_val[scene_idx, v, 0].item(), hist_val[scene_idx, v, 1].item()
    ax.plot(cx, cy, 'o', color=c, markersize=8, zorder=5)
    ax.text(cx, cy + 0.5, f'V{v}', ha='center', fontsize=7, color=c, fontweight='bold')

    # True future
    true_traj = fut_val[scene_idx, v].numpy()  # (horizon, 2)
    true_x = cx + true_traj[:, 0]
    true_y = cy + true_traj[:, 1]
    ax.plot(true_x, true_y, '--', color=c, lw=1.5, alpha=0.5)

    # Predicted future (ring)
    pred_traj = pred_ring[0, v].numpy()
    pred_x = cx + pred_traj[:, 0]
    pred_y = cy + pred_traj[:, 1]
    ax.plot(pred_x, pred_y, '-', color=c, lw=2)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.legend(['true (dashed)', 'predicted (solid)'], fontsize=7, loc='upper left')
ax.grid(True, alpha=0.15)

# (b) Intent similarity matrix (ring model)
ax = axes[0, 1]
ax.set_title('Intent Similarity (ring, scene 0)', fontsize=11, fontweight='bold')
ring_model.eval()
with torch.no_grad():
    _, intents = ring_model(hist_val[:32], lane_val[:32])
    # Average over batch for stable picture
    avg_intent = intents.mean(dim=0)  # (n_veh, 16)
    intent_norm = F.normalize(avg_intent, dim=-1)
    sim_matrix = (intent_norm @ intent_norm.T).numpy()

im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(N_VEHICLES))
ax.set_yticks(range(N_VEHICLES))
ax.set_xticklabels([f'V{i}' for i in range(N_VEHICLES)], fontsize=8)
ax.set_yticklabels([f'V{i}' for i in range(N_VEHICLES)], fontsize=8)
for i in range(N_VEHICLES):
    for j in range(N_VEHICLES):
        ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center', fontsize=6)

# (c) Training curves
ax = axes[0, 2]
ax.set_title('Training Loss', fontsize=11, fontweight='bold')
w = 30
def smooth(a, w=w): return np.convolve(a, np.ones(w)/w, mode='valid')
for name, losses, color in [
    ('flat', flat_losses, TOPO_COLORS['flat']),
    ('isolated', iso_losses, TOPO_COLORS['isolated']),
    ('ring', ring_losses, TOPO_COLORS['ring']),
    ('dense', dense_losses, TOPO_COLORS['dense']),
]:
    ax.plot(smooth(losses), color=color, lw=1.5, label=name)
ax.legend(fontsize=8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid(True, alpha=0.2)
ax.set_yscale('log')

# (d) ADE/FDE comparison bar chart
ax = axes[1, 0]
ax.set_title('Trajectory Error (ADE / FDE)', fontsize=11, fontweight='bold')
models_data = []
for name, model, color in [
    ('flat', flat_model, TOPO_COLORS['flat']),
    ('isolated', iso_model, TOPO_COLORS['isolated']),
    ('ring', ring_model, TOPO_COLORS['ring']),
    ('dense', dense_model, TOPO_COLORS['dense']),
]:
    ade, fde, cr = compute_metrics(model, hist_val, fut_val, lane_val)
    models_data.append((name, ade, fde, cr, color))

x_pos = np.arange(len(models_data))
width = 0.35
ade_vals = [m[1] for m in models_data]
fde_vals = [m[2] for m in models_data]
colors = [m[4] for m in models_data]
bars1 = ax.bar(x_pos - width/2, ade_vals, width, label='ADE', color=colors, alpha=0.7)
bars2 = ax.bar(x_pos + width/2, fde_vals, width, label='FDE', color=colors, alpha=0.4,
               edgecolor=colors, linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels([m[0] for m in models_data], fontsize=9)
ax.legend(fontsize=8)
ax.set_ylabel('Error (m)')
ax.grid(True, alpha=0.2, axis='y')

# Annotate best
best_ade_idx = np.argmin(ade_vals)
ax.annotate(f'{ade_vals[best_ade_idx]:.3f}',
            (best_ade_idx - width/2, ade_vals[best_ade_idx]),
            textcoords="offset points", xytext=(0, 5),
            ha='center', fontsize=8, fontweight='bold', color=colors[best_ade_idx])

# (e) Collision rate comparison
ax = axes[1, 1]
ax.set_title('Collision Rate (lower is better)', fontsize=11, fontweight='bold')
cr_vals = [m[3] for m in models_data]
bars = ax.bar(x_pos, cr_vals, 0.5, color=colors, alpha=0.7, edgecolor=colors, linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels([m[0] for m in models_data], fontsize=9)
ax.set_ylabel('Collision Rate')
ax.grid(True, alpha=0.2, axis='y')
for i, v in enumerate(cr_vals):
    ax.text(i, v + max(cr_vals) * 0.02, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')

# (f) Canvas layout (ring topology)
ax = axes[1, 2]
ax.set_title('Canvas Layout (ring topology)', fontsize=11, fontweight='bold')
H, W = bound_ring.layout.H, bound_ring.layout.W
grid = np.ones((H, W, 3)) * 0.95
field_colors = {
    'traffic_state': '#2C3E50',
    'position': '#E74C3C', 'heading': '#E67E22',
    'lane_context': '#95A5A6', 'intent': '#9B59B6',
    'trajectory': '#3498DB',
}

for name, bf in bound_ring.fields.items():
    # Find which color to use
    color_key = name.split('.')[-1] if '.' in name else name
    color = field_colors.get(color_key, '#BDC3C7')
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]

ax.imshow(grid, aspect='equal', interpolation='nearest')

# Legend for field types
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=n) for n, c in field_colors.items()]
ax.legend(handles=legend_elements, fontsize=6, loc='lower right', ncol=2)
ax.set_xlabel('W')
ax.set_ylabel('H')

plt.tight_layout(rect=[0, 0, 1, 0.97])
path = os.path.join(ASSETS, "04_fleet.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\nSaved {path}")


# ── 7. Animation: bird's-eye vehicle simulation ─────────────────────

import matplotlib.animation as animation

print("Generating fleet animation...")

# Run a full social-force simulation for animation (longer, single scene)
np.random.seed(7)
N_ANIM_STEPS = 60
lane_assignments = np.random.randint(0, len(LANES), N_VEHICLES)
ax_pos = np.random.uniform(0, 30, N_VEHICLES).astype(np.float32)
ay_pos = LANES[lane_assignments].astype(np.float32)
avx = np.random.uniform(5, 12, N_VEHICLES).astype(np.float32)
avy = np.zeros(N_VEHICLES, dtype=np.float32)

anim_traj_x = [ax_pos.copy()]
anim_traj_y = [ay_pos.copy()]

for step in range(N_ANIM_STEPS):
    fx = np.zeros(N_VEHICLES, dtype=np.float32)
    fy = np.zeros(N_VEHICLES, dtype=np.float32)
    for i in range(N_VEHICLES):
        fy[i] += 2.0 * (LANES[lane_assignments[i]] - ay_pos[i])
        fx[i] += 0.5 * (10.0 - avx[i])
        for j in range(N_VEHICLES):
            if i == j: continue
            dx = ax_pos[i] - ax_pos[j]
            dy = ay_pos[i] - ay_pos[j]
            dist = max(np.sqrt(dx**2 + dy**2), 0.5)
            if dist < 8.0:
                force = 15.0 / (dist ** 2)
                fx[i] += force * dx / dist
                fy[i] += force * dy / dist
    fx += np.random.randn(N_VEHICLES).astype(np.float32) * 0.3
    fy += np.random.randn(N_VEHICLES).astype(np.float32) * 0.2
    avx += fx * DT
    avy += fy * DT
    avx = np.clip(avx, 2, 20)
    avy = np.clip(avy, -3, 3)
    ax_pos += avx * DT
    ay_pos += avy * DT
    anim_traj_x.append(ax_pos.copy())
    anim_traj_y.append(ay_pos.copy())

anim_traj_x = np.array(anim_traj_x)  # (N_ANIM_STEPS+1, N_VEHICLES)
anim_traj_y = np.array(anim_traj_y)

fig_anim, ax_anim = plt.subplots(1, 1, figsize=(10, 4), dpi=100)
fig_anim.patch.set_facecolor('#2C3E50')

def animate_fleet(frame):
    ax_anim.clear()
    ax_anim.set_facecolor('#34495E')

    # Draw lanes
    for lc in LANES:
        ax_anim.axhline(y=lc, color='white', ls='--', lw=0.5, alpha=0.3)
        ax_anim.axhspan(lc - LANE_WIDTH / 2, lc + LANE_WIDTH / 2,
                        color='#445566', alpha=0.3)

    # Draw road edges
    ax_anim.axhline(y=LANES[0] - LANE_WIDTH / 2, color='white', lw=2)
    ax_anim.axhline(y=LANES[-1] + LANE_WIDTH / 2, color='white', lw=2)

    for v in range(N_VEHICLES):
        c = COLORS[v % len(COLORS)]
        x = anim_traj_x[frame, v]
        y = anim_traj_y[frame, v]

        # Trail
        trail_start = max(0, frame - 8)
        trail_x = anim_traj_x[trail_start:frame+1, v]
        trail_y = anim_traj_y[trail_start:frame+1, v]
        ax_anim.plot(trail_x, trail_y, '-', color=c, lw=1.5, alpha=0.3)

        # Vehicle (rectangle-ish)
        ax_anim.plot(x, y, 's', color=c, markersize=10, zorder=5,
                     markeredgecolor='white', markeredgewidth=0.5)
        ax_anim.text(x, y, f'{v}', ha='center', va='center',
                     fontsize=6, color='white', fontweight='bold', zorder=6)

    # Camera follows the fleet
    x_center = anim_traj_x[frame].mean()
    ax_anim.set_xlim(x_center - 25, x_center + 25)
    ax_anim.set_ylim(LANES[0] - LANE_WIDTH, LANES[-1] + LANE_WIDTH)
    ax_anim.set_title(f'Fleet Simulation — t={frame * DT:.1f}s',
                      color='white', fontsize=12, fontweight='bold')
    ax_anim.tick_params(colors='#888888')
    for spine in ax_anim.spines.values():
        spine.set_color('#555555')

anim = animation.FuncAnimation(fig_anim, animate_fleet,
                                frames=N_ANIM_STEPS + 1, interval=100)
gif_path = os.path.join(ASSETS, "04_fleet.gif")
anim.save(gif_path, writer='pillow', fps=10)
plt.close()
print(f"Saved {gif_path}")

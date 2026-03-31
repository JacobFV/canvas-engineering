"""World model with mixed carriers: deterministic, diffusive, and filter dynamics.

Demonstrates v2 carriers through compile_program. Different regions use
different update dynamics:
  - observed_video (observation, carrier=deterministic): direct frame input
  - predicted_video (observation, carrier=diffusive): noisy prediction refined
  - belief (state, carrier=filter): predict-then-correct Bayesian update
  - action (action, carrier=deterministic): standard control output

The model predicts future frames from past observations and actions. The
belief state tracks latent dynamics via a simulated filtering process.

Run:  python examples/15_world_model_carriers.py
Out:  assets/examples/15_world_model_carriers.png
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from canvas_engineering import (
    Field, compile_program, ConnectivityPolicy,
    CanvasProgram, RegionProgram,
    CARRIERS,
)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)

# ── 1. Declare types with carrier annotations ────────────────────────

@dataclass
class WorldModel:
    observed_video: Field = Field(3, 3, family="observation",
                                  carrier="deterministic",
                                  semantic_type="observed video frame 3x3")
    predicted_video: Field = Field(3, 3, family="observation",
                                   carrier="diffusive", loss_weight=2.0,
                                   semantic_type="predicted future frame 3x3")
    belief: Field = Field(2, 4, family="state",
                          carrier="filter", tags=("belief",),
                          loss_weight=1.5,
                          semantic_type="latent belief state for dynamics")
    action: Field = Field(1, 4, family="action",
                          carrier="deterministic",
                          semantic_type="control action 4-dim")


# ── 2. Compile with program semantics ─────────────────────────────────

wm = WorldModel()
bound, program = compile_program(
    wm, T=1, H=8, W=8, d_model=48,
    connectivity=ConnectivityPolicy(intra="dense", temporal="same_frame"),
)

print("=== World Model with Mixed Carriers ===")
print(bound.summary())
print()
print(program.summary())

# Show carrier assignments
print("\nCarrier assignments:")
for name, rp in program.regions.items():
    if name in bound:
        print(f"  {name}: family={rp.family}, carrier={rp.carrier}")
print(f"\nAvailable carriers: {sorted(CARRIERS)}")


# ── 3. Generate synthetic data ────────────────────────────────────────
# Simple world dynamics: bouncing ball. Observation = position on 3x3 grid.
# Action = force applied. Next frame = ball moves with dynamics.

OBS_DIM = 9    # 3x3 grid
BELIEF_DIM = 8  # 2x4 latent state
ACTION_DIM = 4

def generate_world_data(n_samples=2048, noise_levels=None):
    """Generate (obs, action) -> next_obs with latent dynamics."""
    if noise_levels is None:
        noise_levels = {'obs': 0.05, 'pred': 0.2, 'belief': 0.1}

    # Latent state: ball position (x, y) + velocity (vx, vy)
    x = torch.rand(n_samples) * 2 - 1
    y = torch.rand(n_samples) * 2 - 1
    vx = torch.randn(n_samples) * 0.3
    vy = torch.randn(n_samples) * 0.3

    # Action: force in 4 directions (up, down, left, right)
    action = torch.softmax(torch.randn(n_samples, ACTION_DIM), dim=-1)
    force_x = action[:, 2] - action[:, 3]  # right - left
    force_y = action[:, 0] - action[:, 1]  # up - down

    # Next state (simple physics)
    next_vx = vx + force_x * 0.5
    next_vy = vy + force_y * 0.5
    next_x = (x + next_vx * 0.3).clamp(-1, 1)
    next_y = (y + next_vy * 0.3).clamp(-1, 1)

    # Render to 3x3 grid (soft Gaussian)
    def render(px, py):
        grid = torch.zeros(n_samples, 3, 3)
        for i in range(3):
            for j in range(3):
                cx = (i - 1) / 1.5  # grid centers at -0.67, 0, 0.67
                cy = (j - 1) / 1.5
                d2 = (px - cx)**2 + (py - cy)**2
                grid[:, i, j] = torch.exp(-d2 / 0.3)
        return grid.reshape(n_samples, OBS_DIM)

    obs = render(x, y) + torch.randn(n_samples, OBS_DIM) * noise_levels['obs']
    next_obs = render(next_x, next_y) + torch.randn(n_samples, OBS_DIM) * noise_levels['obs']

    # Belief target: latent state representation
    belief_target = torch.stack([x, y, vx, vy, next_x, next_y, next_vx, next_vy], dim=-1)
    belief_target = belief_target + torch.randn_like(belief_target) * noise_levels['belief']

    return {
        'obs': obs, 'action': action, 'next_obs': next_obs,
        'belief': belief_target,
        'x': x, 'y': y, 'next_x': next_x, 'next_y': next_y,
    }

data_tr = generate_world_data()
data_val = generate_world_data(512)


# ── 4. Build model with carrier-aware processing ─────────────────────

class CarrierWorldModel(nn.Module):
    """World model that processes each region according to its carrier type."""

    def __init__(self, bound_schema, d_model=48, nhead=4):
        super().__init__()
        self.bound = bound_schema
        self.d = d_model
        N = bound_schema.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, N, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=192,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        mask = bound_schema.topology.to_additive_mask(bound_schema.layout)
        self.register_buffer('attn_mask', mask)

        obs_n = len(bound_schema.layout.region_indices("observed_video"))
        pred_n = len(bound_schema.layout.region_indices("predicted_video"))
        belief_n = len(bound_schema.layout.region_indices("belief"))
        act_n = len(bound_schema.layout.region_indices("action"))

        # Deterministic carrier: direct projection
        self.obs_proj = nn.Linear(OBS_DIM, obs_n * d_model)
        self.act_proj = nn.Linear(ACTION_DIM, act_n * d_model)

        # Diffusive carrier: prediction output with noise layer
        self.pred_out = nn.Linear(pred_n * d_model, OBS_DIM)
        self.pred_noise_scale = nn.Parameter(torch.tensor(0.1))

        # Filter carrier: predict + correct heads
        self.belief_predict = nn.Linear(belief_n * d_model, BELIEF_DIM)
        self.belief_correct = nn.Linear(OBS_DIM, BELIEF_DIM)
        self.belief_gate = nn.Linear(BELIEF_DIM * 2, BELIEF_DIM)

        self.obs_n = obs_n
        self.pred_n = pred_n
        self.belief_n = belief_n
        self.act_n = act_n

    def forward(self, obs, action, add_diffusion_noise=False):
        B = obs.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        obs_idx = self.bound.layout.region_indices("observed_video")
        act_idx = self.bound.layout.region_indices("action")
        pred_idx = self.bound.layout.region_indices("predicted_video")
        belief_idx = self.bound.layout.region_indices("belief")

        # Deterministic carrier: direct scatter
        canvas[:, obs_idx] = canvas[:, obs_idx] + \
            self.obs_proj(obs).reshape(B, self.obs_n, self.d)
        canvas[:, act_idx] = canvas[:, act_idx] + \
            self.act_proj(action).reshape(B, self.act_n, self.d)

        # Transformer processes all regions
        canvas = self.encoder(canvas, mask=self.attn_mask)

        # Diffusive carrier: output + optional noise injection
        pred_emb = canvas[:, pred_idx].reshape(B, -1)
        predicted = self.pred_out(pred_emb)
        if add_diffusion_noise and self.training:
            noise = torch.randn_like(predicted) * torch.abs(self.pred_noise_scale)
            predicted = predicted + noise

        # Filter carrier: predict-correct update
        belief_emb = canvas[:, belief_idx].reshape(B, -1)
        belief_prior = self.belief_predict(belief_emb)  # prediction step
        belief_innovation = self.belief_correct(obs)     # correction from observation
        gate_input = torch.cat([belief_prior, belief_innovation], dim=-1)
        gate = torch.sigmoid(self.belief_gate(gate_input))
        belief = gate * belief_prior + (1 - gate) * belief_innovation

        return predicted, belief

    def get_noise_scale(self):
        return torch.abs(self.pred_noise_scale).item()


model = CarrierWorldModel(bound)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400)


# ── 5. Train ──────────────────────────────────────────────────────────

losses_total = []
losses_pred = []
losses_belief = []
noise_scales = []
n_epochs = 400
batch_size = 64

print("\nTraining world model with mixed carriers...")
for epoch in range(n_epochs):
    idx = torch.randint(0, len(data_tr['obs']), (batch_size,))
    predicted, belief = model(
        data_tr['obs'][idx], data_tr['action'][idx],
        add_diffusion_noise=True,
    )

    # Prediction loss (diffusive carrier: MSE)
    pred_loss = ((predicted - data_tr['next_obs'][idx]) ** 2).mean() * 2.0

    # Belief loss (filter carrier: MSE on latent state)
    belief_loss = ((belief - data_tr['belief'][idx]) ** 2).mean() * 1.5

    loss = pred_loss + belief_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()

    losses_total.append(loss.item())
    losses_pred.append(pred_loss.item())
    losses_belief.append(belief_loss.item())
    noise_scales.append(model.get_noise_scale())

    if epoch % 50 == 0:
        print(f"  epoch {epoch:3d}: pred={pred_loss.item():.4f} belief={belief_loss.item():.4f} noise={model.get_noise_scale():.4f}")


# ── 6. Evaluate ───────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    pred_val, belief_val = model(data_val['obs'], data_val['action'])
    pred_mse = ((pred_val - data_val['next_obs']) ** 2).mean().item()
    belief_mse = ((belief_val - data_val['belief']) ** 2).mean().item()

    # Carrier-specific quality: track per-position errors
    print(f"\n  Prediction MSE (diffusive): {pred_mse:.4f}")
    print(f"  Belief MSE (filter): {belief_mse:.4f}")
    print(f"  Final noise scale: {model.get_noise_scale():.4f}")


# ── 7. Visualize ──────────────────────────────────────────────────────

CARRIER_COLORS = {
    'deterministic': '#4A90D9',
    'diffusive': '#E74C3C',
    'filter': '#E67E22',
    'memory': '#9B59B6',
    'residual': '#2ECC71',
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('World Model: Mixed Carriers (deterministic / diffusive / filter)',
             fontsize=14, fontweight='bold', y=0.98)

# (a) Canvas layout color-coded by carrier
ax = axes[0, 0]
ax.set_title('Canvas Layout (colored by carrier)', fontsize=11, fontweight='bold')
H, W = bound.layout.H, bound.layout.W
grid = np.ones((H, W, 3)) * 0.93
for name, rp in program.regions.items():
    if name not in bound:
        continue
    bf = bound[name]
    color = CARRIER_COLORS.get(rp.carrier, '#95A5A6')
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]
    ax.text((w0 + w1) / 2 - 0.5, (h0 + h1) / 2 - 0.5,
            f'{name}\n[{rp.carrier}]',
            ha='center', va='center', fontsize=5, fontweight='bold', color='white')
ax.imshow(grid, aspect='equal', interpolation='nearest')
ax.set_xlabel('W'); ax.set_ylabel('H')
for carr, col in CARRIER_COLORS.items():
    if carr in [rp.carrier for rp in program.regions.values()]:
        ax.plot([], [], 's', color=col, markersize=8, label=carr)
ax.legend(fontsize=7, loc='lower right', ncol=2)

# (b) Prediction quality by carrier type
ax = axes[0, 1]
ax.set_title('Quality by Carrier Type', fontsize=11, fontweight='bold')
carrier_metrics = {
    'deterministic\n(obs input)': 0.0,  # input, no prediction error
    'diffusive\n(predicted)': pred_mse,
    'filter\n(belief)': belief_mse,
}
colors_bar = [CARRIER_COLORS['deterministic'], CARRIER_COLORS['diffusive'], CARRIER_COLORS['filter']]
bars = ax.bar(list(carrier_metrics.keys()), list(carrier_metrics.values()),
              color=colors_bar, edgecolor='white', linewidth=2)
for bar, val in zip(bars, carrier_metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('MSE')
ax.grid(True, alpha=0.2, axis='y')

# (c) Belief state evolution: show filter's predict-correct dynamics
ax = axes[1, 0]
ax.set_title('Belief State: Filter Dynamics', fontsize=11, fontweight='bold')
# Plot belief predictions vs targets for first 8 dimensions
with torch.no_grad():
    sample_idx = 0
    belief_single = belief_val[sample_idx].numpy()
    belief_true = data_val['belief'][sample_idx].numpy()

x_pos = np.arange(BELIEF_DIM)
width = 0.35
ax.bar(x_pos - width/2, belief_true, width, color=CARRIER_COLORS['filter'],
       alpha=0.7, label='true state')
ax.bar(x_pos + width/2, belief_single, width, color=CARRIER_COLORS['deterministic'],
       alpha=0.7, label='predicted (filter)')
ax.set_xlabel('State dimension')
ax.set_ylabel('Value')
ax.set_xticks(x_pos)
ax.set_xticklabels(['x', 'y', 'vx', 'vy', 'nx', 'ny', 'nvx', 'nvy'], fontsize=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# (d) Training curves + noise scale
ax = axes[1, 1]
ax.set_title('Training Curves', fontsize=11, fontweight='bold')
w = 20
def smooth(a): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.semilogy(smooth(losses_pred), color=CARRIER_COLORS['diffusive'], lw=2, label='prediction (diffusive)')
ax.semilogy(smooth(losses_belief), color=CARRIER_COLORS['filter'], lw=2, label='belief (filter)')

ax2 = ax.twinx()
ax2.plot(noise_scales, color='#95A5A6', lw=1.5, alpha=0.7, label='noise scale')
ax2.set_ylabel('Noise Scale', color='#95A5A6', fontsize=9)
ax2.tick_params(axis='y', labelcolor='#95A5A6')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "15_world_model_carriers.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")

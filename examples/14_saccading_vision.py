"""Saccading vision: event-triggered fovea, residual-driven scheduling, families.

Demonstrates v2 process semantics: compile_program, families, event-triggered
scheduling via residual summaries, and the RegionScheduler.

Architecture:
  periphery (observation) - low-res view of the full scene
  fovea (observation)     - high-res view at gaze point
  scene_belief (state, tags=belief,object) - accumulated scene understanding
  error (residual)        - prediction error signal
  gaze (action)           - where to look next

The fovea fires only when prediction error exceeds a threshold. The model
learns to saccade to informative locations to build a scene representation.

Run:  python examples/14_saccading_vision.py
Out:  assets/examples/14_saccading_vision.png
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
    CanvasProgram, RegionProgram, ClockSpec,
    RegionScheduler, ResidualSpec, ResidualAccumulator,
    REGION_FAMILIES,
)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)

# ── 1. Declare types with families ────────────────────────────────────

@dataclass
class SaccadingVision:
    periphery: Field = Field(3, 3, family="observation",
                             semantic_type="low-res peripheral view 3x3")
    fovea: Field = Field(2, 2, family="observation",
                         semantic_type="high-res foveal view 2x2")
    scene_belief: Field = Field(3, 3, family="state", tags=("belief", "object"),
                                loss_weight=2.0,
                                semantic_type="accumulated scene belief")
    error: Field = Field(1, 2, family="residual",
                         semantic_type="prediction error signal")
    gaze: Field = Field(1, 2, family="action", loss_weight=3.0,
                        semantic_type="gaze position x,y")


# ── 2. Compile with program semantics ─────────────────────────────────

schema_obj = SaccadingVision()
bound, program = compile_program(
    schema_obj, T=1, H=8, W=8, d_model=48,
    connectivity=ConnectivityPolicy(intra="dense", temporal="same_frame"),
)

print("=== Saccading Vision ===")
print(bound.summary())
print()
print(program.summary())

# Add clock specs to the program: fovea is event-triggered
program.regions["fovea"] = RegionProgram(
    family="observation",
    clock=ClockSpec(
        mode="on_event",
        event_source="error.prediction",
        event_threshold=0.3,
        cooldown=2,
        max_silence=5,
    ),
)
program.regions["error"] = RegionProgram(
    family="residual",
    carrier="residual",
)

# Show families
print("\nRegion families:")
for name, rp in program.regions.items():
    tags_str = f" tags={rp.tags}" if rp.tags else ""
    clock_str = ""
    if rp.clock:
        clock_str = f" clock={rp.clock.mode}"
    print(f"  {name}: family={rp.family}{tags_str}{clock_str}")

# Set up scheduler and residual accumulator
scheduler = RegionScheduler(program)
accumulator = ResidualAccumulator(["error"], ResidualSpec(kinds=("prediction",)))


# ── 3. Generate synthetic data ────────────────────────────────────────
# Scene: 8x8 grid with "objects" (Gaussians) at random positions.
# Periphery: downsampled 3x3 view. Fovea: 2x2 high-res at gaze point.
# Task: predict object locations from peripheral + foveal input.

SCENE_SIZE = 8
PERIPH_DIM = 9   # 3x3 flattened
FOVEA_DIM = 4    # 2x2 flattened
BELIEF_DIM = 9   # 3x3 scene belief
ERROR_DIM = 2    # error signal
GAZE_DIM = 2     # gaze position (x, y)

def make_scene(n_samples, n_objects=3):
    """Create scenes with Gaussian objects at random positions."""
    scenes = torch.zeros(n_samples, SCENE_SIZE, SCENE_SIZE)
    object_positions = torch.zeros(n_samples, n_objects, 2)

    for i in range(n_samples):
        for j in range(n_objects):
            cx = torch.rand(1).item() * (SCENE_SIZE - 2) + 1
            cy = torch.rand(1).item() * (SCENE_SIZE - 2) + 1
            object_positions[i, j] = torch.tensor([cx / SCENE_SIZE, cy / SCENE_SIZE])
            for x in range(SCENE_SIZE):
                for y in range(SCENE_SIZE):
                    d2 = (x - cx)**2 + (y - cy)**2
                    scenes[i, x, y] += math.exp(-d2 / 2.0)
    return scenes, object_positions

def get_periphery(scenes):
    """Downsample scene to 3x3 peripheral view."""
    # Average pooling to 3x3
    B = scenes.shape[0]
    s = scenes.reshape(B, 1, SCENE_SIZE, SCENE_SIZE)
    p = nn.functional.adaptive_avg_pool2d(s, (3, 3))
    return p.reshape(B, PERIPH_DIM)

def get_fovea(scenes, gaze):
    """Extract 2x2 high-res view at gaze position."""
    B = scenes.shape[0]
    fovea = torch.zeros(B, FOVEA_DIM)
    for i in range(B):
        gx = int(gaze[i, 0].item() * (SCENE_SIZE - 2))
        gy = int(gaze[i, 1].item() * (SCENE_SIZE - 2))
        gx = max(0, min(gx, SCENE_SIZE - 2))
        gy = max(0, min(gy, SCENE_SIZE - 2))
        patch = scenes[i, gx:gx+2, gy:gy+2]
        fovea[i] = patch.reshape(-1)
    return fovea

# Generate data
scenes_tr, objpos_tr = make_scene(1024)
scenes_val, objpos_val = make_scene(256)

periph_tr = get_periphery(scenes_tr)
periph_val = get_periphery(scenes_val)

# Initial gaze: center
gaze_init_tr = torch.ones(1024, 2) * 0.5
gaze_init_val = torch.ones(256, 2) * 0.5

# Target gaze: location of strongest object signal
target_gaze_tr = objpos_tr[:, 0]  # aim at first object
target_gaze_val = objpos_val[:, 0]

# Target belief: downsampled scene as belief
belief_target_tr = get_periphery(scenes_tr)
belief_target_val = get_periphery(scenes_val)


# ── 4. Build transformer model ───────────────────────────────────────

class SaccadeModel(nn.Module):
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

        periph_n = len(bound_schema.layout.region_indices("periphery"))
        fovea_n = len(bound_schema.layout.region_indices("fovea"))
        belief_n = len(bound_schema.layout.region_indices("scene_belief"))
        error_n = len(bound_schema.layout.region_indices("error"))
        gaze_n = len(bound_schema.layout.region_indices("gaze"))

        self.periph_proj = nn.Linear(PERIPH_DIM, periph_n * d_model)
        self.fovea_proj = nn.Linear(FOVEA_DIM, fovea_n * d_model)
        self.belief_out = nn.Linear(belief_n * d_model, BELIEF_DIM)
        self.error_out = nn.Linear(error_n * d_model, ERROR_DIM)
        self.gaze_out = nn.Linear(gaze_n * d_model, GAZE_DIM)

        self.periph_n = periph_n
        self.fovea_n = fovea_n
        self.belief_n = belief_n
        self.error_n = error_n
        self.gaze_n = gaze_n

    def forward(self, periphery, fovea, fovea_active=True):
        B = periphery.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        periph_idx = self.bound.layout.region_indices("periphery")
        fovea_idx = self.bound.layout.region_indices("fovea")
        belief_idx = self.bound.layout.region_indices("scene_belief")
        error_idx = self.bound.layout.region_indices("error")
        gaze_idx = self.bound.layout.region_indices("gaze")

        canvas[:, periph_idx] = canvas[:, periph_idx] + \
            self.periph_proj(periphery).reshape(B, self.periph_n, self.d)

        if fovea_active:
            canvas[:, fovea_idx] = canvas[:, fovea_idx] + \
                self.fovea_proj(fovea).reshape(B, self.fovea_n, self.d)

        canvas = self.encoder(canvas, mask=self.attn_mask)

        belief = self.belief_out(canvas[:, belief_idx].reshape(B, -1))
        error = self.error_out(canvas[:, error_idx].reshape(B, -1))
        gaze = torch.sigmoid(self.gaze_out(canvas[:, gaze_idx].reshape(B, -1)))

        return belief, error, gaze


model = SaccadeModel(bound)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched_opt = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400)


# ── 5. Train with scheduling ─────────────────────────────────────────

losses = []
fovea_fire_rates = []
n_epochs = 400
batch_size = 64

print("\nTraining saccading vision model...")
for epoch in range(n_epochs):
    idx = torch.randint(0, len(scenes_tr), (batch_size,))
    periph_batch = periph_tr[idx]
    gaze_batch = gaze_init_tr[idx]
    fovea_batch = get_fovea(scenes_tr[idx], gaze_batch)

    # Simulate scheduling: check if fovea should fire
    # In training we always fire fovea but track when scheduler would fire
    scheduler.reset()
    summaries = accumulator.summaries()
    active = scheduler.step(epoch, summaries=summaries)
    fovea_active = "fovea" in active

    belief_pred, error_pred, gaze_pred = model(periph_batch, fovea_batch, fovea_active=True)

    # Losses
    belief_loss = ((belief_pred - belief_target_tr[idx]) ** 2).mean() * 2.0
    gaze_loss = ((gaze_pred - target_gaze_tr[idx]) ** 2).mean() * 3.0
    error_magnitude = ((belief_pred.detach() - belief_target_tr[idx]) ** 2).mean(dim=-1)
    error_target = error_magnitude.unsqueeze(-1).expand(-1, ERROR_DIM)
    error_loss = ((error_pred - error_target) ** 2).mean()

    loss = belief_loss + gaze_loss + error_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched_opt.step()
    losses.append(loss.item())

    # Update accumulator for scheduling
    accumulator.update("error", error_pred.detach())
    fovea_fire_rates.append(1.0 if fovea_active else 0.0)

    if epoch % 50 == 0:
        s = accumulator.summaries()
        err_val = s.get("error", {}).get("prediction", 0.0)
        print(f"  epoch {epoch:3d}: loss={loss.item():.4f} error_summary={err_val:.4f} fovea={'ON' if fovea_active else 'OFF'}")


# ── 6. Evaluate ───────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    fovea_val = get_fovea(scenes_val, gaze_init_val)

    # With fovea
    belief_w, _, gaze_w = model(periph_val, fovea_val, fovea_active=True)
    mse_with_fovea = ((belief_w - belief_target_val) ** 2).mean().item()
    gaze_mse = ((gaze_w - target_gaze_val) ** 2).mean().item()

    # Without fovea
    belief_wo, _, _ = model(periph_val, fovea_val, fovea_active=False)
    mse_without_fovea = ((belief_wo - belief_target_val) ** 2).mean().item()

    print(f"\n  Belief MSE with fovea: {mse_with_fovea:.4f}")
    print(f"  Belief MSE without fovea: {mse_without_fovea:.4f}")
    print(f"  Gaze prediction MSE: {gaze_mse:.4f}")

# Simulate scheduling over evaluation steps
scheduler.reset()
accumulator.reset()
schedule_history = []
for t in range(50):
    summaries = accumulator.summaries()
    active = scheduler.step(t, summaries=summaries)
    schedule_history.append({name: (name in active) for name in program.regions})
    # Simulate error signal
    err_signal = torch.rand(1) * (0.5 if t % 7 < 3 else 0.1)
    accumulator.update("error", err_signal)


# ── 7. Visualize ──────────────────────────────────────────────────────

FAMILY_COLORS = {
    'observation': '#4A90D9', 'state': '#E67E22',
    'residual': '#E74C3C', 'action': '#2ECC71', 'memory': '#9B59B6',
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Saccading Vision: Event-Triggered Fovea (v2 Program Semantics)',
             fontsize=14, fontweight='bold', y=0.98)

# (a) Canvas layout color-coded by family
ax = axes[0, 0]
ax.set_title('Canvas Layout (colored by family)', fontsize=11, fontweight='bold')
H, W = bound.layout.H, bound.layout.W
grid = np.ones((H, W, 3)) * 0.93
for name, rp in program.regions.items():
    if name not in bound:
        continue
    bf = bound[name]
    color = FAMILY_COLORS.get(rp.family, '#95A5A6')
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]
    ax.text((w0 + w1) / 2 - 0.5, (h0 + h1) / 2 - 0.5,
            f'{name}\n({rp.family})',
            ha='center', va='center', fontsize=5, fontweight='bold', color='white')
ax.imshow(grid, aspect='equal', interpolation='nearest')
ax.set_xlabel('W'); ax.set_ylabel('H')
# Legend
for fam, col in FAMILY_COLORS.items():
    if fam in [rp.family for rp in program.regions.values()]:
        ax.plot([], [], 's', color=col, markersize=8, label=fam)
ax.legend(fontsize=7, loc='lower right', ncol=2)

# (b) Saccade trajectory (simulated)
ax = axes[0, 1]
ax.set_title('Peripheral vs Foveal Accuracy', fontsize=11, fontweight='bold')
labels = ['With Fovea', 'Without Fovea']
values = [mse_with_fovea, mse_without_fovea]
colors = ['#4A90D9', '#95A5A6']
bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Belief MSE')
ax.grid(True, alpha=0.2, axis='y')
improvement = (mse_without_fovea - mse_with_fovea) / mse_without_fovea * 100
ax.text(0.98, 0.95, f'Fovea improves by {improvement:.1f}%',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        fontweight='bold', color='#4A90D9',
        bbox=dict(boxstyle='round', facecolor='#E8F0FE', alpha=0.8))

# (c) Scheduling activity heatmap
ax = axes[1, 0]
ax.set_title('Region Scheduling Activity', fontsize=11, fontweight='bold')
region_names_sorted = sorted(program.regions.keys())
activity_matrix = np.zeros((len(region_names_sorted), len(schedule_history)))
for t, step_active in enumerate(schedule_history):
    for i, rname in enumerate(region_names_sorted):
        activity_matrix[i, t] = 1.0 if step_active.get(rname, False) else 0.0
# Color by family
family_colors_matrix = np.zeros((len(region_names_sorted), len(schedule_history), 3))
for i, rname in enumerate(region_names_sorted):
    rp = program.regions.get(rname, RegionProgram())
    color = FAMILY_COLORS.get(rp.family, '#95A5A6')
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    for t in range(len(schedule_history)):
        if activity_matrix[i, t] > 0:
            family_colors_matrix[i, t] = [r, g, b]
        else:
            family_colors_matrix[i, t] = [0.95, 0.95, 0.95]
ax.imshow(family_colors_matrix, aspect='auto', interpolation='nearest')
ax.set_yticks(range(len(region_names_sorted)))
ax.set_yticklabels(region_names_sorted, fontsize=7)
ax.set_xlabel('Timestep')
ax.set_ylabel('Region')

# (d) Training curves
ax = axes[1, 1]
ax.set_title('Training Loss', fontsize=11, fontweight='bold')
ax.semilogy(losses, color='#2C3E50', lw=1.5, alpha=0.5)
w = 20
smoothed = np.convolve(losses, np.ones(w)/w, mode='valid')
ax.semilogy(range(w-1, len(losses)), smoothed, color='#E74C3C', lw=2, label='smoothed')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "14_saccading_vision.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")

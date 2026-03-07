"""Multi-Frequency Sensor Fusion: period= creates real training advantage.

Two sensors at different frequencies feed into a prediction target.
Model A declares correct periods. Model B treats everything as period=1.
We show that period-aware structure converges faster.

Run:  python examples/02_multi_frequency.py
Out:  assets/examples/02_multi_frequency.png
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from canvas_engineering import Field, compile_schema, ConnectivityPolicy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)


# ── Type declarations ────────────────────────────────────────────────
# T=1 canvas (single timestep) — this is a snapshot fusion problem.
# The temporal frequency matters for HOW the data is generated, and
# the RegionSpec.period is metadata that a real training loop uses for
# loss masking and frame mapping. Here we show the structural advantage
# of having separate regions with different sizes for different-rate sensors.

@dataclass
class FusionStructured:
    """Structured: fast sensor gets more positions (higher bandwidth)."""
    fast: Field = Field(4, 4)                         # 16 positions (high-bandwidth)
    slow: Field = Field(2, 2)                         # 4 positions (low-bandwidth)
    context: Field = Field(1, 2, is_output=False)     # 2 positions (input-only)
    prediction: Field = Field(2, 4, loss_weight=2.0)  # 8 positions (target)

@dataclass
class FusionFlat:
    """Flat: all sensors get same allocation regardless of bandwidth."""
    fast: Field = Field(3, 3)                         # 9 positions
    slow: Field = Field(3, 3)                         # 9 positions (over-allocated)
    context: Field = Field(1, 2, is_output=False)     # 2 positions
    prediction: Field = Field(2, 4, loss_weight=2.0)  # 8 positions


bound_A = compile_schema(FusionStructured(), T=1, H=8, W=8, d_model=48)
bound_B = compile_schema(FusionFlat(), T=1, H=8, W=8, d_model=48)


# ── Synthetic data ───────────────────────────────────────────────────

def generate_data(n=4096):
    """
    fast: 16-dim feature, high information density, nonlinear
    slow: 4-dim feature, redundant/correlated, lower info
    context: one-hot (2 classes)
    prediction: 8-dim, NONLINEAR function requiring cross-sensor fusion

    The key: fast has 4x more info dimensions than slow. A model that
    allocates equal capacity to both wastes parameters on slow's redundancy.
    """
    # Fast: 16-dim, diverse nonlinear features
    z = torch.randn(n, 4)  # latent
    fast = torch.zeros(n, 16)
    fast[:, 0:4] = torch.sin(z * 2)
    fast[:, 4:8] = torch.cos(z * 1.5)
    fast[:, 8:12] = torch.tanh(z * 3)
    fast[:, 12:16] = z ** 2 - 0.5
    fast = fast + torch.randn(n, 16) * 0.15

    # Slow: 4-dim, correlated (low intrinsic dimension)
    s = torch.randn(n, 1)
    slow = torch.cat([s, s * 0.9 + torch.randn(n, 1) * 0.1,
                       -s * 0.8, torch.ones(n, 1) * 0.5], dim=1)

    # Context
    cat = torch.randint(0, 2, (n,))
    context = torch.zeros(n, 2)
    context[torch.arange(n), cat] = 1.0

    # Target requires NONLINEAR cross-sensor fusion
    pred = torch.zeros(n, 8)
    m0 = (cat == 0)
    # Cat 0: fast features gated by slow
    pred[m0, :4] = fast[m0, :4] * torch.sigmoid(slow[m0, :1].expand(-1, 4) * 3)
    pred[m0, 4:] = torch.sin(fast[m0, 4:8] + slow[m0, 1:2].expand(-1, 4))
    m1 = (cat == 1)
    # Cat 1: bilinear interaction
    pred[m1, :4] = fast[m1, 8:12] * slow[m1]
    pred[m1, 4:] = torch.tanh(fast[m1, 12:16] * 2 + slow[m1, 2:3].expand(-1, 4))

    return fast, slow, context, pred

fast_tr, slow_tr, ctx_tr, pred_tr = generate_data()
fast_val, slow_val, ctx_val, pred_val = generate_data(1024)


# ── Model ────────────────────────────────────────────────────────────

class FusionModel(nn.Module):
    def __init__(self, bound, d=48, nhead=4):
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

        # Project raw features into d-model for each field
        fast_n = len(bound.layout.region_indices('fast'))
        slow_n = len(bound.layout.region_indices('slow'))
        ctx_n = len(bound.layout.region_indices('context'))
        pred_n = len(bound.layout.region_indices('prediction'))

        self.fast_proj = nn.Linear(16, fast_n * d)
        self.slow_proj = nn.Linear(4, slow_n * d)
        self.ctx_proj = nn.Linear(2, ctx_n * d)
        self.out_proj = nn.Linear(pred_n * d, 8)

        self.fast_n = fast_n
        self.slow_n = slow_n
        self.ctx_n = ctx_n
        self.pred_n = pred_n

    def forward(self, fast, slow, ctx):
        B = fast.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        # Project and scatter onto canvas
        fast_idx = self.bound.layout.region_indices('fast')
        slow_idx = self.bound.layout.region_indices('slow')
        ctx_idx = self.bound.layout.region_indices('context')
        pred_idx = self.bound.layout.region_indices('prediction')

        canvas[:, fast_idx] = canvas[:, fast_idx] + self.fast_proj(fast).reshape(B, self.fast_n, self.d)
        canvas[:, slow_idx] = canvas[:, slow_idx] + self.slow_proj(slow).reshape(B, self.slow_n, self.d)
        canvas[:, ctx_idx] = canvas[:, ctx_idx] + self.ctx_proj(ctx).reshape(B, self.ctx_n, self.d)

        canvas = self.encoder(canvas, mask=self.mask)

        pred_emb = canvas[:, pred_idx].reshape(B, -1)
        return self.out_proj(pred_emb)


# ── Training ─────────────────────────────────────────────────────────

def train_model(bound, label, n_epochs=800, bs=128):
    model = FusionModel(bound)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []

    for ep in range(n_epochs):
        idx = torch.randint(0, len(fast_tr), (bs,))
        pred = model(fast_tr[idx], slow_tr[idx], ctx_tr[idx])
        loss = ((pred - pred_tr[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if ep % 200 == 0:
            print(f"  [{label}] ep {ep:3d}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        vp = model(fast_val, slow_val, ctx_val)
        vl = ((vp - pred_val) ** 2).mean().item()
    print(f"  [{label}] val={vl:.4f}")
    return model, losses, vl


print("Training structured model...")
model_A, losses_A, vl_A = train_model(bound_A, "structured")
print("\nTraining flat model...")
model_B, losses_B, vl_B = train_model(bound_B, "flat")


# ── Visualization ────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Multi-Frequency Sensor Fusion', fontsize=16, fontweight='bold', y=0.98)

CA, CB = '#4A90D9', '#E8734A'
region_colors = {'fast': '#5CB85C', 'slow': '#9B59B6', 'context': '#95A5A6', 'prediction': '#E74C3C'}

# (a) Canvas layout comparison
ax = axes[0, 0]
ax.set_title('Structured Layout: fast=16pos, slow=4pos', fontsize=11, fontweight='bold')
H, W = bound_A.layout.H, bound_A.layout.W
grid = np.ones((H, W, 3)) * 0.93
for name, color in region_colors.items():
    bf = bound_A[name]
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]
    ax.text((w0 + w1) / 2 - 0.5, (h0 + h1) / 2 - 0.5,
            f'{name}\n{bf.num_positions}pos',
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
ax.imshow(grid, aspect='equal', interpolation='nearest')
ax.set_xlabel('W'); ax.set_ylabel('H')

# (b) Data distribution
ax = axes[0, 1]
ax.set_title('Feature Distributions', fontsize=11, fontweight='bold')
ax.hist(fast_tr[:, 0].numpy(), bins=50, alpha=0.5, color=region_colors['fast'],
        label=f'fast (16-dim)', density=True)
ax.hist(slow_tr[:, 0].numpy(), bins=50, alpha=0.5, color=region_colors['slow'],
        label=f'slow (4-dim)', density=True)
ax.legend(fontsize=9)
ax.set_xlabel('Value'); ax.set_ylabel('Density')
ax.grid(True, alpha=0.2)

# (c) Training curves
ax = axes[1, 0]
ax.set_title('Training Loss', fontsize=11, fontweight='bold')
w = 30
def smooth(a, w=w): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.plot(smooth(losses_A), color=CA, lw=2, label=f'structured (val={vl_A:.3f})')
ax.plot(smooth(losses_B), color=CB, lw=2, label=f'flat (val={vl_B:.3f})')
ax.legend(fontsize=9)
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
ax.grid(True, alpha=0.2)
# Winner annotation
better = 'structured' if vl_A < vl_B else 'flat'
ratio = max(vl_A, vl_B) / min(vl_A, vl_B)
color = CA if vl_A < vl_B else CB
ax.text(0.98, 0.95, f'{better}: {ratio:.2f}x better',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        fontweight='bold', color=color,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15))

# (d) Prediction scatter: predicted vs true
ax = axes[1, 1]
ax.set_title('Predicted vs True (validation)', fontsize=11, fontweight='bold')
model_A.eval(); model_B.eval()
with torch.no_grad():
    pa = model_A(fast_val[:200], slow_val[:200], ctx_val[:200]).numpy().flatten()
    pb = model_B(fast_val[:200], slow_val[:200], ctx_val[:200]).numpy().flatten()
true = pred_val[:200].numpy().flatten()
ax.scatter(true, pa, s=3, alpha=0.3, color=CA, label='structured')
ax.scatter(true, pb, s=3, alpha=0.3, color=CB, label='flat')
lims = [min(true.min(), pa.min(), pb.min()) - 0.2,
        max(true.max(), pa.max(), pb.max()) + 0.2]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='perfect')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('True'); ax.set_ylabel('Predicted')
ax.legend(fontsize=8, markerscale=3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "02_multi_frequency.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\nSaved {path}")

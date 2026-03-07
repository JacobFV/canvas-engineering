"""Hello Canvas Types: declare, compile, train, visualize.

Three signal fields. One is the product of the other two.
A 1-layer transformer learns to combine them through the canvas.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from dataclasses import dataclass
from canvas_engineering import Field, compile_schema

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)


# ── 1. Declare types ─────────────────────────────────────────────────

@dataclass
class SignalMixer:
    signal_a: Field = Field(1, 4)                        # 4 positions
    signal_b: Field = Field(1, 4)                        # 4 positions
    output: Field = Field(1, 4, loss_weight=2.0)         # predicted product


# ── 2. Compile ────────────────────────────────────────────────────────

bound = compile_schema(SignalMixer(), T=4, H=4, W=4, d_model=64)
layout = bound.layout
print(bound.summary())


# ── 3. Generate synthetic data ────────────────────────────────────────

T_CANVAS = 4  # match canvas T

def generate_data(n_samples=1024, T=T_CANVAS):
    """signal_a = sin, signal_b = cos, output = a * b."""
    t = torch.linspace(0, 2 * math.pi, T).unsqueeze(0).expand(n_samples, T)
    freqs_a = torch.rand(n_samples, 1) * 2 + 0.5
    freqs_b = torch.rand(n_samples, 1) * 2 + 0.5
    phases = torch.rand(n_samples, 1) * 2 * math.pi

    a = torch.sin(freqs_a * t + phases)
    b = torch.cos(freqs_b * t)
    out = a * b

    # Expand to (N, T, 4) — 4 channels per timestep
    def expand(x):
        x = x.unsqueeze(-1).expand(-1, -1, 4)
        return x + torch.randn_like(x) * 0.01

    return expand(a), expand(b), expand(out)

a_data, b_data, out_data = generate_data()
a_val, b_val, out_val = generate_data(256)


# ── 4. Build a tiny transformer on the canvas ─────────────────────────

class CanvasTransformer(nn.Module):
    """1-layer transformer that operates on the compiled canvas."""

    def __init__(self, bound_schema, d_model=64, nhead=4):
        super().__init__()
        self.bound = bound_schema
        self.d = d_model
        n_pos = bound_schema.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, n_pos, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.proj_in = nn.Linear(1, d_model)    # scalar -> d_model
        self.proj_out = nn.Linear(d_model, 1)   # d_model -> scalar

        # Attention mask from topology — handles unused grid positions
        mask = bound_schema.topology.to_additive_mask(bound_schema.layout)
        self.register_buffer('attn_mask', mask)

    def forward(self, a, b):
        """a, b: (batch, T, 4) -> predicted output (batch, T, 4).

        Each field has T*4 = 64 canvas positions. We project each of the
        T*4 input scalars into d_model, place on canvas, attend, extract.
        """
        B, T, W = a.shape
        N = self.bound.layout.num_positions

        # Create canvas
        canvas = torch.zeros(B, N, self.d, device=a.device)
        canvas = canvas + self.pos_emb

        # Place fields: flatten (T, W) -> (T*W,) to match region indices
        a_idx = self.bound.layout.region_indices("signal_a")
        b_idx = self.bound.layout.region_indices("signal_b")
        out_idx = self.bound.layout.region_indices("output")

        # (B, T, W) -> (B, T*W, 1) -> proj_in -> (B, T*W, d_model)
        a_flat = self.proj_in(a.reshape(B, -1).unsqueeze(-1))  # (B, T*W, d)
        b_flat = self.proj_in(b.reshape(B, -1).unsqueeze(-1))

        canvas[:, a_idx] = canvas[:, a_idx] + a_flat[:, :len(a_idx)]
        canvas[:, b_idx] = canvas[:, b_idx] + b_flat[:, :len(b_idx)]

        # Transformer with topology mask
        canvas = self.encoder(canvas, mask=self.attn_mask)

        # Extract output -> project back to scalar per position
        out_emb = canvas[:, out_idx]               # (B, T*W, d)
        pred_flat = self.proj_out(out_emb)          # (B, T*W, 1)
        pred = pred_flat.squeeze(-1).reshape(B, T, W)  # (B, T, W)
        return pred


model = CanvasTransformer(bound)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
weight_mask = bound.layout.loss_weight_mask("cpu")


# ── 5. Train ──────────────────────────────────────────────────────────

losses = []
n_epochs = 500
batch_size = 128

for epoch in range(n_epochs):
    idx = torch.randint(0, len(a_data), (batch_size,))
    a_batch = a_data[idx]
    b_batch = b_data[idx]
    target = out_data[idx]

    pred = model(a_batch, b_batch)   # (B, T, 4)
    loss = ((pred - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 50 == 0:
        print(f"  epoch {epoch:3d}: loss = {loss.item():.4f}")


# ── 6. Evaluate ───────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    pred_val = model(a_val, b_val)  # (128, T, 4)
    val_loss = ((pred_val - out_val) ** 2).mean().item()
    print(f"\n  validation loss: {val_loss:.4f}")


# ── 7. Visualize ──────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Canvas Types: Signal Mixer', fontsize=16, fontweight='bold', y=0.98)

COLORS = {'signal_a': '#4A90D9', 'signal_b': '#E8734A', 'output': '#5CB85C'}

# (a) Canvas layout
ax = axes[0, 0]
ax.set_title('Canvas Layout', fontsize=12, fontweight='bold')
grid = np.zeros((bound.layout.H, bound.layout.W, 3))
for name, color in COLORS.items():
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    bf = bound[name]
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]

ax.imshow(grid, aspect='auto', interpolation='nearest')
for name, color in COLORS.items():
    bf = bound[name]
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    ax.text((w0 + w1) / 2 - 0.5, (h0 + h1) / 2 - 0.5, name.replace('_', '\n'),
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
ax.set_xlabel('W')
ax.set_ylabel('H')

# (b) Input signals
ax = axes[0, 1]
ax.set_title('Input Signals (sample 0)', fontsize=12, fontweight='bold')
t = np.arange(T_CANVAS)
ax.plot(t, a_val[0, :, 0].numpy(), color=COLORS['signal_a'], label='signal_a', lw=2)
ax.plot(t, b_val[0, :, 0].numpy(), color=COLORS['signal_b'], label='signal_b', lw=2)
ax.plot(t, out_val[0, :, 0].numpy(), color=COLORS['output'], label='true output', lw=2, ls='--')
ax.legend(fontsize=8)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.grid(True, alpha=0.2)

# (c) Predicted vs true
ax = axes[1, 0]
ax.set_title('Prediction vs Ground Truth', fontsize=12, fontweight='bold')
with torch.no_grad():
    single_pred = model(a_val[:1], b_val[:1])  # (1, T, 4)
true_plot = out_val[0, :, 0].numpy()
pred_plot = single_pred[0, :, 0].numpy()
ax.plot(t, true_plot, color=COLORS['output'], label='true', lw=2, ls='--')
ax.plot(t, pred_plot, color='#E74C3C', label='predicted', lw=2)
ax.fill_between(t, true_plot, pred_plot, alpha=0.15, color='#E74C3C')
ax.legend(fontsize=8)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.grid(True, alpha=0.2)

# (d) Training loss
ax = axes[1, 1]
ax.set_title('Training Loss', fontsize=12, fontweight='bold')
ax.semilogy(losses, color='#2C3E50', lw=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "01_hello_types.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")

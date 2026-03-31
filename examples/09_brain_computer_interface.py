"""Brain-computer interface: neural decoding for paralyzed patients.

Decode motor intent from intracortical electrode arrays into cursor control
and speech tokens simultaneously. Three cortical area arrays (M1, PMd, S1)
feed through a transformer on the canvas to produce multi-modal outputs.

Synthetic data simulates neural spike trains and local field potentials
that encode cursor velocity and phoneme sequences via noisy nonlinear
mappings. The model learns to decode both modalities from shared
neural representations.

Run:  python examples/09_brain_computer_interface.py
Out:  assets/examples/09_brain_computer_interface.png
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field as dc_field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from canvas_engineering import Field, compile_schema, ConnectivityPolicy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)

# ── 1. Declare types ─────────────────────────────────────────────────

@dataclass
class ElectrodeArray:
    """One Utah array implanted in a cortical area."""
    spikes: Field = Field(3, 4, period=1,
                          semantic_type="binned spike counts 12ch 50ms bins")
    lfp: Field = Field(2, 2, period=1,
                       semantic_type="LFP bandpower features")


@dataclass
class CursorDecoder:
    """2D cursor control output."""
    velocity: Field = Field(1, 2, period=1, loss_weight=5.0,
                            semantic_type="decoded cursor velocity 2D")
    click: Field = Field(1, 1, period=1, loss_weight=3.0,
                         semantic_type="decoded click probability")


@dataclass
class SpeechDecoder:
    """Speech token decoding."""
    phonemes: Field = Field(2, 4, period=1, loss_weight=4.0,
                            semantic_type="decoded phoneme probabilities")


@dataclass
class BCISystem:
    """Intracortical brain-computer interface."""
    m1: ElectrodeArray = dc_field(default_factory=ElectrodeArray)
    pmd: ElectrodeArray = dc_field(default_factory=ElectrodeArray)
    s1: ElectrodeArray = dc_field(default_factory=ElectrodeArray)

    cursor_feedback: Field = Field(1, 2, is_output=False, period=1,
                                   semantic_type="visual cursor position feedback")

    cursor: CursorDecoder = dc_field(default_factory=CursorDecoder)
    speech: SpeechDecoder = dc_field(default_factory=SpeechDecoder)

    intent: Field = Field(2, 4, period=1, loss_weight=2.0,
                          semantic_type="decoded motor intent")


# ── 2. Compile ────────────────────────────────────────────────────────

bci = BCISystem()
bound = compile_schema(
    bci, T=1, d_model=48,
    connectivity=ConnectivityPolicy(
        intra="dense",
        temporal="same_frame",
    ),
)
print("=== Brain-Computer Interface ===")
print(bound.summary())


# ── 3. Generate synthetic data ────────────────────────────────────────
# Simulated neural activity: spike counts + LFP bandpower
# that encode cursor velocity and phoneme sequences.

SPIKE_DIM = 12  # per array
LFP_DIM = 4    # per array
CURSOR_DIM = 2
CLICK_DIM = 1
PHONEME_DIM = 8
FEEDBACK_DIM = 2

def generate_bci_data(n_samples=2048):
    """Generate neural recordings -> cursor + speech targets."""
    # Latent motor intent: 8-dim
    intent = torch.randn(n_samples, 8) * 0.5

    # Cursor velocity: linear decode from intent + noise
    cursor_vel = intent[:, :2] * 1.5 + torch.randn(n_samples, 2) * 0.1

    # Click: sigmoid of intent magnitude
    click_prob = torch.sigmoid(intent[:, 2:3] * 2 - 0.5)
    click = (click_prob > 0.5).float()

    # Phoneme targets: softmax over 8 classes, driven by intent
    W_phon = torch.randn(8, 8) * 0.4
    phoneme_logits = intent @ W_phon
    phonemes = torch.softmax(phoneme_logits, dim=-1)

    # Neural recordings: noisy nonlinear encoding of intent
    # M1 (primary motor): best for hand kinematics
    W_m1_spike = torch.randn(8, SPIKE_DIM) * 0.5
    m1_spikes = torch.relu(intent @ W_m1_spike + torch.randn(n_samples, SPIKE_DIM) * 0.3)
    W_m1_lfp = torch.randn(8, LFP_DIM) * 0.3
    m1_lfp = torch.tanh(intent @ W_m1_lfp) + torch.randn(n_samples, LFP_DIM) * 0.2

    # PMd (dorsal premotor): best for reach planning
    W_pmd_spike = torch.randn(8, SPIKE_DIM) * 0.4
    pmd_spikes = torch.relu(intent @ W_pmd_spike + 0.5 + torch.randn(n_samples, SPIKE_DIM) * 0.35)
    W_pmd_lfp = torch.randn(8, LFP_DIM) * 0.3
    pmd_lfp = torch.sin(intent @ W_pmd_lfp * 2) + torch.randn(n_samples, LFP_DIM) * 0.2

    # S1 (somatosensory): sensory feedback
    W_s1_spike = torch.randn(8, SPIKE_DIM) * 0.3
    s1_spikes = torch.relu(intent @ W_s1_spike - 0.3 + torch.randn(n_samples, SPIKE_DIM) * 0.4)
    W_s1_lfp = torch.randn(8, LFP_DIM) * 0.25
    s1_lfp = torch.tanh(intent @ W_s1_lfp * 1.5) + torch.randn(n_samples, LFP_DIM) * 0.25

    # Cursor feedback (previous cursor position, input-only)
    feedback = cursor_vel * 0.8 + torch.randn(n_samples, 2) * 0.05

    return {
        'm1_spikes': m1_spikes, 'm1_lfp': m1_lfp,
        'pmd_spikes': pmd_spikes, 'pmd_lfp': pmd_lfp,
        's1_spikes': s1_spikes, 's1_lfp': s1_lfp,
        'feedback': feedback,
        'cursor_vel': cursor_vel, 'click': click,
        'phonemes': phonemes, 'intent': intent,
    }

data_tr = generate_bci_data()
data_val = generate_bci_data(512)


# ── 4. Build transformer on the canvas ────────────────────────────────

class BCITransformer(nn.Module):
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

        # Input projections
        self.region_projs = nn.ModuleDict()
        self.region_sizes = {}
        input_specs = {
            'm1.spikes': SPIKE_DIM, 'm1.lfp': LFP_DIM,
            'pmd.spikes': SPIKE_DIM, 'pmd.lfp': LFP_DIM,
            's1.spikes': SPIKE_DIM, 's1.lfp': LFP_DIM,
            'cursor_feedback': FEEDBACK_DIM,
        }
        for rname, in_dim in input_specs.items():
            n = len(bound_schema.layout.region_indices(rname))
            self.region_projs[rname.replace('.', '_')] = nn.Linear(in_dim, n * d_model)
            self.region_sizes[rname] = n

        # Output projections
        cursor_n = len(bound_schema.layout.region_indices("cursor.velocity"))
        click_n = len(bound_schema.layout.region_indices("cursor.click"))
        phon_n = len(bound_schema.layout.region_indices("speech.phonemes"))

        self.cursor_out = nn.Linear(cursor_n * d_model, CURSOR_DIM)
        self.click_out = nn.Linear(click_n * d_model, CLICK_DIM)
        self.phoneme_out = nn.Linear(phon_n * d_model, PHONEME_DIM)

        self.cursor_n = cursor_n
        self.click_n = click_n
        self.phon_n = phon_n

    def forward(self, data):
        B = data['m1_spikes'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        input_map = {
            'm1.spikes': 'm1_spikes', 'm1.lfp': 'm1_lfp',
            'pmd.spikes': 'pmd_spikes', 'pmd.lfp': 'pmd_lfp',
            's1.spikes': 's1_spikes', 's1.lfp': 's1_lfp',
            'cursor_feedback': 'feedback',
        }
        for rname, dkey in input_map.items():
            idx = self.bound.layout.region_indices(rname)
            proj = self.region_projs[rname.replace('.', '_')]
            n = self.region_sizes[rname]
            canvas[:, idx] = canvas[:, idx] + proj(data[dkey]).reshape(B, n, self.d)

        canvas = self.encoder(canvas, mask=self.attn_mask)

        cur_idx = self.bound.layout.region_indices("cursor.velocity")
        clk_idx = self.bound.layout.region_indices("cursor.click")
        phn_idx = self.bound.layout.region_indices("speech.phonemes")

        cursor_pred = self.cursor_out(canvas[:, cur_idx].reshape(B, -1))
        click_pred = torch.sigmoid(self.click_out(canvas[:, clk_idx].reshape(B, -1)))
        phoneme_pred = torch.softmax(self.phoneme_out(canvas[:, phn_idx].reshape(B, -1)), dim=-1)

        return cursor_pred, click_pred, phoneme_pred


model = BCITransformer(bound)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)


# ── 5. Train ──────────────────────────────────────────────────────────

losses_total = []
losses_cursor = []
losses_speech = []
n_epochs = 500
batch_size = 128

print("\nTraining BCI decoder...")
for epoch in range(n_epochs):
    idx = torch.randint(0, len(data_tr['m1_spikes']), (batch_size,))
    batch = {k: v[idx] for k, v in data_tr.items()}
    cursor_pred, click_pred, phoneme_pred = model(batch)

    cursor_loss = ((cursor_pred - batch['cursor_vel']) ** 2).mean() * 5.0
    click_loss = nn.functional.binary_cross_entropy(click_pred, batch['click']) * 3.0
    phoneme_loss = -((batch['phonemes'] * torch.log(phoneme_pred + 1e-8)).sum(dim=-1)).mean() * 4.0

    loss = cursor_loss + click_loss + phoneme_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()

    losses_total.append(loss.item())
    losses_cursor.append(cursor_loss.item())
    losses_speech.append(phoneme_loss.item())

    if epoch % 100 == 0:
        print(f"  epoch {epoch:3d}: total={loss.item():.4f} cursor={cursor_loss.item():.4f} speech={phoneme_loss.item():.4f}")


# ── 6. Evaluate ───────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    cursor_pred, click_pred, phoneme_pred = model(data_val)
    val_cursor_mse = ((cursor_pred - data_val['cursor_vel']) ** 2).mean().item()
    val_click_acc = ((click_pred > 0.5).float() == data_val['click']).float().mean().item()

    # Phoneme accuracy: top-1
    pred_phon = phoneme_pred.argmax(dim=-1)
    true_phon = data_val['phonemes'].argmax(dim=-1)
    val_phon_acc = (pred_phon == true_phon).float().mean().item()

    print(f"\n  Cursor MSE: {val_cursor_mse:.4f}")
    print(f"  Click accuracy: {val_click_acc:.3f}")
    print(f"  Phoneme top-1 accuracy: {val_phon_acc:.3f}")


# ── 7. Visualize ──────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Brain-Computer Interface: Neural Decoding', fontsize=16, fontweight='bold', y=0.98)

COLORS = {
    'm1': '#E74C3C', 'pmd': '#3498DB', 's1': '#2ECC71',
    'cursor': '#E67E22', 'speech': '#9B59B6',
}

# (a) Canvas layout
ax = axes[0, 0]
ax.set_title('Canvas Layout', fontsize=11, fontweight='bold')
H, W = bound.layout.H, bound.layout.W
grid = np.ones((H, W, 3)) * 0.93
region_colors = {
    'm1.spikes': '#E74C3C', 'm1.lfp': '#EC7063',
    'pmd.spikes': '#3498DB', 'pmd.lfp': '#5DADE2',
    's1.spikes': '#2ECC71', 's1.lfp': '#58D68D',
    'cursor.velocity': '#E67E22', 'cursor.click': '#F39C12',
    'speech.phonemes': '#9B59B6',
    'cursor_feedback': '#95A5A6', 'intent': '#34495E',
}
for name, color in region_colors.items():
    if name not in bound:
        continue
    bf = bound[name]
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]
    label = name.split(".")[-1] if "." in name else name
    ax.text((w0 + w1) / 2 - 0.5, (h0 + h1) / 2 - 0.5,
            label, ha='center', va='center', fontsize=5, fontweight='bold', color='white')
ax.imshow(grid, aspect='equal', interpolation='nearest')
ax.set_xlabel('W'); ax.set_ylabel('H')

# (b) Neural-to-cursor accuracy scatter
ax = axes[0, 1]
ax.set_title('Cursor Decoding: Predicted vs True', fontsize=11, fontweight='bold')
with torch.no_grad():
    cp, _, _ = model(data_val)
true_vx = data_val['cursor_vel'][:200, 0].numpy()
true_vy = data_val['cursor_vel'][:200, 1].numpy()
pred_vx = cp[:200, 0].numpy()
pred_vy = cp[:200, 1].numpy()
ax.scatter(true_vx, pred_vx, s=8, alpha=0.4, color=COLORS['cursor'], label='vx')
ax.scatter(true_vy, pred_vy, s=8, alpha=0.4, color=COLORS['speech'], label='vy')
lims = [-2.5, 2.5]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('True velocity')
ax.set_ylabel('Predicted velocity')
ax.legend(fontsize=9, markerscale=2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.text(0.02, 0.98, f'MSE={val_cursor_mse:.4f}',
        transform=ax.transAxes, ha='left', va='top', fontsize=10,
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (c) Multi-decoder comparison (bar chart)
ax = axes[1, 0]
ax.set_title('Multi-Decoder Performance', fontsize=11, fontweight='bold')
metrics = ['Cursor MSE\n(lower=better)', 'Click Acc\n(higher=better)', 'Phoneme Acc\n(higher=better)']
values = [val_cursor_mse, val_click_acc, val_phon_acc]
colors = [COLORS['cursor'], '#F39C12', COLORS['speech']]
bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(values) * 1.25)
ax.grid(True, alpha=0.2, axis='y')

# (d) Training curves
ax = axes[1, 1]
ax.set_title('Training Curves', fontsize=11, fontweight='bold')
w = 20
def smooth(a): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.semilogy(smooth(losses_cursor), color=COLORS['cursor'], lw=2, label='cursor loss')
ax.semilogy(smooth(losses_speech), color=COLORS['speech'], lw=2, label='speech loss')
ax.semilogy(smooth(losses_total), color='#2C3E50', lw=2, ls='--', label='total loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "09_brain_computer_interface.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")

"""Tokamak fusion reactor: disruption prediction and plasma control.

Model a tokamak's diagnostic → control pipeline on the canvas. Multiple
sensor systems at different frequencies (magnetic probes at period=1,
Thomson scattering at period=4) feed a disruption predictor with 10x
loss weight (a missed disruption costs $100M in reactor damage).

Synthetic data simulates plasma diagnostic time series with disruption
precursors. The model learns to predict disruptions from diagnostic
history and to output appropriate control actuator commands.

Run:  python examples/10_nuclear_fusion_reactor.py
Out:  assets/examples/10_nuclear_fusion_reactor.png
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
class MagneticDiagnostics:
    """Fast magnetic probe measurements."""
    probes: Field = Field(3, 4, period=1,
                          semantic_type="magnetic probe array 12ch 1kHz")
    flux_loops: Field = Field(2, 2, period=1,
                              semantic_type="flux loop measurements 4ch")


@dataclass
class ThomsonScattering:
    """Slower electron temperature and density profiles."""
    te_profile: Field = Field(2, 4, period=4,
                              semantic_type="Thomson Te profile 8 points")
    ne_profile: Field = Field(2, 4, period=4,
                              semantic_type="Thomson ne profile 8 points")


@dataclass
class Tokamak:
    """Tokamak fusion reactor control system."""
    magnetics: MagneticDiagnostics = dc_field(default_factory=MagneticDiagnostics)
    thomson: ThomsonScattering = dc_field(default_factory=ThomsonScattering)

    equilibrium: Field = Field(3, 4, period=1, loss_weight=2.0,
                               semantic_type="magnetic equilibrium state")

    machine_config: Field = Field(1, 4, is_output=False,
                                  semantic_type="tokamak geometry parameters")

    # Actuator outputs
    heating: Field = Field(1, 4, period=1, loss_weight=2.0,
                           semantic_type="auxiliary heating power setpoints")
    coil_currents: Field = Field(2, 4, period=1, loss_weight=3.0,
                                 semantic_type="PF coil current setpoints")

    # Disruption prediction (safety critical)
    disruption_risk: Field = Field(1, 4, period=1, loss_weight=10.0,
                                   semantic_type="disruption probability")
    disruption_class: Field = Field(1, 4, period=1, loss_weight=5.0,
                                    semantic_type="disruption type classification")


# ── 2. Compile ────────────────────────────────────────────────────────

reactor = Tokamak()
bound = compile_schema(
    reactor, T=1, d_model=48,
    connectivity=ConnectivityPolicy(
        intra="dense",
        temporal="same_frame",
    ),
)
print("=== Tokamak Fusion Reactor Control ===")
print(bound.summary())

# Show declared temporal rates (period= metadata)
print("\nDeclared diagnostic timescales (period= metadata):")
for period in [1, 4]:
    fields_at_p = [(n, bf) for n, bf in bound.fields.items()
                   if bf.spec.period == period]
    if fields_at_p:
        names = [n.split(".")[-1] for n, _ in fields_at_p]
        print(f"  period={period}: {', '.join(names)}")


# ── 3. Generate synthetic data ────────────────────────────────────────
# Plasma diagnostics with disruption precursors.
# Normal plasma: smooth evolution. Pre-disruption: growing oscillations.

PROBE_DIM = 12
FLUX_DIM = 4
TE_DIM = 8
NE_DIM = 8
CONFIG_DIM = 4
EQUIL_DIM = 12
HEAT_DIM = 4
COIL_DIM = 8
DISRUPT_DIM = 4
DCLASS_DIM = 4

def generate_tokamak_data(n_samples=2048):
    """Generate plasma diagnostics -> disruption labels + control targets."""
    # Machine config (input-only context)
    config = torch.randn(n_samples, CONFIG_DIM) * 0.2

    # Disruption labels: ~20% disruption rate
    is_disruption = (torch.rand(n_samples) < 0.2).float()
    # Disruption class: 4 types (VDE, TQ, CQ, locked-mode)
    dclass_idx = torch.randint(0, 4, (n_samples,))
    dclass = torch.zeros(n_samples, DCLASS_DIM)
    dclass[torch.arange(n_samples), dclass_idx] = 1.0
    dclass = dclass * is_disruption.unsqueeze(1)

    # Disruption risk: continuous probability
    risk_base = is_disruption * (0.5 + 0.5 * torch.rand(n_samples))
    risk_base = risk_base + (1 - is_disruption) * torch.rand(n_samples) * 0.1
    disruption_risk = risk_base.unsqueeze(1).expand(-1, DISRUPT_DIM)
    disruption_risk = disruption_risk + torch.randn(n_samples, DISRUPT_DIM) * 0.05
    disruption_risk = disruption_risk.clamp(0, 1)

    # Normal plasma state
    plasma_state = torch.randn(n_samples, 8) * 0.5
    plasma_state = plasma_state + config @ torch.randn(CONFIG_DIM, 8) * 0.3

    # Pre-disruption: add growing oscillations
    disruption_signal = is_disruption.unsqueeze(1) * torch.randn(n_samples, 8) * 1.5

    full_state = plasma_state + disruption_signal

    # Generate diagnostics from plasma state
    W_probe = torch.randn(8, PROBE_DIM) * 0.4
    probes = full_state @ W_probe + torch.randn(n_samples, PROBE_DIM) * 0.2

    W_flux = torch.randn(8, FLUX_DIM) * 0.3
    flux = torch.tanh(full_state @ W_flux) + torch.randn(n_samples, FLUX_DIM) * 0.15

    W_te = torch.randn(8, TE_DIM) * 0.3
    te = torch.relu(full_state @ W_te + 1.0) + torch.randn(n_samples, TE_DIM) * 0.1

    W_ne = torch.randn(8, NE_DIM) * 0.3
    ne = torch.relu(full_state @ W_ne + 0.5) + torch.randn(n_samples, NE_DIM) * 0.1

    # Equilibrium: derived from state
    W_eq = torch.randn(8, EQUIL_DIM) * 0.3
    equil = torch.tanh(full_state @ W_eq) + torch.randn(n_samples, EQUIL_DIM) * 0.1

    # Control outputs: depends on state (and should respond to disruption risk)
    W_heat = torch.randn(8, HEAT_DIM) * 0.3
    heating = torch.sigmoid(full_state @ W_heat) * (1 - is_disruption.unsqueeze(1) * 0.8)

    W_coil = torch.randn(8, COIL_DIM) * 0.4
    coils = torch.tanh(full_state @ W_coil)
    # During disruption: coils respond more aggressively
    coils = coils + is_disruption.unsqueeze(1) * torch.randn(n_samples, COIL_DIM) * 0.5

    return {
        'probes': probes, 'flux': flux, 'te': te, 'ne': ne,
        'config': config, 'equil': equil,
        'heating': heating, 'coils': coils,
        'disruption_risk': disruption_risk, 'dclass': dclass,
        'is_disruption': is_disruption,
    }

data_tr = generate_tokamak_data()
data_val = generate_tokamak_data(512)


# ── 4. Build transformer on the canvas ────────────────────────────────

class TokamakTransformer(nn.Module):
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
        self.input_projs = nn.ModuleDict()
        self.input_sizes = {}
        specs = {
            'magnetics.probes': PROBE_DIM, 'magnetics.flux_loops': FLUX_DIM,
            'thomson.te_profile': TE_DIM, 'thomson.ne_profile': NE_DIM,
            'machine_config': CONFIG_DIM,
        }
        for rname, in_dim in specs.items():
            n = len(bound_schema.layout.region_indices(rname))
            self.input_projs[rname.replace('.', '_')] = nn.Linear(in_dim, n * d_model)
            self.input_sizes[rname] = n

        # Output projections
        self.output_projs = nn.ModuleDict()
        self.output_sizes = {}
        out_specs = {
            'equilibrium': EQUIL_DIM, 'heating': HEAT_DIM,
            'coil_currents': COIL_DIM,
            'disruption_risk': DISRUPT_DIM, 'disruption_class': DCLASS_DIM,
        }
        for rname, out_dim in out_specs.items():
            n = len(bound_schema.layout.region_indices(rname))
            self.output_projs[rname] = nn.Linear(n * d_model, out_dim)
            self.output_sizes[rname] = n

    def forward(self, data):
        B = data['probes'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        input_map = {
            'magnetics.probes': 'probes', 'magnetics.flux_loops': 'flux',
            'thomson.te_profile': 'te', 'thomson.ne_profile': 'ne',
            'machine_config': 'config',
        }
        for rname, dkey in input_map.items():
            idx = self.bound.layout.region_indices(rname)
            proj = self.input_projs[rname.replace('.', '_')]
            n = self.input_sizes[rname]
            canvas[:, idx] = canvas[:, idx] + proj(data[dkey]).reshape(B, n, self.d)

        canvas = self.encoder(canvas, mask=self.attn_mask)

        outputs = {}
        for rname, proj in self.output_projs.items():
            idx = self.bound.layout.region_indices(rname)
            n = self.output_sizes[rname]
            outputs[rname] = proj(canvas[:, idx].reshape(B, -1))

        # Apply activations
        outputs['disruption_risk'] = torch.sigmoid(outputs['disruption_risk'])
        outputs['disruption_class'] = torch.softmax(outputs['disruption_class'], dim=-1)
        outputs['heating'] = torch.sigmoid(outputs['heating'])

        return outputs


model = TokamakTransformer(bound)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)


# ── 5. Train ──────────────────────────────────────────────────────────

losses_total = []
losses_disruption = []
losses_control = []
n_epochs = 500
batch_size = 128

print("\nTraining tokamak controller...")
for epoch in range(n_epochs):
    idx = torch.randint(0, len(data_tr['probes']), (batch_size,))
    batch = {k: v[idx] for k, v in data_tr.items()}
    outputs = model(batch)

    # Disruption prediction loss (safety critical, high weight)
    risk_loss = nn.functional.binary_cross_entropy(
        outputs['disruption_risk'], batch['disruption_risk']) * 10.0
    class_loss = -((batch['dclass'] * torch.log(outputs['disruption_class'] + 1e-8)).sum(dim=-1)).mean() * 5.0

    # Control losses
    equil_loss = ((outputs['equilibrium'] - batch['equil']) ** 2).mean() * 2.0
    heat_loss = ((outputs['heating'] - batch['heating']) ** 2).mean() * 2.0
    coil_loss = ((outputs['coil_currents'] - batch['coils']) ** 2).mean() * 3.0

    disrupt_loss = risk_loss + class_loss
    control_loss = equil_loss + heat_loss + coil_loss
    loss = disrupt_loss + control_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()

    losses_total.append(loss.item())
    losses_disruption.append(disrupt_loss.item())
    losses_control.append(control_loss.item())

    if epoch % 100 == 0:
        print(f"  epoch {epoch:3d}: total={loss.item():.4f} disruption={disrupt_loss.item():.4f} control={control_loss.item():.4f}")


# ── 6. Evaluate ───────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    outputs_val = model(data_val)
    risk_pred = outputs_val['disruption_risk'].mean(dim=-1)
    true_labels = data_val['is_disruption']

    # ROC curve data
    thresholds = torch.linspace(0, 1, 200)
    tpr_list, fpr_list = [], []
    for th in thresholds:
        pred_pos = (risk_pred >= th).float()
        tp = (pred_pos * true_labels).sum().item()
        fp = (pred_pos * (1 - true_labels)).sum().item()
        fn = ((1 - pred_pos) * true_labels).sum().item()
        tn = ((1 - pred_pos) * (1 - true_labels)).sum().item()
        tpr_list.append(tp / max(tp + fn, 1))
        fpr_list.append(fp / max(fp + tn, 1))

    # AUC (trapezoidal)
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sort_idx]
    tpr_sorted = tpr_arr[sort_idx]
    auc = np.trapezoid(tpr_sorted, fpr_sorted)

    val_equil_mse = ((outputs_val['equilibrium'] - data_val['equil']) ** 2).mean().item()
    val_coil_mse = ((outputs_val['coil_currents'] - data_val['coils']) ** 2).mean().item()

    print(f"\n  Disruption AUC: {auc:.3f}")
    print(f"  Equilibrium MSE: {val_equil_mse:.4f}")
    print(f"  Coil control MSE: {val_coil_mse:.4f}")


# ── 7. Visualize ──────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Tokamak Fusion Reactor: Disruption Prediction & Control',
             fontsize=15, fontweight='bold', y=0.98)

COLORS = {
    'magnetics.probes': '#E74C3C', 'magnetics.flux_loops': '#EC7063',
    'thomson.te_profile': '#3498DB', 'thomson.ne_profile': '#5DADE2',
    'equilibrium': '#AF7AC5', 'machine_config': '#95A5A6',
    'heating': '#F39C12', 'coil_currents': '#E67E22',
    'disruption_risk': '#C0392B', 'disruption_class': '#922B21',
}

# (a) Canvas layout
ax = axes[0, 0]
ax.set_title('Canvas Layout (period in brackets)', fontsize=11, fontweight='bold')
H, W = bound.layout.H, bound.layout.W
grid = np.ones((H, W, 3)) * 0.93
for name, color in COLORS.items():
    if name not in bound:
        continue
    bf = bound[name]
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]
    label = name.split(".")[-1] if "." in name else name
    ax.text((w0 + w1) / 2 - 0.5, (h0 + h1) / 2 - 0.5,
            f'{label}\n[p={bf.spec.period}]',
            ha='center', va='center', fontsize=5, fontweight='bold', color='white')
ax.imshow(grid, aspect='equal', interpolation='nearest')
ax.set_xlabel('W'); ax.set_ylabel('H')

# (b) Disruption prediction ROC
ax = axes[0, 1]
ax.set_title(f'Disruption Prediction ROC (AUC={auc:.3f})', fontsize=11, fontweight='bold')
ax.plot(fpr_sorted, tpr_sorted, color='#C0392B', lw=2.5)
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='random')
ax.fill_between(fpr_sorted, tpr_sorted, alpha=0.15, color='#C0392B')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# (c) Control response: predicted vs true coil currents
ax = axes[1, 0]
ax.set_title('Control Response: Coil Currents', fontsize=11, fontweight='bold')
with torch.no_grad():
    coil_pred = outputs_val['coil_currents'][:100].numpy().flatten()
    coil_true = data_val['coils'][:100].numpy().flatten()
ax.scatter(coil_true, coil_pred, s=6, alpha=0.3, color='#E67E22')
lims = [min(coil_true.min(), coil_pred.min()) - 0.3,
        max(coil_true.max(), coil_pred.max()) + 0.3]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('True coil current')
ax.set_ylabel('Predicted coil current')
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.text(0.02, 0.98, f'MSE={val_coil_mse:.4f}',
        transform=ax.transAxes, ha='left', va='top', fontsize=10,
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (d) Training curves
ax = axes[1, 1]
ax.set_title('Training Curves', fontsize=11, fontweight='bold')
w = 20
def smooth(a): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.semilogy(smooth(losses_disruption), color='#C0392B', lw=2, label='disruption loss')
ax.semilogy(smooth(losses_control), color='#E67E22', lw=2, label='control loss')
ax.semilogy(smooth(losses_total), color='#2C3E50', lw=2, ls='--', label='total loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "10_nuclear_fusion_reactor.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")

"""Mars colony: autonomous multi-system coordination with cascading failure prediction.

An early Mars colony must handle emergencies autonomously (4-24 min comms delay
to Earth). The canvas type hierarchy mirrors the colony's subsystems:
  - Habitat (life support, power, thermal)
  - Greenhouse (food production)
  - ISRU plant (fuel/O2 from regolith)

Synthetic data simulates multi-system telemetry with cascading failures:
when one system degrades, it affects downstream systems. The model learns
cross-system correlations to predict failures before they cascade.

Run:  python examples/11_mars_colony.py
Out:  assets/examples/11_mars_colony.png
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
class LifeSupport:
    atmosphere: Field = Field(2, 2, loss_weight=5.0,
                              semantic_type="habitat atmosphere O2/CO2/pressure/humidity")
    water: Field = Field(1, 2, loss_weight=3.0,
                         semantic_type="water recycling system state")


@dataclass
class PowerSystem:
    generation: Field = Field(1, 4,
                              semantic_type="solar and RTG power output")
    storage: Field = Field(1, 2, loss_weight=2.0,
                           semantic_type="battery SoC and health")


@dataclass
class HabitatModule:
    life_support: LifeSupport = dc_field(default_factory=LifeSupport)
    power: PowerSystem = dc_field(default_factory=PowerSystem)
    structural: Field = Field(1, 2, loss_weight=4.0,
                              semantic_type="structural integrity seals pressure")


@dataclass
class Greenhouse:
    crops: Field = Field(2, 2,
                         semantic_type="crop growth state 4 bays")
    atmosphere: Field = Field(1, 4, loss_weight=2.0,
                              semantic_type="greenhouse CO2/O2/humidity for crops")
    harvest: Field = Field(1, 2, loss_weight=1.5,
                           semantic_type="predicted harvest schedule")


@dataclass
class ISRUPlant:
    feedstock: Field = Field(1, 2, is_output=False,
                             semantic_type="regolith feedstock level")
    electrolysis: Field = Field(1, 2, loss_weight=3.0,
                                semantic_type="O2 production rate")
    fuel_storage: Field = Field(1, 2, loss_weight=3.0,
                                semantic_type="propellant and O2 tank levels")


@dataclass
class MarsColony:
    situation: Field = Field(2, 4, loss_weight=3.0,
                             semantic_type="colony situation awareness state")
    alert_level: Field = Field(1, 4, loss_weight=8.0,
                               semantic_type="colony alert classification")
    weather: Field = Field(1, 4, is_output=False,
                           semantic_type="Mars weather dust/wind/temperature")

    habitat: HabitatModule = dc_field(default_factory=HabitatModule)
    greenhouse: Greenhouse = dc_field(default_factory=Greenhouse)
    isru: ISRUPlant = dc_field(default_factory=ISRUPlant)


# ── 2. Compile ────────────────────────────────────────────────────────

colony = MarsColony()
bound = compile_schema(
    colony, T=1, d_model=48,
    connectivity=ConnectivityPolicy(
        intra="dense",
        temporal="same_frame",
    ),
)
print("=== Mars Colony Autonomous Control ===")
print(bound.summary())


# ── 3. Generate synthetic data ────────────────────────────────────────
# Multi-system telemetry with cascading failures.
# Systems: power -> life_support -> greenhouse -> ISRU
# Failures propagate downstream.

# Dimension constants (matching Field sizes)
ATMO_DIM = 4
WATER_DIM = 2
GEN_DIM = 4
STOR_DIM = 2
STRUCT_DIM = 2
CROP_DIM = 4
GH_ATMO_DIM = 4
HARVEST_DIM = 2
FEED_DIM = 2
ELEC_DIM = 2
FUEL_DIM = 2
SIT_DIM = 8
ALERT_DIM = 4
WEATHER_DIM = 4

def generate_colony_data(n_samples=2048):
    """Generate multi-system telemetry with cascading failures."""
    # Weather (input context)
    weather = torch.randn(n_samples, WEATHER_DIM) * 0.3
    dust_storm = (torch.rand(n_samples) < 0.15).float()
    weather[:, 0] = weather[:, 0] + dust_storm * 2.0  # dust level spikes

    # Failure chain: power failure -> life support degraded -> greenhouse stressed
    power_failure = (torch.rand(n_samples) < 0.12).float()
    ls_failure = ((torch.rand(n_samples) < 0.08) | (power_failure > 0.5)).float()
    gh_failure = ((torch.rand(n_samples) < 0.05) | (ls_failure > 0.5) & (torch.rand(n_samples) < 0.6)).float()
    isru_failure = ((torch.rand(n_samples) < 0.04) | (power_failure > 0.5) & (torch.rand(n_samples) < 0.5)).float()

    any_failure = ((power_failure + ls_failure + gh_failure + isru_failure) > 0).float()

    # Generate system telemetry
    base_state = torch.randn(n_samples, 8) * 0.3

    # Power system
    generation = torch.relu(base_state[:, :4] * 0.3 + 0.8)
    generation = generation * (1 - power_failure.unsqueeze(1) * 0.7)
    generation = generation - dust_storm.unsqueeze(1) * 0.4  # dust reduces solar
    storage = torch.sigmoid(base_state[:, :2] * 0.5 + 0.5)
    storage = storage * (1 - power_failure.unsqueeze(1) * 0.3)

    # Life support (degraded by power failure)
    atmosphere = torch.randn(n_samples, ATMO_DIM) * 0.2 + 0.5
    atmosphere = atmosphere - ls_failure.unsqueeze(1) * torch.randn(n_samples, ATMO_DIM).abs() * 0.5
    atmosphere = atmosphere - power_failure.unsqueeze(1) * 0.2
    water = torch.relu(base_state[:, :2] * 0.3 + 0.6)
    water = water * (1 - ls_failure.unsqueeze(1) * 0.4)

    # Structural integrity
    structural = torch.ones(n_samples, STRUCT_DIM) * 0.9
    structural = structural - dust_storm.unsqueeze(1) * 0.2
    structural = structural + torch.randn(n_samples, STRUCT_DIM) * 0.05

    # Greenhouse (degraded by life support failure)
    crops = torch.relu(base_state[:, :4] * 0.2 + 0.7)
    crops = crops * (1 - gh_failure.unsqueeze(1) * 0.6)
    gh_atmosphere = torch.randn(n_samples, GH_ATMO_DIM) * 0.15 + 0.5
    gh_atmosphere = gh_atmosphere - gh_failure.unsqueeze(1) * 0.4
    harvest = torch.sigmoid(crops[:, :2] * 1.5 - 0.5)

    # ISRU
    feedstock = torch.rand(n_samples, FEED_DIM) * 0.5 + 0.5
    electrolysis = torch.relu(base_state[:, :2] * 0.3 + 0.6)
    electrolysis = electrolysis * (1 - isru_failure.unsqueeze(1) * 0.8)
    fuel_storage = torch.sigmoid(base_state[:, 2:4] * 0.5 + 0.3)
    fuel_storage = fuel_storage * (1 - isru_failure.unsqueeze(1) * 0.5)

    # Situation awareness and alert level
    all_telem = torch.cat([generation, atmosphere, crops[:, :4], electrolysis,
                           storage, water], dim=1)
    W_sit = torch.randn(all_telem.shape[1], SIT_DIM) * 0.3
    situation = torch.tanh(all_telem @ W_sit)

    # Alert level: one-hot over 4 levels (nominal, caution, warning, emergency)
    alert = torch.zeros(n_samples, ALERT_DIM)
    alert_idx = torch.zeros(n_samples, dtype=torch.long)
    alert_idx[any_failure > 0] = 1
    alert_idx[(power_failure + ls_failure) > 1] = 2
    alert_idx[(power_failure + ls_failure + gh_failure) > 2] = 3
    alert[torch.arange(n_samples), alert_idx] = 1.0

    return {
        'weather': weather, 'feedstock': feedstock,
        'generation': generation, 'storage': storage,
        'atmosphere': atmosphere, 'water': water,
        'structural': structural,
        'crops': crops, 'gh_atmosphere': gh_atmosphere, 'harvest': harvest,
        'electrolysis': electrolysis, 'fuel_storage': fuel_storage,
        'situation': situation, 'alert': alert,
        'any_failure': any_failure,
        'power_failure': power_failure, 'ls_failure': ls_failure,
        'gh_failure': gh_failure, 'isru_failure': isru_failure,
    }

data_tr = generate_colony_data()
data_val = generate_colony_data(512)


# ── 4. Build transformer on the canvas ────────────────────────────────

class ColonyTransformer(nn.Module):
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

        # Input projections (input-only fields + all sensors)
        self.input_projs = nn.ModuleDict()
        self.input_sizes = {}
        input_specs = {
            'weather': WEATHER_DIM,
            'isru.feedstock': FEED_DIM,
            'habitat.power.generation': GEN_DIM,
            'habitat.power.storage': STOR_DIM,
            'habitat.life_support.atmosphere': ATMO_DIM,
            'habitat.life_support.water': WATER_DIM,
            'habitat.structural': STRUCT_DIM,
            'greenhouse.crops': CROP_DIM,
            'greenhouse.atmosphere': GH_ATMO_DIM,
        }
        for rname, in_dim in input_specs.items():
            n = len(bound_schema.layout.region_indices(rname))
            key = rname.replace('.', '_')
            self.input_projs[key] = nn.Linear(in_dim, n * d_model)
            self.input_sizes[rname] = n

        # Output projections
        self.output_projs = nn.ModuleDict()
        self.output_sizes = {}
        out_specs = {
            'situation': SIT_DIM, 'alert_level': ALERT_DIM,
            'greenhouse.harvest': HARVEST_DIM,
            'isru.electrolysis': ELEC_DIM, 'isru.fuel_storage': FUEL_DIM,
        }
        for rname, out_dim in out_specs.items():
            n = len(bound_schema.layout.region_indices(rname))
            key = rname.replace('.', '_')
            self.output_projs[key] = nn.Linear(n * d_model, out_dim)
            self.output_sizes[rname] = n

    def forward(self, data):
        B = data['weather'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        input_map = {
            'weather': 'weather', 'isru.feedstock': 'feedstock',
            'habitat.power.generation': 'generation',
            'habitat.power.storage': 'storage',
            'habitat.life_support.atmosphere': 'atmosphere',
            'habitat.life_support.water': 'water',
            'habitat.structural': 'structural',
            'greenhouse.crops': 'crops',
            'greenhouse.atmosphere': 'gh_atmosphere',
        }
        for rname, dkey in input_map.items():
            idx = self.bound.layout.region_indices(rname)
            key = rname.replace('.', '_')
            proj = self.input_projs[key]
            n = self.input_sizes[rname]
            canvas[:, idx] = canvas[:, idx] + proj(data[dkey]).reshape(B, n, self.d)

        canvas = self.encoder(canvas, mask=self.attn_mask)

        result = {}
        for rname in ['situation', 'alert_level', 'greenhouse.harvest',
                       'isru.electrolysis', 'isru.fuel_storage']:
            idx = self.bound.layout.region_indices(rname)
            key = rname.replace('.', '_')
            n = self.output_sizes[rname]
            result[rname] = self.output_projs[key](canvas[:, idx].reshape(B, -1))

        result['alert_level'] = torch.softmax(result['alert_level'], dim=-1)
        result['greenhouse.harvest'] = torch.sigmoid(result['greenhouse.harvest'])
        result['isru.electrolysis'] = torch.relu(result['isru.electrolysis'])
        result['isru.fuel_storage'] = torch.sigmoid(result['isru.fuel_storage'])

        return result


model = ColonyTransformer(bound)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)


# ── 5. Train ──────────────────────────────────────────────────────────

losses_total = []
losses_alert = []
losses_subsys = []
n_epochs = 500
batch_size = 128

print("\nTraining colony controller...")
for epoch in range(n_epochs):
    idx = torch.randint(0, len(data_tr['weather']), (batch_size,))
    batch = {k: v[idx] for k, v in data_tr.items()}
    result = model(batch)

    # Alert loss (safety critical)
    alert_loss = -((batch['alert'] * torch.log(result['alert_level'] + 1e-8)).sum(dim=-1)).mean() * 8.0

    # Subsystem losses
    sit_loss = ((result['situation'] - batch['situation']) ** 2).mean() * 3.0
    harvest_loss = ((result['greenhouse.harvest'] - batch['harvest']) ** 2).mean() * 1.5
    elec_loss = ((result['isru.electrolysis'] - batch['electrolysis']) ** 2).mean() * 3.0
    fuel_loss = ((result['isru.fuel_storage'] - batch['fuel_storage']) ** 2).mean() * 3.0

    subsys_loss = sit_loss + harvest_loss + elec_loss + fuel_loss
    loss = alert_loss + subsys_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()

    losses_total.append(loss.item())
    losses_alert.append(alert_loss.item())
    losses_subsys.append(subsys_loss.item())

    if epoch % 100 == 0:
        print(f"  epoch {epoch:3d}: total={loss.item():.4f} alert={alert_loss.item():.4f} subsys={subsys_loss.item():.4f}")


# ── 6. Evaluate ───────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    result_val = model(data_val)

    # Alert prediction accuracy
    pred_alert = result_val['alert_level'].argmax(dim=-1)
    true_alert = data_val['alert'].argmax(dim=-1)
    alert_acc = (pred_alert == true_alert).float().mean().item()

    # Failure prediction: detect any_failure from alert > 0
    pred_failure = (pred_alert > 0).float()
    true_failure = data_val['any_failure']
    failure_acc = (pred_failure == true_failure).float().mean().item()

    # Per-subsystem prediction quality
    subsys_metrics = {}
    subsys_metrics['harvest'] = ((result_val['greenhouse.harvest'] - data_val['harvest']) ** 2).mean().item()
    subsys_metrics['electrolysis'] = ((result_val['isru.electrolysis'] - data_val['electrolysis']) ** 2).mean().item()
    subsys_metrics['fuel'] = ((result_val['isru.fuel_storage'] - data_val['fuel_storage']) ** 2).mean().item()

    print(f"\n  Alert accuracy: {alert_acc:.3f}")
    print(f"  Failure detection accuracy: {failure_acc:.3f}")
    for k, v in subsys_metrics.items():
        print(f"  {k} MSE: {v:.4f}")


# ── 7. Visualize ──────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Mars Colony: Cascading Failure Prediction', fontsize=16, fontweight='bold', y=0.98)

COLORS_MAP = {
    'habitat.life_support.atmosphere': '#E74C3C',
    'habitat.life_support.water': '#EC7063',
    'habitat.power.generation': '#F39C12',
    'habitat.power.storage': '#F7DC6F',
    'habitat.structural': '#E67E22',
    'greenhouse.crops': '#2ECC71',
    'greenhouse.atmosphere': '#58D68D',
    'greenhouse.harvest': '#27AE60',
    'isru.feedstock': '#3498DB',
    'isru.electrolysis': '#5DADE2',
    'isru.fuel_storage': '#2980B9',
    'situation': '#AF7AC5',
    'alert_level': '#C0392B',
    'weather': '#95A5A6',
}

# (a) Canvas layout
ax = axes[0, 0]
ax.set_title('Canvas Layout', fontsize=11, fontweight='bold')
H, W = bound.layout.H, bound.layout.W
grid = np.ones((H, W, 3)) * 0.93
for name, color in COLORS_MAP.items():
    if name not in bound:
        continue
    bf = bound[name]
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]
    parts = name.split(".")
    label = parts[-1][:6]
    ax.text((w0 + w1) / 2 - 0.5, (h0 + h1) / 2 - 0.5,
            label, ha='center', va='center', fontsize=4, fontweight='bold', color='white')
ax.imshow(grid, aspect='equal', interpolation='nearest')
ax.set_xlabel('W'); ax.set_ylabel('H')

# (b) Failure prediction accuracy per type
ax = axes[0, 1]
ax.set_title('Failure Prediction Accuracy', fontsize=11, fontweight='bold')
with torch.no_grad():
    failure_types = ['power_failure', 'ls_failure', 'gh_failure', 'isru_failure']
    failure_labels = ['Power', 'Life Support', 'Greenhouse', 'ISRU']
    failure_accs = []
    for ft in failure_types:
        mask = data_val[ft] > 0.5
        if mask.sum() > 0:
            # Check if alert predicts non-nominal when this failure occurs
            acc = (pred_alert[mask] > 0).float().mean().item()
        else:
            acc = 0.0
        failure_accs.append(acc)

colors_bar = ['#F39C12', '#E74C3C', '#2ECC71', '#3498DB']
bars = ax.bar(failure_labels, failure_accs, color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, failure_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.set_ylabel('Detection Rate')
ax.grid(True, alpha=0.2, axis='y')
ax.text(0.98, 0.02, f'Overall: {failure_acc:.3f}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (c) Cross-system attention weights (simulated via correlation)
ax = axes[1, 0]
ax.set_title('Cross-System Correlations (learned)', fontsize=11, fontweight='bold')
# Show correlation between subsystem predictions and true values
subsystem_names = ['atmo', 'water', 'power', 'crops', 'elec', 'fuel']
with torch.no_grad():
    # Use model weights as proxy for learned cross-system attention
    # Compute per-output correlation with each input system
    preds_cat = torch.cat([
        result_val['situation'],
        result_val['greenhouse.harvest'],
        result_val['isru.electrolysis'],
        result_val['isru.fuel_storage'],
    ], dim=1)
    inputs_cat = torch.cat([
        data_val['atmosphere'], data_val['water'][:, :2].repeat(1, 2),
        data_val['generation'], data_val['crops'],
        data_val['electrolysis'].repeat(1, 2), data_val['fuel_storage'].repeat(1, 2),
    ], dim=1)

    corr_matrix = np.corrcoef(inputs_cat[:200].numpy().T, preds_cat[:200].numpy().T)
    n_in = len(subsystem_names)
    cross_corr = np.abs(corr_matrix[:n_in * 4:4, n_in * 4::4])

im = ax.imshow(cross_corr, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
ax.set_xticks(range(cross_corr.shape[1]))
ax.set_xticklabels(['sit', 'harv', 'elec', 'fuel'], fontsize=8)
ax.set_yticks(range(cross_corr.shape[0]))
ax.set_yticklabels(subsystem_names, fontsize=8)
ax.set_xlabel('Output')
ax.set_ylabel('Input')
plt.colorbar(im, ax=ax, shrink=0.8, label='|correlation|')

# (d) Training curves
ax = axes[1, 1]
ax.set_title('Training Curves', fontsize=11, fontweight='bold')
w = 20
def smooth(a): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.semilogy(smooth(losses_alert), color='#C0392B', lw=2, label='alert loss')
ax.semilogy(smooth(losses_subsys), color='#3498DB', lw=2, label='subsystem loss')
ax.semilogy(smooth(losses_total), color='#2C3E50', lw=2, ls='--', label='total loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "11_mars_colony.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")

"""Air Traffic Control: conflict detection with canvas types.

Synthetic TRACON scenarios with 3D aircraft trajectories and separation conflicts.
Two key comparisons:
  1. loss_weight=1 vs loss_weight=10 on conflict detection
  2. Isolated vs dense inter-aircraft connectivity
Plus counterfactual training: "what if aircraft X turned 10 degrees?"

Outputs:
  assets/examples/06_atc.png  — static analysis figure
  assets/examples/06_atc.gif  — animated TRACON radar display

Run:  python examples/06_air_traffic_control.py
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
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.animation as animation

from canvas_engineering import Field, compile_schema, ConnectivityPolicy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

N_AIRCRAFT = 6
PRED_STEPS = 8        # 40s lookahead at 5s/step
SEP_H = 3.0           # 3nm horizontal separation minimum
SEP_V = 1000.0        # 1000ft vertical separation minimum
DT = 5.0              # seconds per simulation step


# ── 1. Type declarations ─────────────────────────────────────────────

@dataclass
class Aircraft:
    state: Field = Field(1, 3)                 # x, y, z + hdg, spd, vrate packed
    flight_plan: Field = Field(1, 4, is_output=False)  # route context
    trajectory: Field = Field(1, 4, loss_weight=3.0)   # predicted future
    conflict: Field = Field(1, 2, loss_weight=10.0)    # conflict flag + time-to-conflict

@dataclass
class AircraftLowWeight(Aircraft):
    """Same but conflict gets weight=1 instead of 10."""
    conflict: Field = Field(1, 2, loss_weight=1.0)

@dataclass
class TRACON:
    weather: Field = Field(1, 2, is_output=False)
    sector_load: Field = Field(1, 2)
    aircraft: list = dc_field(default_factory=list)


def make_schema(n_aircraft=N_AIRCRAFT, array_element="dense", conflict_weight=10.0):
    AcClass = Aircraft if conflict_weight == 10.0 else AircraftLowWeight
    tracon = TRACON(aircraft=[AcClass() for _ in range(n_aircraft)])
    return compile_schema(
        tracon, T=1, H=16, W=16, d_model=32,
        connectivity=ConnectivityPolicy(
            intra="dense",
            parent_child="hub_spoke",
            array_element=array_element,
            temporal="dense",
        ),
    )


bound_dense_w10 = make_schema(array_element="dense", conflict_weight=10.0)
bound_dense_w1 = make_schema(array_element="dense", conflict_weight=1.0)
bound_isolated = make_schema(array_element="isolated", conflict_weight=10.0)

print(f"Dense w10: {len(bound_dense_w10.topology.connections)} connections")
print(f"Dense w1:  {len(bound_dense_w1.topology.connections)} connections")
print(f"Isolated:  {len(bound_isolated.topology.connections)} connections")


# ── 2. Synthetic data: TRACON scenarios ──────────────────────────────

def generate_tracon_data(n_scenarios=2048, n_aircraft=N_AIRCRAFT, n_steps=PRED_STEPS):
    """Generate synthetic air traffic scenarios with conflicts.

    Aircraft fly roughly toward the airport (origin) with some spread.
    Conflicts occur when separation < minimums within prediction horizon.
    """
    all_states = []        # current state: (x, y, z, hdg, spd, vrate)
    all_plans = []         # flight plan context
    all_futures = []       # future trajectory (relative)
    all_conflicts = []     # (conflict_flag, time_to_conflict)
    all_weather = []       # weather context
    all_trajectories = []  # full trajectory for animation

    for _ in range(n_scenarios):
        # Initialize aircraft in a ring around the airport
        angles = np.random.uniform(0, 2 * np.pi, n_aircraft)
        radii = np.random.uniform(15, 40, n_aircraft)  # nm from airport
        altitudes = np.random.uniform(3000, 12000, n_aircraft)  # feet

        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        z = altitudes

        # Heading: roughly toward origin + noise
        hdg = np.arctan2(-y, -x) + np.random.randn(n_aircraft) * 0.3
        spd = np.random.uniform(180, 280, n_aircraft)  # knots
        vrate = np.random.uniform(-1500, 500, n_aircraft)  # fpm, mostly descending

        current_state = np.stack([x, y, z, hdg, spd, vrate], axis=-1).astype(np.float32)

        # Flight plan (encoded context)
        dest_angle = np.random.uniform(0, 2 * np.pi, n_aircraft)
        plan = np.stack([
            np.cos(dest_angle), np.sin(dest_angle),  # destination direction
            np.random.uniform(0, 1, n_aircraft),       # priority
            np.random.randint(0, 4, n_aircraft).astype(float),  # aircraft type
            np.random.uniform(100, 300, n_aircraft) / 300,  # wake category
            np.zeros(n_aircraft), np.zeros(n_aircraft), np.zeros(n_aircraft),
        ], axis=-1).astype(np.float32)

        # Simulate future
        traj_x, traj_y, traj_z = [x.copy()], [y.copy()], [z.copy()]
        cx, cy, cz = x.copy(), y.copy(), z.copy()
        chdg = hdg.copy()

        for step in range(n_steps):
            # Speed in nm/step
            spd_nm = spd * (DT / 3600.0)
            cx += spd_nm * np.cos(chdg)
            cy += spd_nm * np.sin(chdg)
            cz += vrate * (DT / 60.0)
            cz = np.clip(cz, 0, 15000)

            # Slight heading changes (random walk)
            chdg += np.random.randn(n_aircraft) * 0.02

            traj_x.append(cx.copy())
            traj_y.append(cy.copy())
            traj_z.append(cz.copy())

        traj_x = np.array(traj_x)  # (n_steps+1, n_aircraft)
        traj_y = np.array(traj_y)
        traj_z = np.array(traj_z)

        # Future relative to current
        fut_x = traj_x[1:] - traj_x[0:1]  # (n_steps, n_aircraft)
        fut_y = traj_y[1:] - traj_y[0:1]
        fut_z = traj_z[1:] - traj_z[0:1]

        # Pack future: interleave x,y for first 8 positions
        future = np.zeros((n_aircraft, 8), dtype=np.float32)
        for i in range(min(4, n_steps)):
            future[:, i*2] = fut_x[i]
            future[:, i*2+1] = fut_y[i]

        # Detect conflicts: check all pairs at all future times
        conflict_flags = np.zeros(n_aircraft, dtype=np.float32)
        time_to_conflict = np.ones(n_aircraft, dtype=np.float32) * n_steps  # max if no conflict

        for t in range(n_steps):
            for i in range(n_aircraft):
                for j in range(i + 1, n_aircraft):
                    dx = traj_x[t+1, i] - traj_x[t+1, j]
                    dy = traj_y[t+1, i] - traj_y[t+1, j]
                    dz = abs(traj_z[t+1, i] - traj_z[t+1, j])
                    h_dist = np.sqrt(dx**2 + dy**2)

                    if h_dist < SEP_H and dz < SEP_V:
                        if conflict_flags[i] == 0:
                            time_to_conflict[i] = t
                        if conflict_flags[j] == 0:
                            time_to_conflict[j] = t
                        conflict_flags[i] = 1.0
                        conflict_flags[j] = 1.0

        # Normalize time_to_conflict to [0, 1]
        ttc = time_to_conflict / n_steps

        conflict = np.stack([conflict_flags, ttc], axis=-1).astype(np.float32)

        # Weather (simple)
        weather = np.random.randn(4).astype(np.float32) * 0.5

        all_states.append(current_state)
        all_plans.append(plan)
        all_futures.append(future)
        all_conflicts.append(conflict)
        all_weather.append(weather)
        all_trajectories.append(np.stack([traj_x, traj_y, traj_z], axis=-1))  # (n_steps+1, n_ac, 3)

    return {
        'states': torch.tensor(np.array(all_states)),       # (N, n_ac, 6)
        'plans': torch.tensor(np.array(all_plans)),          # (N, n_ac, 8)
        'futures': torch.tensor(np.array(all_futures)),      # (N, n_ac, 8)
        'conflicts': torch.tensor(np.array(all_conflicts)),  # (N, n_ac, 2)
        'weather': torch.tensor(np.array(all_weather)),      # (N, 4)
        'trajectories': np.array(all_trajectories),          # for animation
    }


print("Generating TRACON scenarios...")
train_data = generate_tracon_data(2048)
val_data = generate_tracon_data(512)

conflict_rate = train_data['conflicts'][:, :, 0].mean().item()
print(f"  Conflict rate: {conflict_rate:.1%} of aircraft involved in a conflict")


# ── 3. Model ────────────────────────────────────────────────────────

class ATCModel(nn.Module):
    """Canvas-structured air traffic conflict detection model."""

    def __init__(self, bound, d=32, nhead=4, n_aircraft=N_AIRCRAFT):
        super().__init__()
        self.bound = bound
        self.d = d
        self.n_aircraft = n_aircraft
        N = bound.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, N, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=128,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        mask = bound.topology.to_additive_mask(bound.layout)
        self.register_buffer('mask', mask)

        # Field sizes
        def fs(name): return len(bound.layout.region_indices(name))

        self.state_proj = nn.Linear(6, fs('aircraft[0].state') * d)
        self.plan_proj = nn.Linear(8, fs('aircraft[0].flight_plan') * d)
        self.weather_proj = nn.Linear(4, fs('weather') * d)
        self.d = d

        traj_n = fs('aircraft[0].trajectory')
        conf_n = fs('aircraft[0].conflict')
        load_n = fs('sector_load')

        self.traj_head = nn.Linear(traj_n * d, 8)
        self.conflict_head = nn.Linear(conf_n * d, 2)
        self.load_head = nn.Linear(load_n * d, 2)

        self._fs = {
            'state': fs('aircraft[0].state'),
            'plan': fs('aircraft[0].flight_plan'),
            'weather': fs('weather'),
            'traj': traj_n, 'conf': conf_n, 'load': load_n,
        }

    def forward(self, data):
        B = data['states'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        # Place weather
        w_idx = self.bound.layout.region_indices('weather')
        w_emb = self.weather_proj(data['weather']).reshape(B, self._fs['weather'], self.d)
        canvas[:, w_idx] = canvas[:, w_idx] + w_emb

        # Place aircraft
        for i in range(self.n_aircraft):
            s_idx = self.bound.layout.region_indices(f'aircraft[{i}].state')
            p_idx = self.bound.layout.region_indices(f'aircraft[{i}].flight_plan')

            s_emb = self.state_proj(data['states'][:, i]).reshape(B, self._fs['state'], self.d)
            p_emb = self.plan_proj(data['plans'][:, i]).reshape(B, self._fs['plan'], self.d)

            canvas[:, s_idx] = canvas[:, s_idx] + s_emb
            canvas[:, p_idx] = canvas[:, p_idx] + p_emb

        canvas = self.encoder(canvas, mask=self.mask)

        # Read outputs
        trajs, conflicts = [], []
        for i in range(self.n_aircraft):
            t_idx = self.bound.layout.region_indices(f'aircraft[{i}].trajectory')
            c_idx = self.bound.layout.region_indices(f'aircraft[{i}].conflict')

            traj = self.traj_head(canvas[:, t_idx].reshape(B, -1))
            conf = self.conflict_head(canvas[:, c_idx].reshape(B, -1))
            trajs.append(traj)
            conflicts.append(conf)

        return {
            'trajectories': torch.stack(trajs, dim=1),   # (B, n_ac, 8)
            'conflicts': torch.stack(conflicts, dim=1),   # (B, n_ac, 2)
        }


# ── 4. Training ──────────────────────────────────────────────────────

def focal_loss(logits, targets, gamma=2.0, alpha=0.75):
    """Focal loss for imbalanced conflict detection."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def train_atc(bound, label, use_counterfactual=True, n_epochs=300, bs=128):
    model = ATCModel(bound)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []

    for ep in range(n_epochs):
        idx = torch.randint(0, len(train_data['states']), (bs,))
        batch = {k: v[idx] if isinstance(v, torch.Tensor) else v for k, v in train_data.items()}

        out = model(batch)

        # Trajectory loss
        traj_loss = F.mse_loss(out['trajectories'], batch['futures'])

        # Conflict detection (focal loss for imbalance)
        conf_flags = batch['conflicts'][:, :, 0]  # (B, n_ac)
        conf_logits = out['conflicts'][:, :, 0]
        conf_loss = focal_loss(conf_logits, conf_flags)

        # Time-to-conflict regression (only for conflict aircraft)
        ttc_mask = conf_flags > 0.5
        if ttc_mask.any():
            ttc_loss = F.mse_loss(
                torch.sigmoid(out['conflicts'][:, :, 1])[ttc_mask],
                batch['conflicts'][:, :, 1][ttc_mask]
            )
        else:
            ttc_loss = torch.tensor(0.0)

        loss = traj_loss + 3.0 * conf_loss + ttc_loss

        # Counterfactual: perturb one aircraft's heading, check if conflict changes
        if use_counterfactual and ep > 100:
            perturbed = batch.copy()
            states_p = batch['states'].clone()
            # Rotate heading by 10 degrees for random aircraft
            ac_idx = torch.randint(0, N_AIRCRAFT, (1,)).item()
            states_p[:, ac_idx, 3] += 0.17  # ~10 degrees in radians
            perturbed['states'] = states_p

            out_p = model(perturbed)
            # Counterfactual loss: conflict predictions should change smoothly
            cf_loss = F.mse_loss(
                torch.sigmoid(out_p['conflicts'][:, :, 0]),
                torch.sigmoid(out['conflicts'][:, :, 0]).detach()
            ) * 0.1  # gentle — we want sensitivity, not rigidity
            loss = loss - cf_loss  # negative: we WANT the predictions to differ

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if ep % 200 == 0:
            print(f"  [{label}] ep {ep:3d}: loss={loss.item():.4f}")

    return model, losses


print("\nTraining dense w10 (conflict weight=10)...")
model_d10, losses_d10 = train_atc(bound_dense_w10, "dense_w10")
print("Training dense w1 (conflict weight=1)...")
model_d1, losses_d1 = train_atc(bound_dense_w1, "dense_w1")
print("Training isolated w10...")
model_iso, losses_iso = train_atc(bound_isolated, "isolated_w10")


# ── 5. Evaluate ──────────────────────────────────────────────────────

def evaluate_conflict(model, data, threshold=0.5):
    """Compute TP, FP, FN, TN and ROC data for conflict detection."""
    model.eval()
    with torch.no_grad():
        out = model(data)
    probs = torch.sigmoid(out['conflicts'][:, :, 0]).numpy().flatten()
    labels = data['conflicts'][:, :, 0].numpy().flatten()

    # ROC at multiple thresholds
    thresholds = np.linspace(0, 1, 50)
    tprs, fprs = [], []
    for th in thresholds:
        preds = (probs >= th).astype(float)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tprs.append(tpr)
        fprs.append(fpr)

    # Confusion at default threshold
    preds_bin = (probs >= threshold).astype(float)
    tp = ((preds_bin == 1) & (labels == 1)).sum()
    fp = ((preds_bin == 1) & (labels == 0)).sum()
    fn = ((preds_bin == 0) & (labels == 1)).sum()
    tn = ((preds_bin == 0) & (labels == 0)).sum()

    # Trajectory MSE
    traj_mse = F.mse_loss(out['trajectories'],
                          data['futures']).item()

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'tprs': tprs, 'fprs': fprs, 'thresholds': thresholds,
        'probs': probs, 'labels': labels,
        'traj_mse': traj_mse,
    }


print("\nEvaluating...")
eval_d10 = evaluate_conflict(model_d10, val_data)
eval_d1 = evaluate_conflict(model_d1, val_data)
eval_iso = evaluate_conflict(model_iso, val_data)

for name, ev in [("dense_w10", eval_d10), ("dense_w1", eval_d1), ("isolated", eval_iso)]:
    recall = ev['tp'] / max(ev['tp'] + ev['fn'], 1)
    precision = ev['tp'] / max(ev['tp'] + ev['fp'], 1)
    print(f"  {name:12s}: recall={recall:.3f}, precision={precision:.3f}, "
          f"traj_mse={ev['traj_mse']:.4f}")


# ── 6. Visualization: static figure ─────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Air Traffic Control: Conflict Detection with Canvas Types',
             fontsize=16, fontweight='bold', y=0.99)

C10, C1, CISO = '#E74C3C', '#3498DB', '#95A5A6'

# (a) TRACON bird's-eye view
ax = axes[0, 0]
ax.set_title('TRACON Radar (scenario 0)', fontsize=11, fontweight='bold')
ax.set_facecolor('#1a1a2e')
# Airport at origin
ax.plot(0, 0, '*', color='#F39C12', markersize=15, zorder=10)
ax.text(0, -3, 'APT', color='#F39C12', ha='center', fontsize=8, fontweight='bold')

# Range rings
for r in [10, 20, 30, 40]:
    circle = Circle((0, 0), r, fill=False, color='#2d3436', ls='--', lw=0.5)
    ax.add_patch(circle)

states_0 = val_data['states'][0].numpy()
conflicts_0 = val_data['conflicts'][0].numpy()
traj_0 = val_data['trajectories'][0]  # (n_steps+1, n_ac, 3)

for i in range(N_AIRCRAFT):
    x, y, z = states_0[i, 0], states_0[i, 1], states_0[i, 2]
    hdg = states_0[i, 3]
    is_conflict = conflicts_0[i, 0] > 0.5
    color = '#E74C3C' if is_conflict else '#2ECC71'

    # Aircraft position
    ax.plot(x, y, 'o', color=color, markersize=6, zorder=5)

    # Heading arrow
    dx = 2 * np.cos(hdg)
    dy = 2 * np.sin(hdg)
    ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Label
    alt_str = f'{z/100:.0f}'
    ax.text(x + 1, y + 1, f'AC{i}\nFL{alt_str}', color=color, fontsize=6)

    # Future trajectory
    if traj_0 is not None:
        traj_xy = traj_0[:, i, :2]
        ax.plot(traj_xy[:, 0], traj_xy[:, 1], '-', color=color, lw=0.8, alpha=0.4)

ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect('equal')
ax.set_xlabel('nm')
ax.set_ylabel('nm')

# (b) ROC curves
ax = axes[0, 1]
ax.set_title('Conflict Detection ROC', fontsize=11, fontweight='bold')
for name, ev, color in [
    ('dense w=10', eval_d10, C10),
    ('dense w=1', eval_d1, C1),
    ('isolated w=10', eval_iso, CISO),
]:
    ax.plot(ev['fprs'], ev['tprs'], color=color, lw=2, label=name)
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# (c) Confusion matrices
ax = axes[0, 2]
ax.set_title('Confusion Matrix (dense w=10)', fontsize=11, fontweight='bold')
cm = np.array([[eval_d10['tn'], eval_d10['fp']],
               [eval_d10['fn'], eval_d10['tp']]])
im = ax.imshow(cm, cmap='Blues', aspect='equal')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred: No', 'Pred: Yes'])
ax.set_yticklabels(['True: No', 'True: Yes'])
plt.colorbar(im, ax=ax, shrink=0.8)

# (d) Sensitivity analysis: conflict probability vs separation
ax = axes[1, 0]
ax.set_title('Conflict Sensitivity vs Separation', fontsize=11, fontweight='bold')

# Generate scenarios at different separations
model_d10.eval()
separations = np.linspace(1, 10, 20)
conflict_probs = []

for sep in separations:
    # Create synthetic scenario: two aircraft at given separation
    test_states = torch.zeros(1, N_AIRCRAFT, 6)
    test_states[0, 0, :] = torch.tensor([0, 0, 5000, 0, 200, -500])
    test_states[0, 1, :] = torch.tensor([sep, 0, 5000, math.pi, 200, -500])
    # Fill rest with far-away aircraft
    for i in range(2, N_AIRCRAFT):
        test_states[0, i, :] = torch.tensor([50 + i*10, 50 + i*10, 10000, 0, 200, 0])

    test_data = {
        'states': test_states,
        'plans': torch.zeros(1, N_AIRCRAFT, 8),
        'weather': torch.zeros(1, 4),
    }
    with torch.no_grad():
        out = model_d10(test_data)
    prob = torch.sigmoid(out['conflicts'][0, 0, 0]).item()
    conflict_probs.append(prob)

ax.plot(separations, conflict_probs, 'o-', color=C10, lw=2, markersize=4)
ax.axvline(x=SEP_H, color='#E74C3C', ls='--', lw=1.5, label=f'Min separation ({SEP_H}nm)')
ax.set_xlabel('Horizontal Separation (nm)')
ax.set_ylabel('P(conflict)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (e) Training curves
ax = axes[1, 1]
ax.set_title('Training Loss', fontsize=11, fontweight='bold')
w = 30
def smooth(a, w=w): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.plot(smooth(losses_d10), color=C10, lw=1.5, label='dense w=10')
ax.plot(smooth(losses_d1), color=C1, lw=1.5, label='dense w=1')
ax.plot(smooth(losses_iso), color=CISO, lw=1.5, label='isolated w=10')
ax.legend(fontsize=8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid(True, alpha=0.2)

# (f) Loss weight budget
ax = axes[1, 2]
ax.set_title('Loss Weight Budget (dense w=10)', fontsize=11, fontweight='bold')
weights = bound_dense_w10.layout.loss_weight_mask("cpu")
total_w = weights.sum().item()
categories = {'conflict': 0, 'trajectory': 0, 'sector': 0, 'other': 0}
for name, bf in bound_dense_w10.fields.items():
    indices = bf.indices()
    w = sum(weights[i].item() for i in indices)
    if 'conflict' in name:
        categories['conflict'] += w
    elif 'trajectory' in name:
        categories['trajectory'] += w
    elif 'sector' in name or 'weather' in name:
        categories['sector'] += w
    else:
        categories['other'] += w

cats = {k: v for k, v in categories.items() if v > 0}
colors_pie = ['#E74C3C', '#3498DB', '#F39C12', '#95A5A6']
wedges, texts, autotexts = ax.pie(
    cats.values(), labels=cats.keys(), autopct='%1.1f%%',
    colors=colors_pie[:len(cats)], startangle=90)
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontweight('bold')

plt.tight_layout(rect=[0, 0, 1, 0.97])
path = os.path.join(ASSETS, "06_atc.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\nSaved {path}")


# ── 7. Animation: extended TRACON radar rollout ─────────────────────

print("Generating TRACON animation (extended rollout)...")

# Simulate a longer scenario (120 steps = 10 minutes) for animation
np.random.seed(7)
N_ANIM_STEPS = 120
n_ac_anim = N_AIRCRAFT

angles = np.random.uniform(0, 2 * np.pi, n_ac_anim)
radii = np.random.uniform(20, 45, n_ac_anim)
anim_x = radii * np.cos(angles)
anim_y = radii * np.sin(angles)
anim_z = np.random.uniform(4000, 12000, n_ac_anim)
anim_hdg = np.arctan2(-anim_y, -anim_x) + np.random.randn(n_ac_anim) * 0.2
anim_spd = np.random.uniform(180, 260, n_ac_anim)
anim_vrate = np.random.uniform(-1200, 300, n_ac_anim)

anim_traj_x = [anim_x.copy()]
anim_traj_y = [anim_y.copy()]
anim_traj_z = [anim_z.copy()]

for step in range(N_ANIM_STEPS):
    spd_nm = anim_spd * (DT / 3600.0)
    anim_x = anim_x + spd_nm * np.cos(anim_hdg)
    anim_y = anim_y + spd_nm * np.sin(anim_hdg)
    anim_z = np.clip(anim_z + anim_vrate * (DT / 60.0), 500, 15000)
    anim_hdg += np.random.randn(n_ac_anim) * 0.015
    # Gentle pull toward airport
    dist_to_apt = np.sqrt(anim_x**2 + anim_y**2)
    toward_apt = np.arctan2(-anim_y, -anim_x)
    hdg_diff = toward_apt - anim_hdg
    hdg_diff = (hdg_diff + np.pi) % (2 * np.pi) - np.pi
    anim_hdg += hdg_diff * 0.01

    anim_traj_x.append(anim_x.copy())
    anim_traj_y.append(anim_y.copy())
    anim_traj_z.append(anim_z.copy())

anim_traj_x = np.array(anim_traj_x)
anim_traj_y = np.array(anim_traj_y)
anim_traj_z = np.array(anim_traj_z)

# Detect conflict pairs per frame
def find_conflicts_at(frame):
    pairs = []
    for i in range(n_ac_anim):
        for j in range(i+1, n_ac_anim):
            dx = anim_traj_x[frame, i] - anim_traj_x[frame, j]
            dy = anim_traj_y[frame, i] - anim_traj_y[frame, j]
            dz = abs(anim_traj_z[frame, i] - anim_traj_z[frame, j])
            h_dist = np.sqrt(dx**2 + dy**2)
            if h_dist < SEP_H * 2 and dz < SEP_V * 1.5:
                pairs.append((i, j, h_dist))
    return pairs

fig_anim, ax_anim = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
fig_anim.patch.set_facecolor('#0a0a1a')

def animate(frame):
    ax_anim.clear()
    ax_anim.set_facecolor('#0a0a1a')
    ax_anim.set_xlim(-55, 55)
    ax_anim.set_ylim(-55, 55)
    ax_anim.set_aspect('equal')

    # Compass rose
    for angle_deg in range(0, 360, 30):
        a = np.radians(angle_deg)
        ax_anim.plot([48*np.cos(a), 52*np.cos(a)],
                     [48*np.sin(a), 52*np.sin(a)], '-', color='#1a3a2a', lw=0.5)

    # Range rings with labels
    for r in [10, 20, 30, 40]:
        circle = Circle((0, 0), r, fill=False, color='#1a3a2a', ls='-', lw=0.5)
        ax_anim.add_patch(circle)
        ax_anim.text(r + 0.5, 0.5, f'{r}', color='#1a3a2a', fontsize=5)

    # Runway
    ax_anim.plot([-2, 2], [0, 0], '-', color='#F39C12', lw=3, zorder=9)
    ax_anim.plot(0, 0, '*', color='#F39C12', markersize=10, zorder=10)

    conf_pairs = find_conflicts_at(frame)
    conflict_ac = set()
    for i, j, d in conf_pairs:
        conflict_ac.add(i)
        conflict_ac.add(j)

    ac_colors = ['#44FF44', '#55BBFF', '#FFAA44', '#FF66AA', '#AAFFAA', '#FFFF66']

    for i in range(n_ac_anim):
        x = anim_traj_x[frame, i]
        y = anim_traj_y[frame, i]
        z = anim_traj_z[frame, i]
        base_color = ac_colors[i % len(ac_colors)]
        color = '#FF4444' if i in conflict_ac else base_color

        # History trail (last 15 positions, fading)
        trail_start = max(0, frame - 15)
        for t in range(trail_start, frame):
            alpha = 0.05 + 0.2 * (t - trail_start) / max(frame - trail_start, 1)
            ax_anim.plot(anim_traj_x[t:t+2, i], anim_traj_y[t:t+2, i],
                         '-', color=color, lw=1, alpha=alpha)

        # Aircraft symbol (triangle pointing in heading direction)
        if frame > 0:
            dx = anim_traj_x[frame, i] - anim_traj_x[frame-1, i]
            dy = anim_traj_y[frame, i] - anim_traj_y[frame-1, i]
        else:
            dx, dy = 1, 0
        heading = np.arctan2(dy, dx)
        size = 1.2
        tri_x = [x + size*np.cos(heading),
                 x + size*0.5*np.cos(heading + 2.4),
                 x + size*0.5*np.cos(heading - 2.4)]
        tri_y = [y + size*np.sin(heading),
                 y + size*0.5*np.sin(heading + 2.4),
                 y + size*0.5*np.sin(heading - 2.4)]
        ax_anim.fill(tri_x, tri_y, color=color, zorder=5)

        # Data block
        alt_str = f'{z/100:.0f}'
        spd_str = f'{anim_spd[i]:.0f}'
        ax_anim.plot([x, x + 3], [y, y + 2.5], '-', color=color, lw=0.5, alpha=0.5)
        ax_anim.text(x + 3.2, y + 2.5, f'AC{i} FL{alt_str}\n{spd_str}kt',
                     color=color, fontsize=5, fontfamily='monospace',
                     verticalalignment='bottom')

    # Conflict lines
    for i, j, d in conf_pairs:
        xi, yi = anim_traj_x[frame, i], anim_traj_y[frame, i]
        xj, yj = anim_traj_x[frame, j], anim_traj_y[frame, j]
        ax_anim.plot([xi, xj], [yi, yj], '--', color='#FF4444', lw=1.5, alpha=0.8)
        mx, my = (xi+xj)/2, (yi+yj)/2
        ax_anim.text(mx, my - 1, f'{d:.1f}nm', color='#FF4444',
                     fontsize=6, ha='center', fontfamily='monospace')

    minutes = frame * DT / 60
    ax_anim.text(0.02, 0.98, f'T+{minutes:.1f}min', transform=ax_anim.transAxes,
                 color='#44FF44', fontsize=11, fontfamily='monospace',
                 va='top', fontweight='bold')
    ax_anim.text(0.98, 0.98, f'TRACON RADAR', transform=ax_anim.transAxes,
                 color='#1a5a3a', fontsize=9, fontfamily='monospace',
                 va='top', ha='right')
    if conf_pairs:
        ax_anim.text(0.5, 0.02, f'CONFLICT ALERT: {len(conf_pairs)} pair(s)',
                     transform=ax_anim.transAxes, color='#FF4444', fontsize=10,
                     ha='center', fontweight='bold', fontfamily='monospace')

    ax_anim.tick_params(colors='#333333', labelsize=5)
    for spine in ax_anim.spines.values():
        spine.set_color('#1a2a1a')

anim = animation.FuncAnimation(fig_anim, animate,
                                frames=range(0, N_ANIM_STEPS + 1, 2),  # every other frame
                                interval=150)
gif_path = os.path.join(ASSETS, "06_atc.gif")
anim.save(gif_path, writer='pillow', fps=8)
plt.close()
print(f"Saved {gif_path}")

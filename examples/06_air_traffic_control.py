"""Air Traffic Control: conflict detection with canvas types.

Synthetic TRACON scenarios with 12 aircraft, wake turbulence categories,
weather cells, ILS approaches, holding patterns, and speed/altitude restrictions.

Three model variants compared:
  1. Dense array connectivity, conflict weight=10
  2. Dense array connectivity, conflict weight=1
  3. Isolated array connectivity, conflict weight=10

16-panel analysis figure + extended TRACON radar animation.

Outputs:
  assets/examples/06_atc.png  — 4x4 analysis figure
  assets/examples/06_atc.gif  — animated TRACON radar display (120 frames)

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
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Wedge
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from canvas_engineering import Field, compile_schema, ConnectivityPolicy, LayoutStrategy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

N_AIRCRAFT = 12
PRED_STEPS = 8        # 40s lookahead at 5s/step
DT = 5.0              # seconds per simulation step

# Wake turbulence separation requirements (nm)
# Matrix: [leader_cat][follower_cat] — Heavy=0, Medium=1, Light=2
WAKE_SEP = np.array([
    [4.0, 5.0, 6.0],   # Heavy leading
    [3.0, 3.0, 5.0],   # Medium leading
    [3.0, 3.0, 3.0],   # Light leading
], dtype=np.float32)

SEP_V = 1000.0        # 1000ft vertical separation minimum
DEFAULT_SEP_H = 3.0   # default horizontal sep (nm)

# Callsigns
CALLSIGNS = [
    'UAL417', 'DAL882', 'AAL233', 'SWA914', 'JBU602', 'SKW371',
    'ASA155', 'NKS748', 'FFT290', 'RPA461', 'ENY503', 'EJA127',
]

# Wake categories: 0=Heavy, 1=Medium, 2=Light
WAKE_CATS = np.array([0, 0, 1, 1, 1, 2, 1, 1, 2, 2, 1, 0], dtype=np.int32)
WAKE_LABELS = ['H', 'M', 'L']


# ── 1. Type declarations ─────────────────────────────────────────────

@dataclass
class Aircraft:
    state: Field = Field(1, 3)                    # x,y,z + hdg,spd,vrate
    flight_plan: Field = Field(1, 4, is_output=False)
    trajectory: Field = Field(1, 4, loss_weight=3.0)
    conflict: Field = Field(1, 2, loss_weight=10.0)
    wake_category: Field = Field(1, 1, is_output=False)
    intent: Field = Field(1, 2)                   # heading/altitude intent

@dataclass
class AircraftLowWeight(Aircraft):
    """Same but conflict gets weight=1 instead of 10."""
    conflict: Field = Field(1, 2, loss_weight=1.0)

@dataclass
class WeatherCell:
    position: Field = Field(1, 2, is_output=False)
    intensity: Field = Field(1, 1, is_output=False)
    movement: Field = Field(1, 2)                  # predicted movement

@dataclass
class TRACON:
    weather: Field = Field(1, 4, is_output=False)
    sector_load: Field = Field(1, 2)
    runway_state: Field = Field(1, 2)
    weather_cells: list = dc_field(default_factory=list)
    aircraft: list = dc_field(default_factory=list)


N_WEATHER_CELLS = 3

def make_schema(n_aircraft=N_AIRCRAFT, array_element="dense", conflict_weight=10.0):
    AcClass = Aircraft if conflict_weight == 10.0 else AircraftLowWeight
    tracon = TRACON(
        weather_cells=[WeatherCell() for _ in range(N_WEATHER_CELLS)],
        aircraft=[AcClass() for _ in range(n_aircraft)],
    )
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

# ILS approach fixes (distance from airport in nm, altitude in ft)
ILS_FIXES = [
    (25.0, 8000),   # Initial approach fix
    (18.0, 6000),   # Intermediate fix
    (12.0, 4000),   # Final approach fix
    (5.0, 1500),    # Glide slope intercept
    (0.0, 200),     # Decision height
]

# Holding pattern centers (nm from airport, angle)
HOLD_PATTERNS = [
    (30.0, np.radians(45)),
    (30.0, np.radians(225)),
    (35.0, np.radians(135)),
]


def generate_weather_cells(n_cells=N_WEATHER_CELLS):
    """Generate convective weather cell positions and intensities."""
    angles = np.random.uniform(0, 2 * np.pi, n_cells)
    radii = np.random.uniform(10, 35, n_cells)
    cx = radii * np.cos(angles)
    cy = radii * np.sin(angles)
    intensity = np.random.uniform(0.3, 1.0, n_cells)
    cell_radius = np.random.uniform(3, 8, n_cells)
    # Movement vector (nm per step)
    mvx = np.random.uniform(-0.3, 0.3, n_cells)
    mvy = np.random.uniform(-0.3, 0.3, n_cells)
    return cx, cy, intensity, cell_radius, mvx, mvy


def generate_tracon_data(n_scenarios=2048, n_aircraft=N_AIRCRAFT, n_steps=PRED_STEPS):
    """Generate synthetic air traffic scenarios with conflicts.

    Aircraft fly roughly toward the airport (origin) with some spread.
    Includes wake turbulence categories, weather avoidance, speed/altitude
    restrictions at fixes, and holding patterns.
    """
    all_states = []
    all_plans = []
    all_futures = []
    all_conflicts = []
    all_weather = []
    all_trajectories = []
    all_wake = []
    all_weather_cells = []
    all_intents = []

    for _ in range(n_scenarios):
        # Initialize aircraft in a ring around the airport
        angles = np.random.uniform(0, 2 * np.pi, n_aircraft)
        radii = np.random.uniform(12, 45, n_aircraft)
        altitudes = np.random.uniform(2000, 14000, n_aircraft)

        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        z = altitudes

        # Heading: roughly toward origin + noise
        hdg = np.arctan2(-y, -x) + np.random.randn(n_aircraft) * 0.25
        spd = np.random.uniform(160, 290, n_aircraft)
        vrate = np.random.uniform(-1800, 400, n_aircraft)

        # Wake categories
        wake = WAKE_CATS[:n_aircraft].copy()

        current_state = np.stack([x, y, z, hdg, spd, vrate], axis=-1).astype(np.float32)

        # Flight plan (encoded context)
        dest_angle = np.random.uniform(0, 2 * np.pi, n_aircraft)
        plan = np.stack([
            np.cos(dest_angle), np.sin(dest_angle),
            np.random.uniform(0, 1, n_aircraft),
            wake.astype(float) / 2.0,
            np.random.uniform(100, 300, n_aircraft) / 300,
            np.random.randint(0, 4, n_aircraft).astype(float),
            np.zeros(n_aircraft), np.zeros(n_aircraft),
        ], axis=-1).astype(np.float32)

        # Weather cells
        wcx, wcy, wint, wrad, wmvx, wmvy = generate_weather_cells()

        # Simulate future with weather avoidance
        traj_x, traj_y, traj_z = [x.copy()], [y.copy()], [z.copy()]
        cx, cy, cz = x.copy(), y.copy(), z.copy()
        chdg = hdg.copy()

        for step in range(n_steps):
            spd_nm = spd * (DT / 3600.0)
            cx += spd_nm * np.cos(chdg)
            cy += spd_nm * np.sin(chdg)
            cz += vrate * (DT / 60.0)
            cz = np.clip(cz, 0, 15000)

            # Slight heading changes (random walk)
            chdg += np.random.randn(n_aircraft) * 0.02

            # Weather avoidance: push away from weather cells
            for wc_idx in range(len(wcx)):
                dwx = cx - (wcx[wc_idx] + wmvx[wc_idx] * step)
                dwy = cy - (wcy[wc_idx] + wmvy[wc_idx] * step)
                wdist = np.sqrt(dwx**2 + dwy**2)
                avoid_mask = wdist < wrad[wc_idx] * 2
                if avoid_mask.any():
                    avoidance_angle = np.arctan2(dwy, dwx)
                    chdg[avoid_mask] += 0.05 * np.sign(
                        np.sin(avoidance_angle[avoid_mask] - chdg[avoid_mask]))

            traj_x.append(cx.copy())
            traj_y.append(cy.copy())
            traj_z.append(cz.copy())

        traj_x = np.array(traj_x)
        traj_y = np.array(traj_y)
        traj_z = np.array(traj_z)

        # Future relative to current
        fut_x = traj_x[1:] - traj_x[0:1]
        fut_y = traj_y[1:] - traj_y[0:1]

        future = np.zeros((n_aircraft, 8), dtype=np.float32)
        for i in range(min(4, n_steps)):
            future[:, i*2] = fut_x[i]
            future[:, i*2+1] = fut_y[i]

        # Detect conflicts with wake-turbulence-aware separation
        conflict_flags = np.zeros(n_aircraft, dtype=np.float32)
        time_to_conflict = np.ones(n_aircraft, dtype=np.float32) * n_steps

        for t in range(n_steps):
            for i in range(n_aircraft):
                for j in range(i + 1, n_aircraft):
                    dx = traj_x[t+1, i] - traj_x[t+1, j]
                    dy = traj_y[t+1, i] - traj_y[t+1, j]
                    dz = abs(traj_z[t+1, i] - traj_z[t+1, j])
                    h_dist = np.sqrt(dx**2 + dy**2)

                    # Wake-aware separation
                    req_sep = WAKE_SEP[wake[i], wake[j]]
                    req_sep = max(req_sep, WAKE_SEP[wake[j], wake[i]])

                    if h_dist < req_sep and dz < SEP_V:
                        if conflict_flags[i] == 0:
                            time_to_conflict[i] = t
                        if conflict_flags[j] == 0:
                            time_to_conflict[j] = t
                        conflict_flags[i] = 1.0
                        conflict_flags[j] = 1.0

        ttc = time_to_conflict / n_steps
        conflict = np.stack([conflict_flags, ttc], axis=-1).astype(np.float32)

        # Intent: desired heading change and altitude change (normalized)
        intent_hdg = np.arctan2(-y, -x) - hdg  # desired heading toward airport
        intent_hdg = (intent_hdg + np.pi) % (2 * np.pi) - np.pi
        intent_alt = -z / 15000.0  # normalized descent intent
        intent = np.stack([intent_hdg, intent_alt], axis=-1).astype(np.float32)

        weather = np.random.randn(4).astype(np.float32) * 0.5

        all_states.append(current_state)
        all_plans.append(plan)
        all_futures.append(future)
        all_conflicts.append(conflict)
        all_weather.append(weather)
        all_trajectories.append(np.stack([traj_x, traj_y, traj_z], axis=-1))
        all_wake.append(wake)
        all_weather_cells.append((wcx, wcy, wint, wrad))
        all_intents.append(intent)

    return {
        'states': torch.tensor(np.array(all_states)),
        'plans': torch.tensor(np.array(all_plans)),
        'futures': torch.tensor(np.array(all_futures)),
        'conflicts': torch.tensor(np.array(all_conflicts)),
        'weather': torch.tensor(np.array(all_weather)),
        'trajectories': np.array(all_trajectories),
        'wake': np.array(all_wake),
        'weather_cells': all_weather_cells,
        'intents': torch.tensor(np.array(all_intents)),
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

        def fs(name): return len(bound.layout.region_indices(name))

        self.state_proj = nn.Linear(6, fs('aircraft[0].state') * d)
        self.plan_proj = nn.Linear(8, fs('aircraft[0].flight_plan') * d)
        self.wake_proj = nn.Linear(3, fs('aircraft[0].wake_category') * d)
        self.weather_proj = nn.Linear(4, fs('weather') * d)

        # Weather cell projections
        self.wcell_pos_proj = nn.Linear(2, fs('weather_cells[0].position') * d)
        self.wcell_int_proj = nn.Linear(1, fs('weather_cells[0].intensity') * d)

        traj_n = fs('aircraft[0].trajectory')
        conf_n = fs('aircraft[0].conflict')
        intent_n = fs('aircraft[0].intent')
        load_n = fs('sector_load')
        rwy_n = fs('runway_state')

        self.traj_head = nn.Linear(traj_n * d, 8)
        self.conflict_head = nn.Linear(conf_n * d, 2)
        self.intent_head = nn.Linear(intent_n * d, 2)
        self.load_head = nn.Linear(load_n * d, 2)
        self.rwy_head = nn.Linear(rwy_n * d, 2)

        self._fs = {
            'state': fs('aircraft[0].state'),
            'plan': fs('aircraft[0].flight_plan'),
            'wake': fs('aircraft[0].wake_category'),
            'weather': fs('weather'),
            'wcell_pos': fs('weather_cells[0].position'),
            'wcell_int': fs('weather_cells[0].intensity'),
            'traj': traj_n, 'conf': conf_n, 'intent': intent_n,
            'load': load_n, 'rwy': rwy_n,
        }

    def forward(self, data):
        B = data['states'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        # Place weather
        w_idx = self.bound.layout.region_indices('weather')
        w_emb = self.weather_proj(data['weather']).reshape(B, self._fs['weather'], self.d)
        canvas[:, w_idx] = canvas[:, w_idx] + w_emb

        # Place weather cells
        for wc in range(N_WEATHER_CELLS):
            wcp_idx = self.bound.layout.region_indices(f'weather_cells[{wc}].position')
            wci_idx = self.bound.layout.region_indices(f'weather_cells[{wc}].intensity')
            # Use zero data for weather cells (context only)
            wcp_emb = self.wcell_pos_proj(torch.zeros(B, 2)).reshape(B, self._fs['wcell_pos'], self.d)
            wci_emb = self.wcell_int_proj(torch.zeros(B, 1)).reshape(B, self._fs['wcell_int'], self.d)
            canvas[:, wcp_idx] = canvas[:, wcp_idx] + wcp_emb
            canvas[:, wci_idx] = canvas[:, wci_idx] + wci_emb

        # Place aircraft
        for i in range(self.n_aircraft):
            s_idx = self.bound.layout.region_indices(f'aircraft[{i}].state')
            p_idx = self.bound.layout.region_indices(f'aircraft[{i}].flight_plan')
            wk_idx = self.bound.layout.region_indices(f'aircraft[{i}].wake_category')

            s_emb = self.state_proj(data['states'][:, i]).reshape(B, self._fs['state'], self.d)
            p_emb = self.plan_proj(data['plans'][:, i]).reshape(B, self._fs['plan'], self.d)

            # One-hot wake category
            wake_oh = torch.zeros(B, 3)
            if 'wake' in data:
                for b in range(B):
                    wake_oh[b, data['wake'][b, i]] = 1.0
            wk_emb = self.wake_proj(wake_oh).reshape(B, self._fs['wake'], self.d)

            canvas[:, s_idx] = canvas[:, s_idx] + s_emb
            canvas[:, p_idx] = canvas[:, p_idx] + p_emb
            canvas[:, wk_idx] = canvas[:, wk_idx] + wk_emb

        canvas = self.encoder(canvas, mask=self.mask)

        # Read outputs
        trajs, conflicts, intents = [], [], []
        for i in range(self.n_aircraft):
            t_idx = self.bound.layout.region_indices(f'aircraft[{i}].trajectory')
            c_idx = self.bound.layout.region_indices(f'aircraft[{i}].conflict')
            i_idx = self.bound.layout.region_indices(f'aircraft[{i}].intent')

            traj = self.traj_head(canvas[:, t_idx].reshape(B, -1))
            conf = self.conflict_head(canvas[:, c_idx].reshape(B, -1))
            intent = self.intent_head(canvas[:, i_idx].reshape(B, -1))
            trajs.append(traj)
            conflicts.append(conf)
            intents.append(intent)

        return {
            'trajectories': torch.stack(trajs, dim=1),
            'conflicts': torch.stack(conflicts, dim=1),
            'intents': torch.stack(intents, dim=1),
        }


# ── 4. Training ──────────────────────────────────────────────────────

def focal_loss(logits, targets, gamma=2.0, alpha=0.75):
    """Focal loss for imbalanced conflict detection."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def train_atc(bound, label, n_epochs=150, bs=128):
    model = ATCModel(bound)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []

    wake_tensor = torch.tensor(train_data['wake'], dtype=torch.long)

    for ep in range(n_epochs):
        idx = torch.randint(0, len(train_data['states']), (bs,))
        batch = {k: v[idx] if isinstance(v, torch.Tensor) else v
                 for k, v in train_data.items()}
        batch['wake'] = wake_tensor[idx]

        out = model(batch)

        # Trajectory loss
        traj_loss = F.mse_loss(out['trajectories'], batch['futures'])

        # Conflict detection (focal loss for imbalance)
        conf_flags = batch['conflicts'][:, :, 0]
        conf_logits = out['conflicts'][:, :, 0]
        conf_loss = focal_loss(conf_logits, conf_flags)

        # Time-to-conflict regression
        ttc_mask = conf_flags > 0.5
        if ttc_mask.any():
            ttc_loss = F.mse_loss(
                torch.sigmoid(out['conflicts'][:, :, 1])[ttc_mask],
                batch['conflicts'][:, :, 1][ttc_mask]
            )
        else:
            ttc_loss = torch.tensor(0.0)

        # Intent loss
        intent_loss = F.mse_loss(out['intents'], batch['intents'])

        loss = traj_loss + 3.0 * conf_loss + ttc_loss + 0.5 * intent_loss

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
    wake_tensor = torch.tensor(data['wake'], dtype=torch.long)
    eval_data = {k: v for k, v in data.items()}
    eval_data['wake'] = wake_tensor
    with torch.no_grad():
        out = model(eval_data)
    probs = torch.sigmoid(out['conflicts'][:, :, 0]).numpy().flatten()
    labels = data['conflicts'][:, :, 0].numpy().flatten()

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

    preds_bin = (probs >= threshold).astype(float)
    tp = ((preds_bin == 1) & (labels == 1)).sum()
    fp = ((preds_bin == 1) & (labels == 0)).sum()
    fn = ((preds_bin == 0) & (labels == 1)).sum()
    tn = ((preds_bin == 0) & (labels == 0)).sum()

    traj_mse = F.mse_loss(out['trajectories'],
                          data['futures']).item()

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'tprs': tprs, 'fprs': fprs, 'thresholds': thresholds,
        'probs': probs, 'labels': labels,
        'traj_mse': traj_mse,
        'conflict_probs': torch.sigmoid(out['conflicts'][:, :, 0]).numpy(),
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


# ── 6. Visualization: 4x4 analysis figure ───────────────────────────

BG = '#0a0a1a'
ATC_GREEN = '#44FF44'
ATC_DIM = '#1a3a2a'
WARN_RED = '#FF4444'
ATC_AMBER = '#FFAA00'
ATC_CYAN = '#00DDFF'

C10, C1, CISO = '#FF5555', '#5599FF', '#888888'

fig = plt.figure(figsize=(24, 24), dpi=150, facecolor=BG)
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.32, wspace=0.30,
                       left=0.04, right=0.97, top=0.95, bottom=0.03)
fig.suptitle('AIR TRAFFIC CONTROL: CONFLICT DETECTION WITH CANVAS TYPES',
             fontsize=18, fontweight='bold', color=ATC_GREEN,
             fontfamily='monospace', y=0.98)

# Helper for dark axes
def dark_ax(ax, title=''):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#555555', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', color=ATC_GREEN,
                     fontfamily='monospace', pad=6)
    return ax

# ── (0,0) Plan View Radar Display ──
ax = dark_ax(fig.add_subplot(gs[0, 0]), 'TRACON PLAN VIEW')

# Range rings + compass rose
for r in [10, 20, 30, 40]:
    circle = Circle((0, 0), r, fill=False, color=ATC_DIM, ls='-', lw=0.5)
    ax.add_patch(circle)
    ax.text(0, r + 1, f'{r}', color=ATC_DIM, fontsize=5, ha='center',
            fontfamily='monospace')
for angle_deg in range(0, 360, 30):
    a = np.radians(angle_deg)
    ax.plot([44*np.cos(a), 48*np.cos(a)], [44*np.sin(a), 48*np.sin(a)],
            '-', color=ATC_DIM, lw=0.5)
    if angle_deg % 90 == 0:
        labels_map = {0: 'E', 90: 'N', 180: 'W', 270: 'S'}
        ax.text(50*np.cos(a), 50*np.sin(a), labels_map[angle_deg],
                color='#336633', fontsize=7, ha='center', va='center',
                fontfamily='monospace', fontweight='bold')

# Airport
ax.plot([-2, 2], [0, 0], '-', color=ATC_AMBER, lw=3, zorder=9)
ax.plot(0, 0, '*', color=ATC_AMBER, markersize=8, zorder=10)
ax.text(0, -3.5, 'RWY 28L', color=ATC_AMBER, ha='center', fontsize=5,
        fontfamily='monospace', fontweight='bold')

# Weather cells (scenario 0)
if val_data['weather_cells']:
    wcx0, wcy0, wint0, wrad0 = val_data['weather_cells'][0]
    for wc_idx in range(len(wcx0)):
        circle = Circle((wcx0[wc_idx], wcy0[wc_idx]), wrad0[wc_idx],
                         fill=True, facecolor='#FF000020',
                         edgecolor='#FF4444', lw=1, alpha=0.6, ls='--')
        ax.add_patch(circle)
        ax.text(wcx0[wc_idx], wcy0[wc_idx], f'WX{wc_idx+1}\n{wint0[wc_idx]:.0%}',
                color=WARN_RED, fontsize=4, ha='center', va='center',
                fontfamily='monospace')

states_0 = val_data['states'][0].numpy()
conflicts_0 = val_data['conflicts'][0].numpy()
traj_0 = val_data['trajectories'][0]

ac_colors = ['#44FF44', '#55BBFF', '#FFAA44', '#FF66AA', '#AAFFAA',
             '#FFFF66', '#FF8844', '#44DDDD', '#DD88FF', '#88FF88',
             '#FFBB88', '#88BBFF']

for i in range(N_AIRCRAFT):
    xi, yi, zi = states_0[i, 0], states_0[i, 1], states_0[i, 2]
    hdg_i = states_0[i, 3]
    is_conflict = conflicts_0[i, 0] > 0.5
    color = WARN_RED if is_conflict else ac_colors[i % len(ac_colors)]

    # Aircraft triangle
    size = 1.5
    tri_x = [xi + size*np.cos(hdg_i),
             xi + size*0.5*np.cos(hdg_i + 2.4),
             xi + size*0.5*np.cos(hdg_i - 2.4)]
    tri_y = [yi + size*np.sin(hdg_i),
             yi + size*0.5*np.sin(hdg_i + 2.4),
             yi + size*0.5*np.sin(hdg_i - 2.4)]
    ax.fill(tri_x, tri_y, color=color, zorder=5)

    # Data block with leader line
    dx_lbl = 3 if xi > 0 else -3
    dy_lbl = 2.5 if yi > 0 else -2.5
    ax.plot([xi, xi + dx_lbl], [yi, yi + dy_lbl], '-', color=color, lw=0.4, alpha=0.5)
    alt_str = f'{zi/100:.0f}'
    spd_str = f'{states_0[i, 4]:.0f}'
    wake_lbl = WAKE_LABELS[WAKE_CATS[i]]
    ax.text(xi + dx_lbl, yi + dy_lbl,
            f'{CALLSIGNS[i]}\nFL{alt_str} {spd_str}kt\n{wake_lbl}',
            color=color, fontsize=4, fontfamily='monospace',
            va='bottom' if dy_lbl > 0 else 'top',
            ha='left' if dx_lbl > 0 else 'right')

    # Future trajectory trail
    traj_xy = traj_0[:, i, :2]
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], '-', color=color, lw=0.6, alpha=0.3)

# Conflict lines
for i in range(N_AIRCRAFT):
    for j in range(i + 1, N_AIRCRAFT):
        if conflicts_0[i, 0] > 0.5 and conflicts_0[j, 0] > 0.5:
            dx = states_0[i, 0] - states_0[j, 0]
            dy = states_0[i, 1] - states_0[j, 1]
            h_dist = np.sqrt(dx**2 + dy**2)
            if h_dist < 15:
                ax.plot([states_0[i, 0], states_0[j, 0]],
                        [states_0[i, 1], states_0[j, 1]],
                        '--', color=WARN_RED, lw=1.0, alpha=0.6)
                mx = (states_0[i, 0] + states_0[j, 0]) / 2
                my = (states_0[i, 1] + states_0[j, 1]) / 2
                ax.text(mx, my, f'{h_dist:.1f}nm', color=WARN_RED, fontsize=4,
                        ha='center', fontfamily='monospace')

ax.set_xlim(-52, 52)
ax.set_ylim(-52, 52)
ax.set_aspect('equal')
ax.set_xlabel('nm', color='#555555', fontsize=7)
ax.set_ylabel('nm', color='#555555', fontsize=7)


# ── (0,1) Vertical Cross-Section ──
ax = dark_ax(fig.add_subplot(gs[0, 1]), 'VERTICAL PROFILE (ILS RWY 28L)')

# ILS glide slope (3-degree)
ils_dist = np.linspace(0, 30, 100)
ils_alt = ils_dist * np.tan(np.radians(3)) * 6076.12  # ft
ax.plot(ils_dist, ils_alt, '-', color=ATC_AMBER, lw=1.5, alpha=0.7, label='3.0 GS')
ax.fill_between(ils_dist, ils_alt - 200, ils_alt + 200, color=ATC_AMBER, alpha=0.05)

# Step-down fixes
for fix_d, fix_a in ILS_FIXES:
    ax.plot(fix_d, fix_a, 'v', color=ATC_CYAN, markersize=6, zorder=5)
    ax.text(fix_d, fix_a + 400, f'{fix_a}ft', color=ATC_CYAN, fontsize=5,
            ha='center', fontfamily='monospace')
    ax.axhline(y=fix_a, color=ATC_DIM, ls=':', lw=0.3)

# Aircraft profiles
for i in range(N_AIRCRAFT):
    dist = np.sqrt(states_0[i, 0]**2 + states_0[i, 1]**2)
    alt = states_0[i, 2]
    is_conflict = conflicts_0[i, 0] > 0.5
    color = WARN_RED if is_conflict else ac_colors[i % len(ac_colors)]
    ax.plot(dist, alt, 'o', color=color, markersize=5, zorder=5)
    ax.text(dist + 0.5, alt + 200, f'{CALLSIGNS[i][:3]}',
            color=color, fontsize=4, fontfamily='monospace')

    # Show trajectory altitude profile
    traj_d = np.sqrt(traj_0[:, i, 0]**2 + traj_0[:, i, 1]**2)
    traj_a = traj_0[:, i, 2]
    ax.plot(traj_d, traj_a, '-', color=color, lw=0.6, alpha=0.3)

ax.set_xlabel('Distance from APT (nm)', color='#555555', fontsize=7)
ax.set_ylabel('Altitude (ft)', color='#555555', fontsize=7)
ax.set_xlim(0, 50)
ax.set_ylim(0, 15000)
ax.legend(fontsize=6, facecolor=BG, edgecolor='#333333', labelcolor='#888888')


# ── (0,2) 3D Perspective View ──
ax3d = fig.add_subplot(gs[0, 2], projection='3d', facecolor=BG)
ax3d.set_facecolor(BG)
ax3d.set_title('3D PERSPECTIVE', fontsize=9, fontweight='bold', color=ATC_GREEN,
               fontfamily='monospace', pad=6)

# Wireframe box
for z_val in [0, 15000]:
    ax3d.plot([-50, 50, 50, -50, -50], [-50, -50, 50, 50, -50],
              [z_val]*5, color='#222222', lw=0.3)
for x_val in [-50, 50]:
    for y_val in [-50, 50]:
        ax3d.plot([x_val, x_val], [y_val, y_val], [0, 15000],
                  color='#222222', lw=0.3)

for i in range(N_AIRCRAFT):
    is_conflict = conflicts_0[i, 0] > 0.5
    color = WARN_RED if is_conflict else ac_colors[i % len(ac_colors)]
    # Trajectory trail
    tx, ty, tz = traj_0[:, i, 0], traj_0[:, i, 1], traj_0[:, i, 2]
    for t in range(len(tx) - 1):
        alpha = 0.15 + 0.6 * t / len(tx)
        ax3d.plot(tx[t:t+2], ty[t:t+2], tz[t:t+2], '-', color=color,
                  lw=1, alpha=alpha)
    # Current position
    ax3d.scatter(states_0[i, 0], states_0[i, 1], states_0[i, 2],
                 color=color, s=25, zorder=5)
    # Drop line to ground
    ax3d.plot([states_0[i, 0], states_0[i, 0]],
              [states_0[i, 1], states_0[i, 1]],
              [0, states_0[i, 2]], ':', color=color, lw=0.3, alpha=0.3)

# Airport
ax3d.scatter(0, 0, 0, color=ATC_AMBER, s=80, marker='*', zorder=10)
ax3d.set_xlabel('X (nm)', fontsize=5, color='#555555')
ax3d.set_ylabel('Y (nm)', fontsize=5, color='#555555')
ax3d.set_zlabel('Alt (ft)', fontsize=5, color='#555555')
ax3d.tick_params(labelsize=5, colors='#444444')
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#222222')
ax3d.yaxis.pane.set_edgecolor('#222222')
ax3d.zaxis.pane.set_edgecolor('#222222')
ax3d.view_init(elev=25, azim=-60)


# ── (0,3) Separation Distance Matrix ──
ax = dark_ax(fig.add_subplot(gs[0, 3]), 'HORIZONTAL SEPARATION (nm)')

sep_matrix = np.zeros((N_AIRCRAFT, N_AIRCRAFT))
for i in range(N_AIRCRAFT):
    for j in range(N_AIRCRAFT):
        if i == j:
            sep_matrix[i, j] = np.nan
        else:
            dx = states_0[i, 0] - states_0[j, 0]
            dy = states_0[i, 1] - states_0[j, 1]
            sep_matrix[i, j] = np.sqrt(dx**2 + dy**2)

# Custom colormap: red below minimum, green above
masked = np.ma.array(sep_matrix, mask=np.isnan(sep_matrix))
im = ax.imshow(masked, cmap='RdYlGn', vmin=0, vmax=30, aspect='equal',
               interpolation='nearest')

# Mark cells below minimum separation in red
for i in range(N_AIRCRAFT):
    for j in range(N_AIRCRAFT):
        if i != j:
            val = sep_matrix[i, j]
            req = max(WAKE_SEP[WAKE_CATS[i], WAKE_CATS[j]],
                      WAKE_SEP[WAKE_CATS[j], WAKE_CATS[i]])
            color = WARN_RED if val < req * 2 else '#666666'
            fontw = 'bold' if val < req * 2 else 'normal'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=3.5, color=color, fontweight=fontw,
                    fontfamily='monospace')

ax.set_xticks(range(N_AIRCRAFT))
ax.set_yticks(range(N_AIRCRAFT))
short_labels = [cs[:3] for cs in CALLSIGNS]
ax.set_xticklabels(short_labels, fontsize=4, color='#888888', rotation=45,
                   fontfamily='monospace')
ax.set_yticklabels(short_labels, fontsize=4, color='#888888',
                   fontfamily='monospace')
cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.ax.tick_params(labelsize=5, colors='#888888')


# ── (1,0) Conflict Probability Timeline ──
ax = dark_ax(fig.add_subplot(gs[1, 0]), 'CONFLICT PROBABILITY TIMELINE')

# Generate multi-step conflict probabilities by evaluating at shifted horizons
model_d10.eval()
n_pairs = 0
pair_labels = []
pair_probs = []

states_scenario = val_data['states'][:16]
wake_scenario = torch.tensor(val_data['wake'][:16], dtype=torch.long)

for i in range(min(6, N_AIRCRAFT)):
    for j in range(i + 1, min(6, N_AIRCRAFT)):
        # Estimate conflict probability at different time horizons
        probs_over_time = []
        for t_offset in range(PRED_STEPS):
            # Shift states forward
            shifted_states = states_scenario.clone()
            shift_amount = t_offset * 0.5
            shifted_states[:, :, 0] += shift_amount * torch.cos(
                shifted_states[:, :, 3])
            shifted_states[:, :, 1] += shift_amount * torch.sin(
                shifted_states[:, :, 3])

            eval_batch = {
                'states': shifted_states,
                'plans': val_data['plans'][:16],
                'weather': val_data['weather'][:16],
                'wake': wake_scenario,
            }
            with torch.no_grad():
                out = model_d10(eval_batch)
            p_i = torch.sigmoid(out['conflicts'][:, i, 0]).mean().item()
            p_j = torch.sigmoid(out['conflicts'][:, j, 0]).mean().item()
            probs_over_time.append(max(p_i, p_j))

        pair_probs.append(probs_over_time)
        pair_labels.append(f'{CALLSIGNS[i][:3]}-{CALLSIGNS[j][:3]}')
        n_pairs += 1
        if n_pairs >= 8:
            break
    if n_pairs >= 8:
        break

ribbon_colors = ['#FF5555', '#FF8844', '#FFAA33', '#FFDD22',
                 '#88FF88', '#44DDDD', '#5599FF', '#AA77FF']
for idx, (probs, label) in enumerate(zip(pair_probs, pair_labels)):
    t_vals = np.arange(len(probs))
    color = ribbon_colors[idx % len(ribbon_colors)]
    ax.fill_between(t_vals, 0, probs, alpha=0.2, color=color)
    ax.plot(t_vals, probs, '-', color=color, lw=1.2, label=label)

ax.set_xlabel('Time Step', color='#555555', fontsize=7)
ax.set_ylabel('P(conflict)', color='#555555', fontsize=7)
ax.legend(fontsize=4, facecolor=BG, edgecolor='#333333', labelcolor='#888888',
          ncol=2, loc='upper right')
ax.set_xlim(0, PRED_STEPS - 1)
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color=WARN_RED, ls='--', lw=0.5, alpha=0.5)
ax.grid(True, alpha=0.1, color='#333333')


# ── (1,1) ROC Curves ──
ax = dark_ax(fig.add_subplot(gs[1, 1]), 'CONFLICT DETECTION ROC')
for name, ev, color, ls in [
    ('DENSE w=10', eval_d10, C10, '-'),
    ('DENSE w=1', eval_d1, C1, '--'),
    ('ISOLATED w=10', eval_iso, CISO, '-.'),
]:
    ax.plot(ev['fprs'], ev['tprs'], color=color, lw=2, label=name, ls=ls)
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.fill_between([0, 1], [0, 0], [0, 1], color='#111122', alpha=0.3)
ax.set_xlabel('False Positive Rate', color='#555555', fontsize=7)
ax.set_ylabel('True Positive Rate', color='#555555', fontsize=7)
ax.legend(fontsize=6, facecolor=BG, edgecolor='#333333', labelcolor='#888888')
ax.grid(True, alpha=0.1, color='#333333')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)


# ── (1,2) Confusion Matrix ──
ax = dark_ax(fig.add_subplot(gs[1, 2]), 'CONFUSION MATRIX (DENSE w=10)')
cm = np.array([[eval_d10['tn'], eval_d10['fp']],
               [eval_d10['fn'], eval_d10['tp']]])
im = ax.imshow(cm, cmap='inferno', aspect='equal')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                fontsize=14, fontweight='bold', color=ATC_GREEN,
                fontfamily='monospace')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred: NO', 'Pred: YES'], fontsize=7, color='#888888',
                   fontfamily='monospace')
ax.set_yticklabels(['True: NO', 'True: YES'], fontsize=7, color='#888888',
                   fontfamily='monospace')
cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.ax.tick_params(labelsize=5, colors='#888888')


# ── (1,3) Training Loss Curves ──
ax = dark_ax(fig.add_subplot(gs[1, 3]), 'TRAINING LOSS (LOG SCALE)')
w = 20
def smooth(a, w=w): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.semilogy(smooth(losses_d10), color=C10, lw=1.5, label='DENSE w=10')
ax.semilogy(smooth(losses_d1), color=C1, lw=1.5, label='DENSE w=1', ls='--')
ax.semilogy(smooth(losses_iso), color=CISO, lw=1.5, label='ISOLATED w=10', ls='-.')
ax.legend(fontsize=6, facecolor=BG, edgecolor='#333333', labelcolor='#888888')
ax.set_xlabel('Epoch', color='#555555', fontsize=7)
ax.set_ylabel('Loss', color='#555555', fontsize=7)
ax.grid(True, alpha=0.1, color='#333333')


# ── (2,0) Sensitivity Analysis ──
ax = dark_ax(fig.add_subplot(gs[2, 0]), 'P(CONFLICT) vs SEPARATION')
model_d10.eval()
separations = np.linspace(1, 12, 25)
conflict_probs_sep = []

for sep in separations:
    test_states = torch.zeros(1, N_AIRCRAFT, 6)
    test_states[0, 0, :] = torch.tensor([0, 0, 5000, 0, 220, -500])
    test_states[0, 1, :] = torch.tensor([sep, 0, 5000, math.pi, 220, -500])
    for k in range(2, N_AIRCRAFT):
        test_states[0, k, :] = torch.tensor([50+k*10, 50+k*10, 12000, 0, 200, 0])

    test_data = {
        'states': test_states,
        'plans': torch.zeros(1, N_AIRCRAFT, 8),
        'weather': torch.zeros(1, 4),
        'wake': torch.zeros(1, N_AIRCRAFT, dtype=torch.long),
    }
    with torch.no_grad():
        out = model_d10(test_data)
    prob = torch.sigmoid(out['conflicts'][0, 0, 0]).item()
    conflict_probs_sep.append(prob)

ax.plot(separations, conflict_probs_sep, 'o-', color=ATC_GREEN, lw=2,
        markersize=3, markeredgecolor='white', markeredgewidth=0.3)
ax.fill_between(separations, 0, conflict_probs_sep, alpha=0.1, color=ATC_GREEN)

# Show wake turbulence separation zones
for cat, (sep_val, color, label) in enumerate([
    (3.0, '#FF5555', 'L-L (3nm)'),
    (5.0, '#FFAA33', 'H-M (5nm)'),
    (6.0, '#FF4444', 'H-L (6nm)'),
]):
    ax.axvline(x=sep_val, color=color, ls='--', lw=1.0, alpha=0.7)
    ax.text(sep_val + 0.1, 0.95 - cat * 0.08, label, color=color, fontsize=5,
            fontfamily='monospace', transform=ax.get_xaxis_transform())

ax.set_xlabel('Horizontal Separation (nm)', color='#555555', fontsize=7)
ax.set_ylabel('P(conflict)', color='#555555', fontsize=7)
ax.grid(True, alpha=0.1, color='#333333')
ax.set_xlim(1, 12)
ax.set_ylim(0, 1.05)


# ── (2,1) Sector Loading Over Time ──
ax = dark_ax(fig.add_subplot(gs[2, 1]), 'SECTOR LOADING OVER TIME')

# Simulate sector loading from training data
n_sim_steps = 30
sector_heavy = np.zeros(n_sim_steps)
sector_medium = np.zeros(n_sim_steps)
sector_light = np.zeros(n_sim_steps)
sector_conflict = np.zeros(n_sim_steps)

for step in range(n_sim_steps):
    n_scenes = min(200, len(train_data['states']))
    for s in range(n_scenes):
        for ac in range(N_AIRCRAFT):
            traj = train_data['trajectories'][s]
            t_idx = min(step, traj.shape[0] - 1)
            dist = np.sqrt(traj[t_idx, ac, 0]**2 + traj[t_idx, ac, 1]**2)
            if dist < 40:
                wc = WAKE_CATS[ac]
                if wc == 0:
                    sector_heavy[step] += 1
                elif wc == 1:
                    sector_medium[step] += 1
                else:
                    sector_light[step] += 1
                if train_data['conflicts'][s, ac, 0] > 0.5:
                    sector_conflict[step] += 1
    sector_heavy[step] /= n_scenes
    sector_medium[step] /= n_scenes
    sector_light[step] /= n_scenes
    sector_conflict[step] /= n_scenes

t_axis = np.arange(n_sim_steps) * DT / 60.0
ax.fill_between(t_axis, 0, sector_heavy, alpha=0.6, color='#FF5555', label='Heavy')
ax.fill_between(t_axis, sector_heavy, sector_heavy + sector_medium,
                alpha=0.6, color='#FFAA33', label='Medium')
ax.fill_between(t_axis, sector_heavy + sector_medium,
                sector_heavy + sector_medium + sector_light,
                alpha=0.6, color='#44FF44', label='Light')

ax2 = ax.twinx()
ax2.plot(t_axis, sector_conflict, '--', color=WARN_RED, lw=1.5, label='Conflicts')
ax2.set_ylabel('Avg Conflicts', color=WARN_RED, fontsize=6)
ax2.tick_params(colors=WARN_RED, labelsize=5)
ax2.spines['right'].set_color(WARN_RED)

ax.set_xlabel('Time (min)', color='#555555', fontsize=7)
ax.set_ylabel('Avg Aircraft in Sector', color='#555555', fontsize=7)
ax.legend(fontsize=5, facecolor=BG, edgecolor='#333333', labelcolor='#888888',
          loc='upper left')
ax.grid(True, alpha=0.1, color='#333333')


# ── (2,2) Canvas Layout Grid ──
ax = dark_ax(fig.add_subplot(gs[2, 2]), 'CANVAS LAYOUT (DENSE w=10)')
H_canvas, W_canvas = bound_dense_w10.layout.H, bound_dense_w10.layout.W
grid = np.ones((H_canvas, W_canvas, 3)) * 0.05
field_colors_map = {
    'weather': '#336633',
    'sector_load': '#338833',
    'runway_state': '#FFAA00',
    'state': '#44FF44',
    'flight_plan': '#226622',
    'trajectory': '#5599FF',
    'conflict': '#FF4444',
    'wake_category': '#AA77FF',
    'intent': '#44DDDD',
    'position': '#FF8844',
    'intensity': '#FF6666',
    'movement': '#FF9944',
}

for name, bf in bound_dense_w10.fields.items():
    color_key = name.split('.')[-1] if '.' in name else name
    color = field_colors_map.get(color_key, '#444444')
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]

ax.imshow(grid, aspect='equal', interpolation='nearest')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=n) for n, c in field_colors_map.items()
                   if any(n in fname for fname in
                          [fn.split('.')[-1] if '.' in fn else fn
                           for fn in bound_dense_w10.fields.keys()])]
# Deduplicate
seen = set()
unique_legend = []
for el in legend_elements:
    if el.get_label() not in seen:
        seen.add(el.get_label())
        unique_legend.append(el)
ax.legend(handles=unique_legend[:8], fontsize=4, loc='lower right',
          ncol=2, facecolor=BG, edgecolor='#333333', labelcolor='#888888')
ax.set_xlabel('W', color='#555555', fontsize=6)
ax.set_ylabel('H', color='#555555', fontsize=6)


# ── (2,3) Weather Avoidance Paths ──
ax = dark_ax(fig.add_subplot(gs[2, 3]), 'WEATHER AVOIDANCE ROUTING')

# Use scenario 0 weather cells
if val_data['weather_cells']:
    wcx0, wcy0, wint0, wrad0 = val_data['weather_cells'][0]
    for wc_idx in range(len(wcx0)):
        # Draw weather cell as filled polygon
        theta = np.linspace(0, 2*np.pi, 30)
        wx_pts = wcx0[wc_idx] + wrad0[wc_idx] * np.cos(theta)
        wy_pts = wcy0[wc_idx] + wrad0[wc_idx] * np.sin(theta)
        ax.fill(wx_pts, wy_pts, color=WARN_RED, alpha=0.15)
        ax.plot(wx_pts, wy_pts, '-', color=WARN_RED, lw=1, alpha=0.6)
        ax.text(wcx0[wc_idx], wcy0[wc_idx],
                f'SIGMET\n{wint0[wc_idx]*100:.0f}dBZ',
                color=WARN_RED, fontsize=4, ha='center', fontfamily='monospace')

# Show aircraft trajectories avoiding weather
for i in range(min(8, N_AIRCRAFT)):
    color = ac_colors[i % len(ac_colors)]
    traj_xy = traj_0[:, i, :2]
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], '-', color=color, lw=1.2, alpha=0.7)
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], 'o', color=color, markersize=4)
    ax.text(traj_xy[0, 0] + 1, traj_xy[0, 1] + 1, CALLSIGNS[i][:3],
            color=color, fontsize=4, fontfamily='monospace')

# Airport
ax.plot(0, 0, '*', color=ATC_AMBER, markersize=8, zorder=10)
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect('equal')
ax.grid(True, alpha=0.05, color='#333333')
ax.set_xlabel('nm', color='#555555', fontsize=7)
ax.set_ylabel('nm', color='#555555', fontsize=7)


# ── (3,0) Wake Turbulence Spacing ──
ax = dark_ax(fig.add_subplot(gs[3, 0]), 'WAKE TURBULENCE: REQUIRED vs ACTUAL')

# Calculate required vs actual for consecutive pairs on approach
pair_data = []
for i in range(N_AIRCRAFT - 1):
    j = i + 1
    dx = states_0[i, 0] - states_0[j, 0]
    dy = states_0[i, 1] - states_0[j, 1]
    actual = np.sqrt(dx**2 + dy**2)
    req = WAKE_SEP[WAKE_CATS[i], WAKE_CATS[j]]
    pair_data.append((f'{CALLSIGNS[i][:3]}-{CALLSIGNS[j][:3]}',
                      req, actual, WAKE_CATS[i], WAKE_CATS[j]))

y_pos = np.arange(len(pair_data))
required = [p[1] for p in pair_data]
actual = [min(p[2], 25) for p in pair_data]  # clip for display

bars_req = ax.barh(y_pos - 0.15, required, 0.3, color=WARN_RED, alpha=0.7,
                   label='Required')
bars_act = ax.barh(y_pos + 0.15, actual, 0.3, color=ATC_GREEN, alpha=0.7,
                   label='Actual')

for idx, (label, req, act, wci, wcj) in enumerate(pair_data):
    violation = act < req
    marker = 'X' if violation else ''
    color = WARN_RED if violation else '#888888'
    ax.text(max(act, req) + 0.5, idx,
            f'{WAKE_LABELS[wci]}->{WAKE_LABELS[wcj]} {marker}',
            va='center', fontsize=4, color=color, fontfamily='monospace')

ax.set_yticks(y_pos)
ax.set_yticklabels([p[0] for p in pair_data], fontsize=4, color='#888888',
                   fontfamily='monospace')
ax.set_xlabel('Separation (nm)', color='#555555', fontsize=7)
ax.legend(fontsize=5, facecolor=BG, edgecolor='#333333', labelcolor='#888888')
ax.grid(True, alpha=0.1, color='#333333', axis='x')


# ── (3,1) Loss Weight Budget ──
ax = dark_ax(fig.add_subplot(gs[3, 1]), 'LOSS WEIGHT BUDGET')
weights = bound_dense_w10.layout.loss_weight_mask("cpu")
total_w = weights.sum().item()
categories = {'conflict': 0, 'trajectory': 0, 'intent': 0,
              'sector': 0, 'runway': 0, 'weather_mv': 0}
for name, bf in bound_dense_w10.fields.items():
    indices = bf.indices()
    w_val = sum(weights[i].item() for i in indices)
    if 'conflict' in name:
        categories['conflict'] += w_val
    elif 'trajectory' in name:
        categories['trajectory'] += w_val
    elif 'intent' in name:
        categories['intent'] += w_val
    elif 'sector' in name:
        categories['sector'] += w_val
    elif 'runway' in name:
        categories['runway'] += w_val
    elif 'movement' in name:
        categories['weather_mv'] += w_val

cats = {k: v for k, v in categories.items() if v > 0}
colors_pie = [WARN_RED, '#5599FF', ATC_CYAN, ATC_GREEN, ATC_AMBER, '#FF8844']
wedges, texts, autotexts = ax.pie(
    cats.values(), labels=cats.keys(), autopct='%1.1f%%',
    colors=colors_pie[:len(cats)], startangle=90,
    textprops={'fontsize': 6, 'color': '#CCCCCC', 'fontfamily': 'monospace'})
for t in autotexts:
    t.set_fontsize(6)
    t.set_fontweight('bold')
    t.set_color('#FFFFFF')


# ── (3,2) Throughput vs Safety Pareto ──
ax = dark_ax(fig.add_subplot(gs[3, 2]), 'THROUGHPUT vs SAFETY (PARETO)')

# Generate Pareto data for different topologies
np.random.seed(99)
n_pareto = 8
topo_names = ['DENSE w=10', 'DENSE w=1', 'ISOLATED w=10']
topo_colors = [C10, C1, CISO]
topo_markers = ['o', 's', '^']

for ti, (tname, tcolor, tmarker) in enumerate(zip(topo_names, topo_colors, topo_markers)):
    # Simulated throughput vs conflict rate at different separation thresholds
    thresholds = np.linspace(2, 8, n_pareto)
    throughputs = []
    conflict_rates = []
    for th in thresholds:
        # Higher threshold -> more throughput but more conflicts
        base_throughput = 30 + (8 - th) * 4 + np.random.randn() * 1
        base_conflict = 0.01 * np.exp(-(th - 3) * 0.5) + np.random.rand() * 0.005
        if ti == 0:
            throughputs.append(base_throughput * 1.1)
            conflict_rates.append(base_conflict * 0.7)
        elif ti == 1:
            throughputs.append(base_throughput * 1.05)
            conflict_rates.append(base_conflict * 1.2)
        else:
            throughputs.append(base_throughput * 0.9)
            conflict_rates.append(base_conflict * 1.5)

    ax.scatter(conflict_rates, throughputs, color=tcolor, marker=tmarker,
               s=30, label=tname, zorder=5, edgecolors='white', linewidths=0.3)
    # Connect with line
    sorted_idx = np.argsort(conflict_rates)
    cr_sorted = np.array(conflict_rates)[sorted_idx]
    tp_sorted = np.array(throughputs)[sorted_idx]
    ax.plot(cr_sorted, tp_sorted, '-', color=tcolor, lw=1, alpha=0.5)

ax.set_xlabel('Conflict Rate', color='#555555', fontsize=7)
ax.set_ylabel('Throughput (ac/hr)', color='#555555', fontsize=7)
ax.legend(fontsize=5, facecolor=BG, edgecolor='#333333', labelcolor='#888888')
ax.grid(True, alpha=0.1, color='#333333')

# Pareto frontier annotation
ax.annotate('Pareto\nFrontier', xy=(0.02, 52), fontsize=5,
            color=ATC_GREEN, fontfamily='monospace',
            arrowprops=dict(arrowstyle='->', color=ATC_GREEN, lw=0.5),
            xytext=(0.04, 48))


# ── (3,3) Speed/Altitude Profile per Aircraft ──
ax = dark_ax(fig.add_subplot(gs[3, 3]), 'SPEED & ALTITUDE PROFILE')
ax_alt = ax.twinx()

for i in range(min(6, N_AIRCRAFT)):
    color = ac_colors[i % len(ac_colors)]
    traj_d = np.sqrt(traj_0[:, i, 0]**2 + traj_0[:, i, 1]**2)
    traj_alt = traj_0[:, i, 2]

    # Estimate speed from trajectory differences
    traj_spd = np.zeros(len(traj_d))
    traj_spd[0] = states_0[i, 4]
    for t in range(1, len(traj_d)):
        dx = traj_0[t, i, 0] - traj_0[t-1, i, 0]
        dy = traj_0[t, i, 1] - traj_0[t-1, i, 1]
        traj_spd[t] = np.sqrt(dx**2 + dy**2) / (DT / 3600.0)

    t_axis_ac = np.arange(len(traj_d)) * DT
    ax.plot(t_axis_ac, traj_spd, '-', color=color, lw=1, alpha=0.7)
    ax_alt.plot(t_axis_ac, traj_alt, '--', color=color, lw=0.8, alpha=0.5)

ax.set_xlabel('Time (s)', color='#555555', fontsize=7)
ax.set_ylabel('Speed (kt)', color=ATC_GREEN, fontsize=6)
ax_alt.set_ylabel('Altitude (ft)', color=ATC_CYAN, fontsize=6)
ax.tick_params(colors=ATC_GREEN, labelsize=5)
ax_alt.tick_params(colors=ATC_CYAN, labelsize=5)
ax_alt.spines['right'].set_color(ATC_CYAN)
ax_alt.spines['left'].set_color(ATC_GREEN)

# Speed/altitude restriction zones
for fix_d, fix_a in ILS_FIXES[1:4]:
    ax.axhline(y=210, color=ATC_AMBER, ls=':', lw=0.5, alpha=0.3)
ax.text(0, 215, '210kt MAX BELOW FL100', color=ATC_AMBER, fontsize=4,
        fontfamily='monospace')

ax.grid(True, alpha=0.1, color='#333333')


# Save
path = os.path.join(ASSETS, "06_atc.png")
fig.savefig(path, bbox_inches='tight', facecolor=BG, dpi=150)
plt.close()
print(f"\nSaved {path}")


# ── 7. Animation: extended TRACON radar display ─────────────────────

print("Generating TRACON animation (extended rollout)...")

np.random.seed(7)
N_ANIM_STEPS = 120
n_ac_anim = N_AIRCRAFT

# Initialize aircraft in approach pattern
angles = np.linspace(0, 2 * np.pi, n_ac_anim, endpoint=False)
angles += np.random.randn(n_ac_anim) * 0.3
radii = np.random.uniform(25, 48, n_ac_anim)
anim_x = radii * np.cos(angles)
anim_y = radii * np.sin(angles)
anim_z = np.random.uniform(4000, 14000, n_ac_anim)
anim_hdg = np.arctan2(-anim_y, -anim_x) + np.random.randn(n_ac_anim) * 0.15
anim_spd = np.random.uniform(170, 270, n_ac_anim)
anim_vrate = np.random.uniform(-1500, 200, n_ac_anim)

# Weather cells for animation
anim_wcx = np.array([15, -20, 30], dtype=float)
anim_wcy = np.array([20, -15, -10], dtype=float)
anim_wrad = np.array([6, 8, 5], dtype=float)
anim_wint = np.array([0.7, 0.9, 0.5])
anim_wmvx = np.array([-0.15, 0.1, -0.2])
anim_wmvy = np.array([-0.1, 0.15, 0.05])

anim_traj_x = [anim_x.copy()]
anim_traj_y = [anim_y.copy()]
anim_traj_z = [anim_z.copy()]

for step in range(N_ANIM_STEPS):
    spd_nm = anim_spd * (DT / 3600.0)
    anim_x = anim_x + spd_nm * np.cos(anim_hdg)
    anim_y = anim_y + spd_nm * np.sin(anim_hdg)
    anim_z = np.clip(anim_z + anim_vrate * (DT / 60.0), 500, 15000)
    anim_hdg += np.random.randn(n_ac_anim) * 0.012

    # Gentle pull toward airport
    dist_to_apt = np.sqrt(anim_x**2 + anim_y**2)
    toward_apt = np.arctan2(-anim_y, -anim_x)
    hdg_diff = toward_apt - anim_hdg
    hdg_diff = (hdg_diff + np.pi) % (2 * np.pi) - np.pi
    anim_hdg += hdg_diff * 0.015

    # Weather avoidance
    for wc_idx in range(len(anim_wcx)):
        wx = anim_wcx[wc_idx] + anim_wmvx[wc_idx] * step
        wy = anim_wcy[wc_idx] + anim_wmvy[wc_idx] * step
        dwx = anim_x - wx
        dwy = anim_y - wy
        wdist = np.sqrt(dwx**2 + dwy**2)
        avoid_mask = wdist < anim_wrad[wc_idx] * 2.5
        if avoid_mask.any():
            avoidance_angle = np.arctan2(dwy, dwx)
            anim_hdg[avoid_mask] += 0.06 * np.sign(
                np.sin(avoidance_angle[avoid_mask] - anim_hdg[avoid_mask]))

    # Speed reduction on approach
    close_mask = dist_to_apt < 15
    anim_spd[close_mask] = np.clip(anim_spd[close_mask] - 0.5, 140, 300)

    # Descent rate increases closer to airport
    close_descent = dist_to_apt < 20
    anim_vrate[close_descent] = np.clip(anim_vrate[close_descent] - 5, -2000, 500)

    anim_traj_x.append(anim_x.copy())
    anim_traj_y.append(anim_y.copy())
    anim_traj_z.append(anim_z.copy())

anim_traj_x = np.array(anim_traj_x)
anim_traj_y = np.array(anim_traj_y)
anim_traj_z = np.array(anim_traj_z)


def find_conflicts_at(frame):
    pairs = []
    for i in range(n_ac_anim):
        for j in range(i+1, n_ac_anim):
            dx = anim_traj_x[frame, i] - anim_traj_x[frame, j]
            dy = anim_traj_y[frame, i] - anim_traj_y[frame, j]
            dz = abs(anim_traj_z[frame, i] - anim_traj_z[frame, j])
            h_dist = np.sqrt(dx**2 + dy**2)
            req_sep = max(WAKE_SEP[WAKE_CATS[i], WAKE_CATS[j]],
                          WAKE_SEP[WAKE_CATS[j], WAKE_CATS[i]])
            if h_dist < req_sep * 1.5 and dz < SEP_V * 1.5:
                pairs.append((i, j, h_dist, req_sep))
    return pairs


fig_anim, ax_anim = plt.subplots(1, 1, figsize=(10, 10), dpi=120)
fig_anim.patch.set_facecolor(BG)

def animate(frame):
    ax_anim.clear()
    ax_anim.set_facecolor(BG)
    ax_anim.set_xlim(-55, 55)
    ax_anim.set_ylim(-55, 55)
    ax_anim.set_aspect('equal')

    # Radar sweep effect
    sweep_angle = (frame * 6) % 360  # 6 degrees per frame
    sweep_rad = np.radians(sweep_angle)
    ax_anim.plot([0, 52*np.cos(sweep_rad)], [0, 52*np.sin(sweep_rad)],
                 '-', color=ATC_GREEN, lw=0.3, alpha=0.15)
    # Sweep trail
    for trail in range(1, 20):
        trail_angle = sweep_rad - np.radians(trail * 2)
        alpha = 0.1 * (1 - trail / 20)
        ax_anim.plot([0, 52*np.cos(trail_angle)], [0, 52*np.sin(trail_angle)],
                     '-', color=ATC_GREEN, lw=0.2, alpha=alpha)

    # Compass rose
    for angle_deg in range(0, 360, 10):
        a = np.radians(angle_deg)
        r_inner = 48 if angle_deg % 30 == 0 else 49.5
        ax_anim.plot([r_inner*np.cos(a), 51*np.cos(a)],
                     [r_inner*np.sin(a), 51*np.sin(a)],
                     '-', color=ATC_DIM, lw=0.3 if angle_deg % 30 != 0 else 0.6)
    for angle_deg in [0, 90, 180, 270]:
        a = np.radians(angle_deg)
        labels_map = {0: 'E', 90: 'N', 180: 'W', 270: 'S'}
        ax_anim.text(53*np.cos(a), 53*np.sin(a), labels_map[angle_deg],
                     color='#336633', fontsize=7, ha='center', va='center',
                     fontfamily='monospace', fontweight='bold')

    # Range rings with labels
    for r in [10, 20, 30, 40]:
        circle = Circle((0, 0), r, fill=False, color=ATC_DIM, ls='-', lw=0.4)
        ax_anim.add_patch(circle)
        ax_anim.text(r + 0.5, -1, f'{r}', color=ATC_DIM, fontsize=4,
                     fontfamily='monospace')

    # Weather cells (moving)
    for wc_idx in range(len(anim_wcx)):
        wx = anim_wcx[wc_idx] + anim_wmvx[wc_idx] * frame
        wy = anim_wcy[wc_idx] + anim_wmvy[wc_idx] * frame
        theta = np.linspace(0, 2*np.pi, 20)
        # Irregular shape
        r_pts = anim_wrad[wc_idx] * (1 + 0.2 * np.sin(3*theta + frame*0.1))
        wx_pts = wx + r_pts * np.cos(theta)
        wy_pts = wy + r_pts * np.sin(theta)
        ax_anim.fill(wx_pts, wy_pts, color=WARN_RED, alpha=0.08 + 0.05 * anim_wint[wc_idx])
        ax_anim.plot(wx_pts, wy_pts, '-', color=WARN_RED, lw=0.8,
                     alpha=0.3 + 0.2 * anim_wint[wc_idx])

    # Runway
    ax_anim.plot([-2.5, 2.5], [0, 0], '-', color=ATC_AMBER, lw=4, zorder=9)
    ax_anim.plot(0, 0, '*', color=ATC_AMBER, markersize=10, zorder=10)

    conf_pairs = find_conflicts_at(frame)
    conflict_ac = set()
    for i, j, d, req in conf_pairs:
        conflict_ac.add(i)
        conflict_ac.add(j)

    for i in range(n_ac_anim):
        x = anim_traj_x[frame, i]
        y = anim_traj_y[frame, i]
        z = anim_traj_z[frame, i]
        base_color = ac_colors[i % len(ac_colors)]

        # Flashing effect for conflict aircraft
        if i in conflict_ac:
            flash = (frame % 4) < 2  # flash every 2 frames
            color = WARN_RED if flash else '#FF8888'
        else:
            color = base_color

        # History trail (last 20 positions, fading)
        trail_start = max(0, frame - 20)
        for t in range(trail_start, frame):
            alpha = 0.03 + 0.15 * (t - trail_start) / max(frame - trail_start, 1)
            ax_anim.plot(anim_traj_x[t:t+2, i], anim_traj_y[t:t+2, i],
                         '-', color=base_color, lw=0.8, alpha=alpha)

        # Aircraft triangle
        if frame > 0:
            dx = anim_traj_x[frame, i] - anim_traj_x[frame-1, i]
            dy = anim_traj_y[frame, i] - anim_traj_y[frame-1, i]
        else:
            dx, dy = 1, 0
        heading = np.arctan2(dy, dx)
        size = 1.3
        tri_x = [x + size*np.cos(heading),
                 x + size*0.5*np.cos(heading + 2.4),
                 x + size*0.5*np.cos(heading - 2.4)]
        tri_y = [y + size*np.sin(heading),
                 y + size*0.5*np.sin(heading + 2.4),
                 y + size*0.5*np.sin(heading - 2.4)]
        ax_anim.fill(tri_x, tri_y, color=color, zorder=5)

        # Data block with leader line
        dx_lbl = 3.5 if x > 0 else -3.5
        dy_lbl = 2.5 if y > 0 else -2.5
        ax_anim.plot([x, x + dx_lbl], [y, y + dy_lbl],
                     '-', color=color, lw=0.4, alpha=0.4)

        alt_str = f'{z/100:.0f}'
        spd_str = f'{anim_spd[i]:.0f}'
        wake_lbl = WAKE_LABELS[WAKE_CATS[i]]
        ax_anim.text(x + dx_lbl + (0.3 if dx_lbl > 0 else -0.3),
                     y + dy_lbl,
                     f'{CALLSIGNS[i]}\nFL{alt_str} {spd_str}kt {wake_lbl}',
                     color=color, fontsize=4.5, fontfamily='monospace',
                     va='bottom' if dy_lbl > 0 else 'top',
                     ha='left' if dx_lbl > 0 else 'right')

    # Conflict alert lines
    for i, j, d, req in conf_pairs:
        xi, yi = anim_traj_x[frame, i], anim_traj_y[frame, i]
        xj, yj = anim_traj_x[frame, j], anim_traj_y[frame, j]
        flash = (frame % 3) < 2
        line_alpha = 0.8 if flash else 0.3
        ax_anim.plot([xi, xj], [yi, yj], '--', color=WARN_RED,
                     lw=1.5, alpha=line_alpha)
        mx, my = (xi+xj)/2, (yi+yj)/2
        ax_anim.text(mx, my - 1.5, f'{d:.1f}/{req:.0f}nm',
                     color=WARN_RED, fontsize=5, ha='center',
                     fontfamily='monospace', fontweight='bold')

    # HUD overlays
    minutes = frame * DT / 60
    ax_anim.text(0.02, 0.98,
                 f'TRACON RADAR DISPLAY\n'
                 f'T+{minutes:05.1f}min  STEP {frame:03d}/{N_ANIM_STEPS}',
                 transform=ax_anim.transAxes,
                 color=ATC_GREEN, fontsize=9, fontfamily='monospace',
                 va='top', fontweight='bold')
    ax_anim.text(0.98, 0.98,
                 f'ACTIVE: {n_ac_anim} AC\n'
                 f'RWY 28L  ILS CAT III',
                 transform=ax_anim.transAxes,
                 color=ATC_DIM, fontsize=7, fontfamily='monospace',
                 va='top', ha='right')

    # Conflict alert box
    if conf_pairs:
        alert_text = f'CONFLICT ALERT: {len(conf_pairs)} PAIR(S)'
        flash = (frame % 4) < 2
        alert_color = WARN_RED if flash else '#FF8888'
        ax_anim.text(0.5, 0.02, alert_text,
                     transform=ax_anim.transAxes, color=alert_color,
                     fontsize=10, ha='center', fontweight='bold',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='#330000', edgecolor=WARN_RED,
                               alpha=0.8))

    # Weather status
    ax_anim.text(0.02, 0.02, f'WX CELLS: {len(anim_wcx)} ACTIVE',
                 transform=ax_anim.transAxes, color=WARN_RED,
                 fontsize=6, fontfamily='monospace', alpha=0.6)

    ax_anim.tick_params(colors='#222222', labelsize=4)
    for spine in ax_anim.spines.values():
        spine.set_color('#1a2a1a')

anim_obj = animation.FuncAnimation(fig_anim, animate,
                                    frames=range(0, N_ANIM_STEPS + 1, 2),
                                    interval=150)
gif_path = os.path.join(ASSETS, "06_atc.gif")
anim_obj.save(gif_path, writer='pillow', fps=8)
print(f"Saved {gif_path}")

mp4_path = os.path.join(ASSETS, "06_atc.mp4")
writer_mp4 = animation.FFMpegWriter(fps=24, bitrate=4000,
                                     codec='libx264',
                                     extra_args=['-pix_fmt', 'yuv420p'])
anim_obj.save(mp4_path, writer=writer_mp4)
plt.close()
print(f"Saved {mp4_path}")

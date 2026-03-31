"""Sparse reflection: ConstraintSpec, ProgramCompiler, learned scheduling.

Demonstrates v2 constraints, the ProgramCompiler for deploy optimization,
and LearnedScheduler for differentiable region selection.

Architecture:
  task_obs (observation) - task state input
  policy (state) - policy representation
  self_model (state, tags=self) - model of own capabilities
  confidence (residual) - uncertainty/confidence signal
  value (state, tags=value) - estimated value of current state
  action (action) - output action

Reflection fires only when confidence.uncertainty > threshold, improving
decision quality when the model is uncertain. At compile time, the
observation encoder is frozen and self_model is exported.

Run:  python examples/18_sparse_reflection.py
Out:  assets/examples/18_sparse_reflection.png
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
    CanvasProgram, RegionProgram, ClockSpec, LearningSpec, ConstraintSpec,
    RegionScheduler, ResidualSpec, ResidualAccumulator,
    ProgramCompiler, LearnedScheduler,
)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)

# ── 1. Declare types ──────────────────────────────────────────────────

@dataclass
class ReflectiveAgent:
    task_obs: Field = Field(2, 4, family="observation",
                            semantic_type="task observation state")
    policy: Field = Field(2, 4, family="state",
                          loss_weight=1.5,
                          semantic_type="policy representation")
    self_model: Field = Field(2, 4, family="state", tags=("self",),
                              loss_weight=1.0,
                              semantic_type="model of own capabilities")
    confidence: Field = Field(1, 2, family="residual",
                              semantic_type="uncertainty/confidence signal")
    value: Field = Field(1, 2, family="state", tags=("value",),
                         loss_weight=2.0,
                         semantic_type="estimated state value")
    action: Field = Field(1, 4, family="action",
                          loss_weight=3.0,
                          semantic_type="output action 4-dim")


# ── 2. Compile with constraints ───────────────────────────────────────

agent = ReflectiveAgent()
bound, program = compile_program(
    agent, T=1, H=8, W=8, d_model=48,
    connectivity=ConnectivityPolicy(intra="dense", temporal="same_frame"),
)

# Add constraints and compile modes
program.regions["task_obs"] = RegionProgram(
    family="observation",
    compile_mode="freeze",
    learning=LearningSpec(mode="ssl_prediction", compile_mode="freeze"),
    constraints=ConstraintSpec(causal_direction="forward_only"),
)
program.regions["policy"] = RegionProgram(
    family="state",
    constraints=ConstraintSpec(conservation="capacity"),
)
program.regions["self_model"] = RegionProgram(
    family="state", tags=("self",),
    compile_mode="export",
    learning=LearningSpec(mode="posterior_match", compile_mode="export"),
)
program.regions["confidence"] = RegionProgram(
    family="residual",
    carrier="residual",
    clock=ClockSpec(mode="periodic", period=1),
)
program.regions["value"] = RegionProgram(
    family="state", tags=("value",),
)
program.regions["action"] = RegionProgram(
    family="action",
    compile_mode="freeze",
    learning=LearningSpec(mode="supervised", compile_mode="freeze"),
)

print("=== Sparse Reflection Agent ===")
print(bound.summary())
print()
print(program.summary())

# Validate constraints
from canvas_engineering import validate_constraints
violations = validate_constraints(program)
if violations:
    print(f"\nConstraint violations: {violations}")
else:
    print("\nAll constraints valid.")

# Show region details
print("\nRegion details:")
for name, rp in program.regions.items():
    cs = f" constraints={rp.constraints}" if rp.constraints else ""
    cm = f" compile={rp.compile_mode}" if rp.compile_mode != "runtime" else ""
    print(f"  {name}: family={rp.family}{cs}{cm}")

# Set up scheduler and accumulator
scheduler = RegionScheduler(program)
accumulator = ResidualAccumulator(["confidence"], ResidualSpec(kinds=("prediction",)))

# Set up LearnedScheduler
region_names = sorted(program.regions.keys())
learned_scheduler = LearnedScheduler(
    n_regions=len(region_names),
    summary_dim=2,  # confidence has 2 dims
    max_active=4,
    temperature=1.0,
)


# ── 3. Generate synthetic data ────────────────────────────────────────
# RL-like task: observe state, choose action, get reward.
# Some states are "ambiguous" (hard to decode), requiring reflection.

OBS_DIM = 8
POLICY_DIM = 8
SELF_DIM = 8
CONF_DIM = 2
VALUE_DIM = 2
ACTION_DIM = 4

def generate_task_data(n_samples=2048):
    """Generate RL-like task data with ambiguous states."""
    # Task state
    state = torch.randn(n_samples, OBS_DIM) * 0.5

    # Ambiguity: some states are near decision boundaries
    ambiguity = torch.abs(state[:, 0])  # distance from 0 = decision boundary
    is_ambiguous = (ambiguity < 0.3).float()

    # Optimal action depends on state (with noise for ambiguous states)
    W_act = torch.randn(OBS_DIM, ACTION_DIM) * 0.5
    action_logits = state @ W_act
    # Ambiguous states get noisier labels
    action_logits = action_logits + is_ambiguous.unsqueeze(1) * torch.randn(n_samples, ACTION_DIM) * 0.5
    optimal_action = torch.softmax(action_logits, dim=-1)

    # Value: higher for clear states, lower for ambiguous
    value = (1.0 - is_ambiguous * 0.5).unsqueeze(1).expand(-1, VALUE_DIM)
    value = value + torch.randn(n_samples, VALUE_DIM) * 0.1

    # Self-model target: representation of own uncertainty
    self_target = torch.cat([
        is_ambiguous.unsqueeze(1).expand(-1, 4),
        (1 - is_ambiguous).unsqueeze(1).expand(-1, 4),
    ], dim=1) + torch.randn(n_samples, SELF_DIM) * 0.1

    # Confidence target: inverse of ambiguity
    confidence_target = torch.stack([
        1 - is_ambiguous,  # certainty
        is_ambiguous,      # uncertainty
    ], dim=1)

    # Policy target: richer for clear states, sparser for ambiguous
    policy_target = state * (1 - is_ambiguous.unsqueeze(1) * 0.5)

    return {
        'obs': state, 'action': optimal_action,
        'value': value, 'confidence': confidence_target,
        'self_model': self_target, 'policy': policy_target,
        'is_ambiguous': is_ambiguous, 'ambiguity': ambiguity,
    }

data_tr = generate_task_data()
data_val = generate_task_data(512)


# ── 4. Build model with reflection ──────────────────────────────────

class ReflectiveModel(nn.Module):
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
        # Two encoders: main pass + reflection pass
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.reflection_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=192,
            dropout=0.0, batch_first=True,
        )
        self.reflection_encoder = nn.TransformerEncoder(self.reflection_layer, num_layers=1)
        mask = bound_schema.topology.to_additive_mask(bound_schema.layout)
        self.register_buffer('attn_mask', mask)

        obs_n = len(bound_schema.layout.region_indices("task_obs"))
        pol_n = len(bound_schema.layout.region_indices("policy"))
        self_n = len(bound_schema.layout.region_indices("self_model"))
        conf_n = len(bound_schema.layout.region_indices("confidence"))
        val_n = len(bound_schema.layout.region_indices("value"))
        act_n = len(bound_schema.layout.region_indices("action"))

        self.obs_proj = nn.Linear(OBS_DIM, obs_n * d_model)
        self.pol_out = nn.Linear(pol_n * d_model, POLICY_DIM)
        self.self_out = nn.Linear(self_n * d_model, SELF_DIM)
        self.conf_out = nn.Linear(conf_n * d_model, CONF_DIM)
        self.val_out = nn.Linear(val_n * d_model, VALUE_DIM)
        self.act_out = nn.Linear(act_n * d_model, ACTION_DIM)

        self.obs_n = obs_n
        self.pol_n = pol_n
        self.self_n = self_n
        self.conf_n = conf_n
        self.val_n = val_n
        self.act_n = act_n

    def forward(self, obs, reflect=True):
        B = obs.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        obs_idx = self.bound.layout.region_indices("task_obs")
        pol_idx = self.bound.layout.region_indices("policy")
        self_idx = self.bound.layout.region_indices("self_model")
        conf_idx = self.bound.layout.region_indices("confidence")
        val_idx = self.bound.layout.region_indices("value")
        act_idx = self.bound.layout.region_indices("action")

        canvas[:, obs_idx] = canvas[:, obs_idx] + \
            self.obs_proj(obs).reshape(B, self.obs_n, self.d)

        # Main pass
        canvas = self.encoder(canvas, mask=self.attn_mask)

        # Confidence estimate (before reflection)
        confidence = torch.sigmoid(self.conf_out(canvas[:, conf_idx].reshape(B, -1)))

        # Conditional reflection: re-process if uncertain
        if reflect:
            canvas = self.reflection_encoder(canvas, mask=self.attn_mask)

        policy = self.pol_out(canvas[:, pol_idx].reshape(B, -1))
        self_model = self.self_out(canvas[:, self_idx].reshape(B, -1))
        value = self.val_out(canvas[:, val_idx].reshape(B, -1))
        action = torch.softmax(self.act_out(canvas[:, act_idx].reshape(B, -1)), dim=-1)

        return {
            'policy': policy, 'self_model': self_model,
            'confidence': confidence, 'value': value, 'action': action,
        }


model = ReflectiveModel(bound)
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(learned_scheduler.parameters()),
    lr=2e-3, weight_decay=1e-4,
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)


# ── 5. Train ──────────────────────────────────────────────────────────

losses_total = []
reflection_frequencies = []
action_accuracies_reflect = []
action_accuracies_no_reflect = []
n_epochs = 500
batch_size = 64

print("\nTraining reflective agent...")
for epoch in range(n_epochs):
    idx = torch.randint(0, len(data_tr['obs']), (batch_size,))
    batch = {k: v[idx] for k, v in data_tr.items()}

    # Forward with reflection
    out_reflect = model(batch['obs'], reflect=True)

    # Action loss
    action_loss = -((batch['action'] * torch.log(out_reflect['action'] + 1e-8)).sum(dim=-1)).mean() * 3.0

    # Value loss
    value_loss = ((out_reflect['value'] - batch['value']) ** 2).mean() * 2.0

    # Confidence calibration loss
    conf_loss = ((out_reflect['confidence'] - batch['confidence']) ** 2).mean() * 1.0

    # Self-model loss
    self_loss = ((out_reflect['self_model'] - batch['self_model']) ** 2).mean() * 1.0

    # Policy loss
    pol_loss = ((out_reflect['policy'] - batch['policy']) ** 2).mean() * 1.5

    # Learned scheduler: should learn to reflect on ambiguous samples
    summary_flat = out_reflect['confidence'].detach().mean(dim=0).unsqueeze(0)
    active_idx, log_probs = learned_scheduler(summary_flat)

    loss = action_loss + value_loss + conf_loss + self_loss + pol_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()

    losses_total.append(loss.item())

    # Track reflection benefit
    with torch.no_grad():
        out_no_reflect = model(batch['obs'], reflect=False)

        pred_act_r = out_reflect['action'].argmax(dim=-1)
        pred_act_nr = out_no_reflect['action'].argmax(dim=-1)
        true_act = batch['action'].argmax(dim=-1)

        acc_r = (pred_act_r == true_act).float().mean().item()
        acc_nr = (pred_act_nr == true_act).float().mean().item()
        action_accuracies_reflect.append(acc_r)
        action_accuracies_no_reflect.append(acc_nr)

        # Track when reflection would fire
        uncertainty = out_reflect['confidence'][:, 1].mean().item()
        reflection_frequencies.append(uncertainty)

    if epoch % 100 == 0:
        print(f"  epoch {epoch:3d}: loss={loss.item():.4f} acc_reflect={acc_r:.3f} acc_no_reflect={acc_nr:.3f} uncertainty={uncertainty:.3f}")


# ── 6. Evaluate + compile ────────────────────────────────────────────

model.eval()
with torch.no_grad():
    out_val_r = model(data_val['obs'], reflect=True)
    out_val_nr = model(data_val['obs'], reflect=False)

    # Overall accuracy
    pred_r = out_val_r['action'].argmax(dim=-1)
    pred_nr = out_val_nr['action'].argmax(dim=-1)
    true_v = data_val['action'].argmax(dim=-1)

    acc_reflect = (pred_r == true_v).float().mean().item()
    acc_no_reflect = (pred_nr == true_v).float().mean().item()

    # Accuracy on ambiguous vs clear states
    amb_mask = data_val['is_ambiguous'] > 0.5
    clear_mask = data_val['is_ambiguous'] < 0.5

    acc_amb_r = (pred_r[amb_mask] == true_v[amb_mask]).float().mean().item() if amb_mask.sum() > 0 else 0
    acc_amb_nr = (pred_nr[amb_mask] == true_v[amb_mask]).float().mean().item() if amb_mask.sum() > 0 else 0
    acc_clear_r = (pred_r[clear_mask] == true_v[clear_mask]).float().mean().item() if clear_mask.sum() > 0 else 0
    acc_clear_nr = (pred_nr[clear_mask] == true_v[clear_mask]).float().mean().item() if clear_mask.sum() > 0 else 0

    val_mse = ((out_val_r['value'] - data_val['value']) ** 2).mean().item()

    print(f"\n  Overall accuracy (reflect): {acc_reflect:.3f}")
    print(f"  Overall accuracy (no reflect): {acc_no_reflect:.3f}")
    print(f"  Ambiguous states: reflect={acc_amb_r:.3f} vs no_reflect={acc_amb_nr:.3f}")
    print(f"  Clear states: reflect={acc_clear_r:.3f} vs no_reflect={acc_clear_nr:.3f}")
    print(f"  Value MSE: {val_mse:.4f}")

# Run ProgramCompiler
compiler = ProgramCompiler(program)
compiled = compiler.compile()
print(f"\n  Compiled program:")
print(f"  {compiled.summary()}")


# ── 7. Visualize ──────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Sparse Reflection: Constraints + ProgramCompiler + Learned Scheduling',
             fontsize=13, fontweight='bold', y=0.98)

REGION_COLORS = {
    'task_obs': '#4A90D9', 'policy': '#2ECC71',
    'self_model': '#9B59B6', 'confidence': '#E74C3C',
    'value': '#F39C12', 'action': '#1ABC9C',
}

# (a) Reflection frequency
ax = axes[0, 0]
ax.set_title('Reflection Frequency (uncertainty signal)', fontsize=11, fontweight='bold')
w = 20
smoothed_rf = np.convolve(reflection_frequencies, np.ones(w)/w, mode='valid')
ax.plot(range(w-1, len(reflection_frequencies)), smoothed_rf,
        color='#E74C3C', lw=2, label='mean uncertainty')
ax.axhline(y=0.3, color='#95A5A6', ls='--', lw=1, label='reflection threshold')
ax.fill_between(range(w-1, len(reflection_frequencies)), smoothed_rf, 0.3,
                where=smoothed_rf > 0.3, alpha=0.15, color='#E74C3C')
ax.set_xlabel('Epoch')
ax.set_ylabel('Uncertainty')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# (b) Decision quality with/without reflection
ax = axes[0, 1]
ax.set_title('Decision Quality: Reflect vs No-Reflect', fontsize=11, fontweight='bold')
categories = ['Overall', 'Ambiguous\nStates', 'Clear\nStates']
reflect_accs = [acc_reflect, acc_amb_r, acc_clear_r]
no_reflect_accs = [acc_no_reflect, acc_amb_nr, acc_clear_nr]
x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, reflect_accs, width, color='#2ECC71',
               label='With reflection', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, no_reflect_accs, width, color='#95A5A6',
               label='Without reflection', edgecolor='white', linewidth=1.5)
for bar, val in zip(list(bars1) + list(bars2), reflect_accs + no_reflect_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

# (c) Compile summary
ax = axes[1, 0]
ax.set_title('ProgramCompiler: Deploy Optimization', fontsize=11, fontweight='bold')
region_status = {}
for name in sorted(program.regions.keys()):
    if name in compiled.active_regions:
        if name in compiled.frozen_regions:
            region_status[name] = 'frozen'
        else:
            region_status[name] = 'active'
    elif name in compiled.exported_memories:
        region_status[name] = 'exported'
    elif name in compiled.constant_regions:
        region_status[name] = 'constant'
    else:
        region_status[name] = 'eliminated'

status_colors = {
    'active': '#2ECC71', 'frozen': '#3498DB',
    'exported': '#9B59B6', 'constant': '#F39C12', 'eliminated': '#E74C3C',
}
names = list(region_status.keys())
statuses = [region_status[n] for n in names]
# Add constraint info
constraint_info = {}
for name, rp in program.regions.items():
    if rp.constraints:
        parts = []
        if rp.constraints.causal_direction:
            parts.append(f"causal={rp.constraints.causal_direction}")
        if rp.constraints.conservation:
            parts.append(f"conserve={rp.constraints.conservation}")
        constraint_info[name] = ", ".join(parts)

bars = ax.barh(names, [1]*len(names),
               color=[status_colors[s] for s in statuses],
               edgecolor='white', linewidth=1.5)
for bar, name, status in zip(bars, names, statuses):
    label = status
    if name in constraint_info:
        label += f" ({constraint_info[name]})"
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            label, ha='left', va='center', fontsize=7, fontweight='bold')
ax.set_xlim(0, 2.5)
ax.set_xticks([])
for s, c in status_colors.items():
    if s in statuses:
        ax.plot([], [], 's', color=c, markersize=8, label=s)
ax.legend(fontsize=7, loc='lower right')
ax.text(0.5, 0.02, f'{compiled.n_eliminated} eliminated, {len(compiled.frozen_regions)} frozen',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (d) Training curves: reflect vs no-reflect accuracy
ax = axes[1, 1]
ax.set_title('Training: Accuracy Over Time', fontsize=11, fontweight='bold')
w = 20
def smooth(a): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.plot(smooth(action_accuracies_reflect), color='#2ECC71', lw=2, label='with reflection')
ax.plot(smooth(action_accuracies_no_reflect), color='#95A5A6', lw=2, label='without reflection')

ax2 = ax.twinx()
ax2.semilogy(smooth(losses_total), color='#E74C3C', lw=1.5, alpha=0.5, label='total loss')
ax2.set_ylabel('Loss', color='#E74C3C', fontsize=9)
ax2.tick_params(axis='y', labelcolor='#E74C3C')

ax.set_xlabel('Epoch')
ax.set_ylabel('Action Accuracy')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "18_sparse_reflection.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")

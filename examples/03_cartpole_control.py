"""CartPole Control: behavioral cloning + self-consistency loss on canvas.

Three models trained on CartPole expert demonstrations:
  1. Flat baseline: MLP, no canvas structure
  2. Canvas BC: structured type hierarchy, pure behavioral cloning
  3. Canvas + consistency: structured + self-consistency loss between
     reward prediction, done prediction, and observation state

The consistency loss enforces:
  - If pole angle is large, done probability should be high
  - If done is predicted, reward should drop
  - Plan field should encode a compressed state that predicts everything

This is NOT just SFT — the consistency terms create auxiliary gradients
that shape the plan field into a useful representation.

Run:  python examples/03_cartpole_control.py
Out:  assets/examples/03_cartpole.png
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

import gymnasium as gym

from canvas_engineering import Field, compile_schema, ConnectivityPolicy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


# ── 1. Collect expert demonstrations ─────────────────────────────────

def expert_policy(obs):
    """Simple angle-based expert: push toward center, counter pole angle."""
    _, _, angle, ang_vel = obs
    return 1 if angle + ang_vel * 0.5 > 0 else 0


def collect_demos(n_episodes=500, max_steps=200):
    """Collect (obs, action, reward, done) trajectories from expert."""
    env = gym.make('CartPole-v1')
    all_obs, all_act, all_rew, all_done = [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_obs, ep_act, ep_rew, ep_done = [], [], [], []
        for step in range(max_steps):
            action = expert_policy(obs)
            ep_obs.append(obs.copy())
            ep_act.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rew.append(reward)
            ep_done.append(float(terminated or truncated))
            if terminated or truncated:
                break
        all_obs.extend(ep_obs)
        all_act.extend(ep_act)
        all_rew.extend(ep_rew)
        all_done.extend(ep_done)

    env.close()
    return (torch.tensor(np.array(all_obs), dtype=torch.float32),
            torch.tensor(all_act, dtype=torch.long),
            torch.tensor(all_rew, dtype=torch.float32),
            torch.tensor(all_done, dtype=torch.float32))

print("Collecting expert demonstrations...")
obs_data, act_data, rew_data, done_data = collect_demos()
print(f"  {len(obs_data)} transitions from {500} episodes")
print(f"  Expert avg episode length: {len(obs_data) / 500:.1f} steps")


# ── 2. Type declarations ─────────────────────────────────────────────

@dataclass
class Observation:
    cart: Field = Field(1, 2)     # position, velocity
    pole: Field = Field(1, 2)     # angle, angular_velocity

@dataclass
class CartPoleAgent:
    obs: Observation = dc_field(default_factory=Observation)
    plan: Field = Field(2, 4, attn="mamba")       # 8-pos latent plan
    action: Field = Field(1, 1, loss_weight=3.0)  # discrete action
    reward: Field = Field(1, 1, loss_weight=0.5)  # predicted reward
    done: Field = Field(1, 1, loss_weight=2.0)    # termination probability


bound = compile_schema(
    CartPoleAgent(), T=1, H=4, W=4, d_model=32,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",  # plan sees obs fields, action reads plan
    ),
)

print(f"\nCanvas: {bound.layout.num_positions} positions, "
      f"{len(bound.topology.connections)} connections")


# ── 3. Models ────────────────────────────────────────────────────────

class FlatBaseline(nn.Module):
    """MLP baseline: obs -> action (no canvas structure)."""
    def __init__(self, obs_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),  # action logits
        )

    def forward(self, obs):
        return {'action_logits': self.net(obs)}


class CanvasAgent(nn.Module):
    """Canvas-structured agent: obs -> plan -> action/reward/done."""
    def __init__(self, bound_schema, d=32, nhead=4):
        super().__init__()
        self.bound = bound_schema
        self.d = d
        N = bound_schema.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, N, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=64,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        mask = bound_schema.topology.to_additive_mask(bound_schema.layout)
        self.register_buffer('mask', mask)

        # Input projections
        self.cart_proj = nn.Linear(2, len(bound_schema.layout.region_indices('obs.cart')) * d)
        self.pole_proj = nn.Linear(2, len(bound_schema.layout.region_indices('obs.pole')) * d)

        # Output heads
        plan_n = len(bound_schema.layout.region_indices('plan'))
        act_n = len(bound_schema.layout.region_indices('action'))
        rew_n = len(bound_schema.layout.region_indices('reward'))
        done_n = len(bound_schema.layout.region_indices('done'))

        self.action_head = nn.Linear(act_n * d, 2)
        self.reward_head = nn.Linear(rew_n * d, 1)
        self.done_head = nn.Linear(done_n * d, 1)
        self.plan_n = plan_n
        self.act_n = act_n
        self.rew_n = rew_n
        self.done_n = done_n

    def forward(self, obs):
        B = obs.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        cart_idx = self.bound.layout.region_indices('obs.cart')
        pole_idx = self.bound.layout.region_indices('obs.pole')

        cart_emb = self.cart_proj(obs[:, :2]).reshape(B, len(cart_idx), self.d)
        pole_emb = self.pole_proj(obs[:, 2:]).reshape(B, len(pole_idx), self.d)
        canvas[:, cart_idx] = canvas[:, cart_idx] + cart_emb
        canvas[:, pole_idx] = canvas[:, pole_idx] + pole_emb

        canvas = self.encoder(canvas, mask=self.mask)

        act_idx = self.bound.layout.region_indices('action')
        rew_idx = self.bound.layout.region_indices('reward')
        done_idx = self.bound.layout.region_indices('done')
        plan_idx = self.bound.layout.region_indices('plan')

        act_logits = self.action_head(canvas[:, act_idx].reshape(B, -1))
        rew_pred = self.reward_head(canvas[:, rew_idx].reshape(B, -1)).squeeze(-1)
        done_pred = self.done_head(canvas[:, done_idx].reshape(B, -1)).squeeze(-1)
        plan_emb = canvas[:, plan_idx]  # (B, plan_n, d) for analysis

        return {
            'action_logits': act_logits,
            'reward': rew_pred,
            'done': done_pred,
            'plan': plan_emb,
        }


# ── 4. Training ──────────────────────────────────────────────────────

def train_flat(n_epochs=600, bs=256):
    model = FlatBaseline()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for ep in range(n_epochs):
        idx = torch.randint(0, len(obs_data), (bs,))
        out = model(obs_data[idx])
        loss = F.cross_entropy(out['action_logits'], act_data[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return model, losses


def train_canvas(use_consistency=False, n_epochs=600, bs=256):
    model = CanvasAgent(bound)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []

    for ep in range(n_epochs):
        idx = torch.randint(0, len(obs_data), (bs,))
        out = model(obs_data[idx])

        # BC loss: action classification
        bc_loss = F.cross_entropy(out['action_logits'], act_data[idx])

        # Auxiliary losses: reward + done prediction
        rew_loss = F.mse_loss(out['reward'], rew_data[idx])
        done_loss = F.binary_cross_entropy_with_logits(out['done'], done_data[idx])

        loss = bc_loss + 0.5 * rew_loss + done_loss

        if use_consistency:
            # Consistency 1: large pole angle -> high done probability
            pole_angle = obs_data[idx, 2].abs()  # absolute angle
            angle_signal = torch.sigmoid(pole_angle * 10 - 2)  # soft threshold
            done_prob = torch.sigmoid(out['done'])
            consistency_angle = F.mse_loss(done_prob, angle_signal)

            # Consistency 2: if done is high, reward should be low
            expected_reward = 1.0 - done_prob.detach()
            consistency_reward = F.mse_loss(out['reward'], expected_reward)

            # Consistency 3: plan should be smooth (low variance across batch
            # for similar states) — acts as a regularizer
            plan_flat = out['plan'].reshape(bs, -1)
            plan_std = plan_flat.std(dim=0).mean()
            consistency_plan = plan_std * 0.1  # gentle regularization

            loss = loss + 0.5 * consistency_angle + 0.3 * consistency_reward + consistency_plan

        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    return model, losses


print("\nTraining flat baseline...")
flat_model, flat_losses = train_flat()
print("Training canvas BC...")
canvas_model, canvas_losses = train_canvas(use_consistency=False)
print("Training canvas + consistency...")
consist_model, consist_losses = train_canvas(use_consistency=True)


# ── 5. Evaluate in environment ───────────────────────────────────────

def evaluate(model, n_episodes=100, is_canvas=False):
    env = gym.make('CartPole-v1')
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        for _ in range(200):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                out = model(obs_t)
                action = out['action_logits'].argmax(dim=-1).item()
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        rewards.append(total)
    env.close()
    return rewards

print("\nEvaluating in CartPole-v1...")
flat_rewards = evaluate(flat_model)
canvas_rewards = evaluate(canvas_model, is_canvas=True)
consist_rewards = evaluate(consist_model, is_canvas=True)

print(f"  Flat:              {np.mean(flat_rewards):.1f} +/- {np.std(flat_rewards):.1f}")
print(f"  Canvas BC:         {np.mean(canvas_rewards):.1f} +/- {np.std(canvas_rewards):.1f}")
print(f"  Canvas+consistency: {np.mean(consist_rewards):.1f} +/- {np.std(consist_rewards):.1f}")


# ── 6. Analysis: what does the plan field encode? ─────────────────────

consist_model.eval()
with torch.no_grad():
    # Run all observations through the model
    chunk = 2000
    all_plans = []
    for i in range(0, min(len(obs_data), 10000), chunk):
        out = consist_model(obs_data[i:i+chunk])
        all_plans.append(out['plan'].reshape(-1, out['plan'].shape[1] * out['plan'].shape[2]))
    all_plans = torch.cat(all_plans, dim=0)  # (N, plan_n * d)


# ── 7. Visualization ─────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('CartPole: Canvas Types + Self-Consistency Loss',
             fontsize=16, fontweight='bold', y=0.99)

C1, C2, C3 = '#95A5A6', '#4A90D9', '#E74C3C'  # flat, canvas, consistency

# (a) Canvas layout
ax = axes[0, 0]
ax.set_title('Agent Canvas Layout', fontsize=11, fontweight='bold')
rcolors = {
    'obs.cart': '#5CB85C', 'obs.pole': '#3498DB',
    'plan': '#9B59B6', 'action': '#E74C3C',
    'reward': '#F5A623', 'done': '#E67E22',
}
H, W = bound.layout.H, bound.layout.W
grid = np.ones((H, W, 3)) * 0.93
for name, color in rcolors.items():
    if name not in bound:
        continue
    bf = bound[name]
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]
    lbl = name.split('.')[-1]
    lw = bf.spec.loss_weight
    if lw != 1.0:
        lbl += f'\n{lw}x'
    ax.text((w0+w1)/2-0.5, (h0+h1)/2-0.5, lbl,
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
ax.imshow(grid, aspect='equal', interpolation='nearest')
ax.set_xlabel('W'); ax.set_ylabel('H')

# (b) Training curves
ax = axes[0, 1]
ax.set_title('Training Loss', fontsize=11, fontweight='bold')
w = 20
def sm(a): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.plot(sm(flat_losses), color=C1, lw=1.5, label='flat')
ax.plot(sm(canvas_losses), color=C2, lw=1.5, label='canvas BC')
ax.plot(sm(consist_losses), color=C3, lw=1.5, label='canvas+consist.')
ax.legend(fontsize=8); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.grid(True, alpha=0.2)

# (c) Episode reward distribution (box plots)
ax = axes[0, 2]
ax.set_title('Episode Reward (100 episodes)', fontsize=11, fontweight='bold')
bp = ax.boxplot([flat_rewards, canvas_rewards, consist_rewards],
                tick_labels=['Flat', 'Canvas\nBC', 'Canvas+\nConsist.'],
                patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], [C1, C2, C3]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Total Reward')
ax.grid(True, alpha=0.2, axis='y')
# Annotate means
for i, (rews, c) in enumerate([(flat_rewards, C1), (canvas_rewards, C2), (consist_rewards, C3)]):
    ax.text(i + 1, np.mean(rews) + 3, f'{np.mean(rews):.0f}',
            ha='center', fontsize=9, fontweight='bold', color=c)

# (d) Single episode rollout with predictions
ax = axes[1, 0]
ax.set_title('Episode Rollout (canvas+consistency)', fontsize=11, fontweight='bold')
# Run one episode and record everything
env = gym.make('CartPole-v1')
obs_ep, _ = env.reset(seed=42)
angles, actions, done_preds, rew_preds = [], [], [], []
for _ in range(200):
    with torch.no_grad():
        obs_t = torch.tensor(obs_ep, dtype=torch.float32).unsqueeze(0)
        out = consist_model(obs_t)
        action = out['action_logits'].argmax(dim=-1).item()
        angles.append(obs_ep[2])
        actions.append(action)
        done_preds.append(torch.sigmoid(out['done']).item())
        rew_preds.append(out['reward'].item())
    obs_ep, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break
env.close()
t = np.arange(len(angles))
ax.plot(t, angles, color='#2C3E50', lw=1.5, label='pole angle')
ax.plot(t, done_preds, color='#E67E22', lw=1.5, label='P(done)', ls='--')
ax.plot(t, rew_preds, color='#F5A623', lw=1.5, label='pred reward', ls=':')
ax.legend(fontsize=7, loc='upper left')
ax.set_xlabel('Step'); ax.set_ylabel('Value')
ax.grid(True, alpha=0.2)

# (e) Plan field PCA colored by pole angle
ax = axes[1, 1]
ax.set_title('Plan Field (PCA) colored by |pole angle|', fontsize=11, fontweight='bold')
# Simple PCA via SVD
plans_centered = all_plans - all_plans.mean(dim=0)
U, S, V = torch.svd_lowrank(plans_centered, q=2)
pca = (plans_centered @ V).numpy()
angles_all = obs_data[:len(pca), 2].abs().numpy()
sc = ax.scatter(pca[:, 0], pca[:, 1], c=angles_all, s=1, alpha=0.3,
                cmap='RdYlGn_r', vmin=0, vmax=0.3)
plt.colorbar(sc, ax=ax, label='|pole angle|', shrink=0.8)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

# (f) Action accuracy comparison
ax = axes[1, 2]
ax.set_title('Action Accuracy by Pole Angle', fontsize=11, fontweight='bold')
# Bin observations by pole angle, compute accuracy in each bin
angle_abs = obs_data[:, 2].abs().numpy()
bins = np.linspace(0, 0.25, 8)
for model_obj, color, label in [
    (flat_model, C1, 'flat'),
    (canvas_model, C2, 'canvas BC'),
    (consist_model, C3, 'canvas+consist.'),
]:
    model_obj.eval()
    with torch.no_grad():
        preds = model_obj(obs_data)['action_logits'].argmax(dim=-1)
    correct = (preds == act_data).float().numpy()
    bin_acc = []
    bin_centers = []
    for i in range(len(bins) - 1):
        mask = (angle_abs >= bins[i]) & (angle_abs < bins[i+1])
        if mask.sum() > 10:
            bin_acc.append(correct[mask].mean())
            bin_centers.append((bins[i] + bins[i+1]) / 2)
    ax.plot(bin_centers, bin_acc, 'o-', color=color, lw=1.5, label=label, markersize=4)

ax.set_xlabel('|Pole Angle| (rad)')
ax.set_ylabel('Action Accuracy')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_ylim(0.5, 1.02)

plt.tight_layout(rect=[0, 0, 1, 0.97])
path = os.path.join(ASSETS, "03_cartpole.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\nSaved {path}")

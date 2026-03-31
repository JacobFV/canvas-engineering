"""World model: Minecraft next-frame prediction with hierarchical temporal rates.

Three levels of temporal abstraction on the canvas:
- Perception (period=1): raw observation patches, block states, entity positions
- Reasoning (period=4): spatial reasoning, object affordances
- Planning (period=16): high-level strategy, world model state

A transformer on the canvas learns to predict the next observation frame given
the current observation and an action. The imagination buffer produces
counterfactual rollouts with soft supervision.

Run:  python examples/08_world_model_minecraft.py
Out:  assets/examples/08_world_model_minecraft.png
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
class PerceptionModule:
    """Low-level perception at full temporal resolution."""
    visual: Field = Field(4, 4, period=1,
                          semantic_type="observation patches 4x4 grid")
    blocks: Field = Field(2, 2, period=1,
                          semantic_type="nearby block states 2x2")
    entities: Field = Field(1, 4, period=1,
                            semantic_type="entity positions and velocities")


@dataclass
class ReasoningModule:
    """Mid-level reasoning at 4x temporal abstraction."""
    spatial: Field = Field(2, 4, period=4,
                           semantic_type="spatial reasoning state")
    affordances: Field = Field(2, 2, period=4,
                               semantic_type="object affordance predictions")


@dataclass
class PlanningModule:
    """High-level planning at 16x temporal abstraction."""
    strategy: Field = Field(2, 4, period=16,
                            semantic_type="high-level strategy state")
    world_model: Field = Field(2, 4, period=16,
                               semantic_type="learned world dynamics state")


@dataclass
class MinecraftAgent:
    """Simplified Minecraft agent with temporal hierarchy."""
    # Context (input-only)
    inventory: Field = Field(1, 4, is_output=False,
                             semantic_type="inventory contents")
    # Cognitive modules
    perception: PerceptionModule = dc_field(default_factory=PerceptionModule)
    reasoning: ReasoningModule = dc_field(default_factory=ReasoningModule)
    planning: PlanningModule = dc_field(default_factory=PlanningModule)
    # Imagination buffer
    imagined_next: Field = Field(4, 4, period=1, loss_weight=0.5,
                                 semantic_type="imagined next frame")
    # Action output
    action: Field = Field(1, 4, period=1, loss_weight=3.0,
                          semantic_type="agent action")


# ── 2. Compile ────────────────────────────────────────────────────────

agent = MinecraftAgent()
bound = compile_schema(
    agent, T=1, d_model=48,
    connectivity=ConnectivityPolicy(
        intra="dense",
        temporal="same_frame",
    ),
)
layout = bound.layout
print("=== Minecraft World Model Agent ===")
print(bound.summary())

# Show temporal hierarchy declared via period=
print("\nDeclared temporal rates (period= metadata):")
for period in [1, 4, 16]:
    fields_at_period = [(n, bf) for n, bf in bound.fields.items()
                        if bf.spec.period == period]
    total_pos = sum(bf.num_positions for _, bf in fields_at_period)
    if fields_at_period:
        print(f"  period={period}: {len(fields_at_period)} fields, {total_pos} positions")


# ── 3. Generate synthetic data ────────────────────────────────────────
# Simulate a simple grid world: observation = 4x4 "image", blocks = 2x2 state,
# entities = 4 entity features. Actions produce predictable next-frame changes.

OBS_DIM = 16     # 4x4 flattened
BLOCK_DIM = 4    # 2x2 flattened
ENTITY_DIM = 4   # 4 entity features
ACTION_DIM = 4   # 4 action dimensions
INV_DIM = 4      # 4 inventory slots
def generate_data(n_samples=2048):
    """Generate (obs, blocks, entities, inventory, action) -> next_obs."""
    # Current observation: random patterns
    obs = torch.randn(n_samples, OBS_DIM) * 0.5
    blocks = torch.randn(n_samples, BLOCK_DIM) * 0.3
    entities = torch.randn(n_samples, ENTITY_DIM) * 0.4
    inventory = torch.randn(n_samples, INV_DIM) * 0.2

    # Action: discrete-ish (softmax of random logits)
    action_logits = torch.randn(n_samples, ACTION_DIM)
    action = torch.softmax(action_logits, dim=-1)

    # Next observation: nonlinear function of current state + action
    # Action-dependent transformation of observation
    W_act = torch.randn(ACTION_DIM, OBS_DIM) * 0.3
    action_effect = action @ W_act  # (N, OBS_DIM)

    # Entity influence on observation
    W_ent = torch.randn(ENTITY_DIM, OBS_DIM) * 0.2
    entity_effect = entities @ W_ent

    next_obs = torch.tanh(obs + action_effect + entity_effect * 0.5
                          + blocks.repeat(1, OBS_DIM // BLOCK_DIM) * 0.1)
    next_obs = next_obs + torch.randn_like(next_obs) * 0.05

    return obs, blocks, entities, inventory, action, next_obs

obs_tr, blk_tr, ent_tr, inv_tr, act_tr, nobs_tr = generate_data()
obs_val, blk_val, ent_val, inv_val, act_val, nobs_val = generate_data(512)


# ── 4. Build a transformer on the canvas ─────────────────────────────

class MinecraftTransformer(nn.Module):
    """Transformer that operates on the compiled canvas to predict next frame."""

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

        # Projections for each input
        vis_n = len(bound_schema.layout.region_indices("perception.visual"))
        blk_n = len(bound_schema.layout.region_indices("perception.blocks"))
        ent_n = len(bound_schema.layout.region_indices("perception.entities"))
        inv_n = len(bound_schema.layout.region_indices("inventory"))
        act_n = len(bound_schema.layout.region_indices("action"))
        img_n = len(bound_schema.layout.region_indices("imagined_next"))

        self.vis_proj = nn.Linear(OBS_DIM, vis_n * d_model)
        self.blk_proj = nn.Linear(BLOCK_DIM, blk_n * d_model)
        self.ent_proj = nn.Linear(ENTITY_DIM, ent_n * d_model)
        self.inv_proj = nn.Linear(INV_DIM, inv_n * d_model)
        self.act_proj = nn.Linear(ACTION_DIM, act_n * d_model)
        self.out_proj = nn.Linear(img_n * d_model, OBS_DIM)

        self.vis_n = vis_n
        self.blk_n = blk_n
        self.ent_n = ent_n
        self.inv_n = inv_n
        self.act_n = act_n
        self.img_n = img_n

    def forward(self, obs, blocks, entities, inventory, action):
        B = obs.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        vis_idx = self.bound.layout.region_indices("perception.visual")
        blk_idx = self.bound.layout.region_indices("perception.blocks")
        ent_idx = self.bound.layout.region_indices("perception.entities")
        inv_idx = self.bound.layout.region_indices("inventory")
        act_idx = self.bound.layout.region_indices("action")
        img_idx = self.bound.layout.region_indices("imagined_next")

        canvas[:, vis_idx] = canvas[:, vis_idx] + self.vis_proj(obs).reshape(B, self.vis_n, self.d)
        canvas[:, blk_idx] = canvas[:, blk_idx] + self.blk_proj(blocks).reshape(B, self.blk_n, self.d)
        canvas[:, ent_idx] = canvas[:, ent_idx] + self.ent_proj(entities).reshape(B, self.ent_n, self.d)
        canvas[:, inv_idx] = canvas[:, inv_idx] + self.inv_proj(inventory).reshape(B, self.inv_n, self.d)
        canvas[:, act_idx] = canvas[:, act_idx] + self.act_proj(action).reshape(B, self.act_n, self.d)

        canvas = self.encoder(canvas, mask=self.attn_mask)

        img_emb = canvas[:, img_idx].reshape(B, -1)
        return self.out_proj(img_emb)


model = MinecraftTransformer(bound)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400)


# ── 5. Train ──────────────────────────────────────────────────────────

losses = []
n_epochs = 400
batch_size = 128

print("\nTraining next-frame predictor...")
for epoch in range(n_epochs):
    idx = torch.randint(0, len(obs_tr), (batch_size,))
    pred = model(obs_tr[idx], blk_tr[idx], ent_tr[idx], inv_tr[idx], act_tr[idx])
    loss = ((pred - nobs_tr[idx]) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())

    if epoch % 50 == 0:
        print(f"  epoch {epoch:3d}: loss = {loss.item():.4f}")


# ── 6. Evaluate ───────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    pred_val = model(obs_val, blk_val, ent_val, inv_val, act_val)
    val_loss = ((pred_val - nobs_val) ** 2).mean().item()
    print(f"\n  validation loss: {val_loss:.4f}")

# Imagination rollout: iteratively feed predicted obs back as input
with torch.no_grad():
    rollout_steps = 8
    rollout_obs = obs_val[:1].clone()
    rollout_preds = [rollout_obs.squeeze(0).numpy()]
    for step in range(rollout_steps):
        next_pred = model(rollout_obs, blk_val[:1], ent_val[:1], inv_val[:1], act_val[:1])
        rollout_preds.append(next_pred.squeeze(0).numpy())
        rollout_obs = next_pred


# ── 7. Visualize ──────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Minecraft World Model: Next-Frame Prediction', fontsize=16, fontweight='bold', y=0.98)

COLORS = {
    'perception.visual': '#4A90D9', 'perception.blocks': '#5DADE2',
    'perception.entities': '#48C9B0', 'inventory': '#95A5A6',
    'reasoning.spatial': '#AF7AC5', 'reasoning.affordances': '#9B59B6',
    'planning.strategy': '#E67E22', 'planning.world_model': '#D35400',
    'imagined_next': '#E74C3C', 'action': '#2ECC71',
}

# (a) Canvas layout with temporal hierarchy color-coded
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
            ha='center', va='center', fontsize=6, fontweight='bold', color='white')
ax.imshow(grid, aspect='equal', interpolation='nearest')
ax.set_xlabel('W'); ax.set_ylabel('H')

# (b) Training loss
ax = axes[0, 1]
ax.set_title('Training Loss', fontsize=11, fontweight='bold')
ax.semilogy(losses, color='#2C3E50', lw=1.5, alpha=0.7)
w = 20
smoothed = np.convolve(losses, np.ones(w)/w, mode='valid')
ax.semilogy(range(w-1, len(losses)), smoothed, color='#E74C3C', lw=2, label='smoothed')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# (c) Predicted vs true next frame (first 5 samples as heatmaps)
ax = axes[1, 0]
ax.set_title('Predicted vs True Next Frame (sample 0)', fontsize=11, fontweight='bold')
with torch.no_grad():
    single_pred = model(obs_val[:1], blk_val[:1], ent_val[:1], inv_val[:1], act_val[:1])
true_img = nobs_val[0].reshape(4, 4).numpy()
pred_img = single_pred[0].reshape(4, 4).numpy()
combined = np.concatenate([true_img, np.ones((4, 1)) * np.nan, pred_img], axis=1)
im = ax.imshow(combined, aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_xticks([1.5, 6.5])
ax.set_xticklabels(['True', 'Predicted'])
ax.set_ylabel('Patch row')
plt.colorbar(im, ax=ax, shrink=0.8)

# (d) Imagination rollout quality (MSE over rollout steps)
ax = axes[1, 1]
ax.set_title('Imagination Rollout Quality', fontsize=11, fontweight='bold')
rollout_mses = []
true_next = nobs_val[0].numpy()
for i, rp in enumerate(rollout_preds):
    mse = np.mean((rp - true_next) ** 2)
    rollout_mses.append(mse)
ax.plot(range(len(rollout_mses)), rollout_mses, 'o-', color='#E74C3C', lw=2, markersize=6)
ax.set_xlabel('Rollout Step')
ax.set_ylabel('MSE vs True Next Frame')
ax.grid(True, alpha=0.2)
ax.text(0.98, 0.95, f'val MSE={val_loss:.4f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F0FE', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "08_world_model_minecraft.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")

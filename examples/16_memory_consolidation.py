"""Memory consolidation: boundary clocks, compile modes, memory family.

Demonstrates v2 clock expressions, boundary events, the memory family,
and the ProgramCompiler for deploy-time optimization.

Architecture:
  perception (observation) - raw input at every step
  working_memory (memory, tags=working) - short-term, every step
  episodic (memory, tags=episodic) - writes on novelty events
  semantic (memory, tags=semantic, compile_mode=export) - consolidated knowledge

Clocks:
  working = every step
  episodic = on_event (novelty signal > threshold)
  semantic = boundary("episode_end") - consolidates at episode boundaries

The model learns sequences across episodes. Semantic memory (exported at
compile time) captures cross-episode transfer knowledge.

Run:  python examples/16_memory_consolidation.py
Out:  assets/examples/16_memory_consolidation.png
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
    CanvasProgram, RegionProgram, ClockSpec, LearningSpec,
    RegionScheduler, ResidualSpec, ResidualAccumulator,
    ProgramCompiler,
)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)

# ── 1. Declare types ──────────────────────────────────────────────────

@dataclass
class MemorySystem:
    perception: Field = Field(2, 4, family="observation",
                              semantic_type="perceptual input 8-dim")
    working_memory: Field = Field(2, 4, family="memory", tags=("working",),
                                  loss_weight=1.5,
                                  semantic_type="working memory buffer")
    episodic: Field = Field(2, 4, family="memory", tags=("episodic",),
                            loss_weight=2.0,
                            semantic_type="episodic memory store")
    semantic: Field = Field(2, 4, family="memory", tags=("semantic",),
                            loss_weight=1.0,
                            semantic_type="semantic knowledge store")
    prediction: Field = Field(2, 4, family="state", tags=("prediction",),
                              loss_weight=3.0,
                              semantic_type="next-step prediction")


# ── 2. Compile with program semantics ─────────────────────────────────

mem = MemorySystem()
bound, program = compile_program(
    mem, T=1, H=8, W=8, d_model=48,
    connectivity=ConnectivityPolicy(intra="dense", temporal="same_frame"),
)

# Add clocks and compile modes
program.regions["perception"] = RegionProgram(
    family="observation",
    clock=ClockSpec(mode="periodic", period=1),
)
program.regions["working_memory"] = RegionProgram(
    family="memory", tags=("working",),
    clock=ClockSpec(mode="periodic", period=1),
    learning=LearningSpec(mode="retrieval", compile_mode="runtime"),
)
program.regions["episodic"] = RegionProgram(
    family="memory", tags=("episodic",),
    clock=ClockSpec(
        mode="on_event",
        event_source="novelty.prediction",
        event_threshold=0.4,
        cooldown=3,
        max_silence=8,
    ),
    learning=LearningSpec(mode="retrieval", compile_mode="freeze"),
)
program.regions["semantic"] = RegionProgram(
    family="memory", tags=("semantic",),
    clock=ClockSpec(
        mode="boundary",
        event_source="episode_end",
    ),
    compile_mode="export",
    learning=LearningSpec(mode="retrieval", compile_mode="export"),
)

print("=== Memory Consolidation System ===")
print(bound.summary())
print()
print(program.summary())

# Show clock assignments
print("\nClock assignments:")
for name, rp in program.regions.items():
    if rp.clock:
        print(f"  {name}: mode={rp.clock.mode}, compile={rp.compile_mode}")

# Set up scheduler and accumulator
scheduler = RegionScheduler(program)
accumulator = ResidualAccumulator(["novelty"], ResidualSpec(kinds=("prediction",)))


# ── 3. Generate synthetic data ────────────────────────────────────────
# Sequence learning across episodes. Each episode has a repeating pattern.
# Cross-episode transfer: later episodes share structure with earlier ones.

PERCEPT_DIM = 8
MEM_DIM = 8
PRED_DIM = 8
N_EPISODES = 20
EPISODE_LENGTH = 30

def generate_episode_data(n_episodes=N_EPISODES, ep_len=EPISODE_LENGTH):
    """Generate sequences with cross-episode structure."""
    all_inputs = []
    all_targets = []
    episode_ids = []

    # Shared base patterns (semantic knowledge)
    base_patterns = torch.randn(4, PERCEPT_DIM) * 0.5

    for ep in range(n_episodes):
        # Each episode uses 2 base patterns with episode-specific variation
        p1_idx = ep % 4
        p2_idx = (ep + 1) % 4
        pattern1 = base_patterns[p1_idx] + torch.randn(PERCEPT_DIM) * 0.1
        pattern2 = base_patterns[p2_idx] + torch.randn(PERCEPT_DIM) * 0.1

        for t in range(ep_len):
            # Alternating patterns with transition noise
            if t % 6 < 3:
                x = pattern1 + torch.randn(PERCEPT_DIM) * 0.15
                next_x = pattern1 + torch.randn(PERCEPT_DIM) * 0.15
                if t % 6 == 2:
                    next_x = pattern2 + torch.randn(PERCEPT_DIM) * 0.15
            else:
                x = pattern2 + torch.randn(PERCEPT_DIM) * 0.15
                next_x = pattern2 + torch.randn(PERCEPT_DIM) * 0.15
                if t % 6 == 5:
                    next_x = pattern1 + torch.randn(PERCEPT_DIM) * 0.15

            all_inputs.append(x)
            all_targets.append(next_x)
            episode_ids.append(ep)

    return (torch.stack(all_inputs), torch.stack(all_targets),
            torch.tensor(episode_ids))

inputs_tr, targets_tr, eps_tr = generate_episode_data()
inputs_val, targets_val, eps_val = generate_episode_data(10, 20)

# Split into early and late episodes for transfer analysis
early_mask_tr = eps_tr < N_EPISODES // 2
late_mask_tr = eps_tr >= N_EPISODES // 2
early_mask_val = eps_val < 5
late_mask_val = eps_val >= 5


# ── 4. Build model ───────────────────────────────────────────────────

class MemoryModel(nn.Module):
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

        perc_n = len(bound_schema.layout.region_indices("perception"))
        wm_n = len(bound_schema.layout.region_indices("working_memory"))
        ep_n = len(bound_schema.layout.region_indices("episodic"))
        sem_n = len(bound_schema.layout.region_indices("semantic"))
        pred_n = len(bound_schema.layout.region_indices("prediction"))

        self.perc_proj = nn.Linear(PERCEPT_DIM, perc_n * d_model)
        self.wm_out = nn.Linear(wm_n * d_model, MEM_DIM)
        self.ep_out = nn.Linear(ep_n * d_model, MEM_DIM)
        self.sem_out = nn.Linear(sem_n * d_model, MEM_DIM)
        self.pred_out = nn.Linear(pred_n * d_model, PRED_DIM)

        self.perc_n = perc_n
        self.wm_n = wm_n
        self.ep_n = ep_n
        self.sem_n = sem_n
        self.pred_n = pred_n

        # Persistent semantic memory buffer (simulates exported memory)
        self.semantic_buffer = nn.Parameter(torch.zeros(1, sem_n, d_model))

    def forward(self, perception, use_semantic=True):
        B = perception.shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        perc_idx = self.bound.layout.region_indices("perception")
        sem_idx = self.bound.layout.region_indices("semantic")
        pred_idx = self.bound.layout.region_indices("prediction")
        wm_idx = self.bound.layout.region_indices("working_memory")
        ep_idx = self.bound.layout.region_indices("episodic")

        canvas[:, perc_idx] = canvas[:, perc_idx] + \
            self.perc_proj(perception).reshape(B, self.perc_n, self.d)

        if use_semantic:
            canvas[:, sem_idx] = canvas[:, sem_idx] + self.semantic_buffer.expand(B, -1, -1)

        canvas = self.encoder(canvas, mask=self.attn_mask)

        prediction = self.pred_out(canvas[:, pred_idx].reshape(B, -1))
        wm_state = self.wm_out(canvas[:, wm_idx].reshape(B, -1))
        ep_state = self.ep_out(canvas[:, ep_idx].reshape(B, -1))
        sem_state = self.sem_out(canvas[:, sem_idx].reshape(B, -1))

        return prediction, wm_state, ep_state, sem_state


model = MemoryModel(bound)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400)


# ── 5. Train with episode boundaries ─────────────────────────────────

losses = []
memory_write_counts = {'working': [], 'episodic': [], 'semantic': []}
n_epochs = 400
batch_size = 64

print("\nTraining memory consolidation model...")
for epoch in range(n_epochs):
    idx = torch.randint(0, len(inputs_tr), (batch_size,))
    pred, wm, ep, sem = model(inputs_tr[idx])

    # Prediction loss
    pred_loss = ((pred - targets_tr[idx]) ** 2).mean() * 3.0

    # Memory losses: encourage memories to be useful for prediction
    # Working memory should track recent input
    wm_loss = ((wm - inputs_tr[idx]) ** 2).mean() * 1.5
    # Episodic should capture transitions
    ep_loss = ((ep - targets_tr[idx]) ** 2).mean() * 2.0
    # Semantic should capture shared patterns (regularized)
    sem_loss = ((sem - inputs_tr[idx]) ** 2).mean() * 1.0

    loss = pred_loss + wm_loss + ep_loss + sem_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()
    losses.append(loss.item())

    # Simulate scheduling for tracking
    scheduler.reset()
    novelty = torch.rand(1).item() * (0.6 if epoch % 10 < 3 else 0.2)
    accumulator.update("novelty", torch.tensor(novelty))
    summaries = accumulator.summaries()
    is_boundary = (epoch % 30 == 29)
    active = scheduler.step(epoch, summaries=summaries,
                            boundary="episode_end" if is_boundary else None)

    memory_write_counts['working'].append(1)  # always
    memory_write_counts['episodic'].append(1 if 'episodic' in active else 0)
    memory_write_counts['semantic'].append(1 if 'semantic' in active else 0)

    if epoch % 50 == 0:
        ep_rate = sum(memory_write_counts['episodic'][-50:]) / min(50, epoch + 1)
        sem_rate = sum(memory_write_counts['semantic'][-50:]) / min(50, epoch + 1)
        print(f"  epoch {epoch:3d}: loss={loss.item():.4f} ep_write_rate={ep_rate:.2f} sem_write_rate={sem_rate:.2f}")


# ── 6. Evaluate + compile ────────────────────────────────────────────

model.eval()
with torch.no_grad():
    # Cross-episode transfer: compare early vs late episode accuracy
    pred_early, _, _, _ = model(inputs_val[early_mask_val])
    pred_late, _, _, _ = model(inputs_val[late_mask_val])

    early_mse = ((pred_early - targets_val[early_mask_val]) ** 2).mean().item()
    late_mse = ((pred_late - targets_val[late_mask_val]) ** 2).mean().item()

    # Without semantic memory
    pred_no_sem, _, _, _ = model(inputs_val, use_semantic=False)
    no_sem_mse = ((pred_no_sem - targets_val) ** 2).mean().item()
    with_sem_mse = ((model(inputs_val)[0] - targets_val) ** 2).mean().item()

    print(f"\n  Early episode MSE: {early_mse:.4f}")
    print(f"  Late episode MSE: {late_mse:.4f}")
    print(f"  Transfer improvement: {(early_mse - late_mse) / early_mse * 100:.1f}%")
    print(f"  With semantic memory: {with_sem_mse:.4f}")
    print(f"  Without semantic memory: {no_sem_mse:.4f}")

# Run ProgramCompiler
compiler = ProgramCompiler(program)
compiled = compiler.compile()
print(f"\n  Compiled program:")
print(f"  {compiled.summary()}")


# ── 7. Visualize ──────────────────────────────────────────────────────

MEMORY_COLORS = {
    'perception': '#4A90D9',
    'working_memory': '#2ECC71',
    'episodic': '#E67E22',
    'semantic': '#9B59B6',
    'prediction': '#E74C3C',
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Memory Consolidation: Boundary Clocks + ProgramCompiler',
             fontsize=14, fontweight='bold', y=0.98)

# (a) Memory write patterns
ax = axes[0, 0]
ax.set_title('Memory Write Patterns', fontsize=11, fontweight='bold')
w = 20
for mem_type, color in [('working', '#2ECC71'), ('episodic', '#E67E22'), ('semantic', '#9B59B6')]:
    counts = memory_write_counts[mem_type]
    smoothed = np.convolve(counts, np.ones(w)/w, mode='valid')
    ax.plot(range(w-1, len(counts)), smoothed, color=color, lw=2, label=mem_type)
ax.set_xlabel('Epoch')
ax.set_ylabel('Write Rate')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_ylim(-0.05, 1.1)

# (b) Cross-episode transfer accuracy
ax = axes[0, 1]
ax.set_title('Cross-Episode Transfer', fontsize=11, fontweight='bold')
labels = ['Early\nEpisodes', 'Late\nEpisodes', 'With\nSemantic', 'Without\nSemantic']
values = [early_mse, late_mse, with_sem_mse, no_sem_mse]
colors_bar = ['#E67E22', '#2ECC71', '#9B59B6', '#95A5A6']
bars = ax.bar(labels, values, color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Prediction MSE')
ax.grid(True, alpha=0.2, axis='y')

# (c) Compile reduction visualization
ax = axes[1, 0]
ax.set_title('ProgramCompiler Summary', fontsize=11, fontweight='bold')
# Show what happens to each region
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
bars = ax.barh(names, [1]*len(names),
               color=[status_colors[s] for s in statuses],
               edgecolor='white', linewidth=1.5)
for bar, status in zip(bars, statuses):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            status, ha='left', va='center', fontsize=9, fontweight='bold')
ax.set_xlim(0, 1.5)
ax.set_xlabel('')
ax.set_xticks([])
for s, c in status_colors.items():
    if s in statuses:
        ax.plot([], [], 's', color=c, markersize=8, label=s)
ax.legend(fontsize=7, loc='lower right')
ax.text(0.5, 0.02, f'{compiled.n_eliminated} regions eliminated at deploy',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=10,
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (d) Training curves
ax = axes[1, 1]
ax.set_title('Training Loss', fontsize=11, fontweight='bold')
ax.semilogy(losses, color='#2C3E50', lw=1.5, alpha=0.5)
w = 20
smoothed = np.convolve(losses, np.ones(w)/w, mode='valid')
ax.semilogy(range(w-1, len(losses)), smoothed, color='#E74C3C', lw=2, label='smoothed')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "16_memory_consolidation.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")

"""Canvas layout examples for different applications."""

from canvas_engineering import CanvasLayout, SpatiotemporalCanvas
import torch

# ─── Robot Manipulation (Bridge V2) ──────────────────────────────────────────
robot_layout = CanvasLayout(
    T=5, H=8, W=8, d_model=256,
    regions={
        "visual":  (0, 5, 0, 6, 0, 6),    # 5 frames x 6x6 patches = 180 positions (56%)
        "action":  (0, 5, 6, 7, 0, 1),    # 5 action slots = 5 positions (2%)
        "proprio": (0, 3, 6, 7, 1, 2),    # 3 frames of proprioception = 3 positions
        "reward":  (2, 3, 7, 8, 0, 1),    # single reward slot = 1 position
    },
    t_current=2,
)
print(f"Robot canvas: {robot_layout.num_positions} positions")
for name in robot_layout.regions:
    print(f"  {name}: {robot_layout.region_numel(name)} positions")

# ─── Computer Use Agent ──────────────────────────────────────────────────────
computer_layout = CanvasLayout(
    T=16, H=32, W=32, d_model=768,
    regions={
        "screen":   (0, 16, 0, 24, 0, 24),   # 16 frames of 24x24 screen patches
        "mouse":    (0, 16, 24, 26, 0, 4),    # mouse position + click type
        "keyboard": (0, 16, 26, 28, 0, 4),    # keyboard actions
        "llm":      (0, 16, 28, 32, 0, 8),    # LLM steering channel
    },
    t_current=1,
)
print(f"\nComputer canvas: {computer_layout.num_positions:,} positions")
for name in computer_layout.regions:
    pct = computer_layout.region_numel(name) / computer_layout.num_positions * 100
    print(f"  {name}: {computer_layout.region_numel(name):,} positions ({pct:.1f}%)")

# ─── Multi-Robot Control ─────────────────────────────────────────────────────
multi_robot_layout = CanvasLayout(
    T=32, H=16, W=16, d_model=512,
    regions={
        "robot1_cam":    (0, 32, 0, 8, 0, 8),
        "robot1_action": (0, 32, 0, 8, 8, 9),
        "robot2_cam":    (0, 32, 8, 16, 0, 8),
        "robot2_action": (0, 32, 8, 16, 8, 9),
        "shared_task":   (0, 1, 0, 16, 9, 16),
    },
    t_current=1,
)
print(f"\nMulti-robot canvas: {multi_robot_layout.num_positions:,} positions")

# ─── Create and use a canvas ─────────────────────────────────────────────────
canvas = SpatiotemporalCanvas(robot_layout)
batch = canvas.create_empty(batch_size=4)
print(f"\nEmpty canvas shape: {batch.shape}")

# Place visual embeddings
visual_embs = torch.randn(4, robot_layout.region_numel("visual"), 256)
batch = canvas.place(batch, visual_embs, "visual")

# Extract action embeddings
action_embs = canvas.extract(batch, "action")
print(f"Action region shape: {action_embs.shape}")

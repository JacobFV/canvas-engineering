"""Quick start: Graft looped attention onto CogVideoX-2B in 30 lines.

This example:
1. Defines a canvas layout for video + actions
2. Grafts 3-loop attention onto CogVideoX-2B's frozen transformer
3. Shows how to set up curriculum training

Requirements: pip install canvas-engine[cogvideox]
"""

import torch
from canvas_engine import CanvasLayout, graft_looped_blocks, CurriculumScheduler

# 1. Define canvas layout (Bridge V2 robot video)
layout = CanvasLayout(
    T=13, H=60, W=90, d_model=1920,
    regions={
        "visual": (0, 13, 0, 60, 0, 90),   # full video frames
        "action": (0, 13, 0, 1, 0, 1),      # per-frame 7D actions
    },
    t_current=1,
)
print(f"Canvas: {layout.num_positions:,} positions, {layout.d_model}d")

# 2. Load CogVideoX-2B and graft looped blocks
from diffusers import CogVideoXTransformer3DModel

transformer = CogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-2b", subfolder="transformer", torch_dtype=torch.bfloat16
)

looped_blocks, action_head = graft_looped_blocks(
    transformer,
    max_loops=3,          # 3 loops is optimal (empirically validated)
    freeze="full",        # freeze backbone, train only loop params (~350K)
    action_dim=7,         # Bridge V2: 6DOF + gripper
)

# 3. Set up optimizer (only loop params + action head)
trainable = [p for b in looped_blocks for p in b.parameters() if p.requires_grad]
trainable += list(action_head.parameters())
optimizer = torch.optim.AdamW(trainable, lr=1e-4)

# 4. Curriculum: ramp from 1 to 3 loops over 5000 steps
curriculum = CurriculumScheduler(max_loops=3, total_steps=5000)

# 5. Training loop skeleton
for step in range(5000):
    n_loops = curriculum.step(looped_blocks, step)
    # ... your training code here ...
    if step % 1000 == 0:
        n_trainable = sum(p.numel() for p in trainable)
        print(f"Step {step}: {n_loops} loops, {n_trainable:,} trainable params")

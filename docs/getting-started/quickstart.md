# Quick Start

## Graft looped attention onto CogVideoX-2B

```python
from canvas_engineering import graft_looped_blocks, CurriculumScheduler
from diffusers import CogVideoXTransformer3DModel
import torch

# Load pretrained video diffusion model
transformer = CogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-2b", subfolder="transformer", torch_dtype=torch.bfloat16
)

# Graft 3-loop attention onto all 30 frozen DiT blocks
looped_blocks, action_head = graft_looped_blocks(
    transformer,
    max_loops=3,       # 3 is optimal (empirically validated)
    freeze="full",     # freeze backbone, train only loop params
    action_dim=7,      # 6DOF end-effector + gripper
)

# Only 350K params to optimize
optimizer = torch.optim.AdamW(
    [p for b in looped_blocks for p in b.parameters() if p.requires_grad]
    + list(action_head.parameters()),
    lr=1e-4,
)

# Curriculum: gradually ramp from 1 to 3 loops during training
scheduler = CurriculumScheduler(max_loops=3, total_steps=5000)
```

That's it. The frozen 1.69B-parameter backbone now loops its computation 3 times per forward pass, with learned iteration embeddings that cost 0.02% of the model.

## Define a canvas layout

```python
from canvas_engineering import CanvasLayout, RegionSpec, SpatiotemporalCanvas

layout = CanvasLayout(
    T=5, H=8, W=8, d_model=256,
    regions={
        "visual":  (0, 5, 0, 6, 0, 6),    # 180 positions — video patches
        "action":  RegionSpec(
            bounds=(0, 5, 6, 7, 0, 1),
            loss_weight=2.0,               # emphasize action accuracy
        ),
        "reward":  RegionSpec(
            bounds=(2, 3, 7, 8, 0, 1),
            period=5,                      # low-frequency
        ),
    },
    t_current=2,
)

canvas = SpatiotemporalCanvas(layout)
batch = canvas.create_empty(batch_size=4)          # (4, 320, 256)
batch = canvas.place(batch, visual_embs, "visual") # write video patches
actions = canvas.extract(batch, "action")          # read predictions
```

## Define a topology

```python
from canvas_engineering import Connection, CanvasTopology

topology = CanvasTopology(connections=[
    Connection(src="visual", dst="visual"),                    # self-attention
    Connection(src="action", dst="visual"),                    # action reads visual
    Connection(src="action", dst="action"),                    # action self-attention
    Connection(src="reward", dst="visual", fn="pooling"),      # cheap summary
    Connection(src="reward", dst="action", fn="gated"),        # optional conditioning
])

# Compile to attention mask
mask = topology.to_attention_mask(layout)  # (320, 320) float tensor
```

# Recipe: Multi-Agent Coordination

Two robots with shared task context. Each has private perception and policy; they coordinate through a shared hub.

```python
from canvas_engineering import *

layout = CanvasLayout(
    T=8, H=16, W=16, d_model=512,
    regions={
        "r1_cam": RegionSpec(bounds=(0,8, 0,6, 0,6),
                             semantic_type="RGB video 128x128 robot 1 wrist camera"),
        "r1_action": RegionSpec(bounds=(0,8, 6,7, 0,2), loss_weight=2.0,
                                semantic_type="6-DOF EE delta pose robot 1"),
        "r2_cam": RegionSpec(bounds=(0,8, 0,6, 6,12),
                             semantic_type="RGB video 128x128 robot 2 wrist camera"),
        "r2_action": RegionSpec(bounds=(0,8, 6,7, 6,8), loss_weight=2.0,
                                semantic_type="6-DOF EE delta pose robot 2"),
        "shared_task": RegionSpec(bounds=(0,1, 7,8, 0,8), is_output=False,
                                  default_attn="cross_attention",
                                  semantic_type="natural language collaborative task"),
    },
)

topology = CanvasTopology(connections=[
    # Self-attention per region
    Connection(src="r1_cam", dst="r1_cam"),
    Connection(src="r1_action", dst="r1_action"),
    Connection(src="r2_cam", dst="r2_cam"),
    Connection(src="r2_action", dst="r2_action"),
    Connection(src="shared_task", dst="shared_task"),

    # Each robot's action reads its own camera
    Connection(src="r1_action", dst="r1_cam"),
    Connection(src="r2_action", dst="r2_cam"),

    # Cross-robot: see each other's cameras (dampened)
    Connection(src="r1_cam", dst="r2_cam", weight=0.5),
    Connection(src="r2_cam", dst="r1_cam", weight=0.5),

    # Hub: shared task reads cameras, actions read task
    Connection(src="shared_task", dst="r1_cam", fn="perceiver"),
    Connection(src="shared_task", dst="r2_cam", fn="perceiver"),
    Connection(src="r1_action", dst="shared_task", fn="gated"),
    Connection(src="r2_action", dst="shared_task", fn="gated"),
])
```

## Design rationale

- **Private streams** (cam, action) use standard self-attention
- **Cross-robot vision** is dampened (weight=0.5) — they see each other but don't dominate
- **Hub reads cameras via perceiver** — compresses two full camera views into the shared task bottleneck
- **Actions read hub via gated attention** — each robot decides when to incorporate shared context

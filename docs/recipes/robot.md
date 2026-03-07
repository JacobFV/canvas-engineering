# Recipe: Robot Manipulation

A canvas schema for a 6-DOF robot manipulation agent with visual observation, proprioception, and language goal conditioning.

```python
from canvas_engineering import *

layout = CanvasLayout(
    T=16, H=32, W=32, d_model=768,
    regions={
        "visual": RegionSpec(
            bounds=(0, 16, 0, 24, 0, 24),
            semantic_type="RGB video 224x224 30fps front monocular camera",
            semantic_embedding=embed("RGB video 224x224 30fps front monocular camera"),
        ),
        "proprio": RegionSpec(
            bounds=(0, 16, 24, 26, 0, 4),
            period=1, loss_weight=2.0,
            default_attn="linear_attention",  # 12D state, no need for O(N²)
            semantic_type="7-DOF joint positions + velocities 30Hz",
            semantic_embedding=embed("7-DOF joint positions + velocities 30Hz"),
        ),
        "action": RegionSpec(
            bounds=(0, 16, 26, 28, 0, 4),
            loss_weight=2.0,
            semantic_type="6-DOF end-effector delta pose + gripper",
            semantic_embedding=embed("6-DOF end-effector delta pose + gripper"),
        ),
        "goal": RegionSpec(
            bounds=(0, 1, 28, 32, 0, 8),
            is_output=False, period=16,
            semantic_type="natural language task instruction",
            semantic_embedding=embed("natural language task instruction"),
        ),
    },
)

topology = CanvasTopology(connections=[
    # Self-attention (uses region defaults)
    Connection(src="visual", dst="visual"),
    Connection(src="proprio", dst="proprio"),
    Connection(src="action", dst="action"),
    Connection(src="goal", dst="goal"),

    # Action reads from visual (which patches matter for this action?)
    Connection(src="action", dst="visual"),

    # Proprio is cheap conditioning — just pool and inject
    Connection(src="action", dst="proprio", fn="pooling"),

    # Goal conditioning is optional (gated)
    Connection(src="visual", dst="goal", fn="gated"),
    Connection(src="action", dst="goal", fn="gated"),
])

schema = CanvasSchema(layout=layout, topology=topology,
                       metadata={"model": "CogVideoX-2B", "data": "bridge_v2"})
schema.to_json("robot_manipulation_v1.json")
```

## Design rationale

- **Visual** gets standard cross-attention — spatial reasoning requires content-based selection
- **Proprio** uses linear attention by default — it's a 12D vector, quadratic attention adds cost without benefit
- **Action** reads from visual via cross-attention and from proprio via pooling (just needs the state vector)
- **Goal** is gated — the model learns when to pay attention to the instruction vs. relying on visual context
- **Loss weights** emphasize action and proprioception (2x) over visual reconstruction

# Example 08: Minecraft World Model with Imagination

Temporal hierarchy (period 1/4/16) on a world model task. The imagination buffer predicts future states, and the imagination loss provides a self-supervised signal that generalizes beyond behavioral cloning.

**Source**: [`examples/08_minecraft_world_model.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/08_minecraft_world_model.py) *(coming soon)*

## What it demonstrates

- **Three-rate temporal hierarchy** — perception (period=1), planning (period=4), imagination (period=16)
- **Imagination loss** — predict future observations from the imagination buffer; no environment labels needed
- **World model generalization** — imagination-trained agent transfers better to novel grid layouts
- **Grid world** — 16×16 Minecraft-like environment, no actual Minecraft required

## Type hierarchy

```python
@dataclass
class Perception:
    local_view: Field = Field(4, 4, period=1)      # 4x4 local grid view
    inventory: Field = Field(1, 4, period=1)        # held items
    position: Field = Field(1, 2, period=1)         # x, y

@dataclass
class Planning:
    goal: Field = Field(2, 4, period=4)             # target state representation
    path: Field = Field(4, 4, attn="mamba", period=4)  # temporal trajectory
    obstacle_map: Field = Field(4, 4, period=4)    # learned obstacle model

@dataclass
class Imagination:
    future_obs: Field = Field(4, 4, period=16)     # imagined future observation
    future_reward: Field = Field(1, 1, period=16)  # imagined future reward

@dataclass
class MinecraftAgent:
    perception: Perception = field(default_factory=Perception)
    planning: Planning = field(default_factory=Planning)
    imagination: Imagination = field(default_factory=Imagination)
    action: Field = Field(1, 1, loss_weight=2.0)   # next action
```

## The imagination loss

```python
# Imagination loss: predict t+16 observation from imagination buffer
future_obs_true = obs_sequence[:, t + 16]
imagination_loss = mse(imagined_obs, future_obs_true)

# This is self-supervised — no extra labels, just temporal structure
total_loss = bc_loss + 0.5 * imagination_loss
```

The agent receives no reward signal during training. The imagination buffer learns to predict the future, and the path field must encode the trajectory to make this possible — an emergent world model.

!!! note "Task spec"
    Full implementation details in [`examples/tasks/08_minecraft_world_model.md`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/tasks/08_minecraft_world_model.md).

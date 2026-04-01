# Example 08: Minecraft World Model with Imagination

Temporal hierarchy (period 1/4/16) on a world model task. The imagination buffer predicts future states, and the imagination loss provides a self-supervised signal that generalizes beyond behavioral cloning.

**Source**: [`examples/08_minecraft_world_model.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/08_minecraft_world_model.py)

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/08_world_model_minecraft.png" alt="Example 08 results" width="100%">
</p>

**Top left**: Canvas layout with multi-rate regions — perception at period=1, planning at period=4, imagination at period=16, and a shared world model.

**Top right**: Training loss over 400 epochs, dropping from ~1.0 to ~0.01 as the model learns next-frame prediction.

**Bottom left**: Predicted vs true next frame (sample 0) — patch-level comparison showing the model captures spatial structure.

**Bottom right**: Imagination rollout quality — MSE grows with rollout horizon but remains usable (val MSE=0.065 at 8 steps), confirming the imagination buffer learns coherent multi-step predictions.

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

## Run it

```bash
python examples/08_world_model_minecraft.py
# Generates: assets/examples/08_world_model_minecraft.png
```

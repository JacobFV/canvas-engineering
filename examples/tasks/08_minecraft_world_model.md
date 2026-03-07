# 08: Minecraft World Model with Imagination

## Purpose
Show the temporal hierarchy (period 1/4/16) on a world model task.
The imagination buffer is the centerpiece: the agent predicts future
states, and the imagination loss provides a self-supervised signal.

## Environment
Simple grid world (Minecraft-like, not actual Minecraft):
- 16x16 grid with blocks, items, agent
- Agent can move, mine, place, craft
- Simple physics: gravity, block support
- Goal: collect resources and build a structure

## Type hierarchy
Same structure as 08_world_model_minecraft.py but with the grid world.
Key: perception at period=1, reasoning at period=4, planning at period=16.

## Training
- Behavioral cloning on scripted expert trajectories
- **Imagination loss**: at each step, predict the next 4 frames of
  perception from the planning state. Compare imagined vs actual.
  The imagination field is trained with 0.5x loss weight (soft).
- **Plan consistency**: the plan at t and t+16 should be similar if
  the goal hasn't changed. Temporal smoothness on slow fields.
- **Hindsight relabeling**: when the agent fails, relabel the goal
  to what it actually achieved. This creates positive training signal
  from failures.

## Visualization
`assets/examples/08_minecraft.png`:
(a) The grid world with agent, blocks, items
(b) Imagined vs actual next states (side by side)
(c) Plan field activations over time (do they change when goals change?)
(d) Temporal hierarchy: which period fields update when
(e) Training curves: BC only vs BC + imagination vs BC + imagination + hindsight

## Key message
"The temporal hierarchy lets the model think at different timescales.
Imagination provides free self-supervised signal. Canvas types let you
declare this entire cognitive architecture in 40 lines."

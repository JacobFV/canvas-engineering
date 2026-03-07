# 03: CartPole with Latent Planning

## Purpose
Show canvas types on a REAL control problem. Not toy synthetic data.
Demonstrate that the type hierarchy (observation -> plan -> action) with
declared connectivity creates structure the model exploits.

## Environment
OpenAI Gym CartPole-v1 (or gymnasium). No MuJoCo needed — pure Python,
ships with gym. Everyone can run it.

## What it does
1. Collect CartPole trajectories using a simple heuristic policy (angle-based)
2. Declare the agent type:
   ```
   CartPoleAgent:
     observation:
       cart: Field(1, 2)     — position, velocity
       pole: Field(1, 2)     — angle, angular velocity
     plan: Field(2, 4, attn="mamba")  — latent planning state
     action: Field(1, 1, loss_weight=3.0) — discrete action
     reward: Field(1, 1, loss_weight=0.5) — predicted reward
     done: Field(1, 1, loss_weight=2.0)   — termination prediction
   ```
3. Train via behavioral cloning (SFT on expert trajectories)
4. But ALSO train with a self-consistency loss: the predicted reward
   should be consistent with the predicted done signal and the
   observation trajectory. This is the "sophisticated feedback" part:
   - If done=True, reward should drop
   - If pole angle is large, done should be high
   - Plan field should have low variance when the policy is confident
5. Compare:
   - Flat baseline: all fields in one region, no structure
   - Canvas structured: the declared type hierarchy
   - Canvas + consistency: structured + self-consistency loss
6. Deploy all three in the actual CartPole environment and measure
   episode reward

## Visualization
`assets/examples/03_cartpole.png` — 2x3 grid:
Row 1: (a) Canvas layout 3D view, (b) Training curves (3 models),
        (c) Episode reward distribution (box plots, 3 models)
Row 2: (d) Attention heatmap from plan->observation (what does the planner attend to?),
        (e) Single episode rollout: obs, plan activation, action, reward over time,
        (f) Plan field t-SNE colored by pole angle (does the plan encode state?)

## Training
~1000 steps BC + consistency, CPU. Gymnasium rollouts for eval. ~30 seconds.

## Key message
"The type hierarchy isn't just organization — it creates inductive bias.
The plan field learns a compressed state representation. The consistency
loss between reward/done/observation creates a self-supervised signal
that pure BC doesn't have. Canvas types let you declare these feedback
dynamics in 20 lines."

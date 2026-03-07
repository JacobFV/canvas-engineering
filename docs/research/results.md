# Empirical Results

26 experiments, 236 training runs. All on CogVideoX-2B with Bridge V2 robot video data.

## Key findings

| Finding | Multiplier | Significance |
|---------|-----------|--------------|
| Depth vs recurrence | **1.73x** | p<0.001 |
| Per-token adaptive compute | 1.24x | single seed |
| Weight sharing | 1.03x | medium |
| Curriculum vs fixed | 1.05x | -- |
| Frozen 3-loop (350K params) | 0.073 action loss | beats 11.7M unfrozen |

## Falsified hypotheses

| Hypothesis | Result | Evidence |
|---|---|---|
| Looping = iterative reasoning | **Falsified** | 3 independent nulls (p=0.97, p>0.05, p>0.05) |
| Shared canvas = multi-modal binding | **Falsified** | Joint prediction 19% worse (p<0.0001) |
| Token allocation follows power laws | Borderline | R^2=0.902, alpha=0.011 |

## Fixed-point convergence

Loop representations converge toward fixed points:

| Loop | Cosine sim to loop 1 | Velocity |
|------|---------------------|----------|
| 1 | 0.926 | 0.675 |
| 2 | 0.973 | 0.570 |
| 3 | 0.990 | 0.398 |
| 4 | 0.996 | 0.292 |

Token velocities decay exponentially. Visual tokens converge slowest. Action tokens converge fastest. **Looping is weight-sharing regularization, not iterative refinement.**

## Freeze strategy comparison

| Strategy | Trainable | Action loss | Diffusion loss |
|----------|-----------|-------------|---------------|
| Frozen | 350K | **0.073** | 1.48 |
| Half-frozen | 3.7M | 0.107 | 0.19 |
| Unfrozen | 11.7M | 0.088 | 0.18 |

Freeze level doesn't affect action loss (p=0.72). It only affects video generation quality.

## Paper

> **Looped Attention in Video Diffusion Transformers: 26 Experiments on What Works, What Doesn't, and Why**
>
> Jacob Valdez and Claude Opus 4.6

[Paper PDF](https://github.com/JacobFV/recursive-omnimodal-video-action-model/blob/16c4bed/papers/empirical/main.pdf) | [Video](https://youtu.be/LHEhdFAWkEc) | [Experiment data](https://github.com/JacobFV/recursive-omnimodal-video-action-model/tree/main/archive/experiments)

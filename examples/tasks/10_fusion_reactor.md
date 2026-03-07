# 10: Tokamak Plasma Control

## Purpose
Multi-timescale control with safety constraints. Show that the disruption
predictor (loss_weight=10) catches disruptions earlier than a flat model,
and that the multi-rate diagnostics create a natural sensor fusion.

## Data
Synthetic tokamak data:
- Plasma equilibrium: 2D flux surface profiles evolving over time
- Diagnostics: magnetic probes (fast, noisy), Thomson scattering (slow, accurate)
- Actuators: coil currents, heating power
- Disruptions: sudden loss of equilibrium preceded by precursor signals
  (mode locking, density limit, beta limit)
- Generate normal operation + disruption scenarios

## Training
- Equilibrium reconstruction: predict flux surfaces from diagnostics
- Actuator prediction: what should the control system do next?
- Disruption prediction: binary classification with extreme class weight
- **Physics-informed loss**: the predicted equilibrium must satisfy
  force balance (Grad-Shafranov equation in simplified form).
  This is the feedback dynamic: the physics constrains the latent space.
- **Early warning score**: the disruption predictor should fire BEFORE
  the disruption, not at the moment of disruption. Train with a
  time-weighted loss: earlier correct predictions get more reward.

## Visualization
`assets/examples/10_tokamak.png`:
(a) Flux surface visualization (2D cross-section of plasma)
(b) Multi-timescale diagnostic fusion: fast magnetics + slow Thomson
(c) Disruption prediction: probability over time with event marked
(d) Early warning time distribution: how many ms before disruption?
(e) Physics loss vs pure data loss: does physics-informed training help?

## Key message
"Physics-informed losses on the equilibrium field ensure the model's
internal representation respects physical law. The disruption predictor
at 10x loss weight catches disruptions 50ms earlier because the model
dedicates more capacity to the safety-critical path."

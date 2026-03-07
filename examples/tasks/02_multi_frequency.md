# 02: Multi-Frequency Sensor Fusion

## Purpose
Show that declaring different `period` values on fields creates a real
training advantage — the model learns to fuse signals at different
temporal resolutions without any special architecture code.

## What it does
1. Declare a sensor fusion type:
   - fast_sensor: Field(1, 8, period=1) — updates every frame
   - slow_sensor: Field(1, 8, period=4) — updates every 4th frame
   - context: Field(2, 4, is_output=False) — static context
   - prediction: Field(1, 4, loss_weight=2.0) — the target
2. Generate synthetic data where the true signal is a function of BOTH
   the fast and slow sensors plus context
   - Fast: noisy high-freq oscillation
   - Slow: smooth low-freq trend
   - Context: one-hot category that determines the combination rule
   - Prediction: f(fast, slow, context) with different f per category
3. Train two models:
   - Model A: canvas with correct periods declared
   - Model B: canvas with everything at period=1 (no frequency info)
4. Show that Model A converges faster and to lower loss

## Visualization
`assets/examples/02_multi_frequency.png` — 2x2:
(a) The multi-frequency canvas layout (regions colored, periods labeled)
(b) Raw sensor signals at different frequencies (fast=noisy, slow=smooth)
(c) Training curves: Model A vs Model B (loss over steps)
(d) Prediction quality: true vs predicted for both models on held-out data

## Training
Two small transformers, 500 steps each, CPU. ~15 seconds total.

## Key message
"Declaring temporal frequency isn't just metadata — it changes how the
model allocates attention across time and measurably improves fusion."

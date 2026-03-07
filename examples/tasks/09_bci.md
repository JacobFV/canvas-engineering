# 09: Brain-Computer Interface — Neural Decoding

## Purpose
Show real-time multi-output decoding with closed-loop feedback.
The key insight: feedback fields (cursor_feedback, haptic_feedback)
create a closed-loop training signal that pure feedforward decoding misses.

## Data
Synthetic neural data:
- 96-channel spike trains generated from a latent motor intention
- Motor intention = 2D cursor velocity + grasp aperture
- Spike rates are noisy Poisson draws from tuning curves
  (each channel has a preferred direction, cosine tuning)
- Cursor feedback: delayed version of decoded position (50ms delay)
- This is exactly how real BCI decoders work, just synthetic

## Training
- Feedforward baseline: spikes -> velocity (no feedback)
- Canvas with feedback: spikes + cursor_feedback -> velocity
- **Online adaptation**: after initial training, simulate closed-loop
  use where the cursor position depends on the decoder's output.
  The decoder must adapt to its own errors. This is RFT in disguise:
  the model gets feedback from its own predictions and must improve.

## Visualization
`assets/examples/09_bci.png`:
(a) Neural tuning curves (96 channels, color-coded by preferred direction)
(b) Decoded cursor trajectory vs true intention (2D path plot)
(c) Closed-loop vs open-loop decoding accuracy over time
(d) Online adaptation: error decreasing as the model adapts
(e) Attention from decoded velocity -> which electrode channels
    (does the model learn the tuning curves?)

## Key message
"Closed-loop feedback isn't just a feature — it's a training paradigm.
Declaring cursor_feedback as an input-only field creates a recurrent
information pathway through the canvas. Online adaptation shows the
model improving from its own deployment, not just from labeled data."

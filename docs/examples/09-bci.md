# Example 09: Brain-Computer Interface — Neural Decoding

Real-time multi-output decoding with closed-loop feedback. Feedback fields (`cursor_feedback`, `haptic_feedback`) create a closed-loop training signal that pure feedforward decoding misses.

**Source**: [`examples/09_bci.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/09_bci.py)

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/09_brain_computer_interface.png" alt="Example 09 results" width="100%">
</p>

**Top left**: Canvas layout — neural input, motor intention, feedback, and adaptation regions with closed-loop connectivity.

**Top right**: Cursor decoding predicted vs true velocity (MSE=0.004) — tight diagonal confirms accurate real-time cursor control.

**Bottom left**: Multi-decoder performance — cursor MSE 0.004, click accuracy 67.8%, phoneme accuracy 8.2%, showing joint decoding from shared latent.

**Bottom right**: Training curves — cursor loss and speech loss converge over 500 epochs, with total loss stabilizing.

## What it demonstrates

- **Closed-loop feedback fields** — feedback regions attend to motor intention and are trained to be self-consistent with the decoded output
- **Multi-output decoding** — cursor position + velocity + grip force decoded simultaneously from shared latent
- **Spike train inputs** — 96-channel synthetic spike trains generated from a latent motor intention model
- **Adaptation field** — a slow-timescale region (period=30) models neural drift and recalibrates the decoder

## Type hierarchy

```python
@dataclass
class NeuralInput:
    spike_rates: Field = Field(8, 12, is_output=False)   # 96-channel firing rates
    lfp_power: Field = Field(4, 8, is_output=False)      # 32-channel LFP bands

@dataclass
class MotorIntention:
    direction: Field = Field(2, 4, attn="mamba")          # directional intent
    velocity_scale: Field = Field(1, 2)                   # speed intention
    grip: Field = Field(1, 2)                             # grasp intention

@dataclass
class Feedback:
    cursor_feedback: Field = Field(1, 4, period=1)        # visual cursor position
    haptic_feedback: Field = Field(1, 2, period=1)        # haptic return

@dataclass
class BCIDecoder:
    neural: NeuralInput = field(default_factory=NeuralInput)
    intention: MotorIntention = field(default_factory=MotorIntention)
    feedback: Feedback = field(default_factory=Feedback)
    adaptation: Field = Field(2, 4, period=30)            # drift correction
    cursor_pos: Field = Field(1, 2, loss_weight=3.0)
    cursor_vel: Field = Field(1, 2, loss_weight=2.0)
    grip_force: Field = Field(1, 1, loss_weight=1.0)
```

## Closed-loop consistency

```python
# Consistency: predicted cursor should match cursor_feedback
consistency_loss = mse(cursor_pos_pred, cursor_feedback.detach())

# Consistency: grip intention should match haptic feedback
haptic_consistency = mse(grip_force_pred, haptic_feedback.detach())
```

The feedback fields are not targets — they are a closed loop. The model must produce outputs that are consistent with what it receives back, enforcing physically plausible decoding.

!!! note "Task spec"
    Full implementation details in [`examples/tasks/09_bci.md`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/tasks/09_bci.md).

## Run it

```bash
python examples/09_brain_computer_interface.py
# Generates: assets/examples/09_brain_computer_interface.png
```

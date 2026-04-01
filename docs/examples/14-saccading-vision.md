# Example 14: Saccading Vision

Event-triggered visual attention with a fovea that fires only when prediction error exceeds a threshold. Five region families coordinate gaze control: periphery and fovea observe the scene, a state region integrates the scene model, a residual region tracks prediction error, and an action region decides where to look next.

**Source**: [`examples/14_saccading_vision.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/14_saccading_vision.py)

## Result

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/14_saccading_vision.png" alt="Saccading vision results" width="100%">
</p>

**Left**: Gaze trajectory over a cluttered scene -- the fovea saccades to high-error regions, spending more time on novel objects and skipping familiar background.

**Center**: Residual accumulator trace -- prediction error spikes trigger fovea activation, then decays via EMA as the scene model updates.

**Right**: Region scheduling timeline -- periphery fires every step, fovea fires on-event, scene integrates continuously, gaze emits actions after each fovea fixation.

## Type declaration

```python
@dataclass
class SaccadingVision:
    periphery: Field = Field(4, 8, family="observation")
    fovea: Field = Field(4, 4, family="observation")
    scene: Field = Field(4, 8, family="state")
    error: Field = Field(2, 4, family="residual", carrier="residual")
    gaze: Field = Field(1, 2, family="action", loss_weight=2.0)
```

## Compile with scheduling

```python
bound, program = compile_program(SaccadingVision(), T=8, H=8, W=8, d_model=64)

# Override clocks: fovea fires on prediction error, gaze has cooldown
program.regions["fovea"] = RegionProgram(
    family="observation",
    clock=ClockSpec(mode="on_event", event_source="error.prediction", event_threshold=0.5),
)
program.regions["gaze"] = RegionProgram(
    family="action",
    clock=ClockSpec(mode="periodic", period=1, cooldown=3),
)

scheduler = RegionScheduler(program)
accumulator = ResidualAccumulator(["error"], ResidualSpec(decay=0.9))

# Each step: scheduler decides who fires
for t in range(T):
    active = scheduler.step(t, summaries=accumulator.summaries())
    canvas = dispatcher(canvas, active_regions=active)
```

## What this shows

- **`RegionScheduler`** -- clock rules (periodic, on_event) control which regions update each step
- **`ResidualAccumulator`** -- scalar EMA error summaries drive event-triggered scheduling
- **All 5 families in one model** -- observation, state, residual, action working together
- **Compute efficiency** -- fovea only fires when needed, saving ~60% of fovea compute

## Run it

```bash
python examples/14_saccading_vision.py
# Generates: assets/examples/14_saccading_vision.png
```

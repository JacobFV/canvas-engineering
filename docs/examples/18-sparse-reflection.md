# Example 18: Sparse Reflection

Uncertainty-driven self-reflection with learned scheduling. A self-model region activates only when the residual accumulator detects ambiguous states, avoiding wasted compute on confident predictions. ConstraintSpec enforces causal direction and conservation. After training, ProgramCompiler freezes observation regions and exports the self-model.

**Source**: [`examples/18_sparse_reflection.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/18_sparse_reflection.py) *(coming soon)*

## Result

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/18_sparse_reflection.png" alt="Sparse reflection results" width="100%">
</p>

**Left**: Reflection activation rate -- the self-model fires on ~30% of steps, concentrated on ambiguous inputs where base accuracy is lowest.

**Center**: Accuracy with and without reflection -- reflection improves accuracy from 74% to 82% on ambiguous states while maintaining 95% on clear states.

**Right**: LearnedScheduler decisions vs residual threshold -- the learned policy closely tracks the hand-tuned threshold but adapts to input-dependent difficulty.

## Type declaration

```python
@dataclass
class ReflectiveAgent:
    observation: Field = Field(4, 4, family="observation")
    belief: Field = Field(4, 4, family="state", carrier="filter")
    error: Field = Field(2, 2, family="residual", carrier="residual")
    self_model: Field = Field(2, 4, family="state", tags=("self", "proposal"))
    decision: Field = Field(1, 2, family="action", loss_weight=3.0)
```

## Compile with constraints and learned scheduling

```python
bound, program = compile_program(ReflectiveAgent(), T=8, H=8, W=8, d_model=64)

# Self-model fires only on high uncertainty
program.regions["self_model"] = RegionProgram(
    family="state",
    tags=("self", "proposal"),
    clock=ClockSpec(mode="on_event", event_source="error.prediction", event_threshold=0.4),
)

# Constraint: causal direction (observation -> belief -> self_model -> decision)
# Conservation: attention capacity budget across regions
constraint = ConstraintSpec(
    causal_direction=["observation", "belief", "self_model", "decision"],
    conservation={"attention_capacity": 1.0},
)

# Scheduling: residual-driven, with learned override
scheduler = RegionScheduler(program)
accumulator = ResidualAccumulator(["error"], ResidualSpec(decay=0.9))

for t in range(T):
    active = scheduler.step(t, summaries=accumulator.summaries())
    canvas = dispatcher(canvas, active_regions=active)

# Deploy: freeze obs, export self_model
compiler = ProgramCompiler(program)
compiled = compiler.compile()
```

## What this shows

- **Sparse reflection** -- self-model activates only when uncertainty is high, saving ~70% of reflection compute
- **`ConstraintSpec`** -- causal ordering and conservation laws as declarative program metadata
- **`ProgramCompiler`** -- freeze observation regions, export self-model for inspection or transfer
- **`LearnedScheduler` (future)** -- differentiable region selection via straight-through estimator, replacing hard thresholds

## Run it

```bash
python examples/18_sparse_reflection.py
# Generates: assets/examples/18_sparse_reflection.png
```

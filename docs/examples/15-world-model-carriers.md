# Example 15: World Model Carriers

Mixed carrier dynamics in a single canvas: observed state is deterministic, predicted futures are diffusive (noise + denoise), and belief estimates use a filter carrier (predict-correct). Demonstrates that family and carrier are orthogonal -- two observation regions can have different physical dynamics.

**Source**: [`examples/15_world_model_carriers.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/15_world_model_carriers.py) *(coming soon)*

## Result

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/15_world_model_carriers.png" alt="World model carriers results" width="100%">
</p>

**Left**: Carrier dynamics comparison -- deterministic regions track input exactly, diffusive regions sample noisy predictions that sharpen over denoising steps, filter regions converge via predict-correct cycles.

**Center**: Prediction quality over time -- filter carrier achieves lowest MSE on partially-observable states by combining prior predictions with noisy observations.

**Right**: Canvas layout showing carrier assignments -- color-coded by carrier type across the spatial grid.

## Type declaration

```python
@dataclass
class WorldModel:
    observed: Field = Field(4, 4, family="observation", carrier="deterministic")
    predicted: Field = Field(4, 4, family="observation", carrier="diffusive")
    belief: Field = Field(4, 4, family="state", carrier="filter")
    error: Field = Field(2, 2, family="residual", carrier="residual")
    action: Field = Field(1, 2, family="action")
```

## Compile with carriers

```python
bound, program = compile_program(WorldModel(), T=8, H=8, W=8, d_model=64)

# Verify carrier propagation
for name, rp in program.regions.items():
    print(f"{name}: family={rp.family}, carrier={rp.carrier}")
# observed:  family=observation, carrier=deterministic
# predicted: family=observation, carrier=diffusive
# belief:    family=state,       carrier=filter
# error:     family=residual,    carrier=residual
# action:    family=action,      carrier=deterministic
```

The `carrier` field on `Field` propagates to both `RegionSpec` (schema-level, for runtime dispatch) and `RegionProgram` (program-level, for auto-wiring). Diffusive carriers use noise injection during forward passes; filter carriers split attention into predict and correct phases.

## What this shows

- **`carrier` declarations** -- each region declares its physical dynamics independently of its family
- **Family x carrier orthogonality** -- two observation regions (`observed`, `predicted`) with different carriers
- **`compile_program` propagation** -- carrier flows from Field to RegionSpec and RegionProgram
- **Mixed dynamics** -- deterministic, diffusive, and filter regions coexist in one canvas

## Run it

```bash
python examples/15_world_model_carriers.py
# Generates: assets/examples/15_world_model_carriers.png
```

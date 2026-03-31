# Carriers

Not everything in a canvas is diffusive. A robot's joint angles are deterministic measurements. A belief state is a Kalman-style filter. An episodic memory bank is a persistent lookup table. A prediction error signal is an ephemeral trace. The **carrier** declares what dynamics govern a region, independent of what the region *means* (its family).

## The 5 carriers

| Carrier | Dynamics | Update rule | Example regions |
|---------|----------|-------------|-----------------|
| `deterministic` | Standard forward latent updates | Direct write from encoder/decoder | Joint angles, reward scalars, task prompts |
| `diffusive` | Noise/denoise process | DDPM/flow-matching denoising | Video frames, generated images, future plans |
| `filter` | Predict/correct cycle | Kalman-style: predict from prior, correct with evidence | Belief state, object pose estimates |
| `memory` | Persistent lookup | Write/retrieve from key-value store | Episodic memory, semantic knowledge base |
| `residual` | Error traces | EMA-smoothed scalar summaries | Prediction error, novelty signal, uncertainty |

## Family and carrier are orthogonal

The family says *what* a region is. The carrier says *how* it evolves. The same family can use different carriers depending on the application:

| Region | Family | Carrier | Why |
|--------|--------|---------|-----|
| Future video (to generate) | observation | diffusive | Model generates it via denoising |
| Current video (observed) | observation | deterministic | Encoder writes it directly |
| Kalman belief | state | filter | Predict/correct dynamics |
| Goal embedding | state | deterministic | Set once, read many |
| Episodic memory | memory | memory | Persistent write/retrieve |
| Prediction error | residual | residual | EMA scalar summaries |
| Generated plan | action | diffusive | Plan generated via denoising |
| Direct motor command | action | deterministic | MLP decoder output |

## How carrier affects the system

**Residual tracking.** Regions with `carrier="residual"` feed into `ResidualAccumulator`, which maintains running EMA summaries of error signals. These summaries drive event-triggered scheduling: a memory region can fire only when prediction error exceeds a threshold, instead of every step.

**Compile-mode interaction.** The carrier influences what compile modes make sense. A `memory` carrier region naturally maps to `compile_mode="export"` (save the memory bank to disk at deploy). A `deterministic` carrier region can be `compile_mode="constant"` (materialize as a buffer and remove from the compute graph).

**Future: loss computation.** Carrier type will determine which loss function applies. Diffusive regions use diffusion loss (MSE on noise prediction). Deterministic regions use direct reconstruction loss. Filter regions use predictive consistency loss. This is declared, not hard-coded -- the training loop reads the carrier and dispatches accordingly.

## Usage

Carrier is set on `Field` (for `compile_program()`) or on `RegionSpec` (for manual layout construction):

```python
from canvas_engineering import Field

# On Field (compile_program path)
camera = Field(12, 12, family="observation", carrier="deterministic")
future_video = Field(12, 12, family="observation", carrier="diffusive")

# On RegionSpec (manual layout path)
from canvas_engineering import RegionSpec
spec = RegionSpec(bounds=(0, 8, 0, 12, 0, 12), carrier="filter")
```

The default carrier is `"deterministic"`. If you don't set it, the region uses standard forward updates.

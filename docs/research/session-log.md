# Session Log: March 30 - April 4, 2026

Comprehensive log of everything built, tested, and discovered during this multi-day session.

## Day 1 (March 30): Foundation — temporal fill modes

### Built
- Temporal fill modes: DROP, HOLD, INTERPOLATE for cross-frequency attention
- PeriodEmbedding: learned embedding indexed by log-bucketed temporal period
- Higher-order IDW interpolation via `interpolation_order`
- CogVideoX backbone-native attention type
- Mamba readout refactor (pooled-mean → query-based attention)
- Integration tests with synthetic training tasks on Modal

### Key discovery
- INTERPOLATE was broken — fill resolution operated in canvas-frame space, not real-time space. Fixed by converting to real time using region periods.
- PREDICT and DECAY fill modes removed after training tests showed they didn't earn their complexity. Clean taxonomy: DROP (events), HOLD (state), INTERPOLATE (smooth signals).

### Shipped
- v0.2.0 to PyPI (239 tests)

## Day 1-2: v2 typed process compiler

### Built (5 phases)
1. CanvasProgram scaffold: RegionProgram, ConnectionProgram, ClockSpec, LearningSpec
2. Operator/backend split: Connection.operator, DEFAULT_WIRING auto-wiring
3. Carriers + residual summaries: RegionSpec.carrier, ResidualAccumulator
4. Clocks + scheduling: RegionScheduler (periodic, on_event, boundary)
5. Learning recipes + compiler: FAMILY_DEFAULTS, ProgramCompiler

### Key decisions
- 5 region families only (observation, state, memory, residual, action) — everything else = tags
- Carrier and family are orthogonal (future video = observation + diffusive)
- forward() ALWAYS returns Tensor (accumulator as side effect)
- compile_schema() unchanged — compile_program() wraps it

### Shipped
- v0.3.0 to PyPI (361 tests)

## Day 2: Phase 6+ features

### Built
- FamilyCarrierEmbedding (ProgramConditioner)
- Internal microsteps (ClockSpec.max_inner_steps)
- Identity/slot persistence (IdentitySpec, SlotBindingModule)
- MaskSpec (rect_cover for non-rectangular masks)
- CortexSpec (locality domains)
- LearnedScheduler (Gumbel top-k)
- ClockExpr IR (full AST with serialization)
- ConstraintSpec (equivariance, conservation, causal direction)

### Shipped
- v0.4.0 to PyPI (478 tests)

## Day 2-3: Examples

### Fixed
- Removed deprecated `parent_child` kwarg from all 11 examples
- Increased grid sizes for examples 04, 07 (coarse-grained fields need more space)

### Built (new)
- Examples 08-11: upgraded from schema-only stubs to full training examples
- Examples 14-18: new v2 examples showcasing families, carriers, operators, scheduling
- Example 09b: BCI with real TRIBE v2 on Modal GPU (canvas 68.8% vs SVM 59.4%)
- All verified on Modal (13/13 passing)

### Shipped
- v0.4.1, v0.4.2 to PyPI
- All docs updated (23 pages), mkdocs nav, example doc pages

## Day 3-4: Research tracks

### Brain track
- 23 cortical regions mapped to Destrieux atlas
- 42 cortical pathway connections matching known neuroscience
- TRIBE v2 data pipeline (72 stimuli → 20,484 vertex predictions)
- 135-feature extraction (8 per ROI, vs 23 scalar means)
- Classification experiment: flat MLP wins (wrong task — too easy)
- Dynamics prediction: cortical R²=0.825 at 135 features (right task)
- Key finding: topology = convergence prior, helps at higher dimensionality
- Activation snapshots saved (12MB, 10 epochs × 4 layers × 10 stimuli)
- 3D brain surface renders with nilearn

### Browser track
- Canvas agent with screen/DOM/state/action/diagnostics regions
- Synthetic browser environment (4 task types)
- Multi-objective training (BC + SSL + RL)
- Canvas planner fires 12.5% of steps (event-driven, 8× less compute)

### Robotics track
- 4-robot fleet with 51 canvas regions, 189 connections
- 2D physics simulation with lidar, obstacles, formation control
- Scaling analysis (2/4/8 robots)
- Canvas has fewest collisions at 2 robots

### Infrastructure
- Modal runners with volume persistence
- Symlinked results directories for incremental saving
- Detached runs that survive laptop disconnection
- Result collection via `--collect-only` flag

## Presentation materials

### Created
- Scientific HTML report (self-contained, 12 figures, 6 tables, MathJax)
- 3D brain surface animations (activation flow, learning progression)
- Region activation heatmap across training
- Recording outline for video narration
- Unified 3D animation storyboard (brain → canvas → robot)
- Memo figures for compute request

### All at
```
presentation/
├── scientific_report.html
├── plan.md
├── script/recording_outline.md
├── render_brain_animations.py
├── animations/ (17 files)
├── assets/ (19 files)
└── memo_figures/ (5 files)
```

## Numbers

| Metric | Value |
|--------|-------|
| Lines of library code | 6,782 |
| Lines of test code | 6,498 |
| Lines of research code | 8,788 |
| Lines of example code | 11,615 |
| Total tests | 478 |
| PyPI releases | 5 (v0.2.0 → v0.4.2) |
| Research result files | 38+ |
| Presentation assets | 40+ |
| Modal GPU-hours used | ~20 |
| Commits on develop | 50+ |

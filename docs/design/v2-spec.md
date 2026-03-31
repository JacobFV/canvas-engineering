# canvas-engineering v2: typed process compiler

## Thesis

v1 = typed layout + connectivity DSL.
v2 = typed layout + **process semantics** + **scheduling** + **compilation**.

The core invariant: rectangular regions remain the runtime ABI for QKV routing.
Everything else compiles down to that.

---

## The 6 axes

A region is fully described by:

```
(family, carrier, structure, clock, dynamics, learning)
```

A connection is fully described by:

```
(operator, backend, trigger, write_mode)
```

Everything else derives from these or is a tag.

---

## Phase 1: CanvasProgram scaffold

**Goal**: Add `CanvasProgram`, `RegionProgram`, `ConnectionProgram` as data classes.
No runtime behavior change. Old code still works. New objects are informational.

### New file: `canvas_engineering/program.py`

```python
@dataclass(frozen=True)
class RegionProgram:
    """Process semantics for a canvas region."""
    family: str = "state"          # observation|state|memory|residual|action
    tags: Tuple[str, ...] = ()     # belief, object, parser, value, goal, self, proposal, ...
    carrier: str = "deterministic" # deterministic|diffusive|filter|memory|residual
    clock: Optional[ClockSpec] = None
    learning: Optional[LearningSpec] = None
    compile_mode: str = "runtime"  # runtime|freeze|constant|export

@dataclass(frozen=True)
class ConnectionProgram:
    """Process semantics for a canvas connection."""
    operator: str = "attend"       # observe|predict|correct|bind|retrieve|write|act|compress
    trigger: Optional[str] = None  # residual condition expression (serializable string)
    write_mode: str = "add"        # add|replace|gate

@dataclass(frozen=True)
class ClockSpec:
    """When a region updates."""
    domain: str = "external"       # external|boundary
    mode: str = "periodic"         # periodic|on_event|boundary
    period: int = 1                # for periodic mode
    event_source: Optional[str] = None  # region.summary_name for on_event
    event_threshold: float = 0.0   # threshold for on_event
    cooldown: int = 0
    max_silence: Optional[int] = None

@dataclass(frozen=True)
class LearningSpec:
    """How a region learns during training."""
    mode: str = "supervised"       # supervised|ssl_prediction|posterior_match|retrieval|calibration|none
    losses: Tuple[str, ...] = ()   # loss function names
    compile_mode: str = "runtime"  # runtime|freeze|distill|constant|export

@dataclass
class CanvasProgram:
    """Typed process semantics layered on top of CanvasSchema."""
    schema: CanvasSchema
    regions: Dict[str, RegionProgram] = field(default_factory=dict)
    connections: Dict[Tuple[str, str], ConnectionProgram] = field(default_factory=dict)
    version: str = "2.0.0"
```

### Changes to existing files

**`__init__.py`**: Export `CanvasProgram`, `RegionProgram`, `ConnectionProgram`, `ClockSpec`, `LearningSpec`.

**`schema.py`**: Add `CanvasProgram` serialization (to_dict/from_dict/to_json/from_json).

**`types.py`**: Extend `Field` with optional `family`, `tags`, `carrier` kwargs. `compile_schema()` emits a `CanvasProgram` alongside `BoundSchema` when program fields are present. New function: `compile_program(root, ...) -> (BoundSchema, CanvasProgram)`.

### Tests

- RegionProgram/ConnectionProgram creation with defaults
- CanvasProgram serialization round-trip
- compile_schema still works unchanged (backward compat)
- compile_program generates sane RegionPrograms from Field families

### Done when

- All 239 existing tests pass unchanged
- CanvasProgram objects serialize/deserialize
- Field(family="observation") propagates to RegionProgram

---

## Phase 2: operator/backend split + auto-wiring

**Goal**: Formalize the operator/backend distinction. Auto-generate connectivity from region families.

### Changes to `connectivity.py`

Add `operator` field to `Connection`:

```python
@dataclass(frozen=True)
class Connection:
    src: str
    dst: str
    weight: float = 1.0
    t_src: Optional[int] = None
    t_dst: Optional[int] = None
    fn: Optional[str] = None       # RENAMED conceptually to "backend"
    operator: str = "attend"       # NEW: semantic intent
    temporal_fill: TemporalFill = TemporalFill.HOLD
    interpolation_order: int = 1
```

`fn` stays for backward compat. `operator` is new and defaults to `"attend"` (current behavior).

### Changes to `types.py`

Add family-aware auto-wiring to `_generate_connections()`:

```python
DEFAULT_WIRING = {
    ("observation", "state"):    {"operator": "observe"},
    ("state", "observation"):    {"operator": "predict"},
    ("state", "state"):          {"operator": "integrate"},
    ("state", "memory"):         {"operator": "write"},
    ("memory", "state"):         {"operator": "retrieve"},
    ("state", "action"):         {"operator": "act"},
    ("action", "state"):         {"operator": "intervene"},
    ("state", "residual"):       {"operator": "emit_residual"},
    ("observation", "residual"): {"operator": "emit_residual"},
}
```

When `compile_program()` has family info for both endpoints, it sets operator automatically. User can override.

### Changes to `dispatch.py`

`AttentionDispatcher` reads `operator` for logging/summary but doesn't change behavior yet. Operator-specific dispatch logic comes in phase 4.

### Tests

- Connection(operator="predict") works
- Auto-wiring produces correct operators from family pairs
- Existing fn= code unchanged
- Summary includes operator info

### Done when

- Operators appear in topology summaries
- compile_program auto-wires observation→state as "observe", etc.
- All existing tests pass

---

## Phase 3: carriers + residual summaries

**Goal**: Not everything is diffusive. Regions declare their carrier kind.
Residual regions emit scalar summaries that later drive scheduling.

### Changes to `canvas.py`

`RegionSpec` gets a new field:

```python
@dataclass(frozen=True)
class RegionSpec:
    # ... existing fields ...
    carrier: str = "deterministic"  # deterministic|diffusive|filter|memory|residual
```

Backward compat: defaults to "deterministic". Diffusion users explicitly set "diffusive" on future-frame regions.

### New file: `canvas_engineering/residuals.py`

```python
@dataclass(frozen=True)
class ResidualSpec:
    """Declares what error signals a region emits."""
    kinds: Tuple[str, ...] = ("prediction",)  # prediction|uncertainty|novelty
    reduce: str = "max_mean"                   # how to summarize to scalar
    decay: float = 0.95                        # EMA decay for running summaries

class ResidualAccumulator(nn.Module):
    """Tracks running scalar summaries of residual signals."""
    def __init__(self, region_names: List[str], spec: ResidualSpec): ...
    def update(self, region: str, error: torch.Tensor) -> None: ...
    def summaries(self) -> Dict[str, Dict[str, float]]: ...
    def reset(self) -> None: ...
```

### Changes to `dispatch.py`

After each connection's attention output, if dst is a residual region, compute and store the summary. `AttentionDispatcher.forward()` returns `(output, summaries)` when residual regions exist, `output` otherwise.

### Changes to `types.py` / `program.py`

`Field` gets optional `carrier` kwarg. `RegionProgram` already has `carrier`.
`compile_program()` propagates carrier from Field to RegionSpec.

### Tests

- RegionSpec(carrier="diffusive") works
- ResidualAccumulator tracks running summaries
- Dispatcher returns summaries when residual regions exist
- Carrier propagates through compile_program
- loss_weight_mask respects carrier (diffusive regions get diffusion loss, deterministic get prediction loss)

### Done when

- Regions can declare carrier kind
- Residual regions emit scalar summaries
- Dispatcher handles mixed carrier topologies
- All existing tests pass (carrier defaults to "deterministic")

---

## Phase 4: clocks + event triggers

**Goal**: Regions can skip updates. Clock rules determine when a region fires.
Event triggers read residual summaries.

### Changes to `program.py`

ClockSpec already defined in phase 1. Now make it executable.

### New file: `canvas_engineering/scheduling.py`

```python
class RegionScheduler:
    """Evaluates clock rules to decide which regions fire each step."""

    def __init__(self, program: CanvasProgram, layout: CanvasLayout): ...

    def step(
        self,
        external_t: int,
        summaries: Dict[str, Dict[str, float]],
        boundary: Optional[str] = None,
    ) -> Set[str]:
        """Returns the set of region names that should update this step."""
        ...

    def should_fire(self, region: str, clock: ClockSpec, ...) -> bool: ...
```

### Changes to `dispatch.py`

`AttentionDispatcher.forward()` accepts optional `active_regions: Set[str]`.
Inactive regions reuse their cached output (kv cache or last output buffer).

```python
def forward(self, x, active_regions=None):
    # If active_regions is None, all regions fire (backward compat)
    # Otherwise, skip connections where src is inactive
    # Inactive regions' positions pass through unchanged
```

### Changes to `canvas.py` / `SpatiotemporalCanvas`

Add output cache per region for reuse when skipped:

```python
class SpatiotemporalCanvas(nn.Module):
    # ...
    def cache_region(self, canvas, region_name): ...
    def restore_cached(self, canvas, region_name): ...
```

### Tests

- RegionScheduler fires periodic regions correctly
- Event-triggered regions fire when summary > threshold
- Boundary regions fire on boundary events
- Cooldown prevents re-firing
- max_silence forces firing after N silent steps
- Dispatcher skips inactive regions and reuses cache
- All existing tests pass (no scheduler = all regions always fire)

### Done when

- Periodic regions skip correctly
- Event-triggered regions fire on residual summaries
- Skipped regions reuse cached state
- Training loop example with mixed clocks works end-to-end

---

## Phase 5: learning recipes + compile modes

**Goal**: Per-family default training recipes. Compile modes for deploy.

### New file: `canvas_engineering/learning.py`

```python
FAMILY_DEFAULTS = {
    "observation": LearningSpec(
        mode="ssl_prediction",
        losses=("next_step", "masked_prediction"),
        compile_mode="freeze",
    ),
    "state": LearningSpec(
        mode="posterior_match",
        losses=("predictive_consistency", "calibration"),
        compile_mode="runtime",
    ),
    "memory": LearningSpec(
        mode="retrieval",
        losses=("retrieval_accuracy", "write_utility"),
        compile_mode="export",
    ),
    "residual": LearningSpec(
        mode="calibration",
        losses=("calibration", "sparsity"),
        compile_mode="freeze",
    ),
    "action": LearningSpec(
        mode="supervised",
        losses=("task",),
        compile_mode="freeze",
    ),
}

def default_learning(family: str) -> LearningSpec: ...
```

### New file: `canvas_engineering/compiler.py`

```python
class ProgramCompiler:
    """Lowers a CanvasProgram to a deploy-ready execution plan."""

    def __init__(self, program: CanvasProgram): ...

    def compile(self) -> CompiledProgram:
        """Run all compile passes."""
        self._propagate_constants()
        self._eliminate_dead_regions()
        self._freeze_regions()
        self._export_memories()
        return self.result

    def _propagate_constants(self): ...
    def _eliminate_dead_regions(self): ...
    def _freeze_regions(self): ...
    def _export_memories(self): ...

@dataclass
class CompiledProgram:
    """Deploy-ready execution plan."""
    schema: CanvasSchema          # potentially reduced
    frozen_buffers: Dict[str, torch.Tensor]
    exported_memories: Dict[str, torch.Tensor]
    active_regions: Set[str]
    active_connections: List[Connection]
```

### Tests

- Family defaults produce correct LearningSpec
- Compiler freezes observation regions
- Compiler exports memory regions as lookup tables
- Compiler eliminates dead (never-firing) regions
- Compiled program has fewer regions than training program

### Done when

- Per-family learning defaults work
- compile_mode="freeze" removes grad from region parameters
- compile_mode="constant" materializes region as buffer
- compile_mode="export" saves memory bank to disk
- Basic compiler pass runs without error

---

## Phase 6: masks + structure extensions (future)

**Goal**: Support non-rectangular authored masks that compile to rect/tile covers.
Cortex abstraction for locality domains. Internal microsteps.

This is phase 6+ and should not block v2 release.

### Potential additions

- `MaskSpec` for authored masks → `RectSet` compilation
- `CortexSpec` for locality domains with shared cache
- Internal clock domain for microstep loops
- Learned scheduling (straight-through estimator for skip decisions)

---

## File inventory

### New files (6)
```
canvas_engineering/program.py      # CanvasProgram, RegionProgram, ConnectionProgram, ClockSpec, LearningSpec
canvas_engineering/residuals.py    # ResidualSpec, ResidualAccumulator
canvas_engineering/scheduling.py   # RegionScheduler
canvas_engineering/learning.py     # Per-family learning defaults, training recipes
canvas_engineering/compiler.py     # ProgramCompiler, CompiledProgram
tests/test_program.py              # Tests for all new program-layer code
```

### Modified files (7)
```
canvas_engineering/__init__.py     # New exports
canvas_engineering/canvas.py       # RegionSpec.carrier, SpatiotemporalCanvas cache
canvas_engineering/connectivity.py # Connection.operator
canvas_engineering/types.py        # Field(family=, carrier=, tags=), compile_program()
canvas_engineering/schema.py       # CanvasProgram serialization
canvas_engineering/dispatch.py     # active_regions, residual summaries, operator logging
canvas_engineering/semantic.py     # Extend conditioning for family/carrier awareness
```

### Untouched files (6)
```
canvas_engineering/attention.py    # Backends unchanged
canvas_engineering/looped_block.py
canvas_engineering/cogvideox.py
canvas_engineering/graft.py
canvas_engineering/curriculum.py
canvas_engineering/sharpening.py
```

---

## Migration path

### v0.2.0 → v2.0.0

All v0.2.0 code works unchanged. New features are opt-in:

- `Field()` still works → defaults to family="state", carrier="deterministic"
- `Connection()` still works → defaults to operator="attend"
- `compile_schema()` still returns BoundSchema → unchanged
- `compile_program()` is new → returns (BoundSchema, CanvasProgram)
- `CanvasProgram` is informational until phase 4 (scheduling)
- `AttentionDispatcher` gains optional `active_regions` parameter

No breaking changes. All 239 tests pass at every phase.

---

## Ablations needed

1. Families only vs raw schema (does typing regions help?)
2. Operator/backend split vs raw fn (does semantic intent help?)
3. Residual sidecars on/off (do explicit errors help scheduling?)
4. Event clocks vs periodic only (does sparse compute help?)
5. Teacher-posterior training vs end-to-end only (does Bayesian structure help?)
6. Carrier split vs uniform diffusion (does mixed dynamics help?)
7. Compile freeze/export vs live-only (does compilation help deploy?)

---

## What this does NOT include (intentionally)

- Internal microsteps (phase 6+ — variable-depth graphs are hard)
- CortexSpec (nice abstraction, not load-bearing)
- ConstraintSpec (equivariance, conservation — research decorators)
- ClockExpr IR (start with 3 constructors, don't build a language)
- Learned scheduling (start with deterministic rules)
- Teacher implementations (domain-specific, not library features)
- Non-rectangular masks (rectangles are the ABI)
- Multi-agent dedicated semantics (just use regions + connectivity)

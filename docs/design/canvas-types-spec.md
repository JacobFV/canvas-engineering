# Canvas Types: Detailed Design Specification

**Status**: RFC / Pre-implementation
**Module**: `canvas_engineering.types`
**Compilation target**: `CanvasLayout` + `CanvasTopology` → `CanvasSchema`

---

## 1. Problem

Today, canvas-engineering is a memory layout tool with manual wiring. You hand-specify
region bounds, hand-wire connections, hand-tune loss weights. This is like writing
assembly: full control, zero abstraction.

The insight: real-world systems have **compositional structure** — agents contain
subsystems, organizations contain agents, multi-robot cells contain robots. This
structure implies natural connectivity patterns, allocation proportions, and
information flow. Currently the user is the compiler: they see "Company with 10
Employees" and manually translate that into 30+ RegionSpecs and 100+ Connections.

Canvas Types is the compiler.

## 2. Core Abstraction

A **CanvasType** is a declarative struct whose fields are latent regions. Types
compose (nesting), specialize (inheritance), and replicate (arrays). A compiler
flattens any CanvasType into a `CanvasSchema` — a concrete `CanvasLayout` +
`CanvasTopology` ready for the diffusion process.

```
CanvasType (high-level)
    ↓ compile()
CanvasSchema (mid-level)
    ↓ contains
CanvasLayout + CanvasTopology (low-level)
    ↓ produces
attention masks, loss weights, frame maps (runtime)
```

Users pick their level. The low-level API doesn't change.

## 3. Field Types

### 3.1 Vec — atomic latent region

```python
from canvas_engineering.types import Vec

class Robot(CanvasType):
    camera: Vec[144]              # 144 latent positions
    joints: Vec[8]                # 8 positions
    action: Vec[8]                # 8 positions
```

`Vec[N]` declares a field that occupies N latent positions. Vec is parameterized:

```python
Vec[N]                            # N positions, all defaults
Vec[N].configure(
    period=4,                     # temporal frequency
    is_output=False,              # input-only conditioning
    loss_weight=2.0,              # emphasized in loss
    attn="linear_attention",      # default attention type
    semantic_type="RGB video",    # modality description
)
```

Each configured Vec compiles to one `RegionSpec`. The bounds are computed by the
packing algorithm — the user doesn't specify them.

### 3.2 CanvasType — composite struct

```python
class Agent(CanvasType):
    thought: Vec[64]
    goal: Vec[32]

class Employee(Agent):            # inherits thought, goal
    role: Vec[16]

class Product(CanvasType):
    description: Vec[48]
    features: Vec[24]

class Company(Agent):             # inherits thought, goal
    employees: Array[Employee, 10]
    products: Array[Product, 5]
```

A CanvasType is a frozen declaration. Fields are ordered (insertion order). Inherited
fields come first (MRO). The type is not a runtime object — it's a schema that
compiles to canvas primitives.

### 3.3 Array — replicated substructure

```python
Array[Employee, 10]               # 10 Employee instances
Array[Robot, 3].configure(
    layout=ArrayLayout.INTERLEAVED,  # see Section 5
    element_connectivity="isolated",  # elements don't see each other
)
```

An array creates K instances of a type, each with independent latent positions.
The key design decision is **how** these instances are laid out on the grid and
**how** they connect — see Section 5.

### 3.4 Scalar — single-position field

```python
class Robot(CanvasType):
    reward: Scalar                # 1 position — sugar for Vec[1]
    done: Scalar.configure(is_output=False)
```

### 3.5 Shared — aliased field (union type)

```python
class MultiAgent(CanvasType):
    shared_state: Shared[Vec[64]]    # all referencing types write to same positions
    agent_a: AgentType
    agent_b: AgentType
```

A `Shared` field occupies positions that multiple children co-inhabit. This is a
union in C terms — overlapping memory. The diffusion process reads and writes
the same latent positions from multiple semantic perspectives. Radical, but
it's exactly how shared state works in multi-agent latent spaces.

## 4. Addressing

Compiled schemas use dotted paths for region names:

```
"company.thought"                 # Company's own thought field
"company.goal"                    # Company's own goal field
"company.employees[0].thought"    # First employee's thought
"company.employees[3].role"       # Fourth employee's role
"company.products[*].description" # Wildcard: all products' descriptions
```

These are the region names in the compiled `CanvasLayout.regions` dict. The path
structure preserves the type hierarchy — you can always reconstruct what type a
region came from by parsing its path.

Wildcards (`[*]`) are supported in topology declarations and query methods, not
as literal region names in the layout.

## 5. Layout Strategies

This is the critical design decision. Given a type hierarchy, how do you map it
onto a (T, H, W) grid?

### 5.1 Packed (default)

Each type instance gets a contiguous rectangular subvolume. Array elements are
stacked along one axis. The grid is a rectangular bin-packing of all regions.

```
┌─────────────────────────────────┐
│ company.thought  │ company.goal │   Company's own fields
├──────────────────┼──────────────┤
│ emp[0].thought   │ emp[0].goal  │   Employee instances stacked
│ emp[0].role      │              │   along H axis
├──────────────────┼──────────────┤
│ emp[1].thought   │ emp[1].goal  │
│ emp[1].role      │              │
├──────────────────┼──────────────┤
│ ...              │              │
├──────────────────┼──────────────┤
│ prod[0].desc     │ prod[0].feat │   Products stacked below
│ prod[1].desc     │ prod[1].feat │
└──────────────────┴──────────────┘
```

**Pro**: Clean separation. Each instance is a self-contained subgrid. Easy to
reason about. Good for architectures with local attention (nearby positions
within an instance attend strongly; cross-instance requires explicit connections).

**Con**: Company.thought and Employee[i].thought are far apart on the grid.
The pretrained model's spatial priors don't help bridge them — you rely entirely
on explicit cross-attention connections.

**When to use**: Default. Most architectures. Especially when using full
attention (no spatial bias) or when instances should be relatively independent.

### 5.2 Interleaved (field-affinity)

Same-named fields across the hierarchy are spatially co-located. All "thought"
vectors live in one grid region, all "goal" vectors in another.

```
┌─────────────────────────────────────────┐
│ co.thought │ e[0].thought │ e[1].thought│ ... │   All thoughts together
├─────────────────────────────────────────┤
│ co.goal    │ e[0].goal    │ e[1].goal   │ ... │   All goals together
├─────────────────────────────────────────┤
│            │ e[0].role    │ e[1].role   │ ... │   Employee-only fields
├─────────────────────────────────────────┤
│ p[0].desc  │ p[1].desc    │ ...         │     │   Product fields
│ p[0].feat  │ p[1].feat    │ ...         │     │
└─────────────────────────────────────────┘
```

**Pro**: Company.thought is spatially adjacent to Employee[i].thought. For a
pretrained video model with spatial attention priors, adjacent positions already
have strong attention — you get parent-child coupling "for free" via spatial
locality. No explicit cross-attention needed for the pretrained backbone to
transfer information between a company and its employees' thoughts.

**Con**: Employee[i] is no longer a contiguous block — its fields are scattered
across the grid. Harder to extract a single instance's full state. Within-instance
coherence relies on explicit connections rather than spatial locality.

**When to use**: When the semantic relationship between parent and child fields
is the dominant information flow. Multi-agent systems where the group-level
field and individual-level field need tight coupling. When using a pretrained
model whose spatial priors you want to exploit.

This is the user's insight: "map each company latent region directly to each
individual employee latent region." The intuition is that the latent dynamics
of a company's thought and its employees' thoughts are fundamentally
correlated — they should be close in latent space because they're close in
semantic space.

### 5.3 Mirrored (overlapping / union)

Parent and child fields share the SAME grid positions. Company.thought and
Employee[0..9].thought all alias to the same latent subvolume. The diffusion
process resolves them into a shared representation.

```
┌─────────────────────────────────┐
│ co.thought = e[*].thought       │  SAME positions, multiple writers
├─────────────────────────────────┤
│ co.goal = e[*].goal             │  SAME positions
├─────────────────────────────────┤
│ e[0].role │ e[1].role │ ...     │  Instance-specific (not shared)
├─────────────────────────────────┤
│ p[0].desc │ p[1].desc │ ...     │  Products (not shared)
└─────────────────────────────────┘
```

**Pro**: Maximal coupling. The model is forced to learn a shared representation
that serves both company-level and employee-level semantics. Zero overhead for
parent-child communication — it's literally the same latent state. Extreme
parameter efficiency.

**Con**: Employees can't have independent thoughts — they're all writing to
the same positions. Only works when the semantic role is genuinely shared
(e.g., team consensus, broadcast state). Destroys individuality.

**When to use**: Broadcast regions. Shared task descriptions. Consensus state
in multi-agent systems. The "shared_task" pattern from existing topology
constructors is exactly this.

### 5.4 Per-instance (instance-affinity)

Array elements are stacked, but within each element, fields are co-located.

```
┌──────────────────────┐
│ e[0].thought │ e[0].goal │ e[0].role │   Instance 0 contiguous
├──────────────────────┤
│ e[1].thought │ e[1].goal │ e[1].role │   Instance 1 contiguous
├──────────────────────┤
│ ...                  │
├──────────────────────┤
│ co.thought │ co.goal │                   Company fields separate
└──────────────────────┘
```

**Pro**: Each instance is a coherent entity. Spatial locality favors within-
instance coherence. Good when instances are relatively autonomous and
internal state coherence matters more than cross-instance communication.

**Con**: Cross-instance communication (e.g., employee coordination) requires
long-range attention. Parent-child coupling is also long-range.

**When to use**: Autonomous agents with rich internal state. Robots that need
tight sensorimotor coupling within each body but loose coordination between
bodies.

### 5.5 Hybrid / User-specified

The user can override the layout strategy per-field or per-array:

```python
class Company(Agent):
    employees: Array[Employee, 10].configure(
        layout=ArrayLayout.INTERLEAVED,  # thoughts co-located with company.thought
        fields_interleaved=["thought", "goal"],  # only these fields interleave
        # role stays packed within each instance
    )
    products: Array[Product, 5].configure(
        layout=ArrayLayout.PACKED,  # products stay packed (independent)
    )
```

## 6. Connectivity Policies

The type hierarchy implies natural connectivity patterns. The compiler generates
default connections that can be overridden.

### 6.1 Intra-type policy (fields within a type)

Controls how fields of the same type instance connect.

| Policy | Meaning | Use case |
|--------|---------|----------|
| `"dense"` (default) | All fields attend to all fields | General-purpose |
| `"causal_chain"` | Fields connected in declaration order | Sequential processing |
| `"isolated"` | Each field self-attends only | Independent modalities |
| `"star"` | First field is hub, rest are spokes | Central representation + peripherals |

```python
class Robot(CanvasType):
    class Meta:
        intra_connectivity = "dense"

    camera: Vec[144]
    joints: Vec[8]
    action: Vec[8]
```

### 6.2 Parent-child policy

Controls connections between a type and its nested sub-types.

| Policy | Meaning | Use case |
|--------|---------|----------|
| `"matched_fields"` (default) | Parent.X ↔ Child.X for shared field names | Natural hierarchy |
| `"hub_spoke"` | Parent reads from all child fields, children read from parent | Central coordination |
| `"broadcast"` | Parent → all children (one-way) | Top-down commands |
| `"aggregate"` | All children → parent (one-way) | Bottom-up reporting |
| `"none"` | No automatic parent-child connections | Fully manual |

`"matched_fields"` is the powerful default: if Company has `thought` and
Employee has `thought` (inherited from Agent), the compiler creates
`Connection(src="company.thought", dst="company.employees[i].thought")`
and vice versa for each employee. **The shared field name is the semantic
link.** This is type-directed connectivity — the type structure determines
information flow.

### 6.3 Array element policy

Controls connections between elements of the same array.

| Policy | Meaning | Use case |
|--------|---------|----------|
| `"isolated"` (default) | Elements don't see each other | Independent agents |
| `"dense"` | Every element sees every element | Full coordination |
| `"ring"` | Each element sees neighbors | Spatial arrangement |
| `"hub"` | Virtual hub region, elements ↔ hub | Efficient coordination |
| `"matched_fields"` | Element[i].X ↔ Element[j].X | Field-specific cross-talk |

### 6.4 Cross-level policy

Controls whether grandparent connects directly to grandchild, or only through
the intermediate parent.

| Policy | Meaning |
|--------|---------|
| `"direct"` | Skip connections: grandparent directly connects to grandchild |
| `"hierarchical"` (default) | Information flows through intermediate levels only |

### 6.5 Temporal policy

Controls temporal connectivity for all generated connections.

| Policy | Meaning |
|--------|---------|
| `"dense"` (default) | All timesteps see all timesteps |
| `"causal"` | Same-frame self + prev-frame cross |
| `"same_frame"` | Same-frame only, no temporal leakage |

### 6.6 Explicit overrides

Any auto-generated connection can be overridden:

```python
class Company(Agent):
    class Meta:
        intra_connectivity = "dense"
        parent_child = "matched_fields"
        temporal = "causal"

    class Connections:
        # Override specific edges
        ("thought", "employees[*].thought"): {"fn": "gated", "weight": 0.5}
        ("employees[*].role", "goal"): {"fn": "perceiver"}
        # Disable an auto-generated edge
        ("goal", "products[*].description"): None
```

And arbitrary additional connections can be declared:

```python
    class Connections:
        # Add a connection the defaults wouldn't create
        ("employees[*].thought", "products[*].description"): {"fn": "cross_attention"}
```

## 7. Compilation Pipeline

```
CanvasType
    ↓ (1) Flatten
List[(path, Vec)]
    ↓ (2) Size computation
Dict[path → n_positions]
    ↓ (3) Packing
Dict[path → (t0, t1, h0, h1, w0, w1)]
    ↓ (4) RegionSpec generation
Dict[path → RegionSpec]
    ↓ (5) Connectivity generation
List[Connection]
    ↓ (6) Assembly
CanvasSchema
```

### Step 1: Flatten

Walk the type hierarchy depth-first. For arrays, expand each element. Produce
a flat list of `(dotted_path, Vec_config)` pairs.

```
Company.compile() →
  "company.thought"              Vec[64]
  "company.goal"                 Vec[32]
  "company.employees[0].thought" Vec[64]
  "company.employees[0].goal"    Vec[32]
  "company.employees[0].role"    Vec[16]
  "company.employees[1].thought" Vec[64]
  ... (x10 employees)
  "company.products[0].description" Vec[48]
  "company.products[0].features"    Vec[24]
  ... (x5 products)
```

Total positions: 64 + 32 + 10*(64+32+16) + 5*(48+24) = 1,576.

### Step 2: Size computation

Sum total positions. Validate against (T, H, W) grid capacity. If the grid is
too small, raise an error with a suggestion for minimum grid dimensions.

### Step 3: Packing

Map each (path, n_positions) to a (t0, t1, h0, h1, w0, w1) bounding box on the
grid. The packing algorithm depends on the layout strategy:

- **Packed**: recursive rectangle packing. Each type instance is a sub-rectangle.
  Arrays stack along a chosen axis.
- **Interleaved**: group by field name, pack each group as a row/column.
  Shared-name fields from parent and children are placed adjacent.
- **Mirrored**: shared fields map to the same bounds. Non-shared fields are
  packed normally.

The packer is a pure function: `(flattened_fields, T, H, W, strategy) → bounds_map`.
Deterministic. The user can always inspect and override the result.

### Step 4: RegionSpec generation

Convert each (path, bounds, Vec_config) into a `RegionSpec`:

```python
RegionSpec(
    bounds=(t0, t1, h0, h1, w0, w1),
    period=vec_config.period,
    is_output=vec_config.is_output,
    loss_weight=vec_config.loss_weight,
    default_attn=vec_config.attn,
    semantic_type=vec_config.semantic_type,
)
```

### Step 5: Connectivity generation

Apply the connectivity policies (Section 6) to produce a list of `Connection`
objects. The algorithm:

1. For each type instance, apply `intra_connectivity` to generate within-type edges.
2. For each parent-child relationship, apply `parent_child` policy.
3. For each array, apply `element_connectivity` between elements.
4. Apply `cross_level` policy for skip connections.
5. Apply `temporal` policy to all generated connections (setting t_src, t_dst).
6. Apply explicit overrides from `class Connections`.
7. Expand wildcards: `"employees[*].thought"` → per-element connections.

### Step 6: Assembly

Bundle into `CanvasSchema(layout=CanvasLayout(...), topology=CanvasTopology(...))`.

## 8. The Compile API

```python
# Basic compilation
schema = Company.compile(T=8, H=64, W=64, d_model=256)

# With overrides
schema = Company.compile(
    T=8, H=64, W=64, d_model=256,
    layout_strategy=LayoutStrategy.INTERLEAVED,
    temporal_policy="causal",
)

# Inspect before committing
plan = Company.plan(T=8, H=64, W=64, d_model=256)
print(plan.total_positions)        # 1576
print(plan.grid_utilization)       # 0.38 (1576 / 4096)
print(plan.region_summary())       # tabular: path, size, bounds, attn
print(plan.connection_summary())   # tabular: src, dst, fn, temporal

# Override a specific region's placement
plan.override_bounds("company.thought", (0, 8, 0, 8, 0, 1))
schema = plan.finalize()

# Or just get the low-level objects
layout, topology = Company.compile_raw(T=8, H=64, W=64, d_model=256)
```

## 9. Worked Examples

### 9.1 Multi-Robot Cell

```python
class Sensor(CanvasType):
    camera: Vec[144].configure(attn="cross_attention")
    depth: Vec[144].configure(attn="cross_attention")

class Actuator(CanvasType):
    joints: Vec[8].configure(attn="linear_attention", period=1)
    gripper: Scalar.configure(loss_weight=2.0)

class Robot(CanvasType):
    class Meta:
        intra_connectivity = "dense"

    sensor: Sensor
    actuator: Actuator
    plan: Vec[32].configure(attn="mamba")

class Cell(CanvasType):
    class Meta:
        parent_child = "hub_spoke"
        temporal = "causal"

    task: Vec[64].configure(is_output=False, attn="cross_attention")
    robots: Array[Robot, 3].configure(
        layout=ArrayLayout.PACKED,
        element_connectivity="hub",      # efficient coordination via virtual hub
    )

schema = Cell.compile(T=16, H=32, W=32, d_model=512)
```

Compiled regions:
- `cell.task` (64 positions, input-only)
- `cell.robots[0].sensor.camera` (144 positions)
- `cell.robots[0].sensor.depth` (144 positions)
- `cell.robots[0].actuator.joints` (8 positions)
- `cell.robots[0].actuator.gripper` (1 position)
- `cell.robots[0].plan` (32 positions)
- ... (x3 robots)
- `cell.robots._hub` (auto-generated hub for array coordination)

Auto-generated connections include:
- Dense within each Robot (sensor.camera ↔ sensor.depth ↔ actuator.joints ↔ ...)
- Hub-spoke: cell.task ↔ each robot's fields (parent_child="hub_spoke")
- Array hub: robots._hub ↔ each robot[i] (element_connectivity="hub")
- All connections with temporal causal constraints (t_src=0, t_dst=0 for self,
  t_src=0, t_dst=-1 for cross)

### 9.2 Company (interleaved layout)

```python
class Agent(CanvasType):
    thought: Vec[64]
    goal: Vec[32]

class Employee(Agent):
    role: Vec[16]

class Product(CanvasType):
    description: Vec[48]

class Company(Agent):
    class Meta:
        parent_child = "matched_fields"

    employees: Array[Employee, 10].configure(
        layout=ArrayLayout.INTERLEAVED,
        fields_interleaved=["thought", "goal"],  # only inherited Agent fields
        element_connectivity="matched_fields",    # emp[i].thought sees emp[j].thought
    )
    products: Array[Product, 5].configure(
        layout=ArrayLayout.PACKED,
        element_connectivity="isolated",
    )

schema = Company.compile(T=4, H=32, W=32, d_model=256)
```

Interleaved layout result:
```
Row 0: company.thought | emp[0].thought | emp[1].thought | ... | emp[9].thought
Row 1: company.goal    | emp[0].goal    | emp[1].goal    | ... | emp[9].goal
Row 2: emp[0].role | emp[1].role | ... | emp[9].role
Row 3: prod[0].description | prod[1].description | ... | prod[4].description
```

The key: Company.thought and Employee[i].thought are **spatially adjacent**.
A pretrained video model with spatial attention priors will naturally propagate
information between them. The type system's matched_fields connectivity adds
explicit connections, but even without them, the layout creates implicit
coupling via spatial locality.

### 9.3 Hierarchical Organization (recursive nesting)

```python
class Worker(Agent):
    skill: Vec[16]

class Team(Agent):
    members: Array[Worker, 5].configure(
        layout=ArrayLayout.INTERLEAVED,
        element_connectivity="dense",     # team members coordinate
    )

class Division(Agent):
    teams: Array[Team, 3].configure(
        layout=ArrayLayout.PACKED,        # teams are independent units
        element_connectivity="hub",       # lightweight cross-team coordination
    )

class Corporation(Agent):
    class Meta:
        cross_level = "hierarchical"      # no skip connections corp → worker

    divisions: Array[Division, 2]

schema = Corporation.compile(T=4, H=64, W=64, d_model=256)
```

This compiles to:
- 2 divisions × 3 teams × 5 workers = 30 workers
- Each worker: thought(64) + goal(32) + skill(16) = 112 positions
- Each team overhead: thought(64) + goal(32) = 96 positions
- Each division overhead: thought(64) + goal(32) = 96 positions
- Corporation overhead: thought(64) + goal(32) = 96 positions
- Total: 30*112 + 6*96 + 2*96 + 96 = 4,416 positions
- Grid: 4*64*64 = 16,384 → 27% utilization

Information flow: Corporation.thought → Division[i].thought → Team[j].thought
→ Worker[k].thought (hierarchical, no skip connections).

### 9.4 The Layout Strategy Comparison

The same logical type with different layout strategies:

```python
class Swarm(CanvasType):
    hive_mind: Vec[32]
    drones: Array[Drone, 8]

# PACKED: hive_mind is a separate block, drones are stacked
# Good for: drones as independent units, explicit coordination via connections

# INTERLEAVED: hive_mind.thought adjacent to drone[*].thought
# Good for: tight hive-mind coupling, pretrained spatial priors help

# MIRRORED: hive_mind.thought === drone[*].thought (same positions)
# Good for: shared consciousness, forced consensus

# PER_INSTANCE: each drone is a contiguous block, hive_mind separate
# Good for: rich internal drone state, autonomous operation
```

## 10. Interaction with Existing API

Canvas Types compiles DOWN to existing primitives. No changes needed to
CanvasLayout, CanvasTopology, Connection, RegionSpec, SpatiotemporalCanvas,
or any existing code.

```python
schema = Company.compile(T=4, H=32, W=32, d_model=256)

# Everything below works unchanged
canvas = SpatiotemporalCanvas(schema.layout)
batch = canvas.create_empty(batch_size=4)
batch = canvas.place(batch, thought_embs, "company.thought")
emp_thoughts = canvas.extract(batch, "company.employees[3].thought")
weights = schema.layout.loss_weight_mask("cuda")
mask = schema.topology.to_attention_mask(schema.layout)
schema.to_json("company_v1.json")
```

The type system is a compile-time abstraction. At runtime, it's all just
positions, attention masks, and loss weights — exactly what exists today.

## 11. Type Compatibility and Transfer

Two CanvasTypes that share an interface (or just share field names with
compatible semantic embeddings) can transfer latent state between corresponding
regions.

```python
class Agent(CanvasType):
    thought: Vec[64].configure(semantic_type="agent internal state")
    goal: Vec[32].configure(semantic_type="agent goal representation")

class Robot(Agent):
    camera: Vec[144]
    joints: Vec[8]

class SoftwareAgent(Agent):
    screen: Vec[256]
    keyboard: Vec[16]

# These share the Agent interface → thought and goal are transferable
Robot.transfer_map(SoftwareAgent)
# → {"thought": "thought", "goal": "goal"}
# with transfer distances from semantic embeddings

# At runtime: extract Robot.thought, inject into SoftwareAgent.thought
# The matched field names + shared interface = structural subtyping
```

This extends the existing `CanvasSchema.compatible_regions()` with type-level
structure. Instead of brute-force pairwise embedding comparison, the type
hierarchy tells you exactly which regions SHOULD be compatible.

## 12. What This Does NOT Do

- **No runtime dispatch.** The type system is compile-time only. It produces
  static schemas. It doesn't route tokens at inference time.
- **No dynamic sizing.** Array sizes are fixed at compile time. You can't have
  "up to 10 employees" — you have exactly 10 (unused slots get empty tokens).
- **No automatic encoder/decoder generation.** The type system declares the
  schema; you still need to build encoders that project raw modalities into
  the right regions.
- **No training loop.** Compilation produces masks and weights. You still
  write the training loop.

## 13. Open Questions

1. **Packing algorithm.** What's the right bin-packing strategy for 3D grids?
   Strip packing? Guillotine cuts? Or just row-major with user hints?

2. **Temporal field semantics.** Should some fields span all T, others span 1?
   How does this interact with period? Current thought: `Vec[N]` spans all T
   by default; `Vec[N].configure(temporal_extent=1)` for static fields.

3. **Virtual hub regions.** When `element_connectivity="hub"`, the compiler
   creates a synthetic hub region. How big should it be? User-specified?
   Heuristic (e.g., max field size of elements)?

4. **Wildcard expansion.** `"employees[*].thought"` in connectivity declarations
   expands to per-element connections. With 100 elements, that's 10,000
   connections. Need efficient mask generation (block-sparse).

5. **Visualization.** The compiled schema should be visualizable — showing
   which grid positions belong to which type path, colored by type, with
   connections overlaid.

6. **Class Meta vs decorators vs kwargs.** What's the most Pythonic API?
   Django-style Meta classes? Dataclass-style field()? Decorator-based?

## 14. Module Structure

```
canvas_engineering/
  types.py          # CanvasType, Vec, Array, Scalar, Shared
  packing.py        # Layout packing algorithms
  compile.py        # Flatten → Size → Pack → Wire → Assemble
  # existing files unchanged:
  canvas.py         # RegionSpec, CanvasLayout, SpatiotemporalCanvas
  connectivity.py   # Connection, CanvasTopology
  schema.py         # CanvasSchema
```

## 15. Why Same Repo

1. The compilation target IS canvas-engineering's existing primitives.
2. It completes the type system claim — layout is the memory model, types are
   the language.
3. Single package, layered API. Low-level users are unaffected. High-level
   users get the compiler.
4. Prevents drift between the type declarations and the layout engine.
5. The package stays small — three new files, zero changes to existing code.

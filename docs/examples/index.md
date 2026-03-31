# Examples

Runnable examples that train real models on canvas-structured data. Each example generates visualizations and demonstrates a specific capability.

## Core examples (v1 — compile_schema)

| # | Example | What it demonstrates | Key feature |
|---|---------|---------------------|-------------|
| **01** | [Hello Canvas Types](01-hello-types.md) | Declare, compile, train, visualize | `Field`, `compile_schema` |
| **02** | [Multi-Frequency Fusion](02-multi-frequency.md) | Structured vs flat allocation comparison | Bandwidth-proportional allocation |
| **03** | [CartPole Control](03-cartpole.md) | Real gym environment, BC + consistency loss | Self-consistency feedback dynamics |
| **04** | [Vehicle Fleet](04-vehicle-fleet.md) | 64-vehicle cooperative trajectory prediction | Multi-agent with social forces |
| **05** | [Protein Complex](05-protein-complex.md) | 4-chain binding affinity prediction | Molecular structure as canvas types |
| **06** | [Air Traffic Control](06-air-traffic.md) | Conflict detection with 12 aircraft | Safety-critical multi-agent |
| **07** | [Hospital ICU](07-hospital-icu.md) | Full-ward simulation with 6 patients | Deep type hierarchy |
| **08** | [Minecraft World Model](08-minecraft-world-model.md) | Next-frame prediction with imagination | Temporal hierarchy (period=1/4/16) |
| **09** | [Brain-Computer Interface](09-bci.md) | Multi-modal neural decoding (M1/PMd/S1) | Multi-array cursor + speech decoder |
| **09b** | [BCI + TRIBE v2](09b-bci-tribe.md) | Canvas decoder on real cortical predictions | Modal GPU, TRIBE v2, 68.8% accuracy |
| **10** | [Fusion Reactor](10-fusion-reactor.md) | Disruption prediction + plasma control | Multi-timescale diagnostics |
| **11** | [Mars Colony](11-mars-colony.md) | Cascading failure detection | 5+ subsystems, alert prediction |

## v2 examples (compile_program — families, carriers, operators, scheduling)

| # | Example | What it demonstrates | Key v2 feature |
|---|---------|---------------------|----------------|
| **14** | [Saccading Vision](14-saccading-vision.md) | Event-triggered scheduling, residual-driven fovea | `RegionScheduler`, `ResidualAccumulator`, families |
| **15** | [World Model Carriers](15-world-model-carriers.md) | Mixed deterministic/diffusive/filter dynamics | `carrier` field, `compile_program` |
| **16** | [Memory Consolidation](16-memory-consolidation.md) | Boundary clocks, working→episodic→semantic | `ClockSpec`, `ProgramCompiler`, compile modes |
| **17** | [Graph Parser](17-graph-parser.md) | Operator types, fixpoint iteration | `operator`, `ClockExpr` IR |
| **18** | [Sparse Reflection](18-sparse-reflection.md) | Uncertainty-driven self-reflection | `ConstraintSpec`, `LearnedScheduler` |

## Running

```bash
# Install canvas-engineering
pip install canvas-engineering

# Run any example (CPU, 30-120 seconds)
python examples/01_hello_canvas_types.py
python examples/14_saccading_vision.py

# Run TRIBE v2 BCI on Modal (requires GPU)
modal run examples/09b_bci_tribe_modal.py

# Run all examples on Modal
modal run run_examples_modal.py
```

Each example generates a multi-panel visualization to `assets/examples/`.

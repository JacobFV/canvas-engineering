# Example 16: Memory Consolidation

Three memory timescales with boundary clocks: working memory updates every step, episodic memory writes on event triggers, and semantic memory consolidates at episode boundaries. After training, ProgramCompiler exports semantic memory as a frozen buffer and removes episodic storage.

**Source**: [`examples/16_memory_consolidation.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/16_memory_consolidation.py) 
## Result

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/16_memory_consolidation.png" alt="Memory consolidation results" width="100%">
</p>

**Left**: Memory write timeline -- working memory fires every step, episodic writes fire on surprise events, semantic consolidation fires at episode boundaries.

**Center**: Retrieval accuracy across timescales -- working memory excels at recent items, semantic memory at distant items, episodic bridges the gap.

**Right**: Compiled program comparison -- before (all 5 regions active, 8 connections) vs after (3 active regions, semantic exported as frozen buffer).

## Type declaration

```python
@dataclass
class MemorySystem:
    perception: Field = Field(4, 4, family="observation")
    working: Field = Field(4, 4, family="memory", carrier="memory")
    episodic: Field = Field(4, 8, family="memory", carrier="memory")
    semantic: Field = Field(2, 8, family="memory", carrier="memory")
    recall: Field = Field(2, 4, family="action", loss_weight=2.0)
```

## Compile with boundary clocks

```python
bound, program = compile_program(MemorySystem(), T=16, H=8, W=8, d_model=64)

# Clock declarations: three timescales
program.regions["working"] = RegionProgram(
    family="memory", carrier="memory",
    clock=ClockSpec(mode="periodic", period=1),
    learning=LearningSpec(mode="retrieval", compile_mode="runtime"),
)
program.regions["episodic"] = RegionProgram(
    family="memory", carrier="memory",
    clock=ClockSpec(mode="on_event", event_source="perception.prediction", event_threshold=0.6),
    learning=LearningSpec(mode="retrieval", compile_mode="freeze"),
)
program.regions["semantic"] = RegionProgram(
    family="memory", carrier="memory",
    clock=ClockSpec(mode="boundary", event_source="episode_end"),
    learning=LearningSpec(mode="retrieval", compile_mode="export"),
)

# After training: compile for deployment
compiler = ProgramCompiler(program)
compiled = compiler.compile(module=model, export_dir="./deploy")
# compiled.exported_memories == {"semantic"}
# compiled.frozen_regions   == {"episodic", "perception"}
# compiled.active_regions   == {"working", "recall", ...}  # semantic removed
# Files on disk: ./deploy/semantic.pt + ./deploy/semantic.pt.json
```

## Latest run (aarch64)

```
CompiledProgram:
  active: 4 regions, 16 connections
  frozen: ['episodic', 'perception']
  exported: ['semantic']
  eliminated: 1 regions
```

The `compile()` call now actually freezes the live `episodic`/`perception`
parameters, dumps `semantic.state_dict()` to disk, and removes it from
the active graph — not just a stored tag.

## What this shows

- **Boundary clocks** -- `periodic`, `on_event`, and `boundary` modes on a single program
- **`ProgramCompiler`** -- freeze, export, and dead-region elimination in one pass
- **Memory consolidation** -- working (fast) to episodic (event) to semantic (boundary) mirrors cognitive architecture
- **Deployment shrinkage** -- compiled program has fewer active regions and connections than the training program

## Run it

```bash
python examples/16_memory_consolidation.py
# Generates: assets/examples/16_memory_consolidation.png
```

# Example 17: Graph Parser

Operator types and fixpoint iteration for structured prediction. Tokens are observed, spans are bound from tokens, and graph nodes are integrated from spans. Five refinement steps iterate the parse until edge predictions stabilize. ClockExpr IR serializes the scheduling rules for reproducible deployment.

**Source**: [`examples/17_graph_parser.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/17_graph_parser.py) 
## Result

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/17_graph_parser.png" alt="Graph parser results" width="100%">
</p>

**Left**: Parse evolution over 5 refinement steps -- edge confidences sharpen as the fixpoint loop converges, with spurious edges pruned by step 3.

**Center**: Operator flow diagram -- `observe` (tokens to spans), `bind` (spans to nodes), `integrate` (node self-refinement), with connection weights visualized.

**Right**: ClockExpr IR round-trip -- the serialized JSON AST matches the deserialized program exactly, confirming lossless scheduling serialization.

## Type declaration

```python
@dataclass
class GraphParser:
    tokens: Field = Field(1, 16, family="observation")
    spans: Field = Field(2, 8, family="state", tags=("parser",))
    nodes: Field = Field(4, 4, family="state", tags=("parser", "object"))
    edges: Field = Field(4, 4, family="action", loss_weight=2.0)
```

## Compile with operator types

```python
bound, program = compile_program(GraphParser(), T=5, H=8, W=8, d_model=64)

# Auto-wired operators from family pairs:
#   tokens -> spans:  operator="observe"   (observation -> state)
#   spans  -> nodes:  operator="integrate"  (state -> state)
#   nodes  -> edges:  operator="act"        (state -> action)

# Fixpoint iteration: 5 refinement steps
for step in range(5):
    canvas = dispatcher(canvas, active_regions=program.schema.layout.regions.keys())
    edge_delta = (canvas_new - canvas_old).abs().max()
    if edge_delta < 0.01:
        break  # converged

# Serialize scheduling for deployment
program_dict = program.to_dict()
restored = CanvasProgram.from_dict(program_dict)
assert restored.to_dict() == program_dict  # lossless round-trip
```

## What this shows

- **Operator types** -- `observe`, `integrate`, `act` auto-wired from family pairs via `DEFAULT_WIRING`
- **Fixpoint iteration** -- rerunning the canvas converges to stable parses in 3-5 steps
- **Tags for nuance** -- `tags=("parser", "object")` annotates role without changing wiring
- **ClockExpr serialization** -- `to_dict()` / `from_dict()` round-trip preserves the full program spec

## Run it

```bash
python examples/17_graph_parser.py
# Generates: assets/examples/17_graph_parser.png
```

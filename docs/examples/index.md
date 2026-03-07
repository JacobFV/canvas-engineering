# Examples

Runnable examples that train real models on canvas-structured data. Each example generates visualizations and demonstrates a specific capability.

All examples run on CPU in under 60 seconds. No GPU required.

| # | Example | What it demonstrates | Key feature |
|---|---------|---------------------|-------------|
| **01** | [Hello Canvas Types](01-hello-types.md) | Declare, compile, train, visualize | `Field`, `compile_schema` |
| **02** | [Multi-Frequency Fusion](02-multi-frequency.md) | Structured vs flat allocation comparison | Bandwidth-proportional allocation |
| **03** | [CartPole Control](03-cartpole.md) | Real gym environment, BC + consistency loss | Self-consistency feedback dynamics |

## Running

```bash
# Install canvas-engineering
pip install canvas-engineering

# Run any example
python examples/01_hello_canvas_types.py
python examples/02_multi_frequency.py
python examples/03_cartpole_control.py
```

Each example generates a multi-panel visualization to `assets/examples/`.

## Coming soon

| # | Example | Domain |
|---|---------|--------|
| 04 | Autonomous Vehicle Fleet | Multi-agent cooperative perception |
| 05 | Protein Complex | Molecular biology, binding affinity |
| 06 | Air Traffic Control | Safety-critical multi-agent |
| 07 | Hospital ICU | Multi-system physiological model |
| 08 | Minecraft World Model | Temporal hierarchy + imagination |
| 09 | Brain-Computer Interface | Neural decoding with closed-loop feedback |
| 10 | Tokamak Fusion Reactor | Multi-timescale plasma control |
| 11 | Mars Colony | 70+ field cascading failure detection |

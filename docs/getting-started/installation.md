# Installation

```bash
# Core (canvas + looped blocks)
pip install canvas-engineering

# With CogVideoX support
pip install canvas-engineering[cogvideox]

# With video dataset loading
pip install canvas-engineering[data]

# Development
pip install canvas-engineering[dev]
```

Requires Python 3.9+ and PyTorch 2.0+.

## From source

```bash
git clone https://github.com/JacobFV/canvas-engineering.git
cd canvas-engineering
pip install -e ".[dev]"
python -m pytest tests/ -v  # 95 tests
```

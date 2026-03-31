# Recipe: Computer Use Agent

A canvas schema for a computer use agent with screen capture, mouse/keyboard, and language instruction.

```python
from canvas_engineering import *

layout = CanvasLayout(
    T=16, H=32, W=32, d_model=768,
    regions={
        "screen": RegionSpec(
            bounds=(0, 16, 0, 24, 0, 24),
            semantic_type="RGB video 224x224 30fps screen capture",
        ),
        "mouse": RegionSpec(
            bounds=(0, 16, 24, 26, 0, 4),
            loss_weight=2.0,
            default_attn="linear_attention",
            semantic_type="mouse position + button state 30Hz",
        ),
        "keyboard": RegionSpec(
            bounds=(0, 16, 26, 28, 0, 4),
            loss_weight=2.0,
            default_attn="linear_attention",
            semantic_type="keyboard action sequence 30Hz",
        ),
        "instruction": RegionSpec(
            bounds=(0, 1, 28, 32, 0, 8),
            is_output=False, period=16,
            default_attn="cross_attention",
            semantic_type="natural language user instruction",
        ),
    },
)

topology = CanvasTopology(connections=[
    Connection(src="screen", dst="screen"),
    Connection(src="mouse", dst="mouse"),
    Connection(src="keyboard", dst="keyboard"),
    Connection(src="instruction", dst="instruction"),

    # Mouse and keyboard read from screen
    Connection(src="mouse", dst="screen"),
    Connection(src="keyboard", dst="screen"),

    # Instruction conditions everything via gated attention
    Connection(src="screen", dst="instruction", fn="gated"),
    Connection(src="mouse", dst="instruction", fn="gated"),
    Connection(src="keyboard", dst="instruction", fn="gated"),
])
```

## v2: with typed fields

```python
from dataclasses import dataclass
from canvas_engineering import Field, compile_program

@dataclass
class ComputerAgent:
    screen: Field = Field(24, 24, family="observation")
    mouse: Field = Field(2, 4, family="action", loss_weight=2.0, attn="linear_attention")
    keyboard: Field = Field(2, 4, family="action", loss_weight=2.0, attn="linear_attention")
    instruction: Field = Field(4, 8, family="state", tags=("goal",), is_output=False, period=16)

bound, program = compile_program(ComputerAgent(), T=16, H=32, W=32, d_model=768)
```

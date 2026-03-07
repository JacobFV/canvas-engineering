"""World model: Minecraft agent with imagination and hierarchical planning.

A full cognitive architecture for a Minecraft agent, expressed as canvas types.
Three levels of temporal abstraction:
- Perception (period=1): raw pixels, block states, entity positions
- Reasoning (period=4): spatial reasoning, object affordances, goal decomposition
- Planning (period=16): high-level strategy, multi-step plans

The agent can "imagine" — it has an explicit imagination buffer that gets
diffused forward by the same model. The imagined future is compared against
the actual future (loss_weight=0.5 — soft supervision).

The inventory and crafting knowledge are input-only context.

This is the kind of architecture that DIAMOND, Genie, and GameNGen
describe, but here the temporal hierarchy and cognitive structure are
declared, not hand-coded into separate networks.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy


@dataclass
class PerceptionModule:
    """Low-level perception at full temporal resolution."""
    visual: Field = Field(16, 16, period=1,     # 256 pos: screen pixels/patches
                          semantic_type="Minecraft screen 320x240 RGB patches")
    depth: Field = Field(8, 8, period=1,        # 64 pos: depth buffer
                         semantic_type="Minecraft depth buffer 160x120")
    blocks: Field = Field(4, 4, period=1,       # 16 pos: nearby block states
                          semantic_type="3D block states 5x5x5 around player")
    entities: Field = Field(2, 8, period=1,     # 16 pos: entity positions + types
                            attn="sparse_attention",  # few entities, selective binding
                            semantic_type="nearby entity positions and types")


@dataclass
class ReasoningModule:
    """Mid-level reasoning at 4x temporal abstraction."""
    spatial: Field = Field(8, 8, period=4,      # 64 pos: spatial reasoning
                           semantic_type="spatial map and navigation state")
    affordances: Field = Field(4, 4, period=4,  # 16 pos: what can I do here?
                               semantic_type="object affordance predictions")
    goals: Field = Field(4, 8, period=4,        # 32 pos: subgoal decomposition
                         loss_weight=2.0,
                         attn="cross_attention",
                         semantic_type="active subgoal stack")


@dataclass
class PlanningModule:
    """High-level planning at 16x temporal abstraction."""
    strategy: Field = Field(4, 8, period=16,    # 32 pos: overall strategy
                            attn="mamba",        # long-horizon sequential
                            semantic_type="high-level strategy multi-step plan")
    world_model: Field = Field(8, 8, period=16, # 64 pos: learned world dynamics
                               semantic_type="learned world transition model state")


@dataclass
class ImaginationBuffer:
    """Counterfactual simulation: the agent's imagination."""
    imagined_future: Field = Field(8, 8, period=4,  # 64 pos: imagined next states
                                   loss_weight=0.5,  # soft supervision vs. reality
                                   semantic_type="imagined future visual state")
    imagined_reward: Field = Field(1, 4, period=4,   # 4 pos: expected reward
                                   loss_weight=0.5,
                                   semantic_type="imagined future reward signal")


@dataclass
class MinecraftAgent:
    """Full cognitive architecture for a Minecraft agent."""
    # Context (input-only)
    inventory: Field = Field(4, 8, is_output=False,
                             semantic_type="36-slot inventory contents")
    crafting: Field = Field(2, 8, is_output=False,
                            semantic_type="available crafting recipes")
    task: Field = Field(4, 8, is_output=False,
                        semantic_type="natural language task instruction")

    # Cognitive modules
    perception: PerceptionModule = dc_field(default_factory=PerceptionModule)
    reasoning: ReasoningModule = dc_field(default_factory=ReasoningModule)
    planning: PlanningModule = dc_field(default_factory=PlanningModule)
    imagination: ImaginationBuffer = dc_field(default_factory=ImaginationBuffer)

    # Actions (the actual output)
    action: Field = Field(2, 8, period=1,       # 16 pos: action per frame
                          loss_weight=3.0,
                          semantic_type="Minecraft action discrete+continuous")
    camera: Field = Field(1, 4, period=1,       # 4 pos: camera control
                          loss_weight=2.0,
                          semantic_type="camera pitch yaw delta per frame")


agent = MinecraftAgent()
bound = compile_schema(
    agent, T=64, H=64, W=64, d_model=512,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",    # agent-level fields <-> all modules
        temporal="causal",           # causal for real-time play
    ),
)

print("=== Minecraft World Model Agent ===")
print(bound.summary())

# Show the temporal hierarchy
print("\nTemporal hierarchy:")
for period in [1, 4, 16]:
    fields_at_period = [(n, bf) for n, bf in bound.fields.items()
                        if bf.spec.period == period]
    total_pos = sum(bf.num_positions for _, bf in fields_at_period)
    print(f"  period={period} ({['frame', '4-frame', '16-frame'][([1,4,16].index(period))]}): "
          f"{len(fields_at_period)} fields, {total_pos:,} positions")

# Imagination gets soft supervision
print("\nImagination buffer:")
for name, bf in bound.fields.items():
    if "imagination" in name or "imagined" in name:
        print(f"  {name}: loss_weight={bf.spec.loss_weight} (soft supervision)")

# Input context fields
print("\nInput-only context (filed, not predicted):")
for name, bf in bound.fields.items():
    if not bf.spec.is_output:
        print(f"  {name}: {bf.spec.semantic_type or '(no type)'}")

# Loss budget
weights = bound.layout.loss_weight_mask("cpu")
total_w = weights.sum().item()
print(f"\nLoss budget:")
for category, keywords in [
    ("Actions", ["action", "camera"]),
    ("Subgoals", ["goals"]),
    ("Imagination", ["imagin"]),
    ("Perception", ["visual", "depth", "blocks", "entities"]),
    ("Reasoning", ["spatial", "affordances"]),
    ("Planning", ["strategy", "world_model"]),
]:
    w = sum(sum(weights[i].item() for i in bound[n].indices())
            for n in bound.field_names if any(k in n for k in keywords))
    if w > 0:
        print(f"  {category}: {w/total_w*100:.1f}%")

"""Surgical robot: da Vinci-style teleoperation with latent safety monitoring.

A dual-arm surgical system where:
- Each arm has a stereo endoscope, 7-DOF joints, and a tool tip force sensor
- A shared "surgical field" representation fuses both camera views
- A safety monitor runs at low frequency (period=4) to flag unsafe states
- The surgeon's intent (from master controller) is input-only conditioning
- Tool-tissue interaction forces get 3x loss weight (safety-critical)

The type hierarchy mirrors the physical system:
  SurgicalSystem
    surgeon_intent (input-only)
    safety_monitor (low-freq)
    surgical_field (shared visual)
    left_arm: SurgicalArm
    right_arm: SurgicalArm

Each SurgicalArm:
    stereo_cam (high-res visual)
    joints (7-DOF)
    force (tool-tip forces, high loss weight)
    action (predicted motor commands)
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy, LayoutStrategy


@dataclass
class SurgicalArm:
    """One arm of a surgical robot with endoscope, joints, force sensing."""
    stereo_cam: Field = Field(8, 8)    # 64 positions: stereo endoscope patches
    joints: Field = Field(1, 7)        # 7 positions: 7-DOF joint angles
    force: Field = Field(1, 3,         # 3 positions: 3-axis force at tool tip
                         loss_weight=3.0,  # safety-critical — must be accurate
                         attn="linear_attention")  # low-dim, no need for O(N^2)
    action: Field = Field(1, 7,        # 7 positions: predicted motor commands
                          loss_weight=2.0)


@dataclass
class SurgicalSystem:
    """Complete da Vinci-style teleoperated surgical system."""
    # Surgeon's intent from master controller — input only, never predicted
    surgeon_intent: Field = Field(2, 8, is_output=False,
                                  semantic_type="surgeon master controller 14-DOF")

    # Safety monitor: low-frequency holistic assessment
    safety_monitor: Field = Field(4, 4, period=4,  # updates every 4th frame
                                  loss_weight=5.0,  # VERY important to get right
                                  attn="mamba",      # temporal state tracking
                                  semantic_type="surgical safety state assessment")

    # Shared surgical field: fused view from both endoscopes
    surgical_field: Field = Field(12, 12,  # 144 positions: rich spatial representation
                                  semantic_type="fused stereo surgical field view")

    # Dual arms
    left_arm: SurgicalArm = dc_field(default_factory=SurgicalArm)
    right_arm: SurgicalArm = dc_field(default_factory=SurgicalArm)


system = SurgicalSystem()
bound = compile_schema(
    system, T=16, H=48, W=48, d_model=512,
    connectivity=ConnectivityPolicy(
        intra="dense",              # all system-level fields attend to each other
        parent_child="hub_spoke",   # surgical_field sees all arm fields and vice versa
        array_element="isolated",   # left and right arms are independent streams
        temporal="causal",          # strictly causal: no future leakage
    ),
)

print("=== Surgical Robot Canvas ===")
print(bound.summary())

# Show the safety-critical fields
print("\nSafety-critical fields (loss_weight > 1.0):")
for name, bf in bound.fields.items():
    if bf.spec.loss_weight > 1.0:
        print(f"  {name}: loss_weight={bf.spec.loss_weight}, "
              f"period={bf.spec.period}, attn={bf.spec.default_attn}")

# Show the loss weight distribution
weights = bound.layout.loss_weight_mask("cpu")
total = weights.sum().item()
for name, bf in bound.fields.items():
    indices = bf.indices()
    field_weight = sum(weights[i].item() for i in indices)
    if field_weight > 0:
        print(f"  {name}: {field_weight/total*100:.1f}% of total loss")

# Show causal temporal connections
print(f"\nTotal connections: {len(bound.topology.connections)}")
causal = [c for c in bound.topology.connections if c.t_dst == -1]
print(f"  Causal (prev-frame) connections: {len(causal)}")
same_frame = [c for c in bound.topology.connections if c.t_dst == 0]
print(f"  Same-frame connections: {len(same_frame)}")

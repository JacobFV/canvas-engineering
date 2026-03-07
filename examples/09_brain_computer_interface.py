"""Brain-computer interface: neural decoding for paralyzed patients.

Decode motor intent from intracortical electrode arrays (like BrainGate/Neuralink)
into cursor control, speech synthesis, and robotic arm commands simultaneously.

The canvas type hierarchy mirrors the neural decoding pipeline:
- Raw neural signals (high-frequency, 30kHz sampled, binned to features)
- Cortical area decomposition (M1, PMd, S1 — separate arrays)
- Decoded intent at multiple output modalities
- Feedback signals (cursor position, speech audio) for closed-loop decoding

Key insight: different cortical areas decode different things. M1 (primary motor)
is best for hand kinematics. PMd (dorsal premotor) is best for reach planning.
S1 (somatosensory) provides sensory feedback. The type hierarchy makes this
explicit — each array has its own attention type and loss configuration.

The decoder must run at <50ms latency. Everything is causal, period=1.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy


@dataclass
class ElectrodeArray:
    """One Utah array implanted in a cortical area."""
    spikes: Field = Field(8, 12,             # 96 positions (96 electrodes, 1:1)
                          period=1,
                          attn="linear_attention",  # must be fast, O(N)
                          semantic_type="binned spike counts 96ch 50ms bins")
    lfp: Field = Field(4, 8,                 # 32 positions: local field potentials
                       period=1,
                       attn="linear_attention",
                       semantic_type="LFP bandpower delta/theta/alpha/beta/gamma")


@dataclass
class CursorDecoder:
    """2D cursor control for computer access."""
    velocity: Field = Field(1, 2, period=1,   # 2 positions: vx, vy
                            loss_weight=5.0,
                            semantic_type="decoded cursor velocity 2D")
    click: Field = Field(1, 1, period=1,      # 1 position: click probability
                         loss_weight=3.0,
                         semantic_type="decoded mouse click probability")


@dataclass
class SpeechDecoder:
    """Speech synthesis from attempted speech neural activity."""
    phonemes: Field = Field(4, 8, period=1,   # 32 positions: phoneme probabilities
                            loss_weight=4.0,
                            attn="cross_attention",  # content-based selection
                            semantic_type="decoded phoneme posterior probabilities")
    prosody: Field = Field(2, 4, period=1,    # 8 positions: pitch, loudness, rate
                           loss_weight=2.0,
                           semantic_type="decoded speech prosody features")


@dataclass
class RoboticArmDecoder:
    """7-DOF robotic arm control for reaching and grasping."""
    endpoint: Field = Field(1, 6, period=1,   # 6 positions: xyz + rpy
                            loss_weight=5.0,
                            semantic_type="decoded end-effector 6DOF pose")
    grasp: Field = Field(1, 1, period=1,      # 1 position: grasp aperture
                         loss_weight=4.0,
                         semantic_type="decoded grasp aperture")
    force: Field = Field(1, 3, period=1,      # 3 positions: desired contact force
                         loss_weight=3.0,
                         semantic_type="decoded desired contact force 3-axis")


@dataclass
class BCISystem:
    """Complete intracortical brain-computer interface system."""
    # Neural inputs — one array per cortical area
    m1: ElectrodeArray = dc_field(default_factory=ElectrodeArray)    # primary motor
    pmd: ElectrodeArray = dc_field(default_factory=ElectrodeArray)   # dorsal premotor
    s1: ElectrodeArray = dc_field(default_factory=ElectrodeArray)    # somatosensory

    # Feedback (input-only, closed-loop)
    cursor_feedback: Field = Field(1, 4, is_output=False, period=1,
                                   semantic_type="visual cursor position feedback")
    audio_feedback: Field = Field(2, 4, is_output=False, period=1,
                                  semantic_type="auditory speech output feedback")
    haptic_feedback: Field = Field(1, 4, is_output=False, period=1,
                                   semantic_type="haptic force feedback from robot arm")

    # Decoded outputs — multiple simultaneous modalities
    cursor: CursorDecoder = dc_field(default_factory=CursorDecoder)
    speech: SpeechDecoder = dc_field(default_factory=SpeechDecoder)
    arm: RoboticArmDecoder = dc_field(default_factory=RoboticArmDecoder)

    # Internal decoding state
    intent: Field = Field(4, 8, period=1,     # 32 positions: decoded motor intent
                          attn="mamba",        # temporal state tracking
                          semantic_type="decoded high-level motor intent")


bci = BCISystem()
bound = compile_schema(
    bci, T=32, H=32, W=32, d_model=256,
    connectivity=ConnectivityPolicy(
        intra="dense",              # all system-level fields fully connected
        parent_child="hub_spoke",   # intent <-> all arrays and decoders
        temporal="causal",          # real-time, <50ms latency
    ),
)

print("=== Brain-Computer Interface System ===")
print(f"Electrode channels: 3 arrays x 96 = 288")
print(f"Output modalities: cursor + speech + robotic arm")
print(f"Total fields: {len(bound.field_names)}")
print(f"Total connections: {len(bound.topology.connections)}")
print(f"All temporal: causal (prev-frame + same-frame)")
print()

# Show cortical area -> decoder mapping
print("Neural input arrays:")
for area in ["m1", "pmd", "s1"]:
    spikes = bound[f"{area}.spikes"]
    lfp = bound[f"{area}.lfp"]
    print(f"  {area.upper()}: {spikes.num_positions + lfp.num_positions} positions "
          f"(spikes: {spikes.num_positions}, LFP: {lfp.num_positions})")

print("\nDecoder outputs:")
for decoder_name, fields in [
    ("Cursor", ["cursor.velocity", "cursor.click"]),
    ("Speech", ["speech.phonemes", "speech.prosody"]),
    ("Robotic arm", ["arm.endpoint", "arm.grasp", "arm.force"]),
]:
    total_pos = sum(bound[f].num_positions for f in fields)
    max_lw = max(bound[f].spec.loss_weight for f in fields)
    print(f"  {decoder_name}: {total_pos} positions, max loss_weight={max_lw}")

# Closed-loop feedback
print("\nClosed-loop feedback (input-only):")
for name in ["cursor_feedback", "audio_feedback", "haptic_feedback"]:
    bf = bound[name]
    print(f"  {name}: {bf.spec.semantic_type}")

# Loss budget
weights = bound.layout.loss_weight_mask("cpu")
total_w = weights.sum().item()
print(f"\nLoss budget:")
for category, keywords in [
    ("Cursor control", ["cursor.velocity", "cursor.click"]),
    ("Speech synthesis", ["phonemes", "prosody"]),
    ("Robotic arm", ["endpoint", "grasp", "force"]),
    ("Neural encoding", ["spikes", "lfp"]),
    ("Intent decoding", ["intent"]),
]:
    w = sum(sum(weights[i].item() for i in bound[n].indices())
            for n in bound.field_names if any(k in n for k in keywords))
    if w > 0:
        print(f"  {category}: {w/total_w*100:.1f}%")

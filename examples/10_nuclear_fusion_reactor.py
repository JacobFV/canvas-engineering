"""Tokamak fusion reactor: real-time plasma control and disruption prediction.

Model a tokamak fusion reactor's control system as a canvas type hierarchy.
Based on the architecture that DeepMind used for TCV plasma control, but
generalized to a full reactor with multiple diagnostic systems and actuators.

The plasma in a tokamak is a 100-million-degree fluid held in place by
magnetic fields. If the control system fails for even milliseconds, the
plasma disrupts — potentially damaging the reactor wall with forces
equivalent to a small explosion.

Hierarchy:
  Tokamak
    diagnostics: multiple sensor systems at different frequencies
    magnetics: magnetic field coil states
    plasma_state: reconstructed equilibrium
    actuators: heating, fueling, coil currents
    disruption_predictor: the safety-critical output

Key design:
- Magnetic diagnostics at period=1 (1ms control cycle)
- Thomson scattering at period=10 (10ms integration time)
- Disruption prediction at 10x loss weight — a missed disruption can
  cost $100M in reactor damage
- Actuator commands must be causal and low-latency
- Plasma equilibrium reconstruction uses perceiver attention (compress
  thousands of diagnostic channels into a compact state)
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy


@dataclass
class MagneticDiagnostics:
    """Magnetic probe arrays and flux loops (fastest diagnostics)."""
    probes: Field = Field(8, 8, period=1,       # 64 pos: ~200 magnetic probes
                          attn="linear_attention",  # fast, low-latency
                          semantic_type="magnetic probe array 200ch 1kHz")
    flux_loops: Field = Field(4, 4, period=1,   # 16 pos: flux loop measurements
                              attn="linear_attention",
                              semantic_type="flux loop measurements 40ch 1kHz")
    rogowski: Field = Field(1, 4, period=1,     # 4 pos: plasma current + shape
                            semantic_type="Rogowski coil plasma current")


@dataclass
class ThomsonScattering:
    """Electron temperature and density profiles (slower, but critical)."""
    te_profile: Field = Field(4, 8, period=10,  # 32 pos: electron temp profile
                              semantic_type="Thomson Te profile 30 spatial points")
    ne_profile: Field = Field(4, 8, period=10,  # 32 pos: electron density profile
                              semantic_type="Thomson ne profile 30 spatial points")


@dataclass
class Interferometer:
    """Line-integrated electron density."""
    density: Field = Field(2, 4, period=5,      # 8 pos: chord-integrated density
                           semantic_type="interferometer line-integrated density 8ch")


@dataclass
class SpectroscopyDiag:
    """Impurity and radiation diagnostics."""
    impurities: Field = Field(4, 4, period=20,  # 16 pos: impurity concentrations
                              semantic_type="VUV spectroscopy impurity concentrations")
    radiation: Field = Field(4, 4, period=10,   # 16 pos: bolometry radiation profile
                             semantic_type="bolometer array radiation profile")


@dataclass
class HeatingSystem:
    """Auxiliary heating actuator."""
    power: Field = Field(1, 4, period=1,        # 4 pos: NBI/ECH/ICH power setpoints
                         loss_weight=2.0,
                         semantic_type="auxiliary heating power setpoints MW")


@dataclass
class CoilSystem:
    """Magnetic field coil actuator."""
    currents: Field = Field(4, 4, period=1,     # 16 pos: PF/TF coil currents
                            loss_weight=3.0,
                            attn="linear_attention",
                            semantic_type="poloidal field coil current setpoints kA")


@dataclass
class Tokamak:
    """Complete tokamak fusion reactor control system."""
    # Diagnostics (inputs + some predicted)
    magnetics: MagneticDiagnostics = dc_field(default_factory=MagneticDiagnostics)
    thomson: ThomsonScattering = dc_field(default_factory=ThomsonScattering)
    interferometer: Interferometer = dc_field(default_factory=Interferometer)
    spectroscopy: SpectroscopyDiag = dc_field(default_factory=SpectroscopyDiag)

    # Plasma state reconstruction (the core physics)
    equilibrium: Field = Field(8, 8, period=1,   # 64 pos: reconstructed flux surfaces
                               loss_weight=3.0,
                               attn="perceiver",  # compress diagnostics into equilibrium
                               semantic_type="EFIT magnetic equilibrium reconstruction")

    pressure: Field = Field(4, 8, period=5,      # 32 pos: pressure profile
                            loss_weight=2.0,
                            semantic_type="kinetic pressure profile")

    current_profile: Field = Field(4, 8, period=5,  # 32 pos: current density profile
                                   loss_weight=2.0,
                                   semantic_type="toroidal current density profile j(r)")

    # Machine parameters (input-only context)
    machine_config: Field = Field(2, 4, is_output=False,
                                  semantic_type="tokamak geometry Rmaj/a/kappa/delta/Bt")

    # Actuators (the control outputs)
    heating: HeatingSystem = dc_field(default_factory=HeatingSystem)
    coils: CoilSystem = dc_field(default_factory=CoilSystem)

    gas_injection: Field = Field(1, 4, period=1,  # 4 pos: gas valve commands
                                 loss_weight=2.0,
                                 semantic_type="gas injection valve commands")

    # Disruption prediction (SAFETY CRITICAL)
    disruption_risk: Field = Field(2, 4, period=1,  # 8 pos: disruption probability
                                   loss_weight=10.0,  # $100M if you get this wrong
                                   attn="cross_attention",
                                   semantic_type="disruption probability and time-to-disruption")

    disruption_class: Field = Field(2, 4, period=1,  # 8 pos: disruption type
                                    loss_weight=5.0,
                                    semantic_type="disruption classification VDE/TQ/CQ/locked-mode")

    # Mitigation actuator
    mitigation: Field = Field(1, 4, period=1,    # 4 pos: SPI/MGI trigger decision
                              loss_weight=8.0,
                              semantic_type="disruption mitigation system trigger command")


reactor = Tokamak()
bound = compile_schema(
    reactor, T=32, H=64, W=64, d_model=512,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        temporal="causal",          # real-time control, no future information
    ),
)

print("=== Tokamak Fusion Reactor Control ===")
print(f"Total fields: {len(bound.field_names)}")
print(f"Total connections: {len(bound.topology.connections)}")
print(f"Canvas: T={bound.layout.T}, {bound.layout.H}x{bound.layout.W}, "
      f"d={bound.layout.d_model}")
print(f"Total positions: {bound.layout.num_positions:,}")
print()

# Multi-timescale diagnostics
print("Diagnostic timescales:")
for period, label in [(1, "1ms"), (5, "5ms"), (10, "10ms"), (20, "20ms")]:
    fields_at_p = [(n, bf) for n, bf in bound.fields.items()
                   if bf.spec.period == period]
    if fields_at_p:
        names = [n.split(".")[-1] for n, _ in fields_at_p]
        print(f"  {label} (period={period}): {', '.join(names[:4])}"
              + (f" +{len(names)-4} more" if len(names) > 4 else ""))

# Safety-critical loss budget
weights = bound.layout.loss_weight_mask("cpu")
total_w = weights.sum().item()
print(f"\nSafety-critical loss budget:")
for name in ["disruption_risk", "disruption_class", "mitigation"]:
    bf = bound[name]
    w = sum(weights[i].item() for i in bf.indices())
    print(f"  {name}: {w/total_w*100:.1f}% (loss_weight={bf.spec.loss_weight})")

safety_total = sum(
    sum(weights[i].item() for i in bound[n].indices())
    for n in ["disruption_risk", "disruption_class", "mitigation"]
)
print(f"  TOTAL safety: {safety_total/total_w*100:.1f}% of all gradient signal")

# Attention types in use
attn_types = set()
for name, bf in bound.fields.items():
    attn_types.add(bf.spec.default_attn)
print(f"\nAttention function types in use: {sorted(attn_types)}")

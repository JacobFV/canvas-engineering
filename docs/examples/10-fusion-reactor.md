# Example 10: Tokamak Plasma Control

Multi-timescale control with safety constraints. The disruption predictor (`loss_weight=10`) catches disruptions earlier than a flat model, and multi-rate diagnostics create a natural sensor fusion hierarchy.

**Source**: [`examples/10_fusion_reactor.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/10_fusion_reactor.py) *(coming soon)*

## What it demonstrates

- **Extreme multi-rate sensing** — magnetic probes at period=1, Thomson scattering at period=10, neutron diagnostics at period=50
- **Safety-weighted loss** — disruption prediction at 10× forces the model to prioritize plasma stability
- **Actuator fields** — coil currents and gas injection are first-class fields, not external targets
- **Causal control loop** — equilibrium → stability → actuator: the model must route through the plasma physics

## Type hierarchy

```python
@dataclass
class MagneticDiagnostics:
    flux_surface: Field = Field(4, 8, period=1, attn="mamba")  # 2D equilibrium
    rogowski_coils: Field = Field(1, 8, period=1)              # plasma current
    mirnov_coils: Field = Field(2, 8, period=1)                # MHD modes

@dataclass
class ThermalDiagnostics:
    electron_temp: Field = Field(4, 4, period=10)              # Thomson scattering
    electron_density: Field = Field(4, 4, period=10)
    ion_temp: Field = Field(2, 4, period=20)                   # charge exchange

@dataclass
class NeutronDiagnostics:
    neutron_rate: Field = Field(1, 2, period=50)               # fusion rate
    gamma_spectrum: Field = Field(2, 4, period=50)

@dataclass
class Actuators:
    coil_currents: Field = Field(2, 8, loss_weight=1.0)        # shape control
    gas_injection: Field = Field(1, 4, loss_weight=1.0)        # density control
    heating_power: Field = Field(1, 2, loss_weight=1.0)        # NBI + ECRH

@dataclass
class TokamakController:
    magnetic: MagneticDiagnostics = field(default_factory=MagneticDiagnostics)
    thermal: ThermalDiagnostics = field(default_factory=ThermalDiagnostics)
    neutron: NeutronDiagnostics = field(default_factory=NeutronDiagnostics)
    actuators: Actuators = field(default_factory=Actuators)
    disruption_risk: Field = Field(1, 1, loss_weight=10.0)    # safety-critical
    confinement_quality: Field = Field(1, 1, loss_weight=2.0)  # H-mode index
```

## Multi-rate sensor fusion

Period=1 magnetic diagnostics update every step. Period=50 neutron diagnostics update 50× slower. The canvas handles this automatically — slow-period regions attend to fast-period regions with temporal constraints, fusing across timescales without manual interpolation.

!!! note "Task spec"
    Full implementation details in [`examples/tasks/10_fusion_reactor.md`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/tasks/10_fusion_reactor.md).

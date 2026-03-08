# Example 07: Hospital ICU Ward

The deepest type hierarchy in the library. 6 patients with organ-level physiology, 4 nurses with workload/fatigue dynamics, bureaucratic state (insurance, staffing, bed pressure), and family units. Declares the causal structure of a hospital ward as a type hierarchy.

**Source**: [`examples/07_hospital_icu.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/07_hospital_icu.py)

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/07_icu_patient.png" alt="Example 07 results" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/07_icu_patient.gif" alt="Example 07 ICU ward monitor animation" width="100%">
</p>

<video width="100%" controls>
  <source src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/07_icu_patient.mp4" type="video/mp4">
</video>

**5×4 command center figure**: Per-patient vitals (HR, BP, SpO2, RR), organ system heatmaps, deterioration risk trajectories, nurse workload/fatigue, bureaucratic pressure indicators, and training curves — dark command center aesthetic.

**Animation**: Ward monitor dashboard with patient vital signs, nurse status panels, alert system, bed pressure and staffing gauges, shift handoff indicators.

## Type hierarchy

```python
@dataclass
class CardiovascularSystem:
    heart_rate: Field = Field(1, 2, period=1)
    blood_pressure: Field = Field(1, 4, period=2)
    cardiac_output: Field = Field(1, 2, period=5)

@dataclass
class RespiratorySystem:
    spo2: Field = Field(1, 2, period=1)
    respiratory_rate: Field = Field(1, 2, period=1)
    ventilator_settings: Field = Field(1, 4, period=2, is_output=False)

@dataclass
class RenalSystem:
    urine_output: Field = Field(1, 2, period=12)
    creatinine: Field = Field(1, 1, period=24)
    electrolytes: Field = Field(1, 4, period=24)

@dataclass
class NeurologicalSystem:
    consciousness: Field = Field(1, 4, period=6)
    sedation_level: Field = Field(1, 2, period=4)
    pain: Field = Field(1, 2, period=2)
    delirium_risk: Field = Field(1, 2, period=12, loss_weight=3.0)

@dataclass
class PsychologicalState:
    anxiety: Field = Field(1, 2, period=4)
    sleep_quality: Field = Field(1, 2, period=24)
    will_to_recover: Field = Field(1, 2, period=24, loss_weight=2.0)

@dataclass
class Patient:
    cardiovascular: CardiovascularSystem
    respiratory: RespiratorySystem
    renal: RenalSystem
    neurological: NeurologicalSystem
    psychological: PsychologicalState
    deterioration_risk: Field = Field(2, 4, loss_weight=8.0)
    organ_failure_risk: Field = Field(1, 6, loss_weight=5.0)

@dataclass
class Nurse:
    workload: Field = Field(1, 2)
    fatigue: Field = Field(1, 2, loss_weight=2.0)
    stress: Field = Field(1, 2, loss_weight=2.0)
    competence: Field = Field(1, 2, is_output=False)
    rapport: Field = Field(1, 2)

@dataclass
class BureaucraticState:
    insurance_auth: Field = Field(1, 2, is_output=False, period=24)
    bed_pressure: Field = Field(1, 2, period=12)
    staffing_ratio: Field = Field(1, 2, period=8)
    discharge_pressure: Field = Field(1, 2, loss_weight=2.0)

@dataclass
class FamilyUnit:
    presence: Field = Field(1, 2, is_output=False, period=24)
    emotional_state: Field = Field(1, 2)
    communication_quality: Field = Field(1, 2, loss_weight=1.5)

@dataclass
class ICUWard:
    global_acuity: Field = Field(2, 4, loss_weight=3.0)
    resource_state: Field = Field(1, 4, is_output=False)
    patients: list       # 6 patients
    nurses: list         # 4 nurses
    bureaucratic: BureaucraticState
    families: list       # 6 family units
```

## Connectivity

```python
bound = compile_schema(
    ward, T=1, H=24, W=24, d_model=32,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        array_element="ring",
        temporal="dense",
    ),
)
# 152 fields, 576 positions, 1,466 connections
```

## The deterioration pathway

Sepsis trajectory: `renal.creatinine_rise` → `cardiovascular.mean_arterial_pressure_drop` → `neurological.altered_consciousness` → `deterioration_risk`. A flat model learns the correlation. This model is forced to route through the causal pathway — making it interpretable and robust to distribution shift.

!!! warning "This is the flagship example"
    Every field has a physiological reason for existing. Every connection has a known biological pathway. The type system is not decoration — it is the domain knowledge.

## Run it

```bash
python examples/07_hospital_icu.py
# Generates: assets/examples/07_icu_patient.{png,gif,mp4}
```

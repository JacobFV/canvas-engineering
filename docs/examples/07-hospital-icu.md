# Example 07: ICU Patient — Whole-Person Physiological Model

The deepest type hierarchy in the library. Not "vitals as a blob" — every organ system, psychological state, social context. Declaring the causal structure of human physiology as a type hierarchy produces a model that predicts deterioration through mechanistic pathways, not just statistical correlation.

**Source**: [`examples/07_hospital_icu.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/07_hospital_icu.py) *(coming soon)*

## What it demonstrates

- **Organ-level decomposition** — cardiovascular, respiratory, renal, neurological as separate type subtrees
- **Multi-rate sensing** — ECG at period=1, blood pressure at period=2, labs at period=30
- **Psychological + social fields** — pain, anxiety, social support as first-class latent regions
- **Causal deterioration pathways** — the model must route information through physiological connections, not find statistical shortcuts

## Type hierarchy (partial)

```python
@dataclass
class CardiovascularSystem:
    heart_rhythm: Field = Field(4, 8, period=1, attn="mamba")  # ECG morphology
    heart_rate: Field = Field(1, 2, period=1)
    blood_pressure: Field = Field(1, 4, period=2)              # sys/dia/map/pp
    cardiac_output: Field = Field(1, 2, period=5)
    peripheral_resistance: Field = Field(1, 2, period=5)

@dataclass
class RespiratorySystem:
    spo2: Field = Field(1, 2, period=1)
    respiratory_rate: Field = Field(1, 1, period=1)
    tidal_volume: Field = Field(1, 2, period=2)
    airway_resistance: Field = Field(1, 2, period=10)

@dataclass
class PsychologicalState:
    pain: Field = Field(1, 2, period=4)
    anxiety: Field = Field(1, 2, period=4)
    consciousness: Field = Field(1, 4, period=2)               # GCS components
    delirium_risk: Field = Field(1, 1, period=8, loss_weight=3.0)

@dataclass
class SocialContext:
    family_presence: Field = Field(1, 1, period=60, is_output=False)
    care_team_load: Field = Field(1, 2, period=30, is_output=False)

@dataclass
class ICUPatient:
    cardiovascular: CardiovascularSystem = field(default_factory=CardiovascularSystem)
    respiratory: RespiratorySystem = field(default_factory=RespiratorySystem)
    renal: RenalSystem = field(default_factory=RenalSystem)
    neurological: NeurologicalSystem = field(default_factory=NeurologicalSystem)
    psychological: PsychologicalState = field(default_factory=PsychologicalState)
    social: SocialContext = field(default_factory=SocialContext)
    deterioration_risk: Field = Field(1, 1, loss_weight=5.0)   # NEWS2-equivalent
```

## The deterioration pathway

Sepsis trajectory: `renal.creatinine_rise` → `cardiovascular.mean_arterial_pressure_drop` → `neurological.altered_consciousness` → `deterioration_risk`. A flat model learns the correlation. This model is forced to route through the causal pathway — making it interpretable and robust to distribution shift.

!!! warning "This is the flagship example"
    Every field has a physiological reason for existing. Every connection has a known biological pathway. The type system is not decoration — it is the domain knowledge.

!!! note "Task spec"
    Full implementation details in [`examples/tasks/07_hospital_icu.md`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/tasks/07_hospital_icu.md).

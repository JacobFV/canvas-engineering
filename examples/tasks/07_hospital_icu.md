# 07: ICU Patient — Whole-Person Physiological Model

## Purpose
This is the one where we go DEEP on the type hierarchy. Not "vitals" as
a blob — every organ system, psychological state, social context. Show
that declaring the causal structure of human physiology as a type hierarchy
creates a model that predicts deterioration through mechanistic pathways,
not just statistical correlation.

## Type hierarchy (THE REAL ONE)
```
CardiovascularSystem:
  heart_rhythm: Field(4, 8, period=1, attn="mamba")  — ECG morphology
  heart_rate: Field(1, 2, period=1)
  blood_pressure: Field(1, 4, period=2)    — sys/dia/map/pulse_pressure
  cardiac_output: Field(1, 2, period=5)    — CO + stroke volume estimate
  peripheral_resistance: Field(1, 2, period=5)

RespiratorySystem:
  spo2: Field(1, 2, period=1)              — SpO2 + pleth quality
  respiratory_rate: Field(1, 2, period=1)
  etco2: Field(1, 2, period=1)            — end-tidal CO2
  lung_compliance: Field(1, 2, period=10)  — if ventilated
  work_of_breathing: Field(1, 2, period=2) — estimated from waveforms

RenalSystem:
  urine_output: Field(1, 2, period=60)     — hourly
  creatinine: Field(1, 1, period=360)      — q6h labs
  bun: Field(1, 1, period=360)
  electrolytes: Field(1, 4, period=360)    — Na/K/Cl/HCO3

NeurologicalSystem:
  consciousness: Field(1, 4, period=30)    — GCS components
  pupil_reactivity: Field(1, 2, period=30)
  sedation_level: Field(1, 2, period=15)   — RASS score
  pain: Field(1, 2, period=5)              — CPOT/BPS score
  delirium_risk: Field(1, 2, period=60, loss_weight=3.0)

HepaticSystem:
  liver_enzymes: Field(1, 4, period=360)   — AST/ALT/ALP/bilirubin
  coagulation: Field(1, 3, period=360)     — INR/PT/PTT
  albumin: Field(1, 1, period=720)

MetabolicState:
  glucose: Field(1, 2, period=60)          — glucose + insulin rate
  lactate: Field(1, 1, period=120)         — tissue perfusion marker
  temperature: Field(1, 2, period=5)
  nutrition: Field(1, 4, period=480)       — caloric intake, protein, etc.

InfectionState:
  wbc: Field(1, 2, period=360)
  procalcitonin: Field(1, 1, period=720)
  culture_results: Field(1, 4, is_output=False, period=1440) — micro data
  antibiotic_coverage: Field(1, 4, is_output=False)          — current abx

PsychologicalState:
  anxiety: Field(1, 2, period=60)          — estimated from HR variability + behavior
  sleep_quality: Field(1, 2, period=480)   — from overnight monitoring
  agitation: Field(1, 2, period=15)        — behavioral observation
  emotional_regulation: Field(1, 2, period=60)
  will_to_recover: Field(1, 2, period=480, loss_weight=2.0)  — engagement metric

SocialContext:
  family_presence: Field(1, 2, is_output=False, period=480)  — visiting hours
  family_stress: Field(1, 2, period=480)   — from interaction assessment
  spiritual_needs: Field(1, 1, is_output=False, period=1440)

Patient:
  cardiovascular: CardiovascularSystem
  respiratory: RespiratorySystem
  renal: RenalSystem
  neurological: NeurologicalSystem
  hepatic: HepaticSystem
  metabolic: MetabolicState
  infection: InfectionState
  psychological: PsychologicalState
  social: SocialContext

  # The predictions
  deterioration_risk: Field(2, 4, loss_weight=8.0)
  organ_failure_risk: Field(1, 6, loss_weight=5.0)  — per-organ failure prob
  recommended_intervention: Field(2, 8, loss_weight=3.0)
  estimated_los: Field(1, 2, loss_weight=1.0)        — length of stay

  # Context
  demographics: Field(1, 4, is_output=False)
  admission_diagnosis: Field(2, 8, is_output=False)
  medications: Field(2, 8, is_output=False)
  surgical_history: Field(1, 4, is_output=False)
```

## Data
Synthetic but physiologically grounded:
- Simulate 48-hour ICU stays with correlated organ systems
- Cardiovascular drives perfusion -> renal function -> metabolic state
- Infection causes WBC rise -> fever -> tachycardia -> hypotension cascade
- Sedation affects consciousness, respiratory drive, delirium risk
- Psychological state modulates pain perception, heart rate variability
- Family presence modulates anxiety (actually documented in literature)
- Generate deterioration events (sepsis cascade, cardiac arrest, resp failure)
  with realistic multi-system progression

## Training
Multi-task with **causal consistency losses**:
- If lactate rises, cardiac output should drop (or vice versa) — enforce
  physiological consistency between organ systems
- If sedation increases, consciousness should decrease — bidirectional
- Deterioration risk should be an emergent property of organ states,
  not an independent prediction — penalize deterioration risk that
  isn't explained by organ failure risks
- Temporal smoothness on slow fields (albumin shouldn't jump in 1 hour)

## Visualization
`assets/examples/07_icu_patient.png` — large, 3x3:
(a) The type hierarchy as a tree diagram with organ systems colored
(b) Canvas layout: the full grid with all regions, colored by organ system
(c) A simulated sepsis cascade: time series of organ states deteriorating
    in sequence (infection -> cardiovascular -> renal -> metabolic)
(d) Cross-system attention: does the model learn that cardiovascular
    drives renal? (attention from renal.creatinine -> cardiovascular.cardiac_output)
(e) Deterioration risk over time vs true event (how early does it detect?)
(f) Per-organ failure risk: stacked area chart over time
(g) Loss weight budget: pie chart of where gradient signal goes
(h) Psychological state trajectory: anxiety, pain, sleep quality
(i) Multi-frequency timeline: showing how different fields update at
    different rates (ECG every second, labs every 6 hours)

## Key message
"A human isn't 'vitals.' They're 9 interacting organ systems with
psychological state modulated by social context. Declaring this structure
as a type hierarchy means the model learns the causal pathways of
deterioration — not just 'HR went up, patient is sick' but 'infection
caused vasodilation caused hypotension caused renal hypoperfusion
caused creatinine rise.' The loss weights declare clinical priorities.
The periods declare measurement frequencies. The connectivity declares
physiological coupling. This is canvas engineering."

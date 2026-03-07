"""Hospital ICU: real-time patient monitoring and clinical decision support.

Model an entire ICU ward as a canvas type. Each patient has continuous
physiological streams at different frequencies, clinical notes (input-only),
and predicted deterioration risk. Nurses have workload state. The ward
has resource allocation state.

This is the kind of system that could replace early warning scores (NEWS2,
MEWS) with a learned latent model that sees everything simultaneously.

Key design choices:
- Multi-frequency fields: ECG at period=1 (every frame = every second),
  lab results at period=60 (every minute-equivalent), vitals at period=5.
- Clinical notes are input-only (is_output=False) — context, not prediction.
- Deterioration risk gets 8x loss weight. This IS the clinical decision.
- Nurse workload is aggregated from patient states (parent_child="aggregate").
- Causal temporal: you cannot use future vital signs to predict current risk.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy


@dataclass
class Patient:
    """One ICU patient with continuous monitoring."""
    # High-frequency physiological streams
    ecg: Field = Field(4, 8, period=1,         # 32 positions: ECG waveform features
                       attn="mamba",            # temporal sequence modeling
                       semantic_type="12-lead ECG waveform features 250Hz")

    pulse_ox: Field = Field(1, 4, period=1,    # 4 positions: SpO2 + pleth waveform
                            semantic_type="pulse oximetry SpO2 + plethysmograph")

    # Medium-frequency vitals
    vitals: Field = Field(2, 4, period=5,      # 8 positions: HR, BP, RR, Temp
                          semantic_type="vital signs HR/BP/RR/Temp 12s intervals")

    # Low-frequency lab results
    labs: Field = Field(2, 8, period=60,        # 16 positions: CBC, BMP, coag, lactate
                        is_output=False,         # labs are measured, not predicted
                        semantic_type="laboratory results CBC/BMP/coag/lactate")

    # Clinical context (input-only)
    notes: Field = Field(4, 8,                  # 32 positions: clinical note embeddings
                         is_output=False,
                         semantic_type="clinical notes assessment and plan")

    diagnosis: Field = Field(2, 4,              # 8 positions: ICD codes + problem list
                             is_output=False,
                             semantic_type="active diagnoses ICD-10 problem list")

    medications: Field = Field(2, 4,            # 8 positions: active medication list
                               is_output=False,
                               semantic_type="active medications with dosing")

    # Predictions (the actual outputs)
    deterioration: Field = Field(2, 4,          # 8 positions: risk score + trajectory
                                 loss_weight=8.0,  # THE clinical decision
                                 semantic_type="patient deterioration risk 4h horizon")

    intervention: Field = Field(2, 8,           # 16 positions: suggested interventions
                                loss_weight=3.0,
                                semantic_type="recommended clinical interventions")

    ventilator: Field = Field(1, 8,             # 8 positions: vent settings prediction
                              loss_weight=4.0,
                              attn="linear_attention",
                              semantic_type="ventilator settings FiO2/PEEP/TV/RR")


@dataclass
class Nurse:
    """ICU nurse with workload tracking."""
    workload: Field = Field(2, 4,              # 8 positions: cognitive load estimate
                            semantic_type="nurse cognitive workload estimate")
    assignment: Field = Field(1, 4,            # 4 positions: patient assignment
                              is_output=False,
                              semantic_type="nurse patient assignment")


@dataclass
class ICUWard:
    """An entire ICU ward with patients, nurses, and resources."""
    # Ward-level state
    census: Field = Field(2, 4,                # 8 positions: bed occupancy, acuity mix
                          is_output=False,
                          semantic_type="ICU census and acuity distribution")

    resources: Field = Field(2, 4,             # 8 positions: available resources
                             semantic_type="ICU resource availability staff/beds/equipment")

    # Patients and staff
    patients: list = dc_field(default_factory=list)
    nurses: list = dc_field(default_factory=list)


# --- 8-bed ICU pod with 3 nurses ---

icu = ICUWard(
    patients=[Patient() for _ in range(8)],
    nurses=[Nurse() for _ in range(3)],
)

bound = compile_schema(
    icu, T=64, H=64, W=64, d_model=384,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",       # ward state <-> all patients and nurses
        array_element="isolated",       # patients don't directly see each other
        temporal="causal",              # strictly causal
    ),
)

print("=== ICU Ward Clinical Decision Support ===")
print(f"Patients: 8, Nurses: 3")
print(f"Total fields: {len(bound.field_names)}")
print(f"Total connections: {len(bound.topology.connections)}")
print(f"Canvas: T={bound.layout.T}, H={bound.layout.H}, W={bound.layout.W}")
print(f"Total positions: {bound.layout.num_positions:,}")

# Show multi-frequency fields
print("\nMulti-frequency physiological streams (patient 0):")
for name in ["ecg", "pulse_ox", "vitals", "labs"]:
    bf = bound[f"patients[0].{name}"]
    print(f"  {name}: period={bf.spec.period}, "
          f"is_output={bf.spec.is_output}, "
          f"attn={bf.spec.default_attn}")

# Loss budget — where the gradient signal goes
weights = bound.layout.loss_weight_mask("cpu")
total_w = weights.sum().item()
categories = {
    "deterioration risk": 0, "interventions": 0,
    "ventilator settings": 0, "physiological": 0,
    "ward operations": 0,
}
for name, bf in bound.fields.items():
    indices = bf.indices()
    w = sum(weights[i].item() for i in indices)
    if "deterioration" in name:
        categories["deterioration risk"] += w
    elif "intervention" in name:
        categories["interventions"] += w
    elif "ventilator" in name:
        categories["ventilator settings"] += w
    elif any(x in name for x in ["ecg", "pulse_ox", "vitals"]):
        categories["physiological"] += w
    else:
        categories["ward operations"] += w

print(f"\nLoss budget:")
for cat, w in sorted(categories.items(), key=lambda x: -x[1]):
    if w > 0:
        print(f"  {cat}: {w/total_w*100:.1f}%")

# Input-only fields (context, not predicted)
input_only = [n for n, bf in bound.fields.items() if not bf.spec.is_output]
print(f"\nInput-only context fields: {len(input_only)}")
for n in input_only[:5]:
    print(f"  {n}")
if len(input_only) > 5:
    print(f"  ... and {len(input_only)-5} more")

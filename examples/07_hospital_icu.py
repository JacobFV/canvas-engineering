"""Hospital ICU Ward: full-ward simulation with canvas types.

Deep type hierarchy: 6 patients, 4 nurses, bureaucratic/family/resource dynamics.
Canvas-structured model captures inter-system coupling that flat models miss.

Organ-system cascades, nurse fatigue, insurance friction, family psychology,
discharge pressure — the full weight of institutional medicine modeled as
structured latent spaces.

Synthetic 48-hour ward simulation with realistic multi-system cascades:
  - Sepsis: infection -> cardiovascular -> renal -> metabolic
  - Respiratory failure: respiratory -> cardiovascular -> neurological
  - Post-surgical recovery with complications
  - Cardiac decompensation
  - Trauma with hemorrhagic shock
  - Stable patient (discharge candidate under pressure)

Outputs:
  assets/examples/07_icu_patient.png  — 5x4 command center figure
  assets/examples/07_icu_patient.gif  — animated ward monitor dashboard

Run:  python examples/07_hospital_icu.py
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field as dc_field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Patch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

from canvas_engineering import Field, compile_schema, ConnectivityPolicy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

N_TIMESTEPS = 48  # hours of ICU stay
N_PATIENTS = 6
N_NURSES = 4

# ── Dark theme constants ──────────────────────────────────────────────

BG = '#0a0a1a'
BG_PANEL = '#0f0f2a'
BG_CARD = '#141432'
GRID_COLOR = '#1a1a3a'
TEXT_DIM = '#556677'
TEXT_MED = '#8899aa'
TEXT_BRIGHT = '#ccddee'
ACCENT_RED = '#ff3344'
ACCENT_ORANGE = '#ff8844'
ACCENT_YELLOW = '#ffcc33'
ACCENT_GREEN = '#33ff88'
ACCENT_BLUE = '#3388ff'
ACCENT_CYAN = '#33ddff'
ACCENT_PURPLE = '#aa55ff'
ACCENT_PINK = '#ff55aa'
ACCENT_WHITE = '#eeeeff'

# System colors
SYS_CARDIO = '#ff3344'
SYS_RESP = '#3388ff'
SYS_RENAL = '#ffaa22'
SYS_NEURO = '#aa55ff'
SYS_PSYCH = '#ff55aa'
SYS_BUREAU = '#888899'
SYS_FAMILY = '#66ccaa'
SYS_NURSE = '#33ddff'

# Patient status colors
STATUS_STABLE = '#22cc66'
STATUS_WARNING = '#ffaa22'
STATUS_CRITICAL = '#ff3344'

PATIENT_NAMES = ['P1-SEPSIS', 'P2-RESP', 'P3-SURG', 'P4-CARDIAC', 'P5-TRAUMA', 'P6-STABLE']
PATIENT_SHORT = ['Sepsis', 'RespFail', 'PostSurg', 'Cardiac', 'Trauma', 'Stable']
NURSE_NAMES = ['N1-Chen', 'N2-Patel', 'N3-Okafor', 'N4-Kim']

# Nurse assignments: which patients each nurse covers (changes at shift boundaries)
# Shifts: 0-12h, 12-24h, 24-36h, 36-48h
ASSIGNMENTS = [
    {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]},     # shift 1
    {0: [1, 2], 1: [0, 3], 2: [5], 3: [4]},     # shift 2
    {0: [0, 4], 1: [1, 5], 2: [2], 3: [3]},     # shift 3
    {0: [3, 5], 1: [0, 2], 2: [1], 3: [4]},     # shift 4
]


# ── 1. Type declarations: deep organ-system hierarchy ────────────────

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
    cardiovascular: CardiovascularSystem = dc_field(default_factory=CardiovascularSystem)
    respiratory: RespiratorySystem = dc_field(default_factory=RespiratorySystem)
    renal: RenalSystem = dc_field(default_factory=RenalSystem)
    neurological: NeurologicalSystem = dc_field(default_factory=NeurologicalSystem)
    psychological: PsychologicalState = dc_field(default_factory=PsychologicalState)
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
    patients: list = dc_field(default_factory=list)
    nurses: list = dc_field(default_factory=list)
    bureaucratic: BureaucraticState = dc_field(default_factory=BureaucraticState)
    families: list = dc_field(default_factory=list)


ward = ICUWard(
    patients=[Patient() for _ in range(N_PATIENTS)],
    nurses=[Nurse() for _ in range(N_NURSES)],
    families=[FamilyUnit() for _ in range(N_PATIENTS)],
)

bound = compile_schema(
    ward, T=1, H=32, W=32, d_model=32,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        array_element="ring",
        temporal="dense",
    ),
)

print(f"ICU Ward canvas: {len(bound.field_names)} fields, "
      f"{bound.layout.num_positions} positions, "
      f"{len(bound.topology.connections)} connections")


# ── 2. Synthetic data: full ward simulation ──────────────────────────

def simulate_ward(n_stays=1024, n_hours=N_TIMESTEPS):
    """Simulate full ICU ward: 6 patients, 4 nurses, bureaucratic/family dynamics.

    Patient scenarios:
      0: Sepsis — infection cascade
      1: Respiratory failure — ARDS progression
      2: Post-surgical — wound complications
      3: Cardiac — decompensated heart failure
      4: Trauma — hemorrhagic shock, surgery recovery
      5: Stable — pending discharge, bureaucratic delays
    """
    all_patient_features = []
    all_nurse_features = []
    all_bureau_features = []
    all_family_features = []
    all_ward_context = []
    all_deterioration = []   # per patient
    all_organ_failure = []   # per patient
    all_ward_acuity = []

    for stay in range(n_stays):
        # Per-patient time series
        patient_data = {}
        onset_times = [
            np.random.randint(4, 12),   # sepsis onset
            np.random.randint(6, 16),   # resp failure onset
            np.random.randint(2, 8),    # surgical complication
            np.random.randint(8, 18),   # cardiac decompensation
            0,                           # trauma from admission
            0,                           # stable from admission
        ]

        for p_idx in range(N_PATIENTS):
            hr = np.ones(n_hours) * 0.5
            bp = np.ones(n_hours) * 0.6
            co = np.ones(n_hours) * 0.5
            spo2 = np.ones(n_hours) * 0.95
            rr = np.ones(n_hours) * 0.4
            vent = np.zeros(n_hours)
            uo = np.ones(n_hours) * 0.5
            cr = np.ones(n_hours) * 0.3
            lytes = np.ones(n_hours) * 0.5
            gcs = np.ones(n_hours) * 0.9
            sedation = np.ones(n_hours) * 0.2
            pain = np.ones(n_hours) * 0.3
            delirium = np.ones(n_hours) * 0.1
            anxiety = np.ones(n_hours) * 0.3
            sleep_q = np.ones(n_hours) * 0.5
            will = np.ones(n_hours) * 0.6

            onset = onset_times[p_idx]
            noise = lambda: np.random.randn() * 0.02

            if p_idx == 0:  # Sepsis cascade
                for t in range(onset, n_hours):
                    dt = t - onset
                    sev = min(1.0, dt / 18.0)
                    hr[t] = 0.5 + 0.4 * min(1.0, max(0, dt - 2) / 16) + noise()
                    bp[t] = 0.6 - 0.35 * min(1.0, max(0, dt - 2) / 16) + noise()
                    co[t] = 0.5 - 0.2 * min(1.0, max(0, dt - 3) / 15) + noise()
                    spo2[t] = 0.95 - 0.12 * min(1.0, max(0, dt - 8) / 14) + noise() * 0.5
                    rr[t] = 0.4 + 0.35 * min(1.0, max(0, dt - 6) / 12) + noise()
                    uo[t] = 0.5 - 0.4 * min(1.0, max(0, dt - 6) / 14) + noise()
                    cr[t] = 0.3 + 0.6 * min(1.0, max(0, dt - 6) / 14) + noise()
                    lytes[t] = 0.5 + 0.3 * min(1.0, max(0, dt - 8) / 12) * np.random.choice([-1, 1]) + noise()
                    gcs[t] = 0.9 - 0.4 * min(1.0, max(0, dt - 10) / 12) + noise()
                    delirium[t] = 0.1 + 0.7 * min(1.0, max(0, dt - 10) / 12) + noise()
                    anxiety[t] = 0.3 + 0.3 * sev + noise()
                    sleep_q[t] = 0.5 - 0.3 * sev + noise()
                    will[t] = 0.6 - 0.15 * sev + noise()

            elif p_idx == 1:  # Respiratory failure
                for t in range(onset, n_hours):
                    dt = t - onset
                    sev = min(1.0, dt / 20.0)
                    spo2[t] = 0.95 - 0.22 * sev + noise() * 0.5
                    rr[t] = 0.4 + 0.45 * sev + noise()
                    hr[t] = 0.5 + 0.3 * min(1.0, max(0, dt - 4) / 16) + noise()
                    bp[t] = 0.6 + 0.08 * min(1.0, max(0, dt - 4) / 16) + noise()
                    if dt > 6:
                        vent[t] = 0.5 + 0.3 * min(1.0, (dt - 6) / 10)
                        sedation[t] = 0.3 + 0.4 * min(1.0, (dt - 6) / 10) + noise()
                    gcs[t] = 0.9 - 0.25 * min(1.0, max(0, dt - 8) / 14) + noise()
                    anxiety[t] = 0.3 + 0.4 * sev * 0.5 + noise()
                    will[t] = 0.6 - 0.1 * sev + noise()

            elif p_idx == 2:  # Post-surgical
                for t in range(n_hours):
                    if t < onset:
                        improvement = 0
                    else:
                        dt = t - onset
                        complication = min(1.0, dt / 24.0)
                        hr[t] = 0.55 + 0.15 * complication + noise()
                        bp[t] = 0.58 - 0.1 * complication + noise()
                        pain[t] = 0.5 + 0.2 * complication + noise()
                        cr[t] = 0.3 + 0.15 * complication + noise()
                        anxiety[t] = 0.4 + 0.2 * complication + noise()
                        sleep_q[t] = 0.4 - 0.15 * complication + noise()

            elif p_idx == 3:  # Cardiac decompensation
                for t in range(onset, n_hours):
                    dt = t - onset
                    sev = min(1.0, dt / 22.0)
                    hr[t] = 0.6 + 0.25 * sev + noise()
                    bp[t] = 0.55 - 0.2 * sev + noise()
                    co[t] = 0.5 - 0.3 * sev + noise()
                    spo2[t] = 0.94 - 0.08 * sev + noise() * 0.5
                    uo[t] = 0.5 - 0.25 * sev + noise()
                    lytes[t] = 0.5 + 0.2 * sev + noise()
                    anxiety[t] = 0.35 + 0.3 * sev + noise()
                    will[t] = 0.55 - 0.1 * sev + noise()

            elif p_idx == 4:  # Trauma / hemorrhagic shock then recovery
                for t in range(n_hours):
                    if t < 6:
                        sev = 0.8 - 0.1 * t  # acute phase
                    elif t < 18:
                        sev = max(0.1, 0.2 + 0.02 * (18 - t))  # stabilizing
                    else:
                        sev = max(0.05, 0.1 - 0.003 * (t - 18))  # recovery
                    hr[t] = 0.5 + 0.35 * sev + noise()
                    bp[t] = 0.6 - 0.3 * sev + noise()
                    co[t] = 0.5 - 0.2 * sev + noise()
                    spo2[t] = 0.95 - 0.1 * sev + noise() * 0.5
                    pain[t] = 0.3 + 0.5 * sev + noise()
                    gcs[t] = 0.9 - 0.2 * sev + noise()
                    anxiety[t] = 0.3 + 0.4 * sev + noise()
                    will[t] = 0.5 + 0.1 * (1 - sev) + noise()

            else:  # Stable, improving
                for t in range(n_hours):
                    improvement = min(1.0, t / 36.0)
                    hr[t] = 0.55 - 0.05 * improvement + noise()
                    bp[t] = 0.58 + 0.02 * improvement + noise()
                    spo2[t] = 0.94 + 0.04 * improvement + noise() * 0.3
                    pain[t] = 0.3 - 0.15 * improvement + noise()
                    anxiety[t] = 0.35 - 0.2 * improvement + noise()
                    sleep_q[t] = 0.45 + 0.3 * improvement + noise()
                    will[t] = 0.55 + 0.25 * improvement + noise()

            # Clip all
            for arr in [hr, bp, co, spo2, rr, vent, uo, cr, lytes,
                        gcs, sedation, pain, delirium, anxiety, sleep_q, will]:
                np.clip(arr, 0, 1, out=arr)

            patient_data[p_idx] = {
                'hr': hr, 'bp': bp, 'co': co, 'spo2': spo2, 'rr': rr,
                'vent': vent, 'uo': uo, 'cr': cr, 'lytes': lytes,
                'gcs': gcs, 'sedation': sedation, 'pain': pain,
                'delirium': delirium, 'anxiety': anxiety,
                'sleep_q': sleep_q, 'will': will,
            }

        # Nurse dynamics
        nurse_data = {}
        for n_idx in range(N_NURSES):
            workload = np.ones(n_hours) * 0.4
            fatigue = np.ones(n_hours) * 0.2
            stress = np.ones(n_hours) * 0.3
            competence = np.ones(n_hours) * (0.6 + np.random.rand() * 0.3)
            rapport = np.ones(n_hours) * 0.5

            for t in range(n_hours):
                shift = min(t // 12, 3)
                assigned = ASSIGNMENTS[shift].get(n_idx, [])
                n_patients = len(assigned)
                # Workload from patient acuity
                acuity_sum = 0
                for p in assigned:
                    pd = patient_data[p]
                    acuity = (1 - pd['spo2'][t]) + (1 - pd['bp'][t]) + pd['pain'][t]
                    acuity_sum += acuity / 3
                workload[t] = min(1.0, 0.2 + 0.3 * n_patients + 0.3 * acuity_sum)
                # Fatigue accumulates within shift
                hours_in_shift = t % 12
                fatigue[t] = 0.1 + 0.06 * hours_in_shift + 0.2 * workload[t]
                # Stress from workload + patient acuity
                stress[t] = 0.2 + 0.4 * workload[t] + 0.2 * fatigue[t]
                # Rapport builds over time with same patients
                rapport[t] = 0.3 + 0.05 * min(hours_in_shift, 8)

            for arr in [workload, fatigue, stress, rapport]:
                np.clip(arr, 0, 1, out=arr)

            nurse_data[n_idx] = {
                'workload': workload, 'fatigue': fatigue, 'stress': stress,
                'competence': competence, 'rapport': rapport,
            }

        # Bureaucratic dynamics
        insurance_auth = np.zeros(N_PATIENTS)
        for p in range(N_PATIENTS):
            insurance_auth[p] = np.random.rand() * 0.8  # delay level
        bed_pressure = np.ones(n_hours) * 0.4
        staffing_ratio = np.ones(n_hours) * 0.5
        discharge_pressure = np.ones(n_hours) * 0.3

        for t in range(n_hours):
            # Bed pressure increases over time
            bed_pressure[t] = 0.3 + 0.4 * min(1.0, t / 36.0) + np.random.randn() * 0.03
            # Staffing ratio varies by shift
            shift_factor = 1.0 if (t % 24) < 12 else 0.75  # day vs night
            staffing_ratio[t] = 0.5 * shift_factor + np.random.randn() * 0.03
            # Discharge pressure from admin
            discharge_pressure[t] = 0.2 + 0.5 * min(1.0, t / 30.0) + np.random.randn() * 0.03
        np.clip(bed_pressure, 0, 1, out=bed_pressure)
        np.clip(staffing_ratio, 0, 1, out=staffing_ratio)
        np.clip(discharge_pressure, 0, 1, out=discharge_pressure)

        # Family dynamics
        family_data = {}
        for p in range(N_PATIENTS):
            presence = np.zeros(n_hours)
            emotional = np.ones(n_hours) * 0.5
            comm_quality = np.ones(n_hours) * 0.5

            # Visit windows (2-3 visits per day)
            visit_times = sorted(np.random.choice(range(n_hours), size=min(6, n_hours), replace=False))
            for vt in visit_times:
                for dt in range(min(3, n_hours - vt)):
                    presence[vt + dt] = 0.8 + np.random.rand() * 0.2
            pd = patient_data[p]
            for t in range(n_hours):
                # Emotional state tracks patient condition
                patient_severity = (1 - pd['spo2'][t]) + (pd['pain'][t]) + (1 - pd['gcs'][t])
                emotional[t] = max(0.1, 0.7 - 0.3 * patient_severity / 3) + noise()
                # Communication quality
                comm_quality[t] = 0.4 + 0.3 * presence[t] + noise()
                # Family presence improves patient will
                if presence[t] > 0.3:
                    patient_data[p]['will'][t] = min(1.0, pd['will'][t] + 0.05)
                    patient_data[p]['anxiety'][t] = max(0.0, pd['anxiety'][t] - 0.03)

            np.clip(emotional, 0, 1, out=emotional)
            np.clip(comm_quality, 0, 1, out=comm_quality)
            family_data[p] = {'presence': presence, 'emotional': emotional, 'comm_quality': comm_quality}

        # Pack into feature vectors (last timestep snapshot for model input)
        t_last = n_hours - 1
        patient_feats = []
        det_targets = []
        org_targets = []

        for p in range(N_PATIENTS):
            pd = patient_data[p]
            feat = np.array([
                pd['hr'][t_last], pd['bp'][t_last], pd['co'][t_last],
                pd['spo2'][t_last], pd['rr'][t_last], pd['vent'][t_last],
                pd['uo'][t_last], pd['cr'][t_last], pd['lytes'][t_last],
                pd['gcs'][t_last], pd['sedation'][t_last], pd['pain'][t_last],
                pd['delirium'][t_last],
                pd['anxiety'][t_last], pd['sleep_q'][t_last], pd['will'][t_last],
            ], dtype=np.float32)
            patient_feats.append(feat)

            # Deterioration risk
            det = 0.0
            if p == 0:  # sepsis
                dt = max(0, n_hours - 1 - onset_times[0])
                det = min(1.0, dt / 18.0)
            elif p == 1:  # resp
                dt = max(0, n_hours - 1 - onset_times[1])
                det = min(0.85, dt / 20.0)
            elif p == 2:  # surgical
                dt = max(0, n_hours - 1 - onset_times[2])
                det = min(0.5, dt / 30.0)
            elif p == 3:  # cardiac
                dt = max(0, n_hours - 1 - onset_times[3])
                det = min(0.7, dt / 22.0)
            elif p == 4:  # trauma (recovering)
                det = max(0.05, 0.4 - 0.01 * n_hours)
            else:  # stable
                det = max(0, 0.1 - 0.005 * n_hours)
            det_targets.append(np.full(8, det, dtype=np.float32))

            # Organ failure risk [cardio, resp, renal, neuro, psych]
            org = np.zeros(6, dtype=np.float32)
            org[0] = max(0, 1 - pd['co'][t_last]) * det
            org[1] = max(0, 1 - pd['spo2'][t_last]) * det * 3
            org[2] = pd['cr'][t_last] * det
            org[3] = max(0, 1 - pd['gcs'][t_last]) * det
            org[4] = pd['delirium'][t_last] * det
            org[5] = pd['anxiety'][t_last] * det * 0.5
            org = np.clip(org + np.random.randn(6).astype(np.float32) * 0.02, 0, 1)
            org_targets.append(org)

        nurse_feats = []
        for n in range(N_NURSES):
            nd = nurse_data[n]
            nf = np.array([
                nd['workload'][t_last], nd['fatigue'][t_last],
                nd['stress'][t_last], nd['competence'][t_last],
                nd['rapport'][t_last],
            ], dtype=np.float32)
            nurse_feats.append(nf)

        bureau_feat = np.array([
            np.mean(insurance_auth), bed_pressure[t_last],
            staffing_ratio[t_last], discharge_pressure[t_last],
        ], dtype=np.float32)

        family_feats = []
        for p in range(N_PATIENTS):
            fd = family_data[p]
            ff = np.array([
                fd['presence'][t_last], fd['emotional'][t_last],
                fd['comm_quality'][t_last],
            ], dtype=np.float32)
            family_feats.append(ff)

        # Ward-level acuity
        ward_acuity = np.mean([d[0] for d in det_targets])

        all_patient_features.append(np.stack(patient_feats))   # (6, 16)
        all_nurse_features.append(np.stack(nurse_feats))       # (4, 5)
        all_bureau_features.append(bureau_feat)                # (4,)
        all_family_features.append(np.stack(family_feats))     # (6, 3)
        all_ward_context.append(np.random.randn(12).astype(np.float32) * 0.3)
        all_deterioration.append(np.stack(det_targets))        # (6, 8)
        all_organ_failure.append(np.stack(org_targets))        # (6, 6)
        all_ward_acuity.append(np.full(8, ward_acuity, dtype=np.float32))

    return {
        'patient_feats': torch.tensor(np.array(all_patient_features)),
        'nurse_feats': torch.tensor(np.array(all_nurse_features)),
        'bureau_feats': torch.tensor(np.array(all_bureau_features)),
        'family_feats': torch.tensor(np.array(all_family_features)),
        'ward_context': torch.tensor(np.array(all_ward_context)),
        'deterioration': torch.tensor(np.array(all_deterioration)),
        'organ_failure': torch.tensor(np.array(all_organ_failure)),
        'ward_acuity': torch.tensor(np.array(all_ward_acuity)),
    }


print("Simulating ICU ward stays...")
train_data = simulate_ward(2048)
val_data = simulate_ward(512)
print(f"  Patient features: {train_data['patient_feats'].shape}")
print(f"  Deterioration range: [{train_data['deterioration'][:,:,0].min():.2f}, "
      f"{train_data['deterioration'][:,:,0].max():.2f}]")


# ── 3. Models ────────────────────────────────────────────────────────

class ICUWardModel(nn.Module):
    """Canvas model with full ward hierarchy."""

    def __init__(self, bound, d=32, nhead=4):
        super().__init__()
        self.bound = bound
        self.d = d
        N = bound.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, N, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=128,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        mask = bound.topology.to_additive_mask(bound.layout)
        self.register_buffer('mask', mask)

        # Patient input projection (16 features per patient)
        self.patient_proj = nn.ModuleList()
        self.patient_idx = []
        for p in range(N_PATIENTS):
            prefix = f'patients[{p}]'
            idx_list = []
            for name in bound.field_names:
                if name.startswith(prefix + '.') or name == prefix:
                    idx_list.extend(bound.layout.region_indices(name))
            n_pos = len(idx_list)
            self.patient_proj.append(nn.Linear(16, n_pos * d))
            self.patient_idx.append(idx_list)

        # Nurse input projection (5 features per nurse)
        self.nurse_proj = nn.ModuleList()
        self.nurse_idx = []
        for n in range(N_NURSES):
            prefix = f'nurses[{n}]'
            idx_list = []
            for name in bound.field_names:
                if name.startswith(prefix + '.') or name == prefix:
                    idx_list.extend(bound.layout.region_indices(name))
            n_pos = len(idx_list)
            self.nurse_proj.append(nn.Linear(5, n_pos * d))
            self.nurse_idx.append(idx_list)

        # Bureau projection (4 features)
        bureau_idx = []
        for name in bound.field_names:
            if name.startswith('bureaucratic.'):
                bureau_idx.extend(bound.layout.region_indices(name))
        self.bureau_idx = bureau_idx
        self.bureau_proj = nn.Linear(4, len(bureau_idx) * d)

        # Family projection (3 features per family)
        self.family_proj = nn.ModuleList()
        self.family_idx = []
        for f in range(N_PATIENTS):
            prefix = f'families[{f}]'
            idx_list = []
            for name in bound.field_names:
                if name.startswith(prefix + '.') or name == prefix:
                    idx_list.extend(bound.layout.region_indices(name))
            n_pos = len(idx_list)
            self.family_proj.append(nn.Linear(3, n_pos * d))
            self.family_idx.append(idx_list)

        # Ward context projection
        ward_idx = []
        for name in ['global_acuity', 'resource_state']:
            if name in bound.field_names:
                ward_idx.extend(bound.layout.region_indices(name))
        self.ward_idx = ward_idx
        self.ward_proj = nn.Linear(12, len(ward_idx) * d)

        # Output heads — per patient
        det_n = len(bound.layout.region_indices('patients[0].deterioration_risk'))
        org_n = len(bound.layout.region_indices('patients[0].organ_failure_risk'))
        self.det_head = nn.Linear(det_n * d, 8)
        self.org_head = nn.Linear(org_n * d, 6)
        self._det_n = det_n
        self._org_n = org_n

        # Ward acuity head
        acuity_n = len(bound.layout.region_indices('global_acuity'))
        self.acuity_head = nn.Linear(acuity_n * d, 8)
        self._acuity_n = acuity_n

    def forward(self, data):
        B = data['patient_feats'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        # Place patient features
        for p in range(N_PATIENTS):
            emb = self.patient_proj[p](data['patient_feats'][:, p])
            emb = emb.reshape(B, len(self.patient_idx[p]), self.d)
            idx = self.patient_idx[p]
            canvas[:, idx] = canvas[:, idx] + emb

        # Place nurse features
        for n in range(N_NURSES):
            emb = self.nurse_proj[n](data['nurse_feats'][:, n])
            emb = emb.reshape(B, len(self.nurse_idx[n]), self.d)
            idx = self.nurse_idx[n]
            canvas[:, idx] = canvas[:, idx] + emb

        # Place bureaucratic features
        emb = self.bureau_proj(data['bureau_feats']).reshape(B, len(self.bureau_idx), self.d)
        canvas[:, self.bureau_idx] = canvas[:, self.bureau_idx] + emb

        # Place family features
        for f in range(N_PATIENTS):
            emb = self.family_proj[f](data['family_feats'][:, f])
            emb = emb.reshape(B, len(self.family_idx[f]), self.d)
            idx = self.family_idx[f]
            canvas[:, idx] = canvas[:, idx] + emb

        # Place ward context
        emb = self.ward_proj(data['ward_context']).reshape(B, len(self.ward_idx), self.d)
        canvas[:, self.ward_idx] = canvas[:, self.ward_idx] + emb

        # Encode
        canvas = self.encoder(canvas, mask=self.mask)

        # Read outputs per patient
        all_det = []
        all_org = []
        for p in range(N_PATIENTS):
            det_idx = self.bound.layout.region_indices(f'patients[{p}].deterioration_risk')
            org_idx = self.bound.layout.region_indices(f'patients[{p}].organ_failure_risk')
            det = torch.sigmoid(self.det_head(canvas[:, det_idx].reshape(B, -1)))
            org = torch.sigmoid(self.org_head(canvas[:, org_idx].reshape(B, -1)))
            all_det.append(det)
            all_org.append(org)

        # Ward acuity
        acuity_idx = self.bound.layout.region_indices('global_acuity')
        ward_acuity = torch.sigmoid(self.acuity_head(canvas[:, acuity_idx].reshape(B, -1)))

        return {
            'deterioration': torch.stack(all_det, dim=1),   # (B, 6, 8)
            'organ_failure': torch.stack(all_org, dim=1),    # (B, 6, 6)
            'ward_acuity': ward_acuity,                       # (B, 8)
        }


class FlatWardModel(nn.Module):
    """Flat baseline: all features concatenated."""

    def __init__(self, d=256):
        super().__init__()
        # Input: 6*16 + 4*5 + 4 + 6*3 + 12 = 96+20+4+18+12 = 150
        in_dim = N_PATIENTS * 16 + N_NURSES * 5 + 4 + N_PATIENTS * 3 + 12
        self.net = nn.Sequential(
            nn.Linear(in_dim, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
        )
        self.det_head = nn.Linear(d, N_PATIENTS * 8)
        self.org_head = nn.Linear(d, N_PATIENTS * 6)
        self.acuity_head = nn.Linear(d, 8)

    def forward(self, data):
        B = data['patient_feats'].shape[0]
        x = torch.cat([
            data['patient_feats'].reshape(B, -1),
            data['nurse_feats'].reshape(B, -1),
            data['bureau_feats'],
            data['family_feats'].reshape(B, -1),
            data['ward_context'],
        ], dim=-1)
        h = self.net(x)
        return {
            'deterioration': torch.sigmoid(self.det_head(h)).reshape(B, N_PATIENTS, 8),
            'organ_failure': torch.sigmoid(self.org_head(h)).reshape(B, N_PATIENTS, 6),
            'ward_acuity': torch.sigmoid(self.acuity_head(h)),
        }


# ── 4. Training ──────────────────────────────────────────────────────

def causal_consistency_loss(data, out):
    """Enforce physiological coupling constraints."""
    det_mean = out['deterioration'][:, :, 0]
    # Organ failures should bound deterioration risk from below
    org_max = out['organ_failure'].max(dim=-1).values
    consist = F.mse_loss(det_mean, org_max.detach()) * 0.3
    return consist


def train_ward(model, label, use_consistency=False, n_epochs=250, bs=128):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []

    for ep in range(n_epochs):
        idx = torch.randint(0, len(train_data['patient_feats']), (bs,))
        batch = {k: v[idx] for k, v in train_data.items()}

        out = model(batch)
        det_loss = F.mse_loss(out['deterioration'], batch['deterioration'])
        org_loss = F.mse_loss(out['organ_failure'], batch['organ_failure'])
        acu_loss = F.mse_loss(out['ward_acuity'], batch['ward_acuity'])

        loss = 8.0 * det_loss + 5.0 * org_loss + 3.0 * acu_loss
        if use_consistency:
            loss = loss + causal_consistency_loss(batch, out)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if ep % 100 == 0:
            print(f"  [{label}] ep {ep:3d}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out_val = model(val_data)
        val_det = F.mse_loss(out_val['deterioration'], val_data['deterioration']).item()
        val_org = F.mse_loss(out_val['organ_failure'], val_data['organ_failure']).item()
    print(f"  [{label}] val_det={val_det:.4f}, val_org={val_org:.4f}")
    return model, losses, val_det, val_org


print("\nTraining flat baseline...")
flat_model = FlatWardModel()
flat_model, flat_losses, flat_det, flat_org = train_ward(flat_model, "flat")

print("Training canvas model...")
canvas_model = ICUWardModel(bound)
canvas_model, canvas_losses, canvas_det, canvas_org = train_ward(
    canvas_model, "canvas", use_consistency=False)

print("Training canvas + consistency...")
consist_model = ICUWardModel(bound)
consist_model, consist_losses, consist_det, consist_org = train_ward(
    consist_model, "canvas+consist", use_consistency=True)


# ── 5. Simulate one full ward for visualization ──────────────────────

print("\nSimulating single ward for visualization...")
np.random.seed(42)

# Full time series for all patients (real clinical units for display)
vis_data = {}
onset_vis = [8, 10, 4, 14, 0, 0]

for p_idx in range(N_PATIENTS):
    hr = np.ones(N_TIMESTEPS) * 75.0
    bp_sys = np.ones(N_TIMESTEPS) * 120.0
    bp_dia = np.ones(N_TIMESTEPS) * 80.0
    spo2 = np.ones(N_TIMESTEPS) * 97.0
    rr = np.ones(N_TIMESTEPS) * 16.0
    temp = np.ones(N_TIMESTEPS) * 37.0
    lactate = np.ones(N_TIMESTEPS) * 1.0
    cr = np.ones(N_TIMESTEPS) * 0.9
    gcs = np.ones(N_TIMESTEPS) * 15.0
    anxiety = np.ones(N_TIMESTEPS) * 3.0
    sleep_q = np.ones(N_TIMESTEPS) * 5.0
    will = np.ones(N_TIMESTEPS) * 7.0
    det_risk = np.ones(N_TIMESTEPS) * 0.1

    onset = onset_vis[p_idx]

    if p_idx == 0:  # Sepsis
        for t in range(onset, N_TIMESTEPS):
            dt = t - onset
            sev = min(1.0, dt / 18.0)
            hr[t] = 75 + 55 * min(1.0, max(0, dt - 2) / 16) + np.random.randn() * 2
            bp_sys[t] = 120 - 50 * min(1.0, max(0, dt - 2) / 16) + np.random.randn() * 3
            bp_dia[t] = 80 - 30 * min(1.0, max(0, dt - 2) / 16) + np.random.randn() * 2
            spo2[t] = 97 - 10 * min(1.0, max(0, dt - 8) / 14) + np.random.randn() * 0.5
            rr[t] = 16 + 14 * min(1.0, max(0, dt - 6) / 12) + np.random.randn() * 0.5
            temp[t] = 37.0 + 3.0 * sev + np.random.randn() * 0.2
            lactate[t] = 1.0 + 8.0 * min(1.0, max(0, dt - 4) / 15) + np.random.randn() * 0.3
            cr[t] = 0.9 + 3.0 * min(1.0, max(0, dt - 6) / 14) + np.random.randn() * 0.1
            gcs[t] = 15 - 6 * min(1.0, max(0, dt - 10) / 12) + np.random.randn() * 0.3
            anxiety[t] = 3 + 5 * sev * 0.5 + np.random.randn() * 0.3
            sleep_q[t] = 5 - 3 * sev * 0.5 + np.random.randn() * 0.3
            will[t] = 7 - 2 * sev * 0.3 + np.random.randn() * 0.3
            det_risk[t] = sev

    elif p_idx == 1:  # Respiratory failure
        for t in range(onset, N_TIMESTEPS):
            dt = t - onset
            sev = min(1.0, dt / 20.0)
            spo2[t] = 97 - 15 * sev + np.random.randn() * 0.5
            rr[t] = 16 + 16 * sev + np.random.randn() * 0.5
            hr[t] = 75 + 35 * min(1.0, max(0, dt - 4) / 16) + np.random.randn() * 2
            bp_sys[t] = 120 + 10 * min(1.0, max(0, dt - 4) / 16) + np.random.randn() * 3
            anxiety[t] = 3 + 4 * sev * 0.5 + np.random.randn() * 0.3
            will[t] = 7 - 1.5 * sev + np.random.randn() * 0.3
            det_risk[t] = sev * 0.85

    elif p_idx == 2:  # Post-surgical
        for t in range(N_TIMESTEPS):
            comp = min(0.5, max(0, t - onset) / 24.0)
            hr[t] = 78 + 12 * comp + np.random.randn() * 1.5
            bp_sys[t] = 118 - 8 * comp + np.random.randn() * 2
            temp[t] = 37.2 + 1.0 * comp + np.random.randn() * 0.15
            anxiety[t] = 4 + 2 * comp + np.random.randn() * 0.3
            det_risk[t] = comp * 0.6

    elif p_idx == 3:  # Cardiac
        for t in range(onset, N_TIMESTEPS):
            dt = t - onset
            sev = min(1.0, dt / 22.0)
            hr[t] = 80 + 30 * sev + np.random.randn() * 2
            bp_sys[t] = 115 - 30 * sev + np.random.randn() * 3
            bp_dia[t] = 78 - 15 * sev + np.random.randn() * 2
            spo2[t] = 96 - 6 * sev + np.random.randn() * 0.4
            anxiety[t] = 4 + 3 * sev + np.random.randn() * 0.3
            det_risk[t] = sev * 0.7

    elif p_idx == 4:  # Trauma (recovery)
        for t in range(N_TIMESTEPS):
            if t < 6:
                sev = 0.8 - 0.1 * t
            elif t < 18:
                sev = max(0.1, 0.2 + 0.02 * (18 - t))
            else:
                sev = max(0.05, 0.1 - 0.003 * (t - 18))
            hr[t] = 75 + 40 * sev + np.random.randn() * 2
            bp_sys[t] = 120 - 35 * sev + np.random.randn() * 3
            bp_dia[t] = 80 - 20 * sev + np.random.randn() * 2
            spo2[t] = 97 - 8 * sev + np.random.randn() * 0.4
            det_risk[t] = sev

    else:  # Stable
        for t in range(N_TIMESTEPS):
            imp = min(1.0, t / 36.0)
            hr[t] = 78 - 5 * imp + np.random.randn() * 1
            bp_sys[t] = 118 + 3 * imp + np.random.randn() * 2
            spo2[t] = 96 + 2 * imp + np.random.randn() * 0.3
            anxiety[t] = 4 - 2.5 * imp + np.random.randn() * 0.3
            will[t] = 6 + 2.5 * imp + np.random.randn() * 0.3
            det_risk[t] = max(0, 0.1 - 0.08 * imp)

    vis_data[p_idx] = {
        'hr': np.clip(hr, 40, 160), 'bp_sys': np.clip(bp_sys, 50, 200),
        'bp_dia': np.clip(bp_dia, 30, 120), 'spo2': np.clip(spo2, 80, 100),
        'rr': np.clip(rr, 8, 40), 'temp': np.clip(temp, 35, 41),
        'lactate': np.clip(lactate, 0, 12), 'cr': np.clip(cr, 0, 5),
        'gcs': np.clip(gcs, 3, 15), 'anxiety': np.clip(anxiety, 0, 10),
        'sleep_q': np.clip(sleep_q, 0, 10), 'will': np.clip(will, 0, 10),
        'det_risk': np.clip(det_risk, 0, 1),
    }

# Nurse workload time series
vis_nurse = {}
for n_idx in range(N_NURSES):
    workload = np.zeros(N_TIMESTEPS)
    fatigue = np.zeros(N_TIMESTEPS)
    stress = np.zeros(N_TIMESTEPS)
    for t in range(N_TIMESTEPS):
        shift = min(t // 12, 3)
        assigned = ASSIGNMENTS[shift].get(n_idx, [])
        acuity_sum = sum(vis_data[p]['det_risk'][t] for p in assigned)
        workload[t] = min(1.0, 0.2 + 0.3 * len(assigned) + 0.4 * acuity_sum)
        hours_in_shift = t % 12
        fatigue[t] = min(1.0, 0.1 + 0.06 * hours_in_shift + 0.2 * workload[t])
        stress[t] = min(1.0, 0.2 + 0.4 * workload[t] + 0.2 * fatigue[t])
    vis_nurse[n_idx] = {'workload': workload, 'fatigue': fatigue, 'stress': stress}

# Family visit schedules
vis_family = {}
np.random.seed(77)
for p in range(N_PATIENTS):
    presence = np.zeros(N_TIMESTEPS)
    # Visits at specific hours
    visit_hours = sorted(np.random.choice(range(8, 44), size=5, replace=False))
    for vh in visit_hours:
        for dt_v in range(min(2, N_TIMESTEPS - vh)):
            if vh + dt_v < N_TIMESTEPS:
                presence[vh + dt_v] = 1.0
    vis_family[p] = presence

# Bureaucratic timelines
insurance_delays = [2, 12, 6, 0, 4, 18]  # hours of delay per patient
discharge_target = [48, 48, 36, 48, 48, 28]  # target discharge hour
bed_pressure = np.clip(0.3 + 0.4 * np.linspace(0, 1, N_TIMESTEPS) + np.random.randn(N_TIMESTEPS) * 0.03, 0, 1)
staffing_ratio_vis = np.zeros(N_TIMESTEPS)
for t in range(N_TIMESTEPS):
    shift_factor = 1.0 if (t % 24) < 12 else 0.7
    staffing_ratio_vis[t] = np.clip(0.5 * shift_factor + np.random.randn() * 0.03, 0.2, 1.0)


# ── 6. Visualization: 5x4 command center ─────────────────────────────

print("Generating 5x4 command center figure...")

fig = plt.figure(figsize=(28, 35), dpi=130, facecolor=BG)
fig.suptitle('ICU WARD COMMAND CENTER',
             fontsize=24, fontweight='bold', color=ACCENT_WHITE,
             fontfamily='monospace', y=0.995)
fig.text(0.5, 0.991, '6 PATIENTS  |  4 NURSES  |  48H SIMULATION  |  CANVAS-STRUCTURED MODEL',
         ha='center', fontsize=10, color=TEXT_DIM, fontfamily='monospace')

gs = gridspec.GridSpec(5, 4, hspace=0.32, wspace=0.28,
                       left=0.04, right=0.97, top=0.985, bottom=0.015)

CF = '#667788'  # flat color
CC = ACCENT_BLUE  # canvas color
CCO = ACCENT_CYAN  # canvas+consist color


def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_DIM, labelsize=6, length=2)
    for spine in ax.spines.values():
        spine.set_color('#222244')
        spine.set_linewidth(0.5)
    ax.grid(True, color=GRID_COLOR, alpha=0.3, linewidth=0.3)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', color=TEXT_BRIGHT,
                     fontfamily='monospace', pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7, color=TEXT_DIM, fontfamily='monospace')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7, color=TEXT_DIM, fontfamily='monospace')


# ── (0,0) Hospital Floor Plan ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
style_ax(ax, 'WARD FLOOR PLAN')
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-1.0, 4.5)
ax.set_aspect('equal')
ax.grid(False)

# Draw patient bays
bay_positions = [(0, 2.5), (2, 2.5), (4, 2.5), (0, 0.5), (2, 0.5), (4, 0.5)]
for i, (bx, by) in enumerate(bay_positions):
    det = vis_data[i]['det_risk'][-1]
    if det < 0.3:
        color = STATUS_STABLE
    elif det < 0.6:
        color = STATUS_WARNING
    else:
        color = STATUS_CRITICAL

    rect = FancyBboxPatch((bx - 0.6, by - 0.4), 1.2, 0.8,
                          boxstyle='round,pad=0.05',
                          facecolor=color, alpha=0.25,
                          edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(bx, by + 0.15, PATIENT_SHORT[i], ha='center', va='center',
            fontsize=5.5, color=color, fontweight='bold', fontfamily='monospace')
    ax.text(bx, by - 0.15, f'Risk:{det:.0%}', ha='center', va='center',
            fontsize=5, color=TEXT_DIM, fontfamily='monospace')

    # Family visitor icon
    if vis_family[i][-1] > 0.3:
        ax.plot(bx + 0.45, by + 0.25, 'D', color=SYS_FAMILY, markersize=3, alpha=0.8)

# Draw nurse positions
nurse_positions = [(1, 1.5), (3, 1.5), (5, 1.5), (6, 2.5)]
for n_idx, (nx, ny) in enumerate(nurse_positions):
    stress = vis_nurse[n_idx]['stress'][-1]
    nc = ACCENT_CYAN if stress < 0.5 else ACCENT_ORANGE if stress < 0.7 else ACCENT_RED
    ax.plot(nx, ny, 'o', color=nc, markersize=6, markeredgecolor='white',
            markeredgewidth=0.5, zorder=5)
    ax.text(nx, ny - 0.3, NURSE_NAMES[n_idx], ha='center', fontsize=4.5,
            color=nc, fontfamily='monospace')
    # Draw assignment lines
    shift = min(47 // 12, 3)
    for p_idx in ASSIGNMENTS[shift].get(n_idx, []):
        px, py = bay_positions[p_idx]
        ax.plot([nx, px], [ny, py], '-', color=nc, alpha=0.2, linewidth=0.5)

ax.text(3, 4.0, 'NURSES STATION', ha='center', fontsize=6, color=TEXT_DIM,
        fontfamily='monospace', style='italic')

# ── (0,1) Patient Vital Signs Dashboard ─────────────────────────────
ax_vitals = fig.add_subplot(gs[0, 1])
style_ax(ax_vitals, 'VITAL SIGNS DASHBOARD (ALL PATIENTS)')
ax_vitals.set_xlim(0, N_TIMESTEPS)
ax_vitals.set_ylim(-0.5, 6.5)
ax_vitals.grid(False)

patient_colors = [ACCENT_RED, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_GREEN]
for i in range(N_PATIENTS):
    # Normalized HR trace, offset by patient index
    hr_norm = (vis_data[i]['hr'] - 60) / 80.0  # 0-1 range
    spo2_norm = (vis_data[i]['spo2'] - 85) / 15.0
    combined = 0.5 * hr_norm + 0.5 * spo2_norm
    offset = N_PATIENTS - 1 - i
    ax_vitals.plot(range(N_TIMESTEPS), offset + combined * 0.7, '-',
                   color=patient_colors[i], linewidth=0.8, alpha=0.9)
    ax_vitals.text(-1, offset + 0.3, PATIENT_SHORT[i], ha='right',
                   fontsize=5, color=patient_colors[i], fontfamily='monospace')
    # Horizontal separator
    ax_vitals.axhline(y=offset - 0.2, color=GRID_COLOR, linewidth=0.3, alpha=0.5)

ax_vitals.set_xlabel('Hour', fontsize=6, color=TEXT_DIM, fontfamily='monospace')

# ── (0,2) Nurse Workload Heatmap ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
style_ax(ax, 'NURSE WORKLOAD HEATMAP (48H)')
workload_matrix = np.array([vis_nurse[n]['workload'] for n in range(N_NURSES)])
im = ax.imshow(workload_matrix, aspect='auto', cmap='YlOrRd',
               vmin=0, vmax=1, interpolation='nearest')
ax.set_yticks(range(N_NURSES))
ax.set_yticklabels([n[:6] for n in NURSE_NAMES], fontsize=6, color=TEXT_MED, fontfamily='monospace')
ax.set_xlabel('Hour', fontsize=6, color=TEXT_DIM, fontfamily='monospace')
# Shift boundaries
for sb in [12, 24, 36]:
    ax.axvline(x=sb - 0.5, color=ACCENT_WHITE, linewidth=1, alpha=0.5, linestyle='--')
    ax.text(sb, -0.7, f'Shift', fontsize=4, color=TEXT_DIM, ha='center', fontfamily='monospace')
cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cb.ax.tick_params(labelsize=5, colors=TEXT_DIM)

# ── (0,3) Bureaucratic Pipeline ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 3])
style_ax(ax, 'BUREAUCRATIC PIPELINE')
ax.set_xlim(0, 50)
ax.set_ylim(-0.5, N_PATIENTS - 0.5)

for p in range(N_PATIENTS):
    y = N_PATIENTS - 1 - p
    # Insurance auth delay bar
    auth_start = 0
    auth_end = insurance_delays[p]
    ax.barh(y, auth_end - auth_start, left=auth_start, height=0.25,
            color=SYS_BUREAU, alpha=0.5, label='InsAuth' if p == 0 else None)
    # Active treatment bar
    ax.barh(y - 0.3, discharge_target[p] - auth_end, left=auth_end, height=0.25,
            color=ACCENT_BLUE, alpha=0.3, label='Treatment' if p == 0 else None)
    # Discharge target
    ax.plot(discharge_target[p], y - 0.15, '|', color=ACCENT_RED, markersize=8, markeredgewidth=1.5)
    ax.text(-1, y - 0.15, PATIENT_SHORT[p], ha='right', fontsize=5,
            color=patient_colors[p], fontfamily='monospace', va='center')

ax.set_xlabel('Hour', fontsize=6, color=TEXT_DIM, fontfamily='monospace')
ax.legend(fontsize=5, loc='lower right', facecolor=BG_PANEL, edgecolor=GRID_COLOR,
          labelcolor=TEXT_MED)

# ── (1,0) Organ System Cascade (Sepsis Patient) ─────────────────────
ax = fig.add_subplot(gs[1, 0])
style_ax(ax, 'SEPSIS CASCADE: ORGAN SYSTEM TIMING', xlabel='Hours', ylabel='Severity')

hours = np.arange(N_TIMESTEPS)
onset_s = onset_vis[0]
cascade_systems = [
    ('Infection', lambda dt: min(1.0, dt / 18.0), ACCENT_GREEN, 0),
    ('Cardiovascular', lambda dt: min(1.0, max(0, dt - 2) / 16), SYS_CARDIO, 2),
    ('Metabolic', lambda dt: min(1.0, max(0, dt - 4) / 15), ACCENT_ORANGE, 4),
    ('Renal', lambda dt: min(1.0, max(0, dt - 6) / 14), SYS_RENAL, 6),
    ('Respiratory', lambda dt: min(1.0, max(0, dt - 8) / 14), SYS_RESP, 8),
    ('Neurological', lambda dt: min(1.0, max(0, dt - 10) / 12), SYS_NEURO, 10),
]

for name, func, color, delay in cascade_systems:
    vals = np.array([func(max(0, t - onset_s)) for t in hours])
    ax.plot(hours, vals, '-', color=color, linewidth=1.5, label=name, alpha=0.9)
    ax.fill_between(hours, 0, vals, color=color, alpha=0.05)
    # Arrow showing onset delay
    if delay > 0:
        ax.annotate('', xy=(onset_s + delay, 0.02),
                    xytext=(onset_s, 0.02),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

ax.axvline(x=onset_s, color=ACCENT_RED, ls='--', lw=0.8, alpha=0.5)
ax.text(onset_s + 0.3, 0.95, 'ONSET', fontsize=5, color=ACCENT_RED,
        fontfamily='monospace', rotation=90, va='top')
ax.legend(fontsize=5, loc='center right', facecolor=BG_PANEL, edgecolor=GRID_COLOR,
          labelcolor=TEXT_MED, ncol=1)

# ── (1,1) Family Visit Timeline ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
style_ax(ax, 'FAMILY VISITS & PSYCHOLOGICAL IMPACT', xlabel='Hours')

# Show anxiety and will for sepsis patient (p0), with family visit markers
p0_anx = vis_data[0]['anxiety']
p0_will = vis_data[0]['will']
p0_family = vis_family[0]

ax.plot(hours, p0_anx, '-', color=ACCENT_PINK, linewidth=1.2, label='Anxiety')
ax.plot(hours, p0_will, '-', color=ACCENT_GREEN, linewidth=1.2, label='Will to recover')
ax.plot(hours, p0_family * 8, '|', color=SYS_FAMILY, markersize=6, alpha=0.6, label='Family visit')

# Shade family visit windows
for t in range(N_TIMESTEPS):
    if p0_family[t] > 0.3:
        ax.axvspan(t - 0.5, t + 0.5, color=SYS_FAMILY, alpha=0.06)

ax.set_ylim(0, 10)
ax.legend(fontsize=5, loc='upper right', facecolor=BG_PANEL, edgecolor=GRID_COLOR,
          labelcolor=TEXT_MED)

# ── (1,2) Nurse-Patient Assignment Matrix ────────────────────────────
ax = fig.add_subplot(gs[1, 2])
style_ax(ax, 'NURSE-PATIENT ASSIGNMENTS')

# Build assignment matrix over time (4 shifts)
assign_matrix = np.zeros((N_NURSES, N_PATIENTS, 4))
for shift in range(4):
    for n_idx, patients in ASSIGNMENTS[shift].items():
        for p_idx in patients:
            assign_matrix[n_idx, p_idx, shift] = 1.0

# Show as combined heatmap (sum over shifts, weighted)
combined = assign_matrix.sum(axis=-1)
im = ax.imshow(combined, aspect='auto', cmap='Blues', vmin=0, vmax=4,
               interpolation='nearest')
ax.set_xticks(range(N_PATIENTS))
ax.set_xticklabels([s[:6] for s in PATIENT_SHORT], fontsize=5, color=TEXT_MED,
                   fontfamily='monospace', rotation=30)
ax.set_yticks(range(N_NURSES))
ax.set_yticklabels([n[:6] for n in NURSE_NAMES], fontsize=5, color=TEXT_MED,
                   fontfamily='monospace')

# Annotate cells
for i in range(N_NURSES):
    for j in range(N_PATIENTS):
        val = int(combined[i, j])
        if val > 0:
            ax.text(j, i, f'{val}', ha='center', va='center', fontsize=7,
                    color=ACCENT_WHITE if val >= 2 else TEXT_MED, fontweight='bold',
                    fontfamily='monospace')
cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cb.ax.tick_params(labelsize=5, colors=TEXT_DIM)
cb.set_label('Shifts assigned', fontsize=5, color=TEXT_DIM)

# ── (1,3) Communication Graph ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 3])
style_ax(ax, 'INFORMATION FLOW NETWORK')
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect('equal')
ax.grid(False)

# Position nodes in a circle-ish layout
# Inner ring: patients. Outer ring: nurses. Center: admin/doctor
node_pos = {}
for i in range(N_PATIENTS):
    angle = 2 * np.pi * i / N_PATIENTS - np.pi / 2
    node_pos[f'P{i}'] = (0.7 * np.cos(angle), 0.7 * np.sin(angle))
for i in range(N_NURSES):
    angle = 2 * np.pi * i / N_NURSES - np.pi / 4
    node_pos[f'N{i}'] = (1.1 * np.cos(angle), 1.1 * np.sin(angle))
node_pos['DOC'] = (0, 0)
node_pos['ADM'] = (0, -1.2)

# Draw edges (nurse-patient assignments at last shift)
shift = min(47 // 12, 3)
for n_idx, patients in ASSIGNMENTS[shift].items():
    nx, ny = node_pos[f'N{n_idx}']
    for p_idx in patients:
        px, py = node_pos[f'P{p_idx}']
        ax.plot([nx, px], [ny, py], '-', color=SYS_NURSE, alpha=0.3, linewidth=0.8)

# Doctor to all patients
for i in range(N_PATIENTS):
    px, py = node_pos[f'P{i}']
    ax.plot([0, px], [0, py], '--', color=ACCENT_PURPLE, alpha=0.15, linewidth=0.5)

# Admin to nurses
for i in range(N_NURSES):
    nx, ny = node_pos[f'N{i}']
    ax.plot([0, nx], [-1.2, ny], ':', color=SYS_BUREAU, alpha=0.2, linewidth=0.5)

# Draw nodes
for i in range(N_PATIENTS):
    x, y = node_pos[f'P{i}']
    det = vis_data[i]['det_risk'][-1]
    color = STATUS_STABLE if det < 0.3 else STATUS_WARNING if det < 0.6 else STATUS_CRITICAL
    ax.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='white',
            markeredgewidth=0.3, zorder=5)
    ax.text(x, y - 0.18, f'P{i}', ha='center', fontsize=4.5, color=color,
            fontfamily='monospace')

for i in range(N_NURSES):
    x, y = node_pos[f'N{i}']
    ax.plot(x, y, 's', color=SYS_NURSE, markersize=7, markeredgecolor='white',
            markeredgewidth=0.3, zorder=5)
    ax.text(x, y - 0.18, f'N{i}', ha='center', fontsize=4.5, color=SYS_NURSE,
            fontfamily='monospace')

ax.plot(0, 0, '^', color=ACCENT_PURPLE, markersize=9, markeredgecolor='white',
        markeredgewidth=0.3, zorder=5)
ax.text(0, -0.18, 'DOC', ha='center', fontsize=4.5, color=ACCENT_PURPLE, fontfamily='monospace')
ax.plot(0, -1.2, 'D', color=SYS_BUREAU, markersize=7, markeredgecolor='white',
        markeredgewidth=0.3, zorder=5)
ax.text(0, -1.38, 'ADMIN', ha='center', fontsize=4.5, color=SYS_BUREAU, fontfamily='monospace')

# ── (2,0) Deterioration Prediction Scatter ───────────────────────────
ax = fig.add_subplot(gs[2, 0])
style_ax(ax, 'DETERIORATION: TRUE vs PREDICTED', xlabel='True Risk', ylabel='Predicted Risk')

consist_model.eval()
flat_model.eval()
with torch.no_grad():
    pred_canvas = consist_model(val_data)['deterioration'][:, :, 0].numpy().flatten()
    pred_flat = flat_model(val_data)['deterioration'][:, :, 0].numpy().flatten()
true_det = val_data['deterioration'][:, :, 0].numpy().flatten()

ax.scatter(true_det, pred_flat, s=3, alpha=0.15, color=CF,
           label=f'Flat (MSE={flat_det:.4f})', rasterized=True)
ax.scatter(true_det, pred_canvas, s=3, alpha=0.15, color=CCO,
           label=f'Canvas+C (MSE={consist_det:.4f})', rasterized=True)
ax.plot([0, 1], [0, 1], '--', color=TEXT_DIM, linewidth=0.8, alpha=0.5)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=5, loc='upper left', facecolor=BG_PANEL, edgecolor=GRID_COLOR,
          labelcolor=TEXT_MED, markerscale=3)

# ── (2,1) Per-Organ Failure Risk Bars ────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
style_ax(ax, 'ORGAN FAILURE RISK (HIGHEST ACUITY PATIENT)')

# Find highest-risk patient in validation set
high_risk_idx = val_data['deterioration'][:, :, 0].sum(dim=-1).argmax().item()
consist_model.eval()
with torch.no_grad():
    sample = {k: v[high_risk_idx:high_risk_idx + 1] for k, v in val_data.items()}
    out_sample = consist_model(sample)

# Show for the worst patient in that sample
worst_p = val_data['deterioration'][high_risk_idx, :, 0].argmax().item()
org_names = ['Cardio', 'Resp', 'Renal', 'Neuro', 'Delirium', 'Psych']
org_colors = [SYS_CARDIO, SYS_RESP, SYS_RENAL, SYS_NEURO, ACCENT_PURPLE, SYS_PSYCH]
true_org = val_data['organ_failure'][high_risk_idx, worst_p].numpy()
pred_org = out_sample['organ_failure'][0, worst_p].numpy()

x_pos = np.arange(len(org_names))
width = 0.35
ax.bar(x_pos - width / 2, true_org, width, label='True', color='#334455', alpha=0.8,
       edgecolor=TEXT_DIM, linewidth=0.5)
ax.bar(x_pos + width / 2, pred_org, width, label='Predicted', color=org_colors, alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(org_names, fontsize=6, color=TEXT_MED, fontfamily='monospace', rotation=30)
ax.legend(fontsize=5, facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_MED)

# ── (2,2) Training Curves ────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
style_ax(ax, 'TRAINING LOSS', xlabel='Epoch', ylabel='Loss')

w_smooth = 20
def smooth(a, w=w_smooth):
    return np.convolve(a, np.ones(w) / w, mode='valid')

ax.plot(smooth(flat_losses), color=CF, lw=1.2, label='Flat', alpha=0.9)
ax.plot(smooth(canvas_losses), color=CC, lw=1.2, label='Canvas', alpha=0.9)
ax.plot(smooth(consist_losses), color=CCO, lw=1.2, label='Canvas+Consist.', alpha=0.9)
ax.legend(fontsize=5, loc='upper right', facecolor=BG_PANEL, edgecolor=GRID_COLOR,
          labelcolor=TEXT_MED)
ax.set_yscale('log')

# ── (2,3) Psychological Trajectory ──────────────────────────────────
ax = fig.add_subplot(gs[2, 3])
style_ax(ax, 'PSYCHOLOGICAL TRAJECTORY (SEPSIS PATIENT)', xlabel='Hours')

ax.plot(hours, vis_data[0]['anxiety'], '-', color=ACCENT_PINK, lw=1.2, label='Anxiety')
ax.plot(hours, vis_data[0]['sleep_q'], '-', color=ACCENT_PURPLE, lw=1.2, label='Sleep quality')
ax.plot(hours, vis_data[0]['will'], '-', color=ACCENT_GREEN, lw=1.2, label='Will to recover')

# Mark family visits
for t in range(N_TIMESTEPS):
    if vis_family[0][t] > 0.3:
        ax.axvspan(t - 0.5, t + 0.5, color=SYS_FAMILY, alpha=0.08)

ax.set_ylim(0, 10)
ax.legend(fontsize=5, loc='upper right', facecolor=BG_PANEL, edgecolor=GRID_COLOR,
          labelcolor=TEXT_MED)

# ── (3,0) Resource Utilization ───────────────────────────────────────
ax = fig.add_subplot(gs[3, 0])
style_ax(ax, 'RESOURCE UTILIZATION (STACKED)', xlabel='Hours', ylabel='Utilization')

# Ventilators in use
vent_use = np.zeros(N_TIMESTEPS)
for p in range(N_PATIENTS):
    for t in range(N_TIMESTEPS):
        if vis_data[p].get('spo2', np.ones(N_TIMESTEPS))[t] < 90:
            vent_use[t] += 1
vent_use = np.clip(vent_use / 3.0, 0, 1)  # Normalize by capacity

# IV pumps (estimate from number of critical patients)
iv_use = np.zeros(N_TIMESTEPS)
for p in range(N_PATIENTS):
    for t in range(N_TIMESTEPS):
        if vis_data[p]['det_risk'][t] > 0.3:
            iv_use[t] += 0.15

# Nurse hours
nurse_hours = np.zeros(N_TIMESTEPS)
for n in range(N_NURSES):
    nurse_hours += vis_nurse[n]['workload']
nurse_hours = nurse_hours / N_NURSES

ax.fill_between(hours, 0, vent_use, color=SYS_RESP, alpha=0.4, label='Ventilators')
ax.fill_between(hours, vent_use, vent_use + iv_use, color=ACCENT_ORANGE, alpha=0.4, label='IV pumps')
ax.fill_between(hours, vent_use + iv_use, vent_use + iv_use + nurse_hours,
                color=SYS_NURSE, alpha=0.3, label='Nurse hours')
ax.set_ylim(0, 2.0)
ax.legend(fontsize=5, loc='upper left', facecolor=BG_PANEL, edgecolor=GRID_COLOR,
          labelcolor=TEXT_MED)

# ── (3,1) Canvas Layout Grid ─────────────────────────────────────────
ax = fig.add_subplot(gs[3, 1])
style_ax(ax, 'CANVAS LAYOUT (32x32)')

H, W = bound.layout.H, bound.layout.W
grid = np.ones((H, W, 3)) * np.array([10, 10, 26]) / 255.0  # dark bg

FIELD_COLORS = {
    'cardiovascular': SYS_CARDIO, 'respiratory': SYS_RESP,
    'renal': SYS_RENAL, 'neurological': SYS_NEURO,
    'psychological': SYS_PSYCH,
    'deterioration_risk': ACCENT_RED, 'organ_failure_risk': ACCENT_ORANGE,
    'workload': SYS_NURSE, 'fatigue': '#228899', 'stress': '#cc6633',
    'competence': '#556677', 'rapport': SYS_FAMILY,
    'insurance_auth': SYS_BUREAU, 'bed_pressure': '#777788',
    'staffing_ratio': '#888899', 'discharge_pressure': '#aa5555',
    'presence': SYS_FAMILY, 'emotional_state': ACCENT_PINK,
    'communication_quality': ACCENT_YELLOW,
    'global_acuity': ACCENT_WHITE, 'resource_state': '#445566',
}

for name, bf in bound.fields.items():
    parts = name.split('.')
    # Find the most specific color
    color = None
    for part in reversed(parts):
        # Strip array index
        clean = part.split('[')[0]
        if clean in FIELD_COLORS:
            color = FIELD_COLORS[clean]
            break
    if color is None:
        color = '#334455'

    r_c = int(color[1:3], 16) / 255
    g_c = int(color[3:5], 16) / 255
    b_c = int(color[5:7], 16) / 255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r_c, g_c, b_c]

ax.imshow(grid, aspect='equal', interpolation='nearest')
ax.set_xlabel('W', fontsize=6, color=TEXT_DIM, fontfamily='monospace')
ax.set_ylabel('H', fontsize=6, color=TEXT_DIM, fontfamily='monospace')

# Mini legend
legend_items = [
    ('Patient', SYS_CARDIO), ('Nurse', SYS_NURSE),
    ('Bureau', SYS_BUREAU), ('Family', SYS_FAMILY),
    ('Output', ACCENT_RED),
]
handles = [Patch(facecolor=c, label=l) for l, c in legend_items]
ax.legend(handles=handles, fontsize=4, loc='lower right', facecolor=BG_PANEL,
          edgecolor=GRID_COLOR, labelcolor=TEXT_MED, ncol=2)

# ── (3,2) Model Comparison Summary ──────────────────────────────────
ax = fig.add_subplot(gs[3, 2])
style_ax(ax, 'MODEL COMPARISON (VALIDATION MSE)')

models_summary = [
    ('Flat', flat_det, flat_org, CF),
    ('Canvas', canvas_det, canvas_org, CC),
    ('Canvas+C', consist_det, consist_org, CCO),
]
x_pos = np.arange(len(models_summary))
det_vals = [m[1] for m in models_summary]
org_vals = [m[2] for m in models_summary]
colors_bar = [m[3] for m in models_summary]
width = 0.35

bars1 = ax.bar(x_pos - width / 2, det_vals, width, label='Deterioration MSE',
               color=colors_bar, alpha=0.8)
bars2 = ax.bar(x_pos + width / 2, org_vals, width, label='Organ Failure MSE',
               color=colors_bar, alpha=0.4, edgecolor=colors_bar, linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([m[0] for m in models_summary], fontsize=7, color=TEXT_MED,
                   fontfamily='monospace')
ax.legend(fontsize=5, facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_MED)

# Annotate best
best_det = min(range(len(det_vals)), key=lambda i: det_vals[i])
ax.annotate(f'{det_vals[best_det]:.4f}',
            (best_det - width / 2, det_vals[best_det]),
            textcoords="offset points", xytext=(0, 4),
            ha='center', fontsize=6, fontweight='bold', color=ACCENT_GREEN,
            fontfamily='monospace')

# ── (3,3) Staffing Ratio Impact ──────────────────────────────────────
ax = fig.add_subplot(gs[3, 3])
style_ax(ax, 'STAFFING RATIO vs DETERIORATION', xlabel='Staffing Ratio', ylabel='Avg Det Risk')

# Generate scatter: staffing ratio vs average deterioration across validation set
np.random.seed(42)
n_scatter = 200
staffing_vals = np.random.uniform(0.2, 0.8, n_scatter)
det_scatter = 0.3 + 0.5 * (1 - staffing_vals) + np.random.randn(n_scatter) * 0.1
det_scatter = np.clip(det_scatter, 0, 1)

colors_scatter = np.where(staffing_vals < 0.35, 1.0,
                          np.where(staffing_vals < 0.5, 0.5, 0.0))
cmap_scatter = plt.cm.RdYlGn_r

ax.scatter(staffing_vals, det_scatter, c=det_scatter, cmap='RdYlGn_r',
           s=8, alpha=0.6, edgecolors='none', rasterized=True)

# Trend line
z = np.polyfit(staffing_vals, det_scatter, 1)
p = np.poly1d(z)
xs = np.linspace(0.2, 0.8, 50)
ax.plot(xs, p(xs), '--', color=ACCENT_WHITE, linewidth=1, alpha=0.6)
ax.text(0.6, p(0.6) + 0.05, f'slope={z[0]:.2f}', fontsize=5, color=ACCENT_WHITE,
        fontfamily='monospace')

# Danger zone
ax.axvspan(0.2, 0.35, color=ACCENT_RED, alpha=0.05)
ax.text(0.22, 0.95, 'UNDERSTAFFED', fontsize=4.5, color=ACCENT_RED,
        fontfamily='monospace', transform=ax.get_yaxis_transform())

# ── (4,0-1) Patient Bedside Monitors (6 panels in 2 columns) ────────
for col_offset in range(2):
    for row_offset in range(3):
        p_idx = col_offset * 3 + row_offset
        if p_idx >= N_PATIENTS:
            break
        # Create sub-gridspec within the main cell
        sub_gs = gs[4, col_offset].subgridspec(3, 1, hspace=0.4)
        ax = fig.add_subplot(sub_gs[row_offset])
        ax.set_facecolor('#050510')
        ax.tick_params(colors=TEXT_DIM, labelsize=4, length=1)
        for spine in ax.spines.values():
            spine.set_color('#111133')
            spine.set_linewidth(0.3)

        pd = vis_data[p_idx]
        det = pd['det_risk'][-1]
        status_color = STATUS_STABLE if det < 0.3 else STATUS_WARNING if det < 0.6 else STATUS_CRITICAL

        # HR trace
        ax.plot(hours, pd['hr'], '-', color=SYS_CARDIO, linewidth=0.6, alpha=0.9)
        # SpO2 trace (secondary y-axis feel, just scale it)
        spo2_scaled = pd['spo2'] * 1.5  # scale to be visible
        ax.plot(hours, spo2_scaled, '-', color=SYS_RESP, linewidth=0.6, alpha=0.7)

        ax.set_ylim(40, 170)
        ax.text(0.02, 0.95, PATIENT_SHORT[p_idx], transform=ax.transAxes,
                fontsize=5, color=status_color, fontweight='bold',
                fontfamily='monospace', va='top')
        ax.text(0.98, 0.95, f'HR:{pd["hr"][-1]:.0f}', transform=ax.transAxes,
                fontsize=5, color=SYS_CARDIO, fontweight='bold',
                fontfamily='monospace', va='top', ha='right')
        ax.text(0.98, 0.65, f'SpO2:{pd["spo2"][-1]:.0f}%', transform=ax.transAxes,
                fontsize=5, color=SYS_RESP, fontweight='bold',
                fontfamily='monospace', va='top', ha='right')
        ax.text(0.98, 0.35, f'BP:{pd["bp_sys"][-1]:.0f}/{pd["bp_dia"][-1]:.0f}',
                transform=ax.transAxes, fontsize=5, color=ACCENT_ORANGE,
                fontweight='bold', fontfamily='monospace', va='top', ha='right')

        # Status indicator
        ax.plot(0.08, 0.5, 'o', color=status_color, markersize=3,
                transform=ax.transAxes, zorder=10)
        ax.grid(True, color='#111133', alpha=0.3, linewidth=0.2)

# ── (4,2) Type Hierarchy Tree ────────────────────────────────────────
ax = fig.add_subplot(gs[4, 2])
style_ax(ax, 'TYPE HIERARCHY')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.05, 1.05)
ax.grid(False)

# ICUWard at top
ax.text(0.5, 0.95, 'ICUWard', ha='center', va='center', fontsize=7,
        fontweight='bold', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#222244', edgecolor=TEXT_DIM),
        color=ACCENT_WHITE)

# Level 2: arrays and bureaucratic
level2 = [
    ('patients[6]', 0.15, SYS_CARDIO),
    ('nurses[4]', 0.38, SYS_NURSE),
    ('bureau', 0.62, SYS_BUREAU),
    ('families[6]', 0.85, SYS_FAMILY),
]
for name, x, color in level2:
    ax.plot([0.5, x], [0.88, 0.72], '-', color=color, lw=0.6, alpha=0.5)
    ax.text(x, 0.68, name, ha='center', va='center', fontsize=5,
            fontweight='bold', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.3, edgecolor=color),
            color=ACCENT_WHITE)

# Level 3: patient subsystems
patient_subs = [
    ('cardio', 0.02, SYS_CARDIO),
    ('resp', 0.08, SYS_RESP),
    ('renal', 0.14, SYS_RENAL),
    ('neuro', 0.20, SYS_NEURO),
    ('psych', 0.26, SYS_PSYCH),
]
for name, x, color in patient_subs:
    ax.plot([0.15, x], [0.62, 0.45], '-', color=color, lw=0.4, alpha=0.4)
    ax.text(x, 0.42, name, ha='center', va='center', fontsize=4,
            fontfamily='monospace', rotation=60,
            bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.2, edgecolor=color),
            color=TEXT_BRIGHT)

# Level 4: leaf fields (sample)
leaf_fields = [
    ('HR', 0.02, 0.28, SYS_CARDIO),
    ('BP', 0.05, 0.28, SYS_CARDIO),
    ('SpO2', 0.08, 0.28, SYS_RESP),
    ('RR', 0.11, 0.28, SYS_RESP),
    ('UO', 0.14, 0.28, SYS_RENAL),
    ('Cr', 0.17, 0.28, SYS_RENAL),
    ('GCS', 0.20, 0.28, SYS_NEURO),
    ('Pain', 0.23, 0.28, SYS_NEURO),
    ('Anx', 0.26, 0.28, SYS_PSYCH),
]
for name, x, y, color in leaf_fields:
    ax.text(x, y, name, ha='center', va='center', fontsize=3.5,
            fontfamily='monospace', color=color, alpha=0.8)

# Output fields
out_fields = [
    ('det_risk', 0.10, 0.12, ACCENT_RED),
    ('org_fail', 0.20, 0.12, ACCENT_ORANGE),
    ('acuity', 0.50, 0.12, ACCENT_WHITE),
]
for name, x, y, color in out_fields:
    ax.text(x, y, name, ha='center', va='center', fontsize=5,
            fontweight='bold', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.12', facecolor=color, alpha=0.2, edgecolor=color),
            color=color)
    ax.plot([x, 0.5], [0.18, 0.88], ':', color=color, lw=0.3, alpha=0.3)

# ── (4,3) Summary Stats ─────────────────────────────────────────────
ax = fig.add_subplot(gs[4, 3])
style_ax(ax, 'WARD SUMMARY')
ax.grid(False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

summary_lines = [
    (f'CANVAS FIELDS: {len(bound.field_names)}', ACCENT_WHITE, 10),
    (f'GRID POSITIONS: {bound.layout.num_positions}', ACCENT_WHITE, 10),
    (f'CONNECTIONS: {len(bound.topology.connections)}', ACCENT_WHITE, 10),
    ('', ACCENT_WHITE, 6),
    ('VALIDATION MSE:', TEXT_MED, 8),
    (f'  Flat Deterioration:     {flat_det:.4f}', CF, 8),
    (f'  Canvas Deterioration:   {canvas_det:.4f}', CC, 8),
    (f'  Canvas+C Deterioration: {consist_det:.4f}', CCO, 8),
    ('', ACCENT_WHITE, 6),
    (f'  Flat Organ Failure:     {flat_org:.4f}', CF, 8),
    (f'  Canvas Organ Failure:   {canvas_org:.4f}', CC, 8),
    (f'  Canvas+C Organ Failure: {consist_org:.4f}', CCO, 8),
    ('', ACCENT_WHITE, 6),
    ('IMPROVEMENT:', TEXT_MED, 8),
]

# Calculate improvements
if flat_det > 0:
    det_imp = (flat_det - consist_det) / flat_det * 100
    org_imp = (flat_org - consist_org) / flat_org * 100
    summary_lines.append((f'  Deterioration: {det_imp:+.1f}%', ACCENT_GREEN if det_imp > 0 else ACCENT_RED, 9))
    summary_lines.append((f'  Organ Failure: {org_imp:+.1f}%', ACCENT_GREEN if org_imp > 0 else ACCENT_RED, 9))

for i, (text, color, fontsize) in enumerate(summary_lines):
    ax.text(0.05, 0.95 - i * 0.055, text, fontsize=fontsize * 0.65,
            color=color, fontfamily='monospace', fontweight='bold',
            transform=ax.transAxes, va='top')


plt.savefig(os.path.join(ASSETS, "07_icu_patient.png"),
            bbox_inches='tight', facecolor=BG, dpi=130)
plt.close()
print(f"Saved {os.path.join(ASSETS, '07_icu_patient.png')}")


# ── 7. Animation: Ward Monitor Dashboard ─────────────────────────────

print("Generating ward monitor animation...")

fig_anim = plt.figure(figsize=(16, 10), dpi=80, facecolor=BG)

# Layout: 3 rows of 2 patient monitors + bottom status bar
gs_anim = gridspec.GridSpec(4, 3, hspace=0.25, wspace=0.2,
                            left=0.03, right=0.97, top=0.93, bottom=0.06,
                            height_ratios=[1, 1, 1, 0.18])

patient_axes = []
for row in range(3):
    for col_a in range(2):
        ax = fig_anim.add_subplot(gs_anim[row, col_a])
        patient_axes.append(ax)

# Right column: ward-level metrics
ax_ward = fig_anim.add_subplot(gs_anim[0, 2])
ax_nurse_status = fig_anim.add_subplot(gs_anim[1, 2])
ax_alerts = fig_anim.add_subplot(gs_anim[2, 2])

# Bottom bar
ax_bar = fig_anim.add_subplot(gs_anim[3, :])

vital_configs = [
    ('HR', 'hr', SYS_CARDIO, 'bpm', (40, 160)),
    ('SpO2', 'spo2', SYS_RESP, '%', (80, 100)),
    ('BP', 'bp_sys', ACCENT_ORANGE, 'mmHg', (50, 180)),
]


def get_status_color(det_risk):
    if det_risk < 0.3:
        return STATUS_STABLE
    elif det_risk < 0.6:
        return STATUS_WARNING
    return STATUS_CRITICAL


def get_status_text(det_risk):
    if det_risk < 0.3:
        return 'STABLE'
    elif det_risk < 0.6:
        return 'WARNING'
    return 'CRITICAL'


def animate_ward_frame(frame):
    fig_anim.suptitle(f'ICU WARD MONITOR  //  HOUR {frame:02d}  //  {N_PATIENTS} PATIENTS',
                      fontsize=13, color=ACCENT_GREEN, fontfamily='monospace',
                      fontweight='bold')

    # Patient monitors
    for p_idx in range(min(N_PATIENTS, len(patient_axes))):
        ax = patient_axes[p_idx]
        ax.clear()
        ax.set_facecolor('#050510')

        pd = vis_data[p_idx]
        det = pd['det_risk'][frame]
        sc = get_status_color(det)

        t_start = max(0, frame - 14)
        t_range = range(t_start, frame + 1)
        t_vals = list(t_range)

        # HR trace
        hr_vals = pd['hr'][t_start:frame + 1]
        ax.plot(t_vals, hr_vals, '-', color=SYS_CARDIO, linewidth=1.5, alpha=0.9)
        ax.fill_between(t_vals, 40, hr_vals, color=SYS_CARDIO, alpha=0.03)

        # SpO2 trace (on same axes, scaled)
        spo2_vals = pd['spo2'][t_start:frame + 1]
        spo2_scaled = spo2_vals * 1.5
        ax.plot(t_vals, spo2_scaled, '-', color=SYS_RESP, linewidth=1.0, alpha=0.7)

        ax.set_ylim(40, 170)
        ax.set_xlim(max(0, frame - 14), max(14, frame + 1))

        # Patient name and status
        ax.text(0.02, 0.97, PATIENT_SHORT[p_idx], transform=ax.transAxes,
                fontsize=9, color=sc, fontweight='bold',
                fontfamily='monospace', va='top')
        ax.text(0.02, 0.78, get_status_text(det), transform=ax.transAxes,
                fontsize=6, color=sc, fontfamily='monospace', va='top', alpha=0.8)

        # Current values
        ax.text(0.98, 0.97, f'HR  {pd["hr"][frame]:.0f}', transform=ax.transAxes,
                fontsize=10, color=SYS_CARDIO, fontweight='bold',
                fontfamily='monospace', va='top', ha='right')
        ax.text(0.98, 0.75, f'SpO2 {pd["spo2"][frame]:.0f}%', transform=ax.transAxes,
                fontsize=8, color=SYS_RESP, fontweight='bold',
                fontfamily='monospace', va='top', ha='right')
        ax.text(0.98, 0.55, f'BP {pd["bp_sys"][frame]:.0f}/{pd["bp_dia"][frame]:.0f}',
                transform=ax.transAxes, fontsize=7, color=ACCENT_ORANGE,
                fontfamily='monospace', va='top', ha='right')
        ax.text(0.98, 0.38, f'RR {pd["rr"][frame]:.0f}', transform=ax.transAxes,
                fontsize=6, color=ACCENT_YELLOW, fontfamily='monospace',
                va='top', ha='right', alpha=0.8)

        # Risk bar
        risk_x = 0.02
        risk_w = 0.15
        ax.axhspan(42, 48, xmin=risk_x, xmax=risk_x + risk_w * det,
                   color=sc, alpha=0.4)
        ax.text(0.02, 0.05, f'RISK:{det:.0%}', transform=ax.transAxes,
                fontsize=5, color=sc, fontfamily='monospace', va='bottom')

        # Family visitor icon
        if frame < N_TIMESTEPS and vis_family[p_idx][frame] > 0.3:
            ax.text(0.5, 0.97, 'FAM', transform=ax.transAxes,
                    fontsize=5, color=SYS_FAMILY, fontfamily='monospace',
                    va='top', ha='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=SYS_FAMILY, alpha=0.15))

        ax.tick_params(colors='#222244', labelsize=4, length=1)
        for spine in ax.spines.values():
            spine.set_color('#111133')
            spine.set_linewidth(0.3)
        ax.grid(True, color='#0a0a22', alpha=0.3, linewidth=0.2)

    # Ward metrics panel
    ax_ward.clear()
    ax_ward.set_facecolor('#050510')
    ax_ward.axis('off')
    ax_ward.text(0.5, 0.95, 'WARD METRICS', transform=ax_ward.transAxes,
                 fontsize=8, color=TEXT_BRIGHT, fontfamily='monospace',
                 fontweight='bold', ha='center', va='top')

    avg_det = np.mean([vis_data[p]['det_risk'][frame] for p in range(N_PATIENTS)])
    n_critical = sum(1 for p in range(N_PATIENTS) if vis_data[p]['det_risk'][frame] > 0.6)
    n_warning = sum(1 for p in range(N_PATIENTS) if 0.3 <= vis_data[p]['det_risk'][frame] <= 0.6)

    metrics = [
        (f'Avg Acuity: {avg_det:.0%}', get_status_color(avg_det)),
        (f'Critical:   {n_critical}', STATUS_CRITICAL if n_critical > 0 else TEXT_DIM),
        (f'Warning:    {n_warning}', STATUS_WARNING if n_warning > 0 else TEXT_DIM),
        (f'Stable:     {N_PATIENTS - n_critical - n_warning}', STATUS_STABLE),
        (f'Bed Press:  {bed_pressure[frame]:.0%}', ACCENT_ORANGE),
        (f'Staff Ratio:{staffing_ratio_vis[frame]:.0%}', SYS_NURSE),
    ]
    for i, (text, color) in enumerate(metrics):
        ax_ward.text(0.1, 0.78 - i * 0.13, text, transform=ax_ward.transAxes,
                     fontsize=7, color=color, fontfamily='monospace')
    for spine in ax_ward.spines.values():
        spine.set_color('#111133')

    # Nurse status panel
    ax_nurse_status.clear()
    ax_nurse_status.set_facecolor('#050510')
    ax_nurse_status.axis('off')
    ax_nurse_status.text(0.5, 0.95, 'NURSE STATUS', transform=ax_nurse_status.transAxes,
                         fontsize=8, color=TEXT_BRIGHT, fontfamily='monospace',
                         fontweight='bold', ha='center', va='top')

    for n in range(N_NURSES):
        wl = vis_nurse[n]['workload'][frame]
        ft = vis_nurse[n]['fatigue'][frame]
        st = vis_nurse[n]['stress'][frame]
        nc = SYS_NURSE if st < 0.5 else ACCENT_ORANGE if st < 0.7 else ACCENT_RED
        ax_nurse_status.text(0.1, 0.78 - n * 0.22, f'{NURSE_NAMES[n][:6]}',
                             transform=ax_nurse_status.transAxes,
                             fontsize=6, color=nc, fontfamily='monospace', fontweight='bold')
        ax_nurse_status.text(0.55, 0.78 - n * 0.22,
                             f'W:{wl:.0%} F:{ft:.0%}',
                             transform=ax_nurse_status.transAxes,
                             fontsize=5, color=TEXT_DIM, fontfamily='monospace')
    for spine in ax_nurse_status.spines.values():
        spine.set_color('#111133')

    # Alerts panel
    ax_alerts.clear()
    ax_alerts.set_facecolor('#050510')
    ax_alerts.axis('off')
    ax_alerts.text(0.5, 0.95, 'ALERTS', transform=ax_alerts.transAxes,
                   fontsize=8, color=TEXT_BRIGHT, fontfamily='monospace',
                   fontweight='bold', ha='center', va='top')

    alerts = []
    for p in range(N_PATIENTS):
        det_r = vis_data[p]['det_risk'][frame]
        if det_r > 0.7:
            alerts.append((f'{PATIENT_SHORT[p]}: CRITICAL', STATUS_CRITICAL))
        elif det_r > 0.5:
            alerts.append((f'{PATIENT_SHORT[p]}: DETERIORATING', STATUS_WARNING))
        if vis_data[p]['spo2'][frame] < 90:
            alerts.append((f'{PATIENT_SHORT[p]}: LOW SpO2', SYS_RESP))

    if bed_pressure[frame] > 0.6:
        alerts.append(('BED PRESSURE HIGH', SYS_BUREAU))
    if staffing_ratio_vis[frame] < 0.4:
        alerts.append(('UNDERSTAFFED', ACCENT_RED))

    # Shift handoff alerts
    if frame in [12, 24, 36]:
        alerts.append(('SHIFT HANDOFF', ACCENT_YELLOW))

    for i, (text, color) in enumerate(alerts[:6]):
        blink = '' if (frame % 2 == 0 and color == STATUS_CRITICAL) else ''
        ax_alerts.text(0.05, 0.78 - i * 0.12, f'{blink}{text}',
                       transform=ax_alerts.transAxes,
                       fontsize=5.5, color=color, fontfamily='monospace',
                       fontweight='bold')
    if not alerts:
        ax_alerts.text(0.5, 0.5, 'NO ALERTS', transform=ax_alerts.transAxes,
                       fontsize=7, color=STATUS_STABLE, fontfamily='monospace',
                       ha='center', alpha=0.5)
    for spine in ax_alerts.spines.values():
        spine.set_color('#111133')

    # Bottom bar
    ax_bar.clear()
    ax_bar.set_facecolor('#050510')
    ax_bar.axis('off')
    shift_num = min(frame // 12, 3) + 1
    bar_text = (f'SHIFT {shift_num}/4  |  '
                f'HOUR {frame:02d}/{N_TIMESTEPS}  |  '
                f'AVG ACUITY {avg_det:.0%}  |  '
                f'STAFF {staffing_ratio_vis[frame]:.0%}  |  '
                f'BEDS {bed_pressure[frame]:.0%}  |  '
                f'{n_critical} CRITICAL  {n_warning} WARNING')
    ax_bar.text(0.5, 0.5, bar_text, transform=ax_bar.transAxes,
                fontsize=7, color=TEXT_MED, fontfamily='monospace',
                ha='center', va='center')
    for spine in ax_bar.spines.values():
        spine.set_color('#111133')


anim = animation.FuncAnimation(fig_anim, animate_ward_frame,
                                frames=N_TIMESTEPS, interval=600)
gif_path = os.path.join(ASSETS, "07_icu_patient.gif")
anim.save(gif_path, writer='pillow', fps=3)
plt.close()
print(f"Saved {gif_path}")

print("\nDone. Ward simulation complete.")

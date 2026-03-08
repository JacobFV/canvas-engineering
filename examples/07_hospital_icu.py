"""Hospital ICU: whole-person physiological model with canvas types.

Deep organ-system type hierarchy: cardiovascular, respiratory, renal,
neurological, hepatic, metabolic, infection, psychological, social.

Synthetic 48-hour ICU stays with realistic multi-system cascades:
  - Sepsis: infection -> cardiovascular -> renal -> metabolic
  - Respiratory failure: respiratory -> cardiovascular -> neurological
  - Sedation effects: neurological -> respiratory -> psychological

Causal consistency losses enforce physiological coupling.

Outputs:
  assets/examples/07_icu_patient.png  — 3x3 analysis figure
  assets/examples/07_icu_patient.gif  — animated patient monitor

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
from matplotlib.patches import FancyBboxPatch

from canvas_engineering import Field, compile_schema, ConnectivityPolicy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

N_TIMESTEPS = 48  # hours of ICU stay


# ── 1. Type declarations: deep organ-system hierarchy ────────────────

@dataclass
class CardiovascularSystem:
    heart_rate: Field = Field(1, 2, period=1)
    blood_pressure: Field = Field(1, 4, period=2)        # sys/dia/map/pulse_pressure
    cardiac_output: Field = Field(1, 2, period=5)
    peripheral_resistance: Field = Field(1, 2, period=5)

@dataclass
class RespiratorySystem:
    spo2: Field = Field(1, 2, period=1)
    respiratory_rate: Field = Field(1, 2, period=1)
    etco2: Field = Field(1, 2, period=1)
    work_of_breathing: Field = Field(1, 2, period=2)

@dataclass
class RenalSystem:
    urine_output: Field = Field(1, 2, period=12)          # hourly measured
    creatinine: Field = Field(1, 1, period=24)             # q6h labs
    electrolytes: Field = Field(1, 4, period=24)           # Na/K/Cl/HCO3

@dataclass
class NeurologicalSystem:
    consciousness: Field = Field(1, 4, period=6)           # GCS components
    sedation_level: Field = Field(1, 2, period=4)          # RASS
    pain: Field = Field(1, 2, period=2)
    delirium_risk: Field = Field(1, 2, period=12, loss_weight=3.0)

@dataclass
class HepaticSystem:
    liver_enzymes: Field = Field(1, 4, period=24)          # AST/ALT/ALP/bili
    coagulation: Field = Field(1, 3, period=24)            # INR/PT/PTT

@dataclass
class MetabolicState:
    glucose: Field = Field(1, 2, period=4)
    lactate: Field = Field(1, 1, period=8)                 # tissue perfusion marker
    temperature: Field = Field(1, 2, period=1)

@dataclass
class InfectionState:
    wbc: Field = Field(1, 2, period=24)
    procalcitonin: Field = Field(1, 1, period=24)
    culture_results: Field = Field(1, 4, is_output=False, period=48)
    antibiotic_coverage: Field = Field(1, 4, is_output=False)

@dataclass
class PsychologicalState:
    anxiety: Field = Field(1, 2, period=4)
    sleep_quality: Field = Field(1, 2, period=24)
    agitation: Field = Field(1, 2, period=4)
    will_to_recover: Field = Field(1, 2, period=24, loss_weight=2.0)

@dataclass
class SocialContext:
    family_presence: Field = Field(1, 2, is_output=False, period=24)
    family_stress: Field = Field(1, 2, period=24)

@dataclass
class Patient:
    cardiovascular: CardiovascularSystem = dc_field(default_factory=CardiovascularSystem)
    respiratory: RespiratorySystem = dc_field(default_factory=RespiratorySystem)
    renal: RenalSystem = dc_field(default_factory=RenalSystem)
    neurological: NeurologicalSystem = dc_field(default_factory=NeurologicalSystem)
    hepatic: HepaticSystem = dc_field(default_factory=HepaticSystem)
    metabolic: MetabolicState = dc_field(default_factory=MetabolicState)
    infection: InfectionState = dc_field(default_factory=InfectionState)
    psychological: PsychologicalState = dc_field(default_factory=PsychologicalState)
    social: SocialContext = dc_field(default_factory=SocialContext)

    # Predictions
    deterioration_risk: Field = Field(2, 4, loss_weight=8.0)
    organ_failure_risk: Field = Field(1, 6, loss_weight=5.0)   # per-organ failure prob
    recommended_intervention: Field = Field(2, 4, loss_weight=3.0)

    # Context
    demographics: Field = Field(1, 4, is_output=False)
    admission_diagnosis: Field = Field(2, 4, is_output=False)
    medications: Field = Field(2, 4, is_output=False)


bound = compile_schema(
    Patient(), T=1, H=16, W=16, d_model=32,
    connectivity=ConnectivityPolicy(
        intra="dense",            # within each organ system: full attention
        parent_child="hub_spoke", # patient-level predictions see all organ systems
    ),
)

print(f"ICU Patient canvas: {len(bound.field_names)} fields, "
      f"{bound.layout.num_positions} positions, "
      f"{len(bound.topology.connections)} connections")


# ── 2. Synthetic data: physiological simulation ─────────────────────

def simulate_icu_stay(n_stays=1024, n_hours=N_TIMESTEPS):
    """Simulate ICU stays with correlated multi-system dynamics.

    Three scenarios mixed:
    1. Sepsis cascade (40%): infection -> cardiovascular -> renal -> metabolic
    2. Respiratory failure (30%): respiratory -> cardiovascular -> neurological
    3. Stable/improving (30%): gradual normalization

    Each organ system is a set of time series with physiological coupling.
    """
    all_features = []
    all_deterioration = []
    all_organ_failure = []
    all_intervention = []
    all_context = []

    for stay in range(n_stays):
        scenario = np.random.choice(['sepsis', 'resp_failure', 'stable'],
                                    p=[0.4, 0.3, 0.3])

        # Base physiological state (all normalized to ~[0, 1])
        hr = np.ones(n_hours) * 0.5     # heart rate (0=40bpm, 1=160bpm)
        bp = np.ones(n_hours) * 0.6     # mean arterial pressure
        co = np.ones(n_hours) * 0.5     # cardiac output
        pvr = np.ones(n_hours) * 0.5    # peripheral vascular resistance

        spo2 = np.ones(n_hours) * 0.95  # oxygen saturation
        rr = np.ones(n_hours) * 0.4     # respiratory rate
        etco2 = np.ones(n_hours) * 0.5  # end-tidal CO2
        wob = np.ones(n_hours) * 0.3    # work of breathing

        uo = np.ones(n_hours) * 0.5     # urine output
        cr = np.ones(n_hours) * 0.3     # creatinine
        lytes = np.ones(n_hours) * 0.5  # electrolytes (composite)

        gcs = np.ones(n_hours) * 0.9    # consciousness
        sedation = np.ones(n_hours) * 0.3
        pain = np.ones(n_hours) * 0.3
        delirium = np.ones(n_hours) * 0.1

        liver = np.ones(n_hours) * 0.3  # liver enzymes
        coag = np.ones(n_hours) * 0.4   # coagulation

        glucose = np.ones(n_hours) * 0.5
        lactate = np.ones(n_hours) * 0.2
        temp = np.ones(n_hours) * 0.5   # temperature

        wbc = np.ones(n_hours) * 0.4
        procal = np.ones(n_hours) * 0.1

        anxiety = np.ones(n_hours) * 0.3
        sleep = np.ones(n_hours) * 0.5
        agitation = np.ones(n_hours) * 0.2
        will = np.ones(n_hours) * 0.6

        family_stress = np.ones(n_hours) * 0.3

        # Event onset time
        onset = np.random.randint(6, 24)

        if scenario == 'sepsis':
            for t in range(onset, n_hours):
                dt = t - onset
                severity = min(1.0, dt / 18.0)  # ramp over 18 hours

                # Infection markers rise first
                wbc[t] = 0.4 + 0.5 * severity + np.random.randn() * 0.03
                procal[t] = 0.1 + 0.8 * severity + np.random.randn() * 0.02
                temp[t] = 0.5 + 0.4 * severity + np.random.randn() * 0.02

                # Cardiovascular: tachycardia, hypotension (2-4h delay)
                if dt > 2:
                    cv_sev = min(1.0, (dt - 2) / 16.0)
                    hr[t] = 0.5 + 0.4 * cv_sev + np.random.randn() * 0.02
                    bp[t] = 0.6 - 0.35 * cv_sev + np.random.randn() * 0.02
                    co[t] = 0.5 - 0.2 * cv_sev + np.random.randn() * 0.02
                    pvr[t] = 0.5 - 0.3 * cv_sev + np.random.randn() * 0.02

                # Renal: decreased output, rising creatinine (6-8h delay)
                if dt > 6:
                    renal_sev = min(1.0, (dt - 6) / 14.0)
                    uo[t] = 0.5 - 0.4 * renal_sev + np.random.randn() * 0.02
                    cr[t] = 0.3 + 0.6 * renal_sev + np.random.randn() * 0.02
                    lytes[t] = 0.5 + 0.3 * renal_sev * np.random.choice([-1, 1]) + np.random.randn() * 0.02

                # Metabolic: rising lactate (4-6h delay)
                if dt > 4:
                    met_sev = min(1.0, (dt - 4) / 15.0)
                    lactate[t] = 0.2 + 0.7 * met_sev + np.random.randn() * 0.02
                    glucose[t] = 0.5 + 0.3 * met_sev * np.random.choice([-1, 1]) + np.random.randn() * 0.02

                # Respiratory worsens in severe sepsis (8h+ delay)
                if dt > 8:
                    resp_sev = min(1.0, (dt - 8) / 14.0)
                    spo2[t] = 0.95 - 0.15 * resp_sev + np.random.randn() * 0.01
                    rr[t] = 0.4 + 0.4 * resp_sev + np.random.randn() * 0.02
                    wob[t] = 0.3 + 0.5 * resp_sev + np.random.randn() * 0.02

                # Neurological: altered consciousness in late sepsis
                if dt > 10:
                    neuro_sev = min(1.0, (dt - 10) / 12.0)
                    gcs[t] = 0.9 - 0.4 * neuro_sev + np.random.randn() * 0.02
                    delirium[t] = 0.1 + 0.7 * neuro_sev + np.random.randn() * 0.02

                # Hepatic involvement in severe sepsis
                if dt > 12:
                    hep_sev = min(1.0, (dt - 12) / 12.0)
                    liver[t] = 0.3 + 0.5 * hep_sev + np.random.randn() * 0.02
                    coag[t] = 0.4 + 0.4 * hep_sev + np.random.randn() * 0.02

                # Psychological: anxiety rises, will drops
                anxiety[t] = 0.3 + 0.4 * severity * 0.5 + np.random.randn() * 0.03
                agitation[t] = 0.2 + 0.3 * severity * 0.5 + np.random.randn() * 0.03
                sleep[t] = 0.5 - 0.3 * severity * 0.5 + np.random.randn() * 0.03
                will[t] = 0.6 - 0.2 * severity * 0.3 + np.random.randn() * 0.03

                # Family stress tracks patient severity
                family_stress[t] = 0.3 + 0.5 * severity * 0.6 + np.random.randn() * 0.03

        elif scenario == 'resp_failure':
            for t in range(onset, n_hours):
                dt = t - onset
                severity = min(1.0, dt / 20.0)

                # Respiratory first
                spo2[t] = 0.95 - 0.25 * severity + np.random.randn() * 0.01
                rr[t] = 0.4 + 0.5 * severity + np.random.randn() * 0.02
                etco2[t] = 0.5 + 0.3 * severity + np.random.randn() * 0.02
                wob[t] = 0.3 + 0.6 * severity + np.random.randn() * 0.02

                # Cardiovascular compensation (4h delay)
                if dt > 4:
                    cv_sev = min(1.0, (dt - 4) / 16.0)
                    hr[t] = 0.5 + 0.3 * cv_sev + np.random.randn() * 0.02
                    bp[t] = 0.6 + 0.1 * cv_sev + np.random.randn() * 0.02

                # Neurological from hypoxia (8h delay)
                if dt > 8:
                    neuro_sev = min(1.0, (dt - 8) / 14.0)
                    gcs[t] = 0.9 - 0.3 * neuro_sev + np.random.randn() * 0.02

                # Sedation for intubated patients
                if dt > 6:
                    sedation[t] = 0.3 + 0.4 * min(1.0, (dt-6)/10) + np.random.randn() * 0.02
                    pain[t] = 0.3 + 0.2 * min(1.0, (dt-6)/10) + np.random.randn() * 0.02

                anxiety[t] = 0.3 + 0.5 * severity * 0.5 + np.random.randn() * 0.03
                will[t] = 0.6 - 0.1 * severity * 0.3 + np.random.randn() * 0.03
                family_stress[t] = 0.3 + 0.4 * severity * 0.5 + np.random.randn() * 0.03

        else:  # stable/improving
            for t in range(n_hours):
                improvement = min(1.0, t / 36.0)
                # Gradual normalization with noise
                hr[t] = 0.6 - 0.1 * improvement + np.random.randn() * 0.02
                bp[t] = 0.55 + 0.05 * improvement + np.random.randn() * 0.02
                spo2[t] = 0.93 + 0.04 * improvement + np.random.randn() * 0.005
                temp[t] = 0.55 - 0.05 * improvement + np.random.randn() * 0.01
                anxiety[t] = 0.4 - 0.15 * improvement + np.random.randn() * 0.03
                will[t] = 0.5 + 0.2 * improvement + np.random.randn() * 0.03

        # Clip all to [0, 1]
        for arr in [hr, bp, co, pvr, spo2, rr, etco2, wob, uo, cr, lytes,
                    gcs, sedation, pain, delirium, liver, coag, glucose, lactate,
                    temp, wbc, procal, anxiety, sleep, agitation, will, family_stress]:
            np.clip(arr, 0, 1, out=arr)

        # Aggregate features at each timestep
        # For the canvas model we feed the LAST timestep's values (snapshot)
        # but the temporal dynamics are in the targets
        t_last = n_hours - 1
        features = {
            'cardiovascular': np.array([hr[t_last], bp[t_last], co[t_last], pvr[t_last],
                                         hr[t_last], bp[t_last], co[t_last], pvr[t_last],
                                         hr[t_last], bp[t_last]], dtype=np.float32),
            'respiratory': np.array([spo2[t_last], rr[t_last], etco2[t_last], wob[t_last],
                                      spo2[t_last], rr[t_last], etco2[t_last], wob[t_last]], dtype=np.float32),
            'renal': np.array([uo[t_last], cr[t_last], lytes[t_last], uo[t_last],
                                cr[t_last], lytes[t_last], lytes[t_last]], dtype=np.float32),
            'neurological': np.array([gcs[t_last], sedation[t_last], pain[t_last], delirium[t_last],
                                       gcs[t_last], sedation[t_last], pain[t_last], delirium[t_last],
                                       gcs[t_last], delirium[t_last]], dtype=np.float32),
            'hepatic': np.array([liver[t_last], coag[t_last], liver[t_last], coag[t_last],
                                  liver[t_last], coag[t_last], coag[t_last]], dtype=np.float32),
            'metabolic': np.array([glucose[t_last], lactate[t_last], temp[t_last],
                                    glucose[t_last], lactate[t_last]], dtype=np.float32),
            'infection': np.array([wbc[t_last], procal[t_last], wbc[t_last], procal[t_last],
                                    0.0, 0.0, 0.0, 0.0,  # culture results (context)
                                    0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # antibiotic coverage
            'psychological': np.array([anxiety[t_last], sleep[t_last], agitation[t_last], will[t_last],
                                        anxiety[t_last], sleep[t_last], agitation[t_last], will[t_last]], dtype=np.float32),
            'social': np.array([0.5, 0.5, family_stress[t_last], family_stress[t_last]], dtype=np.float32),
        }

        # Context
        context = np.random.randn(20).astype(np.float32) * 0.3  # demographics + admission + meds

        # Targets
        # Deterioration: severity at last timestep
        if scenario == 'sepsis':
            dt_from_onset = max(0, n_hours - 1 - onset)
            det_risk = min(1.0, dt_from_onset / 18.0)
        elif scenario == 'resp_failure':
            dt_from_onset = max(0, n_hours - 1 - onset)
            det_risk = min(0.8, dt_from_onset / 20.0)
        else:
            det_risk = max(0, 0.2 - 0.15 * min(1.0, (n_hours - 1) / 36.0))

        deterioration = np.full(8, det_risk, dtype=np.float32)

        # Organ failure risk (6 systems)
        organ_risks = np.zeros(6, dtype=np.float32)
        if scenario == 'sepsis':
            dt_from_onset = max(0, n_hours - 1 - onset)
            organ_risks[0] = min(1.0, max(0, dt_from_onset - 2) / 16.0)   # cardiovascular
            organ_risks[1] = min(1.0, max(0, dt_from_onset - 8) / 14.0)   # respiratory
            organ_risks[2] = min(1.0, max(0, dt_from_onset - 6) / 14.0)   # renal
            organ_risks[3] = min(1.0, max(0, dt_from_onset - 10) / 12.0)  # neurological
            organ_risks[4] = min(1.0, max(0, dt_from_onset - 12) / 12.0)  # hepatic
            organ_risks[5] = min(1.0, max(0, dt_from_onset - 4) / 15.0)   # metabolic
        elif scenario == 'resp_failure':
            dt_from_onset = max(0, n_hours - 1 - onset)
            organ_risks[0] = min(0.5, max(0, dt_from_onset - 4) / 16.0)
            organ_risks[1] = min(1.0, dt_from_onset / 20.0)
            organ_risks[3] = min(0.5, max(0, dt_from_onset - 8) / 14.0)

        organ_risks += np.random.randn(6).astype(np.float32) * 0.03
        organ_risks = np.clip(organ_risks, 0, 1)

        # Intervention (8 dims: fluid resus, vasopressor, antibiotic, ventilator,
        #               sedation, renal replacement, nutrition, mobility)
        intervention = np.zeros(8, dtype=np.float32)
        if scenario == 'sepsis':
            intervention[0] = min(1.0, det_risk)       # fluids
            intervention[1] = min(1.0, det_risk * 0.8)  # vasopressors
            intervention[2] = 1.0                        # antibiotics always
        elif scenario == 'resp_failure':
            intervention[3] = min(1.0, det_risk)       # ventilator
            intervention[4] = min(1.0, det_risk * 0.5)  # sedation
        intervention += np.random.randn(8).astype(np.float32) * 0.05
        intervention = np.clip(intervention, 0, 1)

        all_features.append(features)
        all_deterioration.append(deterioration)
        all_organ_failure.append(organ_risks)
        all_intervention.append(intervention)
        all_context.append(context)

    # Pack into tensors
    def pack_organ(key, dim):
        return torch.tensor(np.array([f[key][:dim] for f in all_features]), dtype=torch.float32)

    return {
        'cv': pack_organ('cardiovascular', 10),
        'resp': pack_organ('respiratory', 8),
        'renal': pack_organ('renal', 7),
        'neuro': pack_organ('neurological', 10),
        'hepatic': pack_organ('hepatic', 7),
        'metabolic': pack_organ('metabolic', 5),
        'infection': pack_organ('infection', 12),
        'psych': pack_organ('psychological', 8),
        'social': pack_organ('social', 4),
        'context': torch.tensor(np.array(all_context)),
        'deterioration': torch.tensor(np.array(all_deterioration)),
        'organ_failure': torch.tensor(np.array(all_organ_failure)),
        'intervention': torch.tensor(np.array(all_intervention)),
    }


print("Simulating ICU stays...")
train_data = simulate_icu_stay(2048)
val_data = simulate_icu_stay(512)
print(f"  Deterioration risk range: "
      f"[{train_data['deterioration'][:, 0].min():.2f}, "
      f"{train_data['deterioration'][:, 0].max():.2f}]")


# ── 3. Model ────────────────────────────────────────────────────────

class ICUModel(nn.Module):
    """Canvas model with organ-system hierarchy."""

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

        # Input projections per organ system
        organ_map = {
            'cv': ('cardiovascular', 10),
            'resp': ('respiratory', 8),
            'renal': ('renal', 7),
            'neuro': ('neurological', 10),
            'hepatic': ('hepatic', 7),
            'metabolic': ('metabolic', 5),
            'infection': ('infection', 12),
            'psych': ('psychological', 8),
            'social': ('social', 4),
        }

        self.organ_projs = nn.ModuleDict()
        self.organ_sizes = {}
        for key, (prefix, in_dim) in organ_map.items():
            # Count total positions for this organ system
            total_pos = 0
            for name in bound.field_names:
                if name.startswith(prefix + '.') or name == prefix:
                    total_pos += len(bound.layout.region_indices(name))
            if total_pos == 0:
                continue
            self.organ_projs[key] = nn.Linear(in_dim, total_pos * d)
            self.organ_sizes[key] = (prefix, total_pos)

        # Context projection (demographics + admission + meds)
        ctx_fields = ['demographics', 'admission_diagnosis', 'medications']
        ctx_total = sum(len(bound.layout.region_indices(f)) for f in ctx_fields
                        if f in bound.field_names)
        self.ctx_proj = nn.Linear(20, ctx_total * d)
        self.ctx_size = ctx_total

        # Output heads
        det_n = len(bound.layout.region_indices('deterioration_risk'))
        org_n = len(bound.layout.region_indices('organ_failure_risk'))
        int_n = len(bound.layout.region_indices('recommended_intervention'))

        self.det_head = nn.Linear(det_n * d, 8)
        self.org_head = nn.Linear(org_n * d, 6)
        self.int_head = nn.Linear(int_n * d, 8)

        self._det_n = det_n
        self._org_n = org_n
        self._int_n = int_n

    def forward(self, data):
        B = data['cv'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        # Place organ system features
        for key, (prefix, total_pos) in self.organ_sizes.items():
            proj = self.organ_projs[key]
            emb = proj(data[key]).reshape(B, total_pos, self.d)

            # Scatter into all fields of this organ system
            all_idx = []
            for name in self.bound.field_names:
                if name.startswith(prefix + '.') or name == prefix:
                    all_idx.extend(self.bound.layout.region_indices(name))
            all_idx = all_idx[:total_pos]  # safety
            canvas[:, all_idx] = canvas[:, all_idx] + emb[:, :len(all_idx)]

        # Place context
        ctx_idx = []
        for f in ['demographics', 'admission_diagnosis', 'medications']:
            if f in self.bound.field_names:
                ctx_idx.extend(self.bound.layout.region_indices(f))
        if ctx_idx:
            ctx_emb = self.ctx_proj(data['context']).reshape(B, self.ctx_size, self.d)
            canvas[:, ctx_idx[:self.ctx_size]] = canvas[:, ctx_idx[:self.ctx_size]] + ctx_emb

        canvas = self.encoder(canvas, mask=self.mask)

        # Read outputs
        det_idx = self.bound.layout.region_indices('deterioration_risk')
        org_idx = self.bound.layout.region_indices('organ_failure_risk')
        int_idx = self.bound.layout.region_indices('recommended_intervention')

        det = torch.sigmoid(self.det_head(canvas[:, det_idx].reshape(B, -1)))
        org = torch.sigmoid(self.org_head(canvas[:, org_idx].reshape(B, -1)))
        interv = torch.sigmoid(self.int_head(canvas[:, int_idx].reshape(B, -1)))

        return {'deterioration': det, 'organ_failure': org, 'intervention': interv}


class FlatICUModel(nn.Module):
    """Flat baseline: all features concatenated."""

    def __init__(self, d=192):
        super().__init__()
        # Total input: 10+8+7+10+7+5+12+8+4+20 = 91
        self.net = nn.Sequential(
            nn.Linear(91, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
        )
        self.det_head = nn.Linear(d, 8)
        self.org_head = nn.Linear(d, 6)
        self.int_head = nn.Linear(d, 8)

    def forward(self, data):
        x = torch.cat([data['cv'], data['resp'], data['renal'], data['neuro'],
                        data['hepatic'], data['metabolic'], data['infection'],
                        data['psych'], data['social'], data['context']], dim=-1)
        h = self.net(x)
        return {
            'deterioration': torch.sigmoid(self.det_head(h)),
            'organ_failure': torch.sigmoid(self.org_head(h)),
            'intervention': torch.sigmoid(self.int_head(h)),
        }


# ── 4. Training ──────────────────────────────────────────────────────

def causal_consistency_loss(data, out):
    """Enforce physiological causal relationships.

    1. High lactate should correlate with high deterioration
    2. Low BP should correlate with cardiovascular failure risk
    3. Organ failure risks should explain deterioration risk
    """
    # Lactate -> deterioration consistency
    lactate = data['metabolic'][:, 1]  # lactate value
    det_mean = out['deterioration'].mean(dim=-1)
    lact_consist = F.mse_loss(det_mean, lactate.clamp(0, 1).detach()) * 0.5

    # Organ failures should explain deterioration
    org_max = out['organ_failure'].max(dim=-1).values
    det_consist = F.mse_loss(det_mean, org_max.detach()) * 0.3

    return lact_consist + det_consist


def train_icu(model, label, use_consistency=False, n_epochs=300, bs=128):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []

    for ep in range(n_epochs):
        idx = torch.randint(0, len(train_data['cv']), (bs,))
        batch = {k: v[idx] if isinstance(v, torch.Tensor) else v for k, v in train_data.items()}

        out = model(batch)

        det_loss = F.mse_loss(out['deterioration'], batch['deterioration'])
        org_loss = F.mse_loss(out['organ_failure'], batch['organ_failure'])
        int_loss = F.mse_loss(out['intervention'], batch['intervention'])

        loss = 8.0 * det_loss + 5.0 * org_loss + 3.0 * int_loss

        if use_consistency:
            loss = loss + causal_consistency_loss(batch, out)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if ep % 200 == 0:
            print(f"  [{label}] ep {ep:3d}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out_val = model(val_data)
        val_det = F.mse_loss(out_val['deterioration'], val_data['deterioration']).item()
        val_org = F.mse_loss(out_val['organ_failure'], val_data['organ_failure']).item()
    print(f"  [{label}] val_det={val_det:.4f}, val_org={val_org:.4f}")
    return model, losses, val_det, val_org


print("\nTraining flat baseline...")
flat_model = FlatICUModel()
flat_model, flat_losses, flat_det, flat_org = train_icu(flat_model, "flat")

print("Training canvas model...")
canvas_model = ICUModel(bound)
canvas_model, canvas_losses, canvas_det, canvas_org = train_icu(
    canvas_model, "canvas", use_consistency=False)

print("Training canvas + consistency...")
consist_model = ICUModel(bound)
consist_model, consist_losses, consist_det, consist_org = train_icu(
    consist_model, "canvas+consist", use_consistency=True)


# ── 5. Visualization: 3x3 analysis figure ───────────────────────────

fig = plt.figure(figsize=(18, 16), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('ICU Patient: Whole-Person Physiological Model',
             fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

CF, CC, CCO = '#95A5A6', '#4A90D9', '#E74C3C'  # flat, canvas, canvas+consist

ORGAN_COLORS = {
    'cardiovascular': '#E74C3C', 'respiratory': '#3498DB',
    'renal': '#F39C12', 'neurological': '#9B59B6',
    'hepatic': '#2ECC71', 'metabolic': '#E67E22',
    'infection': '#1ABC9C', 'psychological': '#E91E63',
    'social': '#795548',
}

# (a) Type hierarchy tree
ax = fig.add_subplot(gs[0, 0])
ax.set_title('Organ System Hierarchy', fontsize=12, fontweight='bold')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.axis('off')

# Draw tree
systems = list(ORGAN_COLORS.keys())
n_sys = len(systems)
# Patient node at top
ax.text(0.5, 0.95, 'Patient', ha='center', va='center', fontsize=10,
        fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#2C3E50', alpha=0.9),
        color='white')

# Organ systems as children
for i, (name, color) in enumerate(ORGAN_COLORS.items()):
    x = (i + 0.5) / n_sys
    y = 0.55
    ax.plot([0.5, x], [0.88, y + 0.08], '-', color='#BDC3C7', lw=1)
    short = name[:5]
    ax.text(x, y, short, ha='center', va='center', fontsize=6,
            fontweight='bold', rotation=45 if n_sys > 7 else 0,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8),
            color='white')

# Prediction nodes at bottom
pred_names = ['deterioration', 'organ_fail', 'intervention']
pred_colors = ['#E74C3C', '#F39C12', '#3498DB']
for i, (name, color) in enumerate(zip(pred_names, pred_colors)):
    x = 0.2 + i * 0.3
    ax.plot([0.5, x], [0.88, 0.2], '--', color=color, lw=1, alpha=0.5)
    ax.text(x, 0.12, name, ha='center', va='center', fontsize=7,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7),
            color='white')

# (b) Canvas layout
ax = fig.add_subplot(gs[0, 1])
ax.set_title('Canvas Layout (32x32)', fontsize=12, fontweight='bold')
H, W = bound.layout.H, bound.layout.W
grid = np.ones((H, W, 3)) * 0.95
for name, bf in bound.fields.items():
    # Find organ system
    parts = name.split('.')
    color_key = parts[0] if parts[0] in ORGAN_COLORS else None
    if color_key:
        color = ORGAN_COLORS[color_key]
    elif 'deterioration' in name:
        color = '#E74C3C'
    elif 'organ_failure' in name:
        color = '#F39C12'
    elif 'intervention' in name:
        color = '#3498DB'
    else:
        color = '#BDC3C7'

    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]

ax.imshow(grid, aspect='equal', interpolation='nearest')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=n[:8]) for n, c in ORGAN_COLORS.items()]
legend_elements.append(Patch(facecolor='#E74C3C', label='det_risk'))
legend_elements.append(Patch(facecolor='#F39C12', label='org_fail'))
ax.legend(handles=legend_elements, fontsize=5, loc='lower right', ncol=2)
ax.set_xlabel('W'); ax.set_ylabel('H')

# (c) Sepsis cascade time series (simulated)
ax = fig.add_subplot(gs[0, 2])
ax.set_title('Sepsis Cascade (simulated)', fontsize=12, fontweight='bold')
hours = np.arange(N_TIMESTEPS)
onset = 8

# Simulate one sepsis case for visualization
wbc_vis = np.ones(N_TIMESTEPS) * 0.4
hr_vis = np.ones(N_TIMESTEPS) * 0.5
bp_vis = np.ones(N_TIMESTEPS) * 0.6
cr_vis = np.ones(N_TIMESTEPS) * 0.3
lactate_vis = np.ones(N_TIMESTEPS) * 0.2

for t in range(onset, N_TIMESTEPS):
    dt = t - onset
    sev = min(1.0, dt / 18.0)
    wbc_vis[t] = 0.4 + 0.5 * sev
    if dt > 2: hr_vis[t] = 0.5 + 0.4 * min(1.0, (dt-2)/16)
    if dt > 2: bp_vis[t] = 0.6 - 0.35 * min(1.0, (dt-2)/16)
    if dt > 6: cr_vis[t] = 0.3 + 0.6 * min(1.0, (dt-6)/14)
    if dt > 4: lactate_vis[t] = 0.2 + 0.7 * min(1.0, (dt-4)/15)

cascade_data = [
    ('WBC', wbc_vis, ORGAN_COLORS['infection']),
    ('HR', hr_vis, ORGAN_COLORS['cardiovascular']),
    ('BP', bp_vis, ORGAN_COLORS['cardiovascular']),
    ('Creatinine', cr_vis, ORGAN_COLORS['renal']),
    ('Lactate', lactate_vis, ORGAN_COLORS['metabolic']),
]
for name, vals, color in cascade_data:
    ax.plot(hours, vals, '-', color=color, lw=2, label=name)
ax.axvline(x=onset, color='#E74C3C', ls='--', lw=1, alpha=0.5, label='onset')
ax.legend(fontsize=7, ncol=2)
ax.set_xlabel('Hours')
ax.set_ylabel('Normalized Value')
ax.grid(True, alpha=0.2)

# (d) Training curves
ax = fig.add_subplot(gs[1, 0])
ax.set_title('Training Loss', fontsize=12, fontweight='bold')
w = 30
def smooth(a, w=w): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.plot(smooth(flat_losses), color=CF, lw=1.5, label='flat')
ax.plot(smooth(canvas_losses), color=CC, lw=1.5, label='canvas')
ax.plot(smooth(consist_losses), color=CCO, lw=1.5, label='canvas+consist.')
ax.legend(fontsize=8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.grid(True, alpha=0.2)

# (e) Deterioration prediction comparison
ax = fig.add_subplot(gs[1, 1])
ax.set_title('Deterioration Risk: Predicted vs True', fontsize=12, fontweight='bold')
consist_model.eval(); flat_model.eval()
with torch.no_grad():
    pred_canvas = consist_model(val_data)['deterioration'][:, 0].numpy()
    pred_flat = flat_model(val_data)['deterioration'][:, 0].numpy()
true_det = val_data['deterioration'][:, 0].numpy()
ax.scatter(true_det, pred_canvas, s=5, alpha=0.3, color=CCO, label=f'canvas+c (MSE={consist_det:.3f})')
ax.scatter(true_det, pred_flat, s=5, alpha=0.3, color=CF, label=f'flat (MSE={flat_det:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
ax.set_xlabel('True Risk')
ax.set_ylabel('Predicted Risk')
ax.legend(fontsize=8, markerscale=3)
ax.grid(True, alpha=0.2)

# (f) Per-organ failure risk (stacked bar for one sample)
ax = fig.add_subplot(gs[1, 2])
ax.set_title('Organ Failure Risk (high-risk patient)', fontsize=12, fontweight='bold')
# Find a high-deterioration patient
high_risk_idx = val_data['deterioration'][:, 0].argmax().item()
consist_model.eval()
with torch.no_grad():
    sample = {k: v[high_risk_idx:high_risk_idx+1] for k, v in val_data.items() if isinstance(v, torch.Tensor)}
    out_sample = consist_model(sample)

org_names = ['Cardio', 'Resp', 'Renal', 'Neuro', 'Hepatic', 'Metabolic']
true_org = val_data['organ_failure'][high_risk_idx].numpy()
pred_org = out_sample['organ_failure'][0].numpy()
x_pos = np.arange(len(org_names))
width = 0.35
ax.bar(x_pos - width/2, true_org, width, label='True', color='#2C3E50', alpha=0.7)
ax.bar(x_pos + width/2, pred_org, width, label='Predicted',
       color=[list(ORGAN_COLORS.values())[i] for i in range(6)], alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(org_names, fontsize=8, rotation=30)
ax.set_ylabel('Failure Risk')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# (g) Loss weight budget
ax = fig.add_subplot(gs[2, 0])
ax.set_title('Loss Weight Budget', fontsize=12, fontweight='bold')
weights = bound.layout.loss_weight_mask("cpu")
total_w = weights.sum().item()
categories = {}
for name, bf in bound.fields.items():
    indices = bf.indices()
    w = sum(weights[i].item() for i in indices)
    if w == 0:
        continue
    parts = name.split('.')
    cat = parts[0] if parts[0] in ORGAN_COLORS else name.split('_')[0]
    categories[cat] = categories.get(cat, 0) + w

cats_sorted = sorted(categories.items(), key=lambda x: -x[1])
cat_names = [c[0][:10] for c in cats_sorted]
cat_vals = [c[1] / total_w * 100 for c in cats_sorted]
cat_colors = [ORGAN_COLORS.get(c[0], '#BDC3C7') for c in cats_sorted]
bars = ax.barh(range(len(cat_names)), cat_vals, color=cat_colors, alpha=0.8)
ax.set_yticks(range(len(cat_names)))
ax.set_yticklabels(cat_names, fontsize=8)
ax.set_xlabel('% of Total Loss Weight')
ax.grid(True, alpha=0.2, axis='x')
for bar, val in zip(bars, cat_vals):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=7)

# (h) Psychological state trajectory
ax = fig.add_subplot(gs[2, 1])
ax.set_title('Psychological Trajectory (sepsis sim)', fontsize=12, fontweight='bold')

# Resimulate one case for psychological visualization
np.random.seed(123)
anx_vis = np.ones(N_TIMESTEPS) * 0.3
sleep_vis = np.ones(N_TIMESTEPS) * 0.5
agit_vis = np.ones(N_TIMESTEPS) * 0.2
will_vis = np.ones(N_TIMESTEPS) * 0.6
fam_vis = np.ones(N_TIMESTEPS) * 0.3
onset_p = 10

for t in range(onset_p, N_TIMESTEPS):
    dt = t - onset_p
    sev = min(1.0, dt / 18.0)
    anx_vis[t] = 0.3 + 0.4 * sev * 0.5
    sleep_vis[t] = 0.5 - 0.3 * sev * 0.5
    agit_vis[t] = 0.2 + 0.3 * sev * 0.5
    will_vis[t] = 0.6 - 0.2 * sev * 0.3
    # Family visits at hours 14, 22, 30, 38, 46
    if t in [14, 22, 30, 38, 46]:
        anx_vis[t] -= 0.1  # family presence reduces anxiety
        will_vis[t] += 0.05
    fam_vis[t] = 0.3 + 0.5 * sev * 0.6

psych_data = [
    ('Anxiety', anx_vis, '#E91E63'),
    ('Sleep quality', sleep_vis, '#673AB7'),
    ('Agitation', agit_vis, '#FF5722'),
    ('Will to recover', will_vis, '#4CAF50'),
    ('Family stress', fam_vis, '#795548'),
]
for name, vals, color in psych_data:
    ax.plot(hours, vals, '-', color=color, lw=1.5, label=name)

# Mark family visits
for t in [14, 22, 30, 38, 46]:
    if t < N_TIMESTEPS:
        ax.axvline(x=t, color='#795548', ls=':', lw=0.8, alpha=0.4)
ax.legend(fontsize=6, ncol=2)
ax.set_xlabel('Hours')
ax.set_ylabel('Normalized Value')
ax.grid(True, alpha=0.2)

# (i) Model comparison summary
ax = fig.add_subplot(gs[2, 2])
ax.set_title('Model Comparison Summary', fontsize=12, fontweight='bold')
models_summary = [
    ('Flat', flat_det, flat_org),
    ('Canvas', canvas_det, canvas_org),
    ('Canvas+\nConsist.', consist_det, consist_org),
]
x_pos = np.arange(len(models_summary))
det_vals = [m[1] for m in models_summary]
org_vals = [m[2] for m in models_summary]
width = 0.35
ax.bar(x_pos - width/2, det_vals, width, label='Deterioration MSE',
       color=[CF, CC, CCO], alpha=0.7)
ax.bar(x_pos + width/2, org_vals, width, label='Organ Failure MSE',
       color=[CF, CC, CCO], alpha=0.4, edgecolor=[CF, CC, CCO], linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels([m[0] for m in models_summary], fontsize=9)
ax.legend(fontsize=8)
ax.set_ylabel('MSE')
ax.grid(True, alpha=0.2, axis='y')

# Annotate best
best_det = min(range(len(det_vals)), key=lambda i: det_vals[i])
ax.annotate(f'{det_vals[best_det]:.4f}',
            (best_det - width/2, det_vals[best_det]),
            textcoords="offset points", xytext=(0, 5),
            ha='center', fontsize=8, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.97])
path = os.path.join(ASSETS, "07_icu_patient.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\nSaved {path}")


# ── 6. Animation: patient monitor ───────────────────────────────────

print("Generating patient monitor animation...")

# Simulate a full sepsis case for the animation
np.random.seed(42)
onset_anim = 10
hr_anim = np.ones(N_TIMESTEPS) * 75
bp_sys = np.ones(N_TIMESTEPS) * 120
bp_dia = np.ones(N_TIMESTEPS) * 80
spo2_anim = np.ones(N_TIMESTEPS) * 97
rr_anim = np.ones(N_TIMESTEPS) * 16
temp_anim = np.ones(N_TIMESTEPS) * 37.0
lactate_anim = np.ones(N_TIMESTEPS) * 1.0
cr_anim = np.ones(N_TIMESTEPS) * 0.9
wbc_anim = np.ones(N_TIMESTEPS) * 8.0
gcs_anim = np.ones(N_TIMESTEPS) * 15

for t in range(onset_anim, N_TIMESTEPS):
    dt = t - onset_anim
    sev = min(1.0, dt / 20.0)

    wbc_anim[t] = 8.0 + 14.0 * sev + np.random.randn() * 0.5
    temp_anim[t] = 37.0 + 3.0 * sev + np.random.randn() * 0.2

    if dt > 2:
        cv = min(1.0, (dt-2)/16)
        hr_anim[t] = 75 + 55 * cv + np.random.randn() * 2
        bp_sys[t] = 120 - 50 * cv + np.random.randn() * 3
        bp_dia[t] = 80 - 30 * cv + np.random.randn() * 2
    if dt > 4:
        lactate_anim[t] = 1.0 + 8.0 * min(1.0, (dt-4)/15) + np.random.randn() * 0.3
    if dt > 6:
        cr_anim[t] = 0.9 + 3.0 * min(1.0, (dt-6)/14) + np.random.randn() * 0.1
    if dt > 8:
        spo2_anim[t] = 97 - 12 * min(1.0, (dt-8)/14) + np.random.randn() * 0.5
        rr_anim[t] = 16 + 14 * min(1.0, (dt-8)/14) + np.random.randn() * 0.5
    if dt > 10:
        gcs_anim[t] = 15 - 6 * min(1.0, (dt-10)/12) + np.random.randn() * 0.3

fig_anim, ax_anim = plt.subplots(4, 2, figsize=(12, 8), dpi=80)
fig_anim.patch.set_facecolor('#0a0a1a')
fig_anim.suptitle('', fontsize=14, color='#44FF44', fontfamily='monospace')

vital_config = [
    (0, 0, 'HR', hr_anim, '#FF4444', 'bpm', (40, 160)),
    (0, 1, 'BP', bp_sys, '#FF8844', 'mmHg', (50, 180)),
    (1, 0, 'SpO2', spo2_anim, '#44AAFF', '%', (80, 100)),
    (1, 1, 'RR', rr_anim, '#44FF44', '/min', (8, 40)),
    (2, 0, 'Temp', temp_anim, '#FFAA44', '°C', (35, 41)),
    (2, 1, 'Lactate', lactate_anim, '#FF44FF', 'mmol/L', (0, 12)),
    (3, 0, 'Creatinine', cr_anim, '#FFFF44', 'mg/dL', (0, 5)),
    (3, 1, 'GCS', gcs_anim, '#44FFAA', '', (3, 15)),
]

def animate_monitor(frame):
    fig_anim.suptitle(f'ICU Patient Monitor — Hour {frame}',
                      fontsize=14, color='#44FF44', fontfamily='monospace')
    for row, col, name, data, color, unit, ylim in vital_config:
        ax = ax_anim[row, col]
        ax.clear()
        ax.set_facecolor('#0a0a1a')

        t_range = slice(max(0, frame - 12), frame + 1)
        t_vals = np.arange(max(0, frame - 12), frame + 1)
        d_vals = data[t_range]

        ax.plot(t_vals, d_vals, '-', color=color, lw=2)
        ax.fill_between(t_vals, d_vals, ylim[0], color=color, alpha=0.1)

        # Current value
        current = data[frame]
        ax.text(0.98, 0.95, f'{current:.1f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=16, fontweight='bold',
                color=color, fontfamily='monospace')
        ax.text(0.98, 0.75, unit, transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color=color, alpha=0.6,
                fontfamily='monospace')
        ax.text(0.02, 0.95, name, transform=ax.transAxes,
                ha='left', va='top', fontsize=10, color=color, alpha=0.8,
                fontfamily='monospace')

        ax.set_ylim(ylim)
        ax.set_xlim(max(0, frame - 12), max(12, frame + 1))
        ax.tick_params(colors='#333333', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#222222')
        ax.grid(True, color='#1a1a2e', alpha=0.3)

anim = animation.FuncAnimation(fig_anim, animate_monitor,
                                frames=N_TIMESTEPS, interval=500)
gif_path = os.path.join(ASSETS, "07_icu_patient.gif")
anim.save(gif_path, writer='pillow', fps=3)
plt.close()
print(f"Saved {gif_path}")

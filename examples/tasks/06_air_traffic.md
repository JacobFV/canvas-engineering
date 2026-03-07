# 06: Air Traffic Conflict Detection

## Purpose
Safety-critical multi-agent. Show that loss_weight=10 on conflict detection
actually produces a model that catches more conflicts, and that dense
inter-aircraft connectivity is essential (unlike the vehicle fleet where
ring sufficed).

## Data
Synthetic TRACON scenarios:
- N aircraft (6-12) with 3D positions, headings, speeds, altitude rates
- Trajectories: straight-line + random perturbations + altitude changes
- Conflicts: defined as separation < 3nm horizontal + 1000ft vertical
  within next 60 seconds
- Generate mix of conflict and non-conflict scenarios

## Type hierarchy
```
Aircraft:
  state: Field(1, 6)                 — x, y, z, hdg, spd, vrate
  flight_plan: Field(2, 4, is_output=False) — route context
  trajectory: Field(4, 4, loss_weight=3.0)  — predicted 60s
  conflict: Field(1, 2, loss_weight=10.0)   — conflict flag + time-to-conflict

TRACON:
  weather: Field(2, 2, is_output=False)
  sector_load: Field(1, 2)
  aircraft: list[Aircraft]
```

## Training
- Trajectory prediction: SFT on true future positions
- Conflict detection: binary classification with focal loss (class imbalance)
- **Counterfactual training**: for each real conflict, generate a "what if
  aircraft X turned 10 degrees" — does the conflict resolve? Train the
  model to predict this counterfactual too. This is the feedback dynamic:
  the conflict field must be sensitive to small state changes.
- Compare: loss_weight=1 vs loss_weight=10 on conflict, isolated vs dense aircraft

## Visualization
`assets/examples/06_atc.png`:
(a) Bird's-eye TRACON view with aircraft, trajectories, conflict pairs
(b) ROC curves: conflict detection at different loss weights
(c) Confusion matrix: TP/FP/FN/TN for conflict detection
(d) Sensitivity analysis: how conflict probability changes as aircraft
    separation decreases (should be steep sigmoid)
(e) Attention heatmap: which aircraft pairs attend most strongly?

## Key message
"10x loss weight on conflict detection isn't a tuning hack — it's a
declaration of priorities. The model allocates more representational
capacity to the thing you told it matters most. Counterfactual training
ensures the conflict field is causally grounded, not just correlated."

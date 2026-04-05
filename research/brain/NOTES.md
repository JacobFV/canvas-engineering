# Brain Dynamics Modeling — Research Notes

April 2026. What we learned from building a cortical dynamics predictor with canvas-engineering + TRIBE v2.

## What worked

**Dynamics prediction was the right task.** Classification (which category?) was trivially solvable by an MLP. Predicting what happens next in the brain requires routing activation through pathways — exactly what the cortical topology declares.

**135 features was the right dimensionality.** At 23 scalars (1 mean per ROI), the space is too low-dimensional — any model memorizes it. At 135 features (8 subsampled vertices per ROI), the topology becomes valuable because the model needs guidance on which features to route where. The cortical model achieved R²=0.825 with only 19.6% of possible connections.

**Sparse connectivity was a computational advantage.** The cortical model (3,579 connections) trained 5× faster than dense (18,225 connections) because our dispatcher iterates sequentially over connections. This mirrors the brain — sparse wiring is more energy-efficient. Dense kept timing out on Modal; cortical finished every time.

**The architecture is genuinely general.** Same canvas-engineering code ran brain dynamics, BCI decoding (68.8% vs SVM 59.4%), robot fleet control, and browser agents. Not a neuroscience-specific tool.

## What didn't work

**Classification as a benchmark.** A 28K-param flat MLP beat our 150K-param cortical model on category classification. The task was too easy — it didn't exercise the declared pathways at all.

**fMRI temporal resolution for wave visualization.** TRIBE v2 operates at ~1-2 seconds per timestep. The V1→fusiform cascade happens in 100-200ms. Our animations show the steady state, not the propagation wave. All layers look similar because by the time fMRI measures, activation has already spread everywhere.

**Dense model completion at 135 features.** 18,225 connections × 200 epochs × sequential dispatch = too slow for CPU within any reasonable timeout. We never got the definitive dense-vs-cortical comparison at 135 features. Dense was at R²=0.826 at epoch 100 when it died — tracking close to cortical.

## Key insight

**Topology is a developmental prior, not a capacity constraint.** The cortical model reaches R²>0.76 by epoch 60. Dense gets there too, just later. Given enough training, a fully connected network finds the same routing. The cortical wiring diagram is like the brain's genetic blueprint — it accelerates learning but doesn't set the ceiling. The value is sample efficiency and convergence speed.

This is actually consistent with neuroscience: the brain's gross connectivity is genetically specified (topology), but synaptic weights are learned (attention parameters). Nature provides the wiring diagram; nurture tunes the weights.

## Numbers

```
Cortical 135-feature:  R²=0.825  |  3,579 connections  |  19.6% density  |  149,889 params
Dense 135-feature:     R²≈0.826  |  18,225 connections  |  100% density   |  149,889 params  (epoch 100, incomplete)
Flat MLP 23-scalar:    R²=0.832  |  N/A                |  N/A            |  28,439 params

BCI TRIBE v2:          68.8% accuracy  (vs SVM 59.4%, chance 25%)

Data: 72 TRIBE v2 stimuli, 9 categories, 20,484 vertices/timestep, 135 features after ROI mapping
```

## What to do next (with 8×H100 compute)

1. **Per-vertex, not per-ROI.** Skip scalar means and low-dim subsampling. Use the full 20,484 vertex space with canvas regions that have actual spatial extent (V1 as Field(12,12) not Field(1,1)).

2. **Real fMRI, not TRIBE v2 predictions.** HCP (Human Connectome Project), NSD (Natural Scenes Dataset), BOLD5000. Real data has richer temporal dynamics and noise structure.

3. **Millisecond temporal resolution.** EEG source localization or MEG. This is where the activation wave animations become dramatic — you'd see V1 light up, then the wave ripple through the ventral stream over 100-200ms.

4. **500+ regions.** Schaefer parcellation instead of 23 Destrieux ROIs. More regions = sparser relative connectivity = bigger topology advantage.

5. **Train for days, not hours.** Let dense fully converge so we can precisely measure the convergence speed advantage of cortical topology.

6. **Multi-modal stimuli.** Text, audio, and video through TRIBE v2 or real fMRI. Cross-modal dynamics (hearing a word → activating visual imagery) would strongly exercise the cross-network pathways.

## Files

```
research/brain/
├── cortical_canvas.py              # 23 regions, 42 pathways, 288 stimuli
├── run_dynamics_pipeline.py        # 135-feature TRIBE v2 → dynamics prediction
├── train_from_saved_modal.py       # Train from saved data on Modal
├── results/
│   ├── activations_cortical.npz    # 12MB: 10 epochs × 4 layers × 10 stimuli
│   ├── checkpoint_cortical_*.pt    # 12 checkpoints spanning training
│   ├── dynamics_data.npz           # 135-feature dataset (270 train, 67 val)
│   └── tribe_data.npz             # Raw TRIBE v2 predictions (23MB)
└── report/
    └── research_report.html        # Self-contained report with 3D brains

data/modal_backup/                  # Local copy of all Modal volume data (60MB, gitignored)
presentation/
├── scientific_report.html          # Full scientific report with equations
├── animations/                     # Brain activation GIFs
└── script/recording_outline.md     # Video narration outline
```

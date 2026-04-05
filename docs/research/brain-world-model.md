# Brain World Model Research

## Overview

This research track uses canvas-engineering to build a neural architecture that mirrors real cortical wiring. Instead of discovering structure through training, we declare it: 23 brain regions from the Destrieux atlas connected by 42 known cortical pathways, trained on real cortical predictions from Facebook's TRIBE v2 brain encoding model.

## Key Results

| Metric | Value |
|--------|-------|
| Cortical dynamics prediction R² (135 features) | **0.825** |
| Connectivity density | 19.6% (3,579 / 18,225 possible connections) |
| BCI classification accuracy | 68.8% (vs 59.4% SVM, chance 25%) |
| Cortical regions | 23 (from Destrieux atlas) |
| Cortical pathways | 42 (matching known neuroscience) |
| TRIBE v2 stimuli | 72 text descriptions across 9 categories |
| Features per region | 8 (subsampled vertices, 135 total) |

## Architecture

The cortical brain model uses `compile_program()` with:

- **16 leaf regions** across 7 sub-networks (visual, auditory, language, frontal, default mode, subcortical, plus prediction error)
- **5 region families**: observation (V1, A1, somatosensory), state (association areas), memory (temporal pole), action (motor cortex), residual (prediction error)
- **42 cortical pathway connections** implementing known neuroscience:
  - Ventral visual stream: V1 → V2/V4 → Fusiform
  - Language pathway: A1 → Wernicke → Broca
  - Executive control: Prefrontal → Premotor → Motor
  - Default mode network: Precuneus ↔ Cingulate ↔ Temporal pole

## Data Pipeline

1. **Stimulus generation**: 288 vivid text descriptions across 9 categories (animal, music, danger, emotion, language, motor, social, spatial, visual)
2. **TRIBE v2 inference**: Facebook's brain encoding model predicts cortical activation (20,484 vertices on fsaverage5) per stimulus, per timestep
3. **ROI mapping**: Destrieux atlas maps vertices to 20 named brain regions
4. **Feature extraction**: 8 evenly-spaced vertices subsampled per ROI = 135 features
5. **Dynamics dataset**: sliding window of 3 timesteps → predict next timestep

## Experiments

### Experiment 1: Classification (wrong task)
- Task: classify stimulus category from mean ROI activations
- Result: flat MLP wins (57.9%) — too low-dimensional, topology doesn't help
- Lesson: classification doesn't exercise cortical pathways

### Experiment 2: Dynamics prediction (right task)
- Task: predict next-timestep regional activation
- 23 scalar features: flat MLP wins (R²=0.832) — space too simple
- 135 features (8 per ROI): cortical topology achieves R²=0.825 with 19.6% connectivity
- Key finding: topology = convergence prior, not capacity advantage

### Experiment 3: BCI decoding
- Task: classify 4 stimulus categories from virtual 20-channel EEG
- Canvas decoder: 68.8% (vs SVM 59.4%, chance 25%)
- Used real TRIBE v2 cortical predictions sampled at 10-20 electrode positions

## Files

```
research/brain/
├── cortical_canvas.py          # Type declarations + pathway connections
├── data_pipeline.py            # Synthetic + TRIBE v2 data generation
├── train.py                    # 3-model comparison training
├── evaluate.py                 # Visualization generation
├── run_tribe_pipeline.py       # Full TRIBE v2 → train → evaluate pipeline
├── run_dynamics_pipeline.py    # 135-feature dynamics prediction pipeline
├── run_dynamics_modal.py       # Modal launcher with volume persistence
├── train_from_saved_modal.py   # Train from saved data (no TRIBE inference)
├── generate_report.py          # HTML report with 3D brain renders
└── results/
    ├── tribe_data.npz              # TRIBE v2 temporal predictions (23MB)
    ├── dynamics_data.npz           # 135-feature train/val dataset (1.4MB)
    ├── activations_cortical.npz    # Activation snapshots (12MB)
    ├── checkpoint_cortical_*.pt    # 12 checkpoints across training
    ├── training_cortical.jsonl     # Per-epoch metrics
    ├── connectivity_matrix.png     # 23×23 connectivity visualization
    └── dynamics_comparison.png     # R² comparison plot
```

## Scaling Plan

| Level | Regions | Data | GPU-hours | Outcome |
|-------|---------|------|-----------|---------|
| Current | 23 | TRIBE v2 (72 stimuli) | ~10 | Proof of concept |
| Level 2 | 200+ | TRIBE v2 (1000+ stimuli) | 50-500 | Cortical simulator |
| **Level 3** | **500** | **HCP + NSD + BOLD5000** | **2,000-5,000** | **Foundation brain model** |
| Level 4 | Full brain | Individual fMRI | 10,000-100,000 | Digital brain twin |

Level 3 requires 2-3 weeks on 8×H100. The architecture and data pipeline are built — only compute is needed.

## Reproducing

```bash
# Generate TRIBE v2 data + train (Modal GPU)
modal run research/brain/run_dynamics_modal.py

# Train from saved data (no GPU needed)
modal run research/brain/train_from_saved_modal.py

# Collect results
modal run research/brain/train_from_saved_modal.py --collect-only

# Render brain animations
python presentation/render_brain_animations.py

# Generate HTML report
python research/brain/generate_report.py
```

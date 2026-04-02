"""Cortical dynamics prediction: predict how activation flows through the brain.

Instead of classifying stimulus categories (which an MLP does trivially),
this task predicts the TEMPORAL DYNAMICS of cortical activation — how
activity propagates from sensory regions through association cortex to
motor/language output regions.

The cortical topology should help here because the task literally requires
routing information through the brain's known pathways:
  V1 → V2/V4 → Fusiform (visual object recognition)
  A1 → Wernicke → Broca (auditory language comprehension → production)
  Prefrontal → Premotor → Motor (executive → planning → action)

TRIBE v2 gives per-timestep cortical predictions. We keep the temporal
dimension and train: given ROI activations at time t, predict ROI
activations at time t+1. The cortical connectivity graph declares
exactly which regions influence which.

Run: modal run --detach research/run_modal.py --track brain
"""

import sys
import os
import json
import hashlib
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.chdir(os.path.dirname(__file__))
os.makedirs("results", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
CACHE = Path("./cache")
CACHE.mkdir(exist_ok=True)


# ── TRIBE v2 loading (same as before) ───────────────────────────────

def load_tribe():
    from tribev2 import TribeModel
    print("Loading TRIBE v2...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
    for attr in ["text_feature", "audio_feature", "video_feature"]:
        feat = getattr(model.data, attr, None)
        if feat is not None and hasattr(feat, "infra"):
            feat.infra.mode = "force"
    print("TRIBE v2 loaded.")
    return model


def text_to_preds(model, text, label=""):
    import pandas as pd
    import soundfile as sf
    from gtts import gTTS
    from langdetect import detect
    from neuralset.events.transforms import (
        AddContextToWords, AddSentenceToWords, AddText,
        ChunkEvents, RemoveMissing, standardize_events,
    )

    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    audio_dir = CACHE / "brain_{}".format(text_hash)
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "audio.mp3"

    if not audio_path.exists():
        tts = gTTS(text, lang=detect(text))
        tts.save(str(audio_path))

    info = sf.info(str(audio_path))
    duration = info.duration
    words = text.split()
    word_dur = duration / max(len(words), 1)

    audio_event = {
        "type": "Audio", "filepath": str(audio_path),
        "start": 0.0, "duration": duration,
        "timeline": "default", "subject": "default",
    }
    word_events = [{
        "type": "Word", "text": w.strip('.,;:!?"\'()-'),
        "start": i * word_dur, "duration": word_dur * 0.8,
        "timeline": "default", "subject": "default",
        "language": "english", "sequence_id": 0,
    } for i, w in enumerate(words)]

    df = pd.DataFrame([audio_event] + word_events)
    transforms = [
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        AddText(),
        AddSentenceToWords(max_unmatched_ratio=0.99),
        AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
        RemoveMissing(),
    ]
    df = standardize_events(df)
    for t in transforms:
        df = t(df)
    df = standardize_events(df, auto_fill=False)

    tag = " [{}]".format(label) if label else ""
    print("  Predicting{}...".format(tag))
    preds, _ = model.predict(events=df)
    return preds  # (n_timesteps, n_vertices)


def get_roi_indices():
    from nilearn import datasets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas = datasets.fetch_atlas_surf_destrieux()
    lh = np.array(atlas["map_left"])
    rh = np.array(atlas["map_right"])
    full_map = np.concatenate([lh, rh])
    labels = [str(l) for l in atlas["labels"]]

    ROI_LABEL_MAP = {
        "Visual (V1/V2)": ["S_calcarine", "G_cuneus"],
        "Occipital": ["G_occipital_sup", "G_occipital_middle", "Pole_occipital"],
        "Auditory (A1)": ["G_temp_sup-G_T_transv"],
        "Broca area": ["G_front_inf-Opercular", "G_front_inf-Triangul"],
        "Wernicke area": ["G_temp_sup-Lateral", "G_temp_sup-Plan_tempo"],
        "Fusiform (FFA)": ["G_oc-temp_lat-fusifor"],
        "Parahipp. (PPA)": ["G_oc-temp_med-Parahip"],
        "Frontal sup.": ["G_front_sup"],
        "Angular/TPJ": ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
        "Precuneus": ["G_precuneus"],
        "Motor": ["G_precentral"],
        "Somatosensory": ["G_postcentral"],
        "Temporal mid.": ["G_temporal_middle"],
        "Temporal inf.": ["G_temporal_inf"],
        "Insula": ["G_insular_short", "G_Ins_lg_and_S_cent_ins"],
        "Cingulate ant.": ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
        "Cingulate post.": ["G_cingul-Post-dorsal", "G_cingul-Post-ventral"],
        "Orbital frontal": ["G_orbital", "G_rectus"],
        "Frontal mid.": ["G_front_middle"],
        "Temporal pole": ["Pole_temporal"],
    }

    roi_indices = {}
    for name, atlas_names in ROI_LABEL_MAP.items():
        indices = []
        for aname in atlas_names:
            if aname in labels:
                label_idx = labels.index(aname)
                indices.append(np.where(full_map == label_idx)[0])
        if indices:
            roi_indices[name] = np.concatenate(indices)
    return roi_indices


ROI_TO_CANVAS = {
    "Visual (V1/V2)": "visual.v1",
    "Occipital": "visual.v2_v4",
    "Fusiform (FFA)": "visual.fusiform",
    "Auditory (A1)": "auditory.a1",
    "Wernicke area": "auditory.wernicke",
    "Broca area": "language.broca",
    "Angular/TPJ": "language.angular",
    "Temporal mid.": "language.temporal_mid",
    "Frontal sup.": "frontal.prefrontal",
    "Motor": "frontal.motor",
    "Frontal mid.": "frontal.premotor",
    "Precuneus": "default_mode.precuneus",
    "Cingulate ant.": "default_mode.cingulate",
    "Cingulate post.": "default_mode.cingulate",
    "Temporal pole": "default_mode.temporal_pole",
    "Insula": "subcortical.insula",
    "Somatosensory": "subcortical.somatosensory",
    "Orbital frontal": "frontal.prefrontal",
    "Parahipp. (PPA)": "visual.fusiform",
    "Temporal inf.": "language.temporal_mid",
}


# ── Data generation: keep temporal dynamics ─────────────────────────

def generate_temporal_data(tribe_model, roi_indices, canvas_region_names,
                           max_stimuli=64, min_timesteps=3):
    """Generate temporal ROI activation sequences from TRIBE v2.

    Returns:
        sequences: list of (T, n_regions) arrays — temporal activation per stimulus
        labels: category index per stimulus
        category_names: list of category names
    """
    from cortical_canvas import STIMULUS_CATEGORIES

    cat_names = sorted(STIMULUS_CATEGORIES.keys())
    sequences = []
    labels = []
    stim_per_cat = max(1, max_stimuli // len(cat_names))

    for cat_idx, cat_name in enumerate(cat_names):
        stimuli = STIMULUS_CATEGORIES[cat_name][:stim_per_cat]
        for text in stimuli:
            preds = text_to_preds(tribe_model, text, label=cat_name)
            # preds: (n_timesteps, 20484)

            if preds.shape[0] < min_timesteps:
                print("    Skipping (only {} timesteps)".format(preds.shape[0]))
                continue

            # Map each timestep to ROI means → canvas region activations
            n_t = preds.shape[0]
            region_seq = np.zeros((n_t, len(canvas_region_names)))

            for t in range(n_t):
                for roi_name, vertex_idx in roi_indices.items():
                    canvas_name = ROI_TO_CANVAS.get(roi_name)
                    if canvas_name and canvas_name in canvas_region_names:
                        idx = canvas_region_names.index(canvas_name)
                        region_seq[t, idx] += preds[t, vertex_idx].mean()

            sequences.append(region_seq)
            labels.append(cat_idx)

    return sequences, np.array(labels), cat_names


# ── Training: next-timestep regional prediction ─────────────────────

def build_dynamics_dataset(sequences, window=3):
    """Build (input_window, target_next) pairs from temporal sequences.

    For each sequence of length T, creates T-window pairs:
      input: region activations at times [t, t+1, ..., t+window-1]
      target: region activations at time t+window

    Returns:
        X: (N, window, n_regions) input sequences
        Y: (N, n_regions) target next-step activations
        seq_ids: which original sequence each pair came from
    """
    X_list, Y_list, ids = [], [], []
    for seq_idx, seq in enumerate(sequences):
        T, n_regions = seq.shape
        for t in range(T - window):
            X_list.append(seq[t:t + window])
            Y_list.append(seq[t + window])
            ids.append(seq_idx)

    return np.stack(X_list), np.stack(Y_list), np.array(ids)


def train_dynamics_model(X_train, Y_train, X_val, Y_val,
                         topology, canvas_region_names,
                         mode="cortical", n_epochs=200, d_model=64,
                         n_heads=4, lr=1e-3):
    """Train a next-timestep prediction model.

    Three modes:
    - cortical: uses the declared cortical topology for attention routing
    - dense: fully connected attention
    - flat: MLP (no attention structure)
    """
    import torch
    import torch.nn as nn
    from canvas_engineering import CanvasTopology, CanvasLayout, RegionSpec, Connection
    from canvas_engineering.dispatch import AttentionDispatcher

    torch.manual_seed(42)
    n_regions = X_train.shape[2]
    window = X_train.shape[1]

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val, dtype=torch.float32)
    Y_vl = torch.tensor(Y_val, dtype=torch.float32)

    if mode == "flat":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * n_regions, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, n_regions),
        )
    else:
        # Build dispatch layout: 1 position per region
        dispatch_regions = {}
        for i, name in enumerate(canvas_region_names):
            dispatch_regions[name] = RegionSpec(bounds=(0, 1, i, i + 1, 0, 1))
        dispatch_layout = CanvasLayout(
            T=1, H=n_regions, W=1, d_model=d_model, regions=dispatch_regions,
        )

        if mode == "cortical":
            topo = topology
        else:  # dense
            topo = CanvasTopology.dense(canvas_region_names)

        class DynamicsTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(window, d_model)
                self.layers = nn.ModuleList()
                self.norms = nn.ModuleList()
                self.ffns = nn.ModuleList()
                for _ in range(3):
                    self.layers.append(AttentionDispatcher(
                        topo, dispatch_layout, d_model, n_heads, dropout=0.1,
                    ))
                    self.norms.append(nn.LayerNorm(d_model))
                    self.ffns.append(nn.Sequential(
                        nn.Linear(d_model, d_model * 4), nn.GELU(),
                        nn.Dropout(0.1), nn.Linear(d_model * 4, d_model),
                    ))
                self.output_proj = nn.Linear(d_model, 1)

            def forward(self, x):
                # x: (B, window, n_regions)
                # Transpose to (B, n_regions, window), project each region
                h = self.input_proj(x.transpose(1, 2))  # (B, n_regions, d_model)
                for layer, norm, ffn in zip(self.layers, self.norms, self.ffns):
                    h2 = layer(h)
                    h = norm(h + h2)
                    h = h + ffn(h)
                out = self.output_proj(h).squeeze(-1)  # (B, n_regions)
                return out

        model = DynamicsTransformer()

    n_params = sum(p.numel() for p in model.parameters())
    print("  {} model: {} params".format(mode, n_params))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = {"train_loss": [], "val_loss": [], "val_r2": []}
    batch_size = min(64, len(X_tr))

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(len(X_tr))[:batch_size]
        pred = model(X_tr[idx])
        loss = ((pred - Y_tr[idx]) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl)
            val_loss = ((val_pred - Y_vl) ** 2).mean().item()

            # R² score
            ss_res = ((val_pred - Y_vl) ** 2).sum().item()
            ss_tot = ((Y_vl - Y_vl.mean(dim=0)) ** 2).sum().item()
            r2 = 1 - ss_res / max(ss_tot, 1e-8)

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_r2"].append(r2)

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print("  [{}] Epoch {:3d}/{} | train {:.4f} | val {:.4f} | R² {:.4f}".format(
                mode, epoch, n_epochs, loss.item(), val_loss, r2))

    return model, history, n_params


# ── Evaluation and visualization ────────────────────────────────────

def plot_dynamics_results(results, canvas_region_names):
    """Generate comprehensive visualization of dynamics prediction results."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Cortical Dynamics Prediction: Next-Timestep Regional Activation",
                 fontsize=16, fontweight="bold", y=0.98)

    colors = {"cortical": "#E74C3C", "dense": "#3498DB", "flat": "#95A5A6"}

    # 1. Training loss curves
    ax = axes[0, 0]
    ax.set_title("Training Loss", fontweight="bold")
    for mode, r in results.items():
        ax.semilogy(r["history"]["train_loss"], color=colors[mode], label=mode, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 2. Validation loss curves
    ax = axes[0, 1]
    ax.set_title("Validation Loss", fontweight="bold")
    for mode, r in results.items():
        ax.semilogy(r["history"]["val_loss"], color=colors[mode], label=mode, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 3. R² curves
    ax = axes[0, 2]
    ax.set_title("Prediction R²", fontweight="bold")
    for mode, r in results.items():
        ax.plot(r["history"]["val_r2"], color=colors[mode], label=mode, lw=2)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 4. Final R² comparison
    ax = axes[1, 0]
    ax.set_title("Final R² (higher = better)", fontweight="bold")
    modes = list(results.keys())
    final_r2 = [results[m]["history"]["val_r2"][-1] for m in modes]
    bars = ax.bar(modes, final_r2, color=[colors[m] for m in modes], edgecolor="white")
    for bar, val in zip(bars, final_r2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                "{:.3f}".format(val), ha="center", fontweight="bold")
    ax.set_ylabel("R²")
    ax.grid(True, alpha=0.2, axis="y")

    # 5. Per-region prediction error
    ax = axes[1, 1]
    ax.set_title("Per-Region Val MSE (cortical vs dense)", fontweight="bold")
    if "per_region_mse" in results.get("cortical", {}):
        cortical_mse = results["cortical"]["per_region_mse"]
        dense_mse = results["dense"]["per_region_mse"]
        x = np.arange(len(canvas_region_names))
        w = 0.35
        ax.barh(x - w / 2, cortical_mse, w, color=colors["cortical"], label="cortical")
        ax.barh(x + w / 2, dense_mse, w, color=colors["dense"], label="dense")
        ax.set_yticks(x)
        ax.set_yticklabels([n.split(".")[-1] for n in canvas_region_names], fontsize=6)
        ax.set_xlabel("MSE")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Per-region MSE\nnot computed", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)

    # 6. Parameter efficiency
    ax = axes[1, 2]
    ax.set_title("Parameter Efficiency", fontweight="bold")
    params = [results[m]["n_params"] for m in modes]
    r2_vals = [max(results[m]["history"]["val_r2"]) for m in modes]
    for m, p, r in zip(modes, params, r2_vals):
        ax.scatter(p, r, color=colors[m], s=200, label=m, zorder=5)
        ax.annotate("{}\n{:.3f}".format(m, r), (p, r), textcoords="offset points",
                    xytext=(10, 5), fontsize=9, fontweight="bold")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Peak R²")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = "results/dynamics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: {}".format(path))


# ── Main pipeline ───────────────────────────────────────────────────

def main():
    import torch
    from cortical_canvas import build_cortical_program

    # Phase 1: Generate temporal data
    print("=" * 60)
    print("PHASE 1: Generate TRIBE v2 temporal cortical data")
    print("=" * 60)

    tribe = load_tribe()
    roi_indices = get_roi_indices()
    print("Mapped {} ROIs".format(len(roi_indices)))

    bound, program, _ = build_cortical_program()
    canvas_region_names = list(bound.field_names)
    print("Canvas regions: {}".format(len(canvas_region_names)))

    # Generate temporal sequences (use 72 stimuli for speed, 8 per category)
    sequences, labels, cat_names = generate_temporal_data(
        tribe, roi_indices, canvas_region_names, max_stimuli=72,
    )
    print("\nGenerated {} temporal sequences".format(len(sequences)))
    lengths = [s.shape[0] for s in sequences]
    print("  Timesteps per sequence: min={}, max={}, mean={:.1f}".format(
        min(lengths), max(lengths), np.mean(lengths)))
    print("  Total timesteps: {}".format(sum(lengths)))

    # Build next-timestep prediction dataset
    window = 3
    X, Y, seq_ids = build_dynamics_dataset(sequences, window=window)
    print("\nDynamics dataset: {} samples, window={}, {} regions".format(
        X.shape[0], window, X.shape[2]))

    # Train/val split (by sequence, not by sample)
    unique_seqs = np.unique(seq_ids)
    np.random.seed(42)
    np.random.shuffle(unique_seqs)
    n_val_seqs = max(1, len(unique_seqs) // 5)
    val_seqs = set(unique_seqs[:n_val_seqs])
    train_mask = np.array([s not in val_seqs for s in seq_ids])
    val_mask = ~train_mask

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]
    print("  Train: {} samples, Val: {} samples".format(len(X_train), len(X_val)))

    # Save data
    np.savez("results/dynamics_data.npz",
             X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
             canvas_region_names=canvas_region_names, labels=labels,
             category_names=cat_names)

    # Phase 2: Train all three models
    print("\n" + "=" * 60)
    print("PHASE 2: Train dynamics prediction models (200 epochs)")
    print("=" * 60)

    topology = program.schema.topology
    results = {}

    for mode in ["cortical", "dense", "flat"]:
        print("\n--- {} ---".format(mode))
        model, history, n_params = train_dynamics_model(
            X_train, Y_train, X_val, Y_val,
            topology, canvas_region_names,
            mode=mode, n_epochs=200, d_model=64,
        )

        # Per-region MSE
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val, dtype=torch.float32))
            per_region = ((val_pred - torch.tensor(Y_val, dtype=torch.float32)) ** 2).mean(dim=0)

        results[mode] = {
            "history": history,
            "n_params": n_params,
            "per_region_mse": per_region.numpy().tolist(),
        }

        # Save checkpoint
        torch.save(model.state_dict(),
                   "results/dynamics_checkpoint_{}.pt".format(mode))

    # Save comparison
    with open("results/dynamics_comparison.json", "w") as f:
        json.dump({
            mode: {
                "n_params": r["n_params"],
                "final_val_loss": r["history"]["val_loss"][-1],
                "final_r2": r["history"]["val_r2"][-1],
                "peak_r2": max(r["history"]["val_r2"]),
            }
            for mode, r in results.items()
        }, f, indent=2)

    # Phase 3: Visualize
    print("\n" + "=" * 60)
    print("PHASE 3: Generate visualizations")
    print("=" * 60)

    plot_dynamics_results(results, canvas_region_names)

    # Print summary
    print("\n" + "=" * 60)
    print("DYNAMICS PREDICTION RESULTS")
    print("=" * 60)
    for mode, r in results.items():
        print("  {}: R²={:.4f} (peak {:.4f}), val_loss={:.4f}, params={}".format(
            mode, r["history"]["val_r2"][-1], max(r["history"]["val_r2"]),
            r["history"]["val_loss"][-1], r["n_params"]))

    print("\nDone!")


if __name__ == "__main__":
    main()

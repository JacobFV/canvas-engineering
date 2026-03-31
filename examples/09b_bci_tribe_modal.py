"""Brain-computer interface with TRIBE v2 cortical predictions.

Uses Facebook's TRIBE v2 brain encoding model to generate realistic
cortical surface predictions from text stimuli, then trains a
canvas-engineering BCI decoder to classify stimulus categories from
virtual electrode readings.

This example requires GPU and runs on Modal. It:
1. Loads TRIBE v2 to generate cortical predictions for 48 stimuli (6 categories)
2. Samples virtual EEG via 10-20 electrode patches on fsaverage5
3. Trains a canvas-based transformer decoder on the virtual EEG
4. Compares canvas decoder vs SVM baseline
5. Generates visualization

Run:  modal run examples/09b_bci_tribe_modal.py
Out:  assets/examples/09b_bci_tribe.png
"""

import modal
import os
import sys
import base64
import io
from pathlib import Path

app = modal.App("canvas-bci-tribe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git", "libgl1", "libglib2.0-0", "libxrender1")
    .pip_install("torch", "torchaudio")
    .run_commands("pip install git+https://github.com/facebookresearch/tribev2.git")
    .run_commands("python -m spacy download en_core_web_lg")
    .pip_install(
        "canvas-engineering>=0.4.0",
        "gtts", "langdetect", "soundfile", "matplotlib", "numpy",
        "nilearn", "scikit-learn", "mne",
        "transformers>=4.45,<4.50",
    )
)

# Upload brain-model core utilities
brain_model_dir = Path(__file__).resolve().parent.parent.parent / "fun" / "brain-model"

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "examples"
ASSETS.mkdir(parents=True, exist_ok=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    cpu=4,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret", environment_name="test-20260327")],
)
def run_bci_experiment():
    """Full BCI experiment: TRIBE v2 → virtual EEG → canvas decoder."""
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    from dataclasses import dataclass, field as dc_field
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from canvas_engineering import (
        Field, compile_program, RegionProgram, CanvasProgram,
        CanvasTopology, Connection, TemporalFill,
    )

    # ── 1. Load TRIBE v2 and generate cortical predictions ──────────────

    print("=" * 60)
    print("PHASE 1: Generate cortical predictions with TRIBE v2")
    print("=" * 60)

    # Inline the brain-model utilities to avoid complex imports
    from pathlib import Path as P
    import hashlib

    CACHE = P("./cache")
    CACHE.mkdir(exist_ok=True)

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
        audio_dir = CACHE / f"bci_{text_hash}"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / "audio.mp3"

        if not audio_path.exists():
            tts = gTTS(text, lang=detect(text))
            tts.save(str(audio_path))

        info = sf.info(str(audio_path))
        duration = info.duration
        words = text.split()
        n_words = len(words)
        word_dur = duration / max(n_words, 1)

        audio_event = {"type": "Audio", "filepath": str(audio_path),
                       "start": 0.0, "duration": duration,
                       "timeline": "default", "subject": "default"}
        word_events = [{"type": "Word", "text": w.strip(".,;:!?\"'()-"),
                        "start": i * word_dur, "duration": word_dur * 0.8,
                        "timeline": "default", "subject": "default",
                        "language": "english", "sequence_id": 0}
                       for i, w in enumerate(words)]

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

        tag = f" [{label}]" if label else ""
        print(f"  Predicting{tag}...")
        preds, _ = model.predict(events=df)
        return preds

    def build_electrode_patches(radius_mm=10.0):
        import mne
        from nilearn import datasets
        import nibabel as nib

        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        coords_lh = nib.load(fsaverage["pial_left"]).agg_data()[0]
        coords_rh = nib.load(fsaverage["pial_right"]).agg_data()[0]
        all_coords = np.vstack([coords_lh, coords_rh])

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_pos = montage.get_positions()["ch_pos"]

        electrodes = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
                      "T3","C3","Cz","C4","T4",
                      "T5","P3","Pz","P4","T6","O1","O2","Oz"]
        patches = {}
        for name in electrodes:
            if name not in ch_pos:
                continue
            scalp_pos = ch_pos[name] * 1000
            dists = np.linalg.norm(all_coords - scalp_pos, axis=1)
            nearest = np.argmin(dists)
            cortex_pos = all_coords[nearest]
            patch_dists = np.linalg.norm(all_coords - cortex_pos, axis=1)
            patches[name] = np.where(patch_dists <= radius_mm)[0]
        return patches

    # Stimulus categories (8 per category)
    CATEGORIES = {
        "motor": [
            "Reaching forward to pick up a coffee mug from the table, wrapping fingers around the warm ceramic handle",
            "Typing rapidly on a keyboard, fingers dancing across the keys, each keystroke precise and automatic",
            "Throwing a ball overhand with full force, the arm whipping forward, releasing at exactly the right moment",
            "Walking up a steep flight of stairs, each step requiring careful balance and muscle coordination",
            "Playing piano scales with both hands, fingers moving independently in complex rhythmic patterns",
            "Catching a falling glass before it hits the floor, hand shooting out reflexively to grab it",
            "Threading a needle, holding the thread steady between thumb and forefinger, eyes focused on the tiny eye",
            "Juggling three balls in a smooth figure-eight pattern, hands moving in coordinated arcs",
        ],
        "language": [
            "Reading a dense paragraph of philosophy, each word building on the last into a complex argument",
            "The word petrichor meaning the smell of rain on dry earth, naming something you always knew but never could say",
            "Composing a sentence in a foreign language, searching for grammar rules stored somewhere deep in memory",
            "Listening to a fast-talking auctioneer and somehow understanding every word despite the incredible speed",
            "A poet choosing between two synonyms, feeling the subtle difference in weight and color between them",
            "Translating humor across languages and watching the joke evaporate despite perfect technical accuracy",
            "A child inventing a new word for something that has no name, and the word being somehow exactly right",
            "Reading braille with fingertips, each raised dot pattern resolving into a letter then a word then meaning",
        ],
        "visual": [
            "A sunset painting the sky in layers of orange and purple, the horizon line razor sharp against dark mountains",
            "Staring at an optical illusion where the image flips between a vase and two faces staring at each other",
            "Watching a flock of starlings forming murmurations, thousands of birds moving as one shifting shape",
            "A photograph so detailed you can count the water droplets on a spider web caught in morning light",
            "The moment your eyes adjust to darkness and shapes slowly emerge from what seemed like nothing",
            "Looking through a kaleidoscope as the colored glass fragments tumble into perfect symmetrical patterns",
            "A time-lapse of a flower blooming, petals unfolding in seconds what took hours in real life",
            "Staring at a fractal pattern zooming infinitely inward, the same shape repeating at every scale",
        ],
        "emotion": [
            "The rush of joy when a song you love comes on unexpectedly, your whole body responding before your mind does",
            "Holding back tears at a funeral, the pressure building behind your eyes, throat tight, jaw clenched",
            "The surge of anger when someone cuts in line, fists clenching involuntarily, jaw tightening",
            "Butterflies in your stomach before a first date, excitement and anxiety tangled together inseparably",
            "The warm contentment of sitting by a fire with people you love, needing nothing, wanting nothing",
            "A wave of nostalgia triggered by a childhood smell, time collapsing as memories flood back unbidden",
            "The relief of finally solving a problem that had been nagging you for days, tension releasing from your shoulders",
            "Stage fright freezing your body moments before stepping onto the stage, every eye about to be on you",
        ],
    }

    tribe = load_tribe()
    patches = build_electrode_patches()
    electrode_names = sorted(patches.keys())
    n_electrodes = len(electrode_names)

    print(f"\n{n_electrodes} electrodes mapped")

    # Generate dataset
    all_eeg = []
    all_labels = []
    cat_names = sorted(CATEGORIES.keys())
    for cat_idx, cat_name in enumerate(cat_names):
        for text in CATEGORIES[cat_name]:
            preds = text_to_preds(tribe, text, label=cat_name)
            eeg = np.zeros((preds.shape[0], n_electrodes))
            for j, name in enumerate(electrode_names):
                eeg[:, j] = preds[:, patches[name]].mean(axis=1)
            all_eeg.append(eeg.mean(axis=0))  # mean across time
            all_labels.append(cat_idx)

    X = np.stack(all_eeg)  # (32, 20)
    y = np.array(all_labels)
    print(f"\nDataset: {X.shape[0]} samples, {n_electrodes} electrodes, {len(cat_names)} categories")

    # ── 2. SVM baseline ─────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PHASE 2: SVM baseline")
    print("=" * 60)

    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
    loo = LeaveOneOut()
    svm_scores = cross_val_score(clf, X, y, cv=loo, scoring="accuracy")
    svm_acc = svm_scores.mean()
    print(f"SVM LOOCV accuracy: {svm_acc:.1%}")

    # ── 3. Canvas decoder ────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PHASE 3: Canvas-based transformer decoder")
    print("=" * 60)

    @dataclass
    class BCIDecoder:
        """Canvas BCI decoder with v2 families."""
        frontal: Field = Field(2, 4, family="observation",
                               semantic_type="frontal electrode readings (Fp1 Fp2 F3 F4 F7 F8 Fz)")
        central: Field = Field(2, 3, family="observation",
                               semantic_type="central electrode readings (C3 Cz C4)")
        temporal: Field = Field(2, 3, family="observation",
                                semantic_type="temporal electrode readings (T3 T4 T5 T6)")
        parietal: Field = Field(2, 3, family="observation",
                                semantic_type="parietal electrode readings (P3 Pz P4)")
        occipital: Field = Field(1, 3, family="observation",
                                 semantic_type="occipital electrode readings (O1 O2 Oz)")
        intent: Field = Field(2, 4, family="state", tags=("belief",),
                              semantic_type="decoded cognitive intent representation")
        category: Field = Field(1, len(cat_names), family="action", loss_weight=3.0,
                                semantic_type="stimulus category prediction")

    bound, program = compile_program(BCIDecoder(), T=1, d_model=48)
    print(bound.summary())
    print(program.summary())

    # Map electrodes to canvas regions
    electrode_region_map = {
        "frontal": ["F3", "F4", "F7", "F8", "Fp1", "Fp2", "Fz"],
        "central": ["C3", "C4", "Cz"],
        "temporal": ["T3", "T4", "T5", "T6"],
        "parietal": ["P3", "P4", "Pz"],
        "occipital": ["O1", "O2", "Oz"],
    }

    # Build input tensors per region
    def build_inputs(X_batch, electrode_names):
        inputs = {}
        for region, elecs in electrode_region_map.items():
            idxs = [electrode_names.index(e) for e in elecs if e in electrode_names]
            inputs[region] = torch.tensor(X_batch[:, idxs], dtype=torch.float32)
        return inputs

    class CanvasBCIDecoder(nn.Module):
        def __init__(self, bound_schema, d_model=48, nhead=4, n_categories=4):
            super().__init__()
            self.bound = bound_schema
            self.d = d_model
            N = bound_schema.layout.num_positions
            self.pos_emb = nn.Parameter(torch.randn(1, N, d_model) * 0.02)

            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=192,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=3)
            mask = bound_schema.topology.to_additive_mask(bound_schema.layout)
            self.register_buffer("attn_mask", mask)

            # Per-region input projections
            self.projs = nn.ModuleDict()
            for region, elecs in electrode_region_map.items():
                n_pos = len(bound_schema.layout.region_indices(region))
                n_elec = len([e for e in elecs if e in electrode_names])
                self.projs[region] = nn.Linear(n_elec, n_pos * d_model)

            # Output projection
            cat_n = len(bound_schema.layout.region_indices("category"))
            self.out_proj = nn.Linear(cat_n * d_model, n_categories)

        def forward(self, inputs):
            B = list(inputs.values())[0].shape[0]
            canvas = self.pos_emb.expand(B, -1, -1).clone()

            for region, data in inputs.items():
                idx = self.bound.layout.region_indices(region)
                n = len(idx)
                proj = self.projs[region](data).reshape(B, n, self.d)
                canvas[:, idx] = canvas[:, idx] + proj

            canvas = self.encoder(canvas, mask=self.attn_mask)

            cat_idx = self.bound.layout.region_indices("category")
            cat_emb = canvas[:, cat_idx].reshape(B, -1)
            return self.out_proj(cat_emb)

    # Train with LOOCV
    torch.manual_seed(42)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    n_samples = len(y)
    canvas_correct = 0

    print("Training canvas decoder (LOOCV)...")
    for i in range(n_samples):
        # Leave one out
        train_mask = torch.ones(n_samples, dtype=torch.bool)
        train_mask[i] = False
        X_train, y_train = X_t[train_mask], y_t[train_mask]
        X_test, y_test = X_t[i:i+1], y_t[i:i+1]

        model = CanvasBCIDecoder(bound, n_categories=len(cat_names))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)

        # Quick training (small dataset)
        for epoch in range(200):
            inputs = build_inputs(X_train.numpy(), electrode_names)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            test_inputs = build_inputs(X_test.numpy(), electrode_names)
            pred = model(test_inputs).argmax(dim=-1)
            if pred.item() == y_test.item():
                canvas_correct += 1

        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{n_samples} done, running acc: {canvas_correct/(i+1):.1%}")

    canvas_acc = canvas_correct / n_samples
    print(f"\nCanvas decoder LOOCV accuracy: {canvas_acc:.1%}")

    # ── 4. Visualization ─────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PHASE 4: Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("white")
    fig.suptitle("Brain-Computer Interface: TRIBE v2 → Canvas Decoder",
                 fontsize=16, fontweight="bold", y=0.98)

    # (a) Canvas layout with families
    ax = axes[0, 0]
    ax.set_title("Canvas Layout (v2 families)", fontsize=11, fontweight="bold")
    H, W = bound.layout.H, bound.layout.W
    grid = np.ones((H, W, 3)) * 0.93
    family_colors = {
        "observation": "#3498DB",
        "state": "#2ECC71",
        "action": "#E74C3C",
    }
    for name in bound.field_names:
        bf = bound[name]
        rp = program.regions.get(name)
        if rp:
            color = family_colors.get(rp.family, "#95A5A6")
        else:
            color = "#95A5A6"
        r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
        h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
        w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
        grid[h0:h1, w0:w1] = [r, g, b]
        label = name.split(".")[-1] if "." in name else name
        if (h1-h0)*(w1-w0) >= 2:
            ax.text((w0+w1)/2-0.5, (h0+h1)/2-0.5, label,
                    ha="center", va="center", fontsize=5, fontweight="bold", color="white")
    ax.imshow(grid, aspect="equal", interpolation="nearest")
    ax.set_xlabel("W"); ax.set_ylabel("H")
    # Legend
    for family, color in family_colors.items():
        ax.plot([], [], 's', color=color, label=family, markersize=8)
    ax.legend(fontsize=7, loc="upper right")

    # (b) Accuracy comparison
    ax = axes[0, 1]
    ax.set_title("Classification Accuracy (LOOCV)", fontsize=11, fontweight="bold")
    chance = 1.0 / len(cat_names)
    bars = ax.bar(
        ["Chance", "SVM\n(20 electrodes)", "Canvas Decoder\n(v2 families)"],
        [chance, svm_acc, canvas_acc],
        color=["#BDC3C7", "#3498DB", "#E74C3C"],
        edgecolor="white", linewidth=1.5,
    )
    for bar, val in zip(bars, [chance, svm_acc, canvas_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.2, axis="y")

    # (c) Per-region electrode distribution
    ax = axes[1, 0]
    ax.set_title("Electrode Distribution by Canvas Region", fontsize=11, fontweight="bold")
    region_sizes = {r: len(e) for r, e in electrode_region_map.items()}
    colors_list = ["#3498DB"] * len(region_sizes)
    ax.barh(list(region_sizes.keys()), list(region_sizes.values()),
            color=colors_list, edgecolor="white")
    for i, (r, n) in enumerate(region_sizes.items()):
        ax.text(n + 0.1, i, f"{n} ch", va="center", fontsize=9)
    ax.set_xlabel("Number of electrodes")

    # (d) Mean EEG by category
    ax = axes[1, 1]
    ax.set_title("Mean EEG Activation by Category", fontsize=11, fontweight="bold")
    cat_colors = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"]
    for ci, cat in enumerate(cat_names):
        cat_mask = y == ci
        cat_mean = X[cat_mask].mean(axis=0)
        ax.plot(range(n_electrodes), cat_mean, label=cat, color=cat_colors[ci], lw=2)
    ax.set_xticks(range(n_electrodes))
    ax.set_xticklabels(electrode_names, rotation=90, fontsize=7)
    ax.set_ylabel("Mean activation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to buffer and return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=150)
    plt.close()
    buf.seek(0)

    results = {
        "svm_accuracy": float(svm_acc),
        "canvas_accuracy": float(canvas_acc),
        "chance": float(chance),
        "n_samples": n_samples,
        "n_electrodes": n_electrodes,
        "categories": cat_names,
        "image_bytes": base64.b64encode(buf.read()).decode(),
    }
    return results


@app.local_entrypoint()
def main():
    print("Running BCI experiment on Modal with GPU...")
    results = run_bci_experiment.remote()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Categories: {results['categories']}")
    print(f"  Samples: {results['n_samples']}")
    print(f"  Electrodes: {results['n_electrodes']}")
    print(f"  Chance: {results['chance']:.1%}")
    print(f"  SVM: {results['svm_accuracy']:.1%}")
    print(f"  Canvas: {results['canvas_accuracy']:.1%}")

    # Save image
    img_bytes = base64.b64decode(results["image_bytes"])
    path = ASSETS / "09b_bci_tribe.png"
    with open(path, "wb") as f:
        f.write(img_bytes)
    print(f"\n  Saved {path}")

"""Generate TRIBE v2 cortical data, train, and evaluate.

Run this directly on Modal or locally with GPU + tribev2 installed.
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

from pathlib import Path
CACHE = Path("./cache")
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
    audio_dir = CACHE / "brain_{}".format(text_hash)
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

    audio_event = {
        "type": "Audio", "filepath": str(audio_path),
        "start": 0.0, "duration": duration,
        "timeline": "default", "subject": "default",
    }
    word_events = []
    for i, w in enumerate(words):
        clean = w.strip('.,;:!?"\'()-')
        word_events.append({
            "type": "Word", "text": clean,
            "start": i * word_dur, "duration": word_dur * 0.8,
            "timeline": "default", "subject": "default",
            "language": "english", "sequence_id": 0,
        })

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
    return preds


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
    for friendly_name, atlas_names in ROI_LABEL_MAP.items():
        indices = []
        for aname in atlas_names:
            if aname in labels:
                label_idx = labels.index(aname)
                indices.append(np.where(full_map == label_idx)[0])
        if indices:
            roi_indices[friendly_name] = np.concatenate(indices)

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


def main():
    from cortical_canvas import STIMULUS_CATEGORIES, build_cortical_program

    # Phase 1: Generate TRIBE v2 data
    print("=" * 60)
    print("PHASE 1: Generate TRIBE v2 cortical predictions")
    print("=" * 60)

    tribe = load_tribe()
    roi_indices = get_roi_indices()
    print("Mapped {} ROIs".format(len(roi_indices)))

    bound, program = build_cortical_program()
    canvas_region_names = list(bound.field_names)
    print("Canvas regions: {}".format(len(canvas_region_names)))

    all_activations = []
    all_labels = []
    all_raw_preds = []
    cat_names = sorted(STIMULUS_CATEGORIES.keys())

    for cat_idx, cat_name in enumerate(cat_names):
        stimuli = STIMULUS_CATEGORIES[cat_name]
        for text in stimuli:
            preds = text_to_preds(tribe, text, label=cat_name)

            # Save raw mean activation for brain surface rendering
            all_raw_preds.append(preds.mean(axis=0))

            # Map to ROI means
            roi_means = {}
            for roi_name, vertex_idx in roi_indices.items():
                roi_means[roi_name] = float(preds[:, vertex_idx].mean())

            # Map to canvas regions
            region_act = np.zeros(len(canvas_region_names))
            for roi_name, canvas_name in ROI_TO_CANVAS.items():
                if roi_name in roi_means and canvas_name in canvas_region_names:
                    idx = canvas_region_names.index(canvas_name)
                    region_act[idx] += roi_means[roi_name]

            all_activations.append(region_act)
            all_labels.append(cat_idx)

    X = np.stack(all_activations)
    y = np.array(all_labels)
    raw_preds = np.stack(all_raw_preds)
    print("\nDataset: {} samples, {} regions, {} categories".format(
        X.shape[0], X.shape[1], len(cat_names)))

    np.savez("results/tribe_data.npz",
             region_activations=X, labels=y, raw_preds=raw_preds)
    with open("results/tribe_data_meta.json", "w") as f:
        json.dump({
            "n_samples": len(y),
            "n_categories": len(cat_names),
            "categories": cat_names,
            "regions": canvas_region_names,
        }, f, indent=2)

    # Phase 2: Train
    print("\n" + "=" * 60)
    print("PHASE 2: Train cortical canvas model (200 epochs)")
    print("=" * 60)

    from train import run_all_baselines
    run_all_baselines(
        data_path="results/tribe_data.npz",
        results_dir="results",
        n_epochs=200,
        d_model=128,
        n_layers=3,
        n_heads=8,
        lr=1e-3,
    )

    # Phase 3: Evaluate
    print("\n" + "=" * 60)
    print("PHASE 3: Generate visualizations")
    print("=" * 60)

    from evaluate import generate_all_plots
    generate_all_plots("results")

    print("\nDone!")


if __name__ == "__main__":
    main()

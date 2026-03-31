"""Modal launcher: generate data on GPU, train, and evaluate.

Single script that orchestrates the full cortical canvas experiment:
1. Generate cortical activation data on Modal GPU (TRIBE v2 inference)
2. Download data locally
3. Train cortical / dense / flat models
4. Generate evaluation plots

Usage:
    # Full pipeline (generate data on Modal, train locally):
    modal run research/brain/run_modal.py

    # Generate data only:
    modal run research/brain/run_modal.py --data-only

    # Train + evaluate locally with synthetic data (no Modal GPU):
    python research/brain/run_modal.py --local --synthetic

    # Train + evaluate locally with existing data:
    python research/brain/run_modal.py --local --data research/brain/results/cortical_dataset.npz

    # Train on Modal GPU:
    modal run research/brain/run_modal.py --train-on-modal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

RESULTS_DIR = Path(_CE_ROOT) / "research" / "brain" / "results"

# ---- Modal setup ----

try:
    import modal

    app = modal.App("cortical-canvas-brain")

    # Image for data generation (needs TRIBE v2 + nilearn)
    data_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "ffmpeg", "libsndfile1")
        .pip_install(
            "torch>=2.0",
            "numpy",
            "nilearn",
            "tribev2",
            "gtts",
            "langdetect",
            "soundfile",
            "pandas",
            "neuralset",
            "transformers",
            "accelerate",
        )
        .add_local_dir(
            str(_CE_ROOT / "research" / "brain"),
            "/root/research/brain",
            copy=True,
        )
        .add_local_dir(
            str(_CE_ROOT / "canvas_engineering"),
            "/root/canvas_engineering",
            copy=True,
        )
    )

    # Image for training (lighter, just needs torch + canvas_engineering)
    train_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch>=2.0",
            "numpy",
            "matplotlib",
            "canvas-engineering>=0.4.0",
        )
        .add_local_dir(
            str(_CE_ROOT / "research" / "brain"),
            "/root/research/brain",
            copy=True,
        )
        .add_local_dir(
            str(_CE_ROOT / "canvas_engineering"),
            "/root/canvas_engineering",
            copy=True,
        )
    )

    @app.function(
        image=data_image,
        gpu="A10G",
        timeout=7200,
        memory=32768,
        secrets=[modal.Secret.from_name("huggingface", environment_name="test-20260327")],
    )
    def generate_data_on_modal() -> dict:
        """Run TRIBE v2 inference on Modal GPU.

        Returns dict with numpy arrays (serialized via modal).
        """
        import sys
        sys.path.insert(0, "/root")

        # Add brain-model core to path for TRIBE v2 loading
        # We inline the necessary functions since brain-model isn't installed
        from research.brain.data_pipeline import (
            generate_dataset_from_model,
            ROI_LABEL_MAP,
            get_roi_indices,
        )
        from research.brain.cortical_canvas import STIMULUS_CATEGORIES

        # Load TRIBE v2 model
        print("Loading TRIBE v2...")
        from tribev2 import TribeModel

        cache = Path("/root/cache")
        cache.mkdir(exist_ok=True)
        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache)

        for attr in ["text_feature", "audio_feature", "video_feature"]:
            feat = getattr(model.data, attr, None)
            if feat is not None and hasattr(feat, "infra"):
                feat.infra.mode = "force"

        print("Model loaded.")

        # Build events function (inline to avoid cross-repo dependency)
        import hashlib
        import pandas as pd
        import soundfile as sf
        from gtts import gTTS
        from langdetect import detect
        from neuralset.events.transforms import (
            AddContextToWords,
            AddSentenceToWords,
            AddText,
            ChunkEvents,
            RemoveMissing,
            standardize_events,
        )

        def build_events_for_text(text, cache_folder):
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            audio_dir = cache_folder / f"brain_radio_{text_hash}"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / "audio.mp3"

            if not audio_path.exists():
                lang = detect(text)
                tts = gTTS(text, lang=lang)
                tts.save(str(audio_path))

            info = sf.info(str(audio_path))
            duration = info.duration

            audio_event = {
                "type": "Audio",
                "filepath": str(audio_path),
                "start": 0.0,
                "duration": duration,
                "timeline": "default",
                "subject": "default",
            }

            words = text.split()
            n_words = len(words)
            word_duration = duration / max(n_words, 1)
            word_events = []
            for i, word in enumerate(words):
                word_events.append({
                    "type": "Word",
                    "text": word.strip(".,;:!?\"'()-"),
                    "start": i * word_duration,
                    "duration": word_duration * 0.8,
                    "timeline": "default",
                    "subject": "default",
                    "language": "english",
                    "sequence_id": 0,
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
            for transform in transforms:
                df = transform(df)
            return standardize_events(df, auto_fill=False)

        def text_to_predictions(model, text, label=""):
            tag = f" [{label}]" if label else ""
            print(f"Predicting brain response{tag}...")
            df = build_events_for_text(text, cache)
            preds, segments = model.predict(events=df)
            print(f"  -> {preds.shape[0]} timesteps x {preds.shape[1]} vertices")
            return preds, segments

        # Get ROI indices
        import warnings
        from nilearn import datasets

        print("Building ROI index map...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atlas = datasets.fetch_atlas_surf_destrieux()

        lh = np.array(atlas["map_left"])
        rh = np.array(atlas["map_right"])
        full_map = np.concatenate([lh, rh])
        labels = [str(l) for l in atlas["labels"]]

        roi_idx = {}
        for friendly_name, atlas_names in ROI_LABEL_MAP.items():
            indices = []
            for aname in atlas_names:
                if aname in labels:
                    label_idx = labels.index(aname)
                    indices.append(np.where(full_map == label_idx)[0])
            if indices:
                roi_idx[friendly_name] = np.concatenate(indices)

        roi_names_sorted = sorted(roi_idx.keys())
        print(f"  {len(roi_idx)} ROIs")

        # Get canvas region names
        from research.brain.cortical_canvas import get_region_names, CANVAS_TO_ROIS
        from research.brain.data_pipeline import roi_to_canvas_vector

        canvas_regions = get_region_names()

        # Generate predictions for all stimuli
        category_names = sorted(STIMULUS_CATEGORIES.keys())
        all_region_acts = []
        all_raw_preds = []
        all_labels = []
        all_texts = []
        all_roi_means_list = []

        for cat_idx, cat_name in enumerate(category_names):
            print(f"\nCategory: {cat_name}")
            for text in STIMULUS_CATEGORIES[cat_name]:
                preds, _ = text_to_predictions(model, text, label=cat_name)

                # ROI means
                act = preds.mean(axis=0) if preds.ndim == 2 else preds
                rmeans = {name: float(act[idx].mean()) for name, idx in roi_idx.items()}
                roi_vec = np.array([rmeans.get(r, 0.0) for r in roi_names_sorted], dtype=np.float32)

                # Canvas vector
                canvas_vec = roi_to_canvas_vector(rmeans, canvas_regions)

                all_region_acts.append(canvas_vec)
                all_raw_preds.append(preds.mean(axis=0).astype(np.float32))
                all_labels.append(cat_idx)
                all_texts.append(text)
                all_roi_means_list.append(roi_vec)

        result = {
            "region_activations": np.stack(all_region_acts),
            "raw_preds": np.stack(all_raw_preds),
            "labels": np.array(all_labels),
            "texts": all_texts,
            "category_names": category_names,
            "region_names": canvas_regions,
            "roi_means_all": np.stack(all_roi_means_list),
            "roi_names": roi_names_sorted,
        }

        print(f"\nGenerated dataset: {result['region_activations'].shape}")
        return result

    @app.function(
        image=train_image,
        gpu="T4",
        timeout=3600,
        memory=16384,
    )
    def train_on_modal(data_dict: dict, epochs: int = 100, d_model: int = 128) -> dict:
        """Train all model variants on Modal GPU."""
        import sys
        sys.path.insert(0, "/root")

        from research.brain.train import run_all_baselines
        return run_all_baselines(data_dict, epochs=epochs, d_model=d_model)

    @app.local_entrypoint()
    def main(
        data_only: bool = False,
        train_on_modal_flag: bool = False,
        local: bool = False,
        synthetic: bool = False,
        data: str = "",
        epochs: int = 100,
        d_model: int = 128,
    ):
        """Orchestrate the full experiment."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        if local:
            # Pure local mode (no Modal GPU)
            _run_local(synthetic=synthetic, data_path=data,
                       epochs=epochs, d_model=d_model)
            return

        # Step 1: Generate data on Modal
        if data and Path(data).exists():
            print(f"Using existing dataset: {data}")
            from research.brain.data_pipeline import load_dataset
            data_dict = load_dataset(data)
        else:
            print("=== Step 1: Generating cortical data on Modal GPU ===")
            t0 = time.time()
            data_dict = generate_data_on_modal.remote()
            elapsed = time.time() - t0
            print(f"Data generation completed in {elapsed:.0f}s")

            # Save locally
            from research.brain.data_pipeline import save_dataset
            output_path = str(RESULTS_DIR / "cortical_dataset.npz")
            save_dataset(data_dict, output_path)
            print(f"Saved dataset to: {output_path}")

        if data_only:
            print("Data generation complete. Exiting.")
            return

        # Step 2: Train
        if train_on_modal_flag:
            print("\n=== Step 2: Training on Modal GPU ===")
            results = train_on_modal.remote(data_dict, epochs=epochs, d_model=d_model)
        else:
            print("\n=== Step 2: Training locally ===")
            from research.brain.train import run_all_baselines
            results = run_all_baselines(data_dict, epochs=epochs, d_model=d_model)

        # Step 3: Evaluate
        print("\n=== Step 3: Generating evaluation plots ===")
        from research.brain.evaluate import run_evaluation
        run_evaluation(data=data_dict, run_training=False, epochs=epochs, d_model=d_model)

        print(f"\nAll results in: {RESULTS_DIR}")

except ImportError:
    # Modal not available
    app = None


def _run_local(
    synthetic: bool = False,
    data_path: str = "",
    epochs: int = 100,
    d_model: int = 128,
):
    """Run the full pipeline locally (no Modal)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if data_path and Path(data_path).exists():
        print(f"Loading dataset from: {data_path}")
        from research.brain.data_pipeline import load_dataset
        data = load_dataset(data_path)
    elif synthetic:
        print("Generating synthetic dataset...")
        from research.brain.data_pipeline import generate_synthetic_dataset
        data = generate_synthetic_dataset()

        # Save synthetic data
        from research.brain.data_pipeline import save_dataset
        save_dataset(data, str(RESULTS_DIR / "cortical_dataset_synthetic.npz"))
    else:
        print("ERROR: Specify --synthetic or --data <path>")
        sys.exit(1)

    print(f"Dataset: {data['region_activations'].shape[0]} samples, "
          f"{data['region_activations'].shape[1]} regions, "
          f"{len(data['category_names'])} categories")

    # Train all models
    print("\n=== Training all models ===")
    from research.brain.train import run_all_baselines
    results = run_all_baselines(data, epochs=epochs, d_model=d_model)

    # Evaluate
    print("\n=== Generating evaluation plots ===")
    from research.brain.evaluate import run_evaluation
    run_evaluation(data=data, run_training=False, epochs=epochs, d_model=d_model)

    print(f"\nAll results in: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cortical Canvas Brain Experiment Launcher",
    )
    parser.add_argument("--local", action="store_true",
                        help="Run entirely locally (no Modal)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no TRIBE v2)")
    parser.add_argument("--data", type=str, default="",
                        help="Path to existing .npz dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    args = parser.parse_args()

    if args.local or app is None:
        _run_local(
            synthetic=args.synthetic,
            data_path=args.data,
            epochs=args.epochs,
            d_model=args.d_model,
        )
    else:
        print("Use 'modal run research/brain/run_modal.py' for Modal execution")
        print("Or use 'python research/brain/run_modal.py --local --synthetic' for local")

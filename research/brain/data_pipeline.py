"""Data pipeline: generate cortical activation data from TRIBE v2.

Runs TRIBE v2 on text stimuli to produce cortical predictions (20,484
vertices on fsaverage5), then maps them to ROI means using the Destrieux
atlas. Each ROI maps to a canvas region in the CorticalBrain hierarchy.

The heavy inference runs on Modal with GPU. Results are saved as .npz
for offline training.

Usage (local, with pre-generated data):
    from research.brain.data_pipeline import load_dataset
    data = load_dataset("research/brain/results/cortical_dataset.npz")

Usage (Modal, generate fresh):
    modal run research/brain/data_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is importable
_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

from research.brain.cortical_canvas import (
    ROI_TO_CANVAS,
    CANVAS_TO_ROIS,
    STIMULUS_CATEGORIES,
    get_region_names,
)


# ---- ROI label map (duplicated from brain-model/core/roi.py to avoid
#      cross-repo imports at definition time) ----

ROI_LABEL_MAP = {
    "Visual (V1/V2)":     ["S_calcarine", "G_cuneus"],
    "Occipital":          ["G_occipital_sup", "G_occipital_middle", "Pole_occipital"],
    "Auditory (A1)":      ["G_temp_sup-G_T_transv"],
    "Broca's area":       ["G_front_inf-Opercular", "G_front_inf-Triangul"],
    "Wernicke's area":    ["G_temp_sup-Lateral", "G_temp_sup-Plan_tempo"],
    "Fusiform (FFA)":     ["G_oc-temp_lat-fusifor"],
    "Parahipp. (PPA)":    ["G_oc-temp_med-Parahip"],
    "Frontal sup.":       ["G_front_sup"],
    "Angular/TPJ":        ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    "Precuneus":          ["G_precuneus"],
    "Motor":              ["G_precentral"],
    "Somatosensory":      ["G_postcentral"],
    "Temporal mid.":      ["G_temporal_middle"],
    "Temporal inf.":      ["G_temporal_inf"],
    "Insula":             ["G_insular_short", "G_Ins_lg_and_S_cent_ins"],
    "Cingulate ant.":     ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
    "Cingulate post.":    ["G_cingul-Post-dorsal", "G_cingul-Post-ventral"],
    "Orbital frontal":    ["G_orbital", "G_rectus"],
    "Frontal mid.":       ["G_front_middle"],
    "Temporal pole":      ["Pole_temporal"],
}


def get_roi_indices(roi_map: Optional[Dict[str, List[str]]] = None) -> Dict[str, np.ndarray]:
    """Build ROI -> vertex index mapping on fsaverage5 (20,484 vertices).

    Reimplemented here to avoid cross-repo dependency at import time.
    Uses nilearn's Destrieux atlas.
    """
    import warnings
    from nilearn import datasets

    if roi_map is None:
        roi_map = ROI_LABEL_MAP

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas = datasets.fetch_atlas_surf_destrieux()

    lh = np.array(atlas["map_left"])
    rh = np.array(atlas["map_right"])
    full_map = np.concatenate([lh, rh])

    labels = [str(l) for l in atlas["labels"]]

    roi_indices = {}
    for friendly_name, atlas_names in roi_map.items():
        indices = []
        for aname in atlas_names:
            if aname in labels:
                label_idx = labels.index(aname)
                indices.append(np.where(full_map == label_idx)[0])
        if indices:
            roi_indices[friendly_name] = np.concatenate(indices)

    return roi_indices


def roi_means(preds: np.ndarray, roi_indices: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute mean activation per ROI from vertex predictions."""
    act = preds.mean(axis=0) if preds.ndim == 2 else preds
    return {name: float(act[idx].mean()) for name, idx in roi_indices.items()}


def roi_to_canvas_vector(
    roi_values: Dict[str, float],
    canvas_region_names: List[str],
) -> np.ndarray:
    """Convert ROI means dict to a vector aligned with canvas region ordering.

    For canvas regions mapped to multiple ROIs, takes the mean.
    For canvas regions with no ROI mapping, uses 0.0.
    """
    vec = np.zeros(len(canvas_region_names), dtype=np.float32)
    for i, region_name in enumerate(canvas_region_names):
        rois_for_region = CANVAS_TO_ROIS.get(region_name, [])
        if rois_for_region:
            vals = [roi_values.get(r, 0.0) for r in rois_for_region]
            vec[i] = np.mean(vals)
    return vec


def generate_dataset_from_model(model, categories: Optional[Dict] = None) -> Dict:
    """Generate the full cortical dataset using a loaded TRIBE v2 model.

    Args:
        model: Loaded TRIBE v2 model (from core.model.load_model)
        categories: Optional stimulus categories dict. Defaults to
            STIMULUS_CATEGORIES from cortical_canvas.py.

    Returns:
        Dict with keys ready for np.savez:
        - region_activations: (n_samples, n_regions) float32
        - raw_preds: (n_samples, 20484) float32  (full vertex predictions)
        - labels: (n_samples,) int category indices
        - texts: list of stimulus texts
        - category_names: list of category name strings
        - region_names: list of canvas region path strings
        - roi_means_all: (n_samples, n_rois) float32
        - roi_names: list of ROI name strings
    """
    from core.model import text_to_predictions

    if categories is None:
        categories = STIMULUS_CATEGORIES

    # Get ROI indices for Destrieux atlas
    print("Building ROI index map...")
    roi_idx = get_roi_indices()
    roi_names_sorted = sorted(roi_idx.keys())
    print(f"  {len(roi_idx)} ROIs, {sum(len(v) for v in roi_idx.values())} total vertices")

    # Get canvas region names
    canvas_regions = get_region_names()
    print(f"  {len(canvas_regions)} canvas regions")

    category_names = sorted(categories.keys())
    all_region_acts = []
    all_raw_preds = []
    all_labels = []
    all_texts = []
    all_roi_means = []

    for cat_idx, cat_name in enumerate(category_names):
        print(f"\nCategory: {cat_name} ({len(categories[cat_name])} stimuli)")
        for text in categories[cat_name]:
            t0 = time.time()
            preds, _ = text_to_predictions(model, text, label=cat_name)
            elapsed = time.time() - t0
            print(f"  [{elapsed:.1f}s] {preds.shape[0]} timesteps x {preds.shape[1]} vertices")

            # Compute ROI means
            rmeans = roi_means(preds, roi_idx)
            roi_vec = np.array([rmeans.get(r, 0.0) for r in roi_names_sorted], dtype=np.float32)

            # Map to canvas regions
            canvas_vec = roi_to_canvas_vector(rmeans, canvas_regions)

            all_region_acts.append(canvas_vec)
            all_raw_preds.append(preds.mean(axis=0).astype(np.float32))
            all_labels.append(cat_idx)
            all_texts.append(text)
            all_roi_means.append(roi_vec)

    return {
        "region_activations": np.stack(all_region_acts),
        "raw_preds": np.stack(all_raw_preds),
        "labels": np.array(all_labels),
        "texts": all_texts,
        "category_names": category_names,
        "region_names": canvas_regions,
        "roi_means_all": np.stack(all_roi_means),
        "roi_names": roi_names_sorted,
    }


def save_dataset(data: Dict, output_path: str) -> None:
    """Save generated dataset to .npz file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Separate array data from string lists
    np.savez(
        path,
        region_activations=data["region_activations"],
        raw_preds=data["raw_preds"],
        labels=data["labels"],
        roi_means_all=data["roi_means_all"],
    )

    # Save metadata as JSON alongside
    meta_path = path.with_suffix(".json")
    meta = {
        "texts": data["texts"],
        "category_names": data["category_names"],
        "region_names": data["region_names"],
        "roi_names": data["roi_names"],
        "n_samples": int(data["labels"].shape[0]),
        "n_regions": int(data["region_activations"].shape[1]),
        "n_categories": len(data["category_names"]),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved dataset: {path} ({data['region_activations'].shape})")
    print(f"Saved metadata: {meta_path}")


def load_dataset(npz_path: str) -> Dict:
    """Load a saved dataset from .npz + .json metadata."""
    path = Path(npz_path)
    data = dict(np.load(path, allow_pickle=False))

    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        meta = json.load(f)

    data["texts"] = meta["texts"]
    data["category_names"] = meta["category_names"]
    data["region_names"] = meta["region_names"]
    data["roi_names"] = meta["roi_names"]

    return data


def generate_synthetic_dataset(
    n_per_category: int = 8,
    n_regions: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """Generate a synthetic dataset for testing without TRIBE v2.

    Creates plausible cortical activation patterns using category-specific
    activation profiles. Useful for testing the training pipeline offline.
    """
    rng = np.random.RandomState(seed)
    canvas_regions = get_region_names()
    if n_regions is None:
        n_regions = len(canvas_regions)

    category_names = sorted(STIMULUS_CATEGORIES.keys())
    n_cats = len(category_names)

    # Define category-specific activation profiles
    # Each category activates certain regions more strongly
    category_profiles = {
        "animal":   {"visual.v1": 0.8, "visual.v2_v4": 0.7, "visual.fusiform": 0.9,
                     "language.temporal_mid": 0.5, "default_mode.temporal_pole": 0.4},
        "music":    {"auditory.a1": 0.9, "auditory.wernicke": 0.7,
                     "frontal.premotor": 0.5, "subcortical.insula": 0.6},
        "danger":   {"subcortical.insula": 0.9, "default_mode.cingulate": 0.8,
                     "frontal.prefrontal": 0.7, "visual.v1": 0.5},
        "spatial":  {"visual.v1": 0.7, "visual.v2_v4": 0.8, "language.angular": 0.7,
                     "default_mode.precuneus": 0.6},
        "social":   {"default_mode.precuneus": 0.7, "default_mode.temporal_pole": 0.8,
                     "language.angular": 0.6, "frontal.prefrontal": 0.7,
                     "visual.fusiform": 0.5},
        "language": {"auditory.wernicke": 0.8, "language.broca": 0.9,
                     "language.angular": 0.7, "language.temporal_mid": 0.8,
                     "frontal.prefrontal": 0.5},
        "motor":    {"frontal.motor": 0.9, "frontal.premotor": 0.8,
                     "subcortical.somatosensory": 0.7, "visual.v1": 0.3},
        "visual":   {"visual.v1": 0.9, "visual.v2_v4": 0.8, "visual.fusiform": 0.6,
                     "language.angular": 0.4},
        "emotion":  {"subcortical.insula": 0.8, "default_mode.cingulate": 0.7,
                     "frontal.prefrontal": 0.6, "default_mode.precuneus": 0.5,
                     "default_mode.temporal_pole": 0.6},
    }

    all_acts = []
    all_labels = []
    all_texts = []

    for cat_idx, cat_name in enumerate(category_names):
        profile = category_profiles.get(cat_name, {})
        stims = STIMULUS_CATEGORIES[cat_name][:n_per_category]

        for text in stims:
            # Base activation: low noise
            act = rng.randn(n_regions).astype(np.float32) * 0.1

            # Add category-specific activations
            for region_name, strength in profile.items():
                if region_name in canvas_regions:
                    idx = canvas_regions.index(region_name)
                    act[idx] += strength + rng.randn() * 0.15

            all_acts.append(act)
            all_labels.append(cat_idx)
            all_texts.append(text)

    # Create dummy ROI means for compatibility
    roi_names = sorted(ROI_LABEL_MAP.keys())
    roi_acts = rng.randn(len(all_labels), len(roi_names)).astype(np.float32) * 0.1

    return {
        "region_activations": np.stack(all_acts),
        "raw_preds": np.zeros((len(all_labels), 20484), dtype=np.float32),
        "labels": np.array(all_labels),
        "texts": all_texts,
        "category_names": category_names,
        "region_names": canvas_regions,
        "roi_means_all": roi_acts,
        "roi_names": roi_names,
    }


# ---- Modal function for GPU inference ----

try:
    import modal

    app = modal.App("cortical-canvas-data")

    brain_image = (
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
    )

    @app.function(
        image=brain_image,
        gpu="A10G",
        timeout=7200,
        memory=32768,
        secrets=[modal.Secret.from_name("huggingface", environment_name="test-20260327")],
    )
    def generate_on_modal() -> bytes:
        """Run TRIBE v2 inference on Modal GPU and return .npz bytes."""
        import io
        import tempfile

        # Load model
        from core.model import load_model
        model = load_model()

        # Generate dataset
        data = generate_dataset_from_model(model)

        # Save to bytes
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            save_dataset(data, tmp.name)
            # Read back the npz
            npz_bytes = Path(tmp.name).read_bytes()
            meta_bytes = Path(tmp.name).with_suffix(".json").read_text()

        return npz_bytes, meta_bytes

except ImportError:
    # Modal not installed -- only local utilities available
    app = None
    generate_on_modal = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate cortical activation dataset")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic dataset (no GPU needed)")
    parser.add_argument("--output", default="research/brain/results/cortical_dataset.npz",
                        help="Output path for .npz file")
    parser.add_argument("--n-per-category", type=int, default=8,
                        help="Number of stimuli per category (synthetic mode)")
    args = parser.parse_args()

    output = Path(_CE_ROOT) / args.output

    if args.synthetic:
        print("Generating synthetic dataset...")
        data = generate_synthetic_dataset(n_per_category=args.n_per_category)
        save_dataset(data, str(output))
        print(f"\nDataset shape: {data['region_activations'].shape}")
        print(f"Categories: {data['category_names']}")
        print(f"Regions: {len(data['region_names'])}")
    else:
        print("Use --synthetic for local generation, or run via Modal:")
        print("  modal run research/brain/data_pipeline.py")

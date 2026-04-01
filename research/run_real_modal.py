"""Run ALL research tracks with REAL data, REAL training, REAL visualizations.

- Brain: TRIBE v2 cortical predictions (GPU), 200 epochs, d_model=128
- Robotics: 50 epochs imitation + 200 episodes self-play, scaling 2/4/8/16
- Browser: train + generate rollout animation

All on Modal. No synthetic data.

Usage:
    modal run research/run_real_modal.py              # all tracks
    modal run research/run_real_modal.py --track brain
"""

import modal
import base64
import io
import os
import sys
import tarfile
from pathlib import Path

app = modal.App("canvas-research-real")

# Shared base for browser + robotics (CPU only)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "canvas-engineering>=0.4.0",
        "torch", "numpy", "matplotlib", "scipy",
        "scikit-learn", "Pillow",
    )
    .apt_install("ffmpeg")
    .add_local_dir("research", "/root/research", copy=True,
                    ignore=["*/results/*", "*/__pycache__/*", "*/.DS_Store"])
    .add_local_dir("canvas_engineering", "/root/canvas_engineering", copy=True,
                    ignore=["__pycache__", ".DS_Store"])
)

# Brain needs TRIBE v2 + GPU
brain_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git", "libgl1", "libglib2.0-0", "libxrender1")
    .pip_install("torch", "torchaudio")
    .run_commands("pip install git+https://github.com/facebookresearch/tribev2.git")
    .run_commands("python -m spacy download en_core_web_lg")
    .pip_install(
        "canvas-engineering>=0.4.0",
        "gtts", "langdetect", "soundfile", "matplotlib", "numpy",
        "nilearn", "scikit-learn", "mne", "scipy", "Pillow",
        "transformers>=4.45,<4.50",
    )
    .add_local_dir("research/brain", "/root/research/brain", copy=True,
                    ignore=["results/*", "__pycache__"])
    .add_local_dir("canvas_engineering", "/root/canvas_engineering", copy=True,
                    ignore=["__pycache__"])
)

RESULTS_DIR = Path(__file__).parent


def _tar_dir(path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in Path(path).rglob("*"):
            if f.is_file() and f.stat().st_size < 100_000_000:
                tar.add(str(f), arcname=str(f.relative_to(path)))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _untar(b64_data, dest):
    tar_bytes = base64.b64decode(b64_data)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(path=dest, filter="data")


# ── BRAIN: Real TRIBE v2 data ──────────────────────────────────────


@app.function(
    image=brain_image, gpu="A10G", timeout=10800, cpu=4, memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret", environment_name="test-20260327")],
)
def run_brain_real():
    """Brain with REAL TRIBE v2 cortical predictions. Full pipeline."""
    import subprocess, sys, os
    os.makedirs("/root/research/brain/results", exist_ok=True)

    script = '''
import sys, os, json, hashlib, numpy as np, torch
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/research/brain')
os.chdir('/root/research/brain')

from cortical_canvas import (
    STIMULUS_CATEGORIES, ROI_TO_CANVAS, build_cortical_brain,
    CORTICAL_PATHWAYS,
)

print("=" * 60)
print("PHASE 1: Generate TRIBE v2 cortical predictions")
print("=" * 60)

# Load TRIBE v2
from pathlib import Path as P
CACHE = P("./cache")
CACHE.mkdir(exist_ok=True)

from tribev2 import TribeModel
print("Loading TRIBE v2...")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
for attr in ["text_feature", "audio_feature", "video_feature"]:
    feat = getattr(model.data, attr, None)
    if feat is not None and hasattr(feat, "infra"):
        feat.infra.mode = "force"
print("TRIBE v2 loaded.")

# Get ROI indices
from nilearn import datasets
import nibabel as nib
import warnings

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

print(f"Mapped {len(roi_indices)} ROIs")

# Generate predictions for all stimuli
import pandas as pd
import soundfile as sf
from gtts import gTTS
from langdetect import detect
from neuralset.events.transforms import (
    AddContextToWords, AddSentenceToWords, AddText,
    ChunkEvents, RemoveMissing, standardize_events,
)

def text_to_preds(text, label=""):
    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    audio_dir = CACHE / f"brain_{text_hash}"
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
    word_events = [{"type": "Word", "text": w.strip(".,;:!?\\\\"\\'-()"),
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

# Map ROI names to canvas region names
roi_to_canvas = {
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

# Get canvas region names from build
from canvas_engineering import compile_program
bound, program = build_cortical_brain()
canvas_region_names = list(bound.field_names)
print(f"Canvas regions: {len(canvas_region_names)}")

# Process all stimuli
all_activations = []
all_labels = []
cat_names = sorted(STIMULUS_CATEGORIES.keys())

for cat_idx, cat_name in enumerate(cat_names):
    stimuli = STIMULUS_CATEGORIES[cat_name]
    for text in stimuli:
        preds = text_to_preds(text, label=cat_name)

        # Map vertex predictions to ROI means
        roi_means = {}
        for roi_name, vertex_idx in roi_indices.items():
            roi_means[roi_name] = float(preds[:, vertex_idx].mean())

        # Map ROI means to canvas region activations
        region_act = np.zeros(len(canvas_region_names))
        for roi_name, canvas_name in roi_to_canvas.items():
            if roi_name in roi_means and canvas_name in canvas_region_names:
                idx = canvas_region_names.index(canvas_name)
                region_act[idx] += roi_means[roi_name]

        all_activations.append(region_act)
        all_labels.append(cat_idx)

X = np.stack(all_activations)
y = np.array(all_labels)
print(f"\\nDataset: {X.shape[0]} samples, {X.shape[1]} regions, {len(cat_names)} categories")

# Save
np.savez('results/tribe_data.npz', region_activations=X, labels=y)
with open('results/tribe_data_meta.json', 'w') as f:
    json.dump({
        'n_samples': len(y),
        'n_categories': len(cat_names),
        'categories': cat_names,
        'regions': canvas_region_names,
    }, f, indent=2)

# PHASE 2: Train
print("\\n" + "=" * 60)
print("PHASE 2: Train cortical canvas model (200 epochs)")
print("=" * 60)

from train import run_all_baselines
run_all_baselines(
    data_path='results/tribe_data.npz',
    results_dir='results',
    n_epochs=200,
    d_model=128,
    n_layers=3,
    n_heads=8,
    lr=1e-3,
)

# PHASE 3: Evaluate
print("\\n" + "=" * 60)
print("PHASE 3: Generate visualizations")
print("=" * 60)

from evaluate import generate_all_plots
generate_all_plots('results')

print("\\nDone!")
'''
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    output = []
    for line in proc.stdout:
        print(f"[brain] {line}", end="", flush=True)
        output.append(line)
    proc.wait()

    return {
        "status": "ok" if proc.returncode == 0 else "fail",
        "results_tar": _tar_dir("/root/research/brain/results"),
        "returncode": proc.returncode,
    }


# ── ROBOTICS: More training + scaling ───────────────────────────────


@app.function(image=cpu_image, timeout=7200, cpu=8, memory=32768)
def run_robotics_real():
    """Robotics with serious training and scaling analysis."""
    import subprocess, sys, os
    os.makedirs("/root/research/robotics/results", exist_ok=True)

    script = '''
import sys, os
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/research/robotics')
os.chdir('/root/research/robotics')

from run import main
import sys
sys.argv = ['run.py']  # no --fast, no --no-scaling = full run
main()
'''
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    output = []
    for line in proc.stdout:
        print(f"[robotics] {line}", end="", flush=True)
        output.append(line)
    proc.wait()

    return {
        "status": "ok" if proc.returncode == 0 else "fail",
        "results_tar": _tar_dir("/root/research/robotics/results"),
        "returncode": proc.returncode,
    }


# ── BROWSER: Train + rollout animation ──────────────────────────────


@app.function(image=cpu_image, timeout=3600, cpu=8, memory=16384)
def run_browser_real():
    """Browser agent with rollout animation."""
    import subprocess, sys, os
    os.makedirs("/root/research/browser/results", exist_ok=True)

    # Run the main script which trains all 3 models
    script = '''
import sys, os
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/research/browser')
os.chdir('/root/research/browser')

from run import main
main()
'''
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    output = []
    for line in proc.stdout:
        print(f"[browser] {line}", end="", flush=True)
        output.append(line)
    proc.wait()

    return {
        "status": "ok" if proc.returncode == 0 else "fail",
        "results_tar": _tar_dir("/root/research/browser/results"),
        "returncode": proc.returncode,
    }


@app.local_entrypoint()
def main(track: str = "all"):
    tracks = []
    if track in ("all", "brain"):
        tracks.append(("brain", run_brain_real))
    if track in ("all", "robotics"):
        tracks.append(("robotics", run_robotics_real))
    if track in ("all", "browser"):
        tracks.append(("browser", run_browser_real))

    print(f"Running {len(tracks)} REAL track(s) on Modal...")
    futures = {name: fn.spawn() for name, fn in tracks}

    for name, future in futures.items():
        print(f"\n{'=' * 60}")
        print(f"REAL RESULTS: {name}")
        print(f"{'=' * 60}")
        try:
            result = future.get()
            print(f"  Status: {result['status']}")
            print(f"  Return code: {result['returncode']}")
            if result.get("results_tar"):
                dest = RESULTS_DIR / name / "results"
                dest.mkdir(parents=True, exist_ok=True)
                _untar(result["results_tar"], str(dest))
                files = [f for f in dest.rglob("*") if f.is_file()]
                print(f"  Downloaded {len(files)} files:")
                for f in sorted(files):
                    print(f"    {f.relative_to(dest)} ({f.stat().st_size // 1024}KB)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nAll done.")

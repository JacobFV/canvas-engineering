"""Run deeper research experiments on Modal with more compute.

Usage:
    modal run research/run_deep_modal.py --track brain     # 200 epochs, d_model=128
    modal run research/run_deep_modal.py --track robotics  # 50 epochs imitation, 100 episodes
    modal run research/run_deep_modal.py --track all
"""

import modal
import base64
import io
import os
import sys
import tarfile
from pathlib import Path

app = modal.App("canvas-research-deep")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "canvas-engineering>=0.4.0",
        "torch", "numpy", "matplotlib", "scipy",
        "scikit-learn", "gymnasium",
    )
    .apt_install("ffmpeg")
    .add_local_dir("research", "/root/research", copy=True,
                    ignore=["*/results/*", "*/__pycache__/*"])
    .add_local_dir("canvas_engineering", "/root/canvas_engineering", copy=True,
                    ignore=["__pycache__"])
)

brain_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git", "libgl1", "libglib2.0-0", "libxrender1")
    .pip_install("torch", "torchaudio")
    .run_commands("pip install git+https://github.com/facebookresearch/tribev2.git")
    .run_commands("python -m spacy download en_core_web_lg")
    .pip_install(
        "canvas-engineering>=0.4.0",
        "gtts", "langdetect", "soundfile", "matplotlib", "numpy",
        "nilearn", "scikit-learn", "mne", "scipy",
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
            if f.is_file() and f.stat().st_size < 50_000_000:
                tar.add(str(f), arcname=str(f.relative_to(path)))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _untar(b64_data, dest):
    tar_bytes = base64.b64decode(b64_data)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(path=dest, filter="data")


@app.function(image=image, timeout=3600, cpu=8, memory=32768)
def run_robotics_deep():
    """Robotics with more training: 20 epochs imitation, 50 episodes self-play."""
    import subprocess, sys, os
    os.makedirs("/root/research/robotics/results", exist_ok=True)

    # Patch run.py to use more training
    run_script = """
import sys, os
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/research/robotics')
os.chdir('/root/research/robotics')

from train import train_all_models
from evaluate import generate_all_plots
import json

results = train_all_models(
    n_robots=4,
    d_model=64,
    n_layers=2,
    imitation_epochs=20,
    selfplay_episodes=50,
    eval_episodes=20,
    results_dir='results',
)
generate_all_plots('results')
print('Done!')
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", run_script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    output = []
    for line in proc.stdout:
        print(f"[robotics] {line}", end="", flush=True)
        output.append(line)
    proc.wait()
    return {"status": "ok" if proc.returncode == 0 else "fail",
            "results_tar": _tar_dir("/root/research/robotics/results"),
            "returncode": proc.returncode}


@app.function(
    image=brain_image, gpu="A10G", timeout=7200, cpu=4, memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret", environment_name="test-20260327")],
)
def run_brain_deep():
    """Brain with TRIBE v2 data + more training: 200 epochs, d_model=128."""
    import subprocess, sys, os
    os.makedirs("/root/research/brain/results", exist_ok=True)

    # Step 1: Generate TRIBE v2 data
    data_script = """
import sys, os, json, numpy as np
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/research/brain')
os.chdir('/root/research/brain')

from cortical_canvas import STIMULUS_CATEGORIES, ROI_TO_CANVAS
from data_pipeline import generate_synthetic_dataset

# Use synthetic for now but with more samples per category
d = generate_synthetic_dataset(samples_per_category=16)
np.savez('results/data.npz',
         region_activations=d['region_activations'],
         labels=d['labels'])
with open('results/data_meta.json', 'w') as f:
    json.dump({
        'n_samples': len(d['labels']),
        'n_categories': len(d['category_names']),
        'categories': d['category_names'],
        'regions': d['region_names'],
    }, f, indent=2)
print(f"Data: {d['region_activations'].shape}")
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", data_script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout:
        print(f"[brain-data] {line}", end="", flush=True)
    proc.wait()

    # Step 2: Train with more epochs
    train_script = """
import sys, os
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/research/brain')
os.chdir('/root/research/brain')

from train import run_all_baselines
run_all_baselines(
    data_path='results/data.npz',
    results_dir='results',
    n_epochs=200,
    d_model=128,
    n_layers=3,
    n_heads=8,
    lr=1e-3,
)
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", train_script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    output = []
    for line in proc.stdout:
        print(f"[brain-train] {line}", end="", flush=True)
        output.append(line)
    proc.wait()

    # Step 3: Evaluate
    eval_script = """
import sys, os
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/research/brain')
os.chdir('/root/research/brain')

from evaluate import generate_all_plots
generate_all_plots('results')
"""
    proc2 = subprocess.Popen(
        [sys.executable, "-u", "-c", eval_script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    for line in proc2.stdout:
        print(f"[brain-eval] {line}", end="", flush=True)
    proc2.wait()

    return {"status": "ok" if proc.returncode == 0 else "fail",
            "results_tar": _tar_dir("/root/research/brain/results"),
            "returncode": proc.returncode}


@app.local_entrypoint()
def main(track: str = "all"):
    tracks = []
    if track in ("all", "robotics"):
        tracks.append(("robotics", run_robotics_deep))
    if track in ("all", "brain"):
        tracks.append(("brain", run_brain_deep))

    print(f"Running {len(tracks)} deep track(s) on Modal...")
    futures = {name: fn.spawn() for name, fn in tracks}

    for name, future in futures.items():
        print(f"\n{'=' * 60}")
        print(f"DEEP RESULTS: {name}")
        print(f"{'=' * 60}")
        try:
            result = future.get()
            print(f"  Status: {result['status']}")
            if result["results_tar"]:
                dest = RESULTS_DIR / name / "results"
                dest.mkdir(parents=True, exist_ok=True)
                _untar(result["results_tar"], str(dest))
                files = [f for f in dest.rglob("*") if f.is_file()]
                print(f"  Downloaded {len(files)} files to {dest}/")
                for f in sorted(files):
                    print(f"    {f.relative_to(dest)} ({f.stat().st_size // 1024}KB)")
        except Exception as e:
            print(f"  ERROR: {e}")
    print("\nDone.")

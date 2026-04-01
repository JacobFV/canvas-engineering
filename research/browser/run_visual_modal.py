"""Run browser agent training + rollout visualization on Modal.

Trains the canvas model, saves checkpoint, then generates a rollout GIF
showing the agent's step-by-step interaction with the browser environment.

Usage:
    modal run research/browser/run_visual_modal.py
"""

import modal
import base64
import io
import tarfile
from pathlib import Path

app = modal.App("canvas-browser-visual")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "canvas-engineering>=0.4.0",
        "torch", "numpy", "matplotlib", "scipy", "scikit-learn", "Pillow",
    )
    .add_local_dir("research/browser", "/root/research/browser", copy=True,
                    ignore=["results/*", "__pycache__"])
    .add_local_dir("canvas_engineering", "/root/canvas_engineering", copy=True,
                    ignore=["__pycache__"])
)

RESULTS_DIR = Path(__file__).parent / "results"


@app.function(image=image, timeout=1800, cpu=8, memory=16384)
def train_and_visualize():
    """Train canvas browser agent, save checkpoint, generate rollout GIF."""
    import subprocess, sys, os

    os.makedirs("/root/research/browser/results", exist_ok=True)

    # Train with checkpoint saving
    train_script = """
import sys, os, torch, json
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/research/browser')
os.chdir('/root/research/browser')

from train import train

print("=== Training Canvas Browser Agent ===")
model, history = train(mode='canvas', d_model=128, n_layers=4, epochs=80)

# Save checkpoint
torch.save(model.state_dict(), 'results/checkpoint_canvas.pt')
print("Saved checkpoint")
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", train_script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout:
        print(f"[train] {line}", end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        return {"status": "train_failed", "returncode": proc.returncode}

    # Generate rollout visualization
    viz_script = """
import sys, os
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/research/browser')
os.chdir('/root/research/browser')

from visualize_rollout import main
main()
"""
    proc2 = subprocess.Popen(
        [sys.executable, "-u", "-c", viz_script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    for line in proc2.stdout:
        print(f"[viz] {line}", end="", flush=True)
    proc2.wait()

    # Tar results
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in Path("/root/research/browser/results").rglob("*"):
            if f.is_file() and f.stat().st_size < 50_000_000:
                tar.add(str(f), arcname=str(f.relative_to("/root/research/browser/results")))
    buf.seek(0)

    return {
        "status": "ok",
        "results_tar": base64.b64encode(buf.read()).decode(),
    }


@app.local_entrypoint()
def main():
    print("Training + visualizing browser agent on Modal...")
    result = train_and_visualize.remote()

    print(f"\nStatus: {result['status']}")
    if result.get("results_tar"):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tar_bytes = base64.b64decode(result["results_tar"])
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(path=str(RESULTS_DIR), filter="data")
        files = [f for f in RESULTS_DIR.rglob("*") if f.is_file()]
        print(f"Downloaded {len(files)} files:")
        for f in sorted(files):
            print(f"  {f.relative_to(RESULTS_DIR)} ({f.stat().st_size // 1024}KB)")

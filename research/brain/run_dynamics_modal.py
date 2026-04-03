"""Run the 135-feature cortical dynamics experiment on Modal.

Standalone script — doesn't use run_modal.py or volumes.
Just runs, streams output, and returns results via function return.

Usage:
    modal run research/brain/run_dynamics_modal.py
"""

import modal
import base64
import io
import tarfile
from pathlib import Path

app = modal.App("brain-dynamics-135")

results_vol = modal.Volume.from_name("brain-dynamics-results", create_if_missing=True)

image = (
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

RESULTS_DIR = Path(__file__).parent / "results"


@app.function(
    image=image, gpu="A10G", timeout=28800, cpu=4, memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret", environment_name="test-20260327")],
    volumes={"/vol": results_vol},
)
def run():
    import subprocess, sys, os, shutil
    # Symlink results to volume so every file write persists immediately
    if os.path.exists("/vol/results"):
        shutil.rmtree("/vol/results")
    os.makedirs("/vol/results", exist_ok=True)
    if os.path.exists("/root/research/brain/results"):
        subprocess.run(["rm", "-rf", "/root/research/brain/results"])
    os.symlink("/vol/results", "/root/research/brain/results")

    proc = subprocess.Popen(
        [sys.executable, "-u", "/root/research/brain/run_dynamics_pipeline.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root/research/brain",
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    results_vol.commit()

    # Also return tar for attached clients
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in Path("/vol/results").rglob("*"):
            if f.is_file() and f.stat().st_size < 100_000_000:
                tar.add(str(f), arcname=str(f.relative_to("/vol/results")))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/vol": results_vol},
)
def collect():
    """Download results from the volume."""
    buf = io.BytesIO()
    results_path = Path("/vol/results")
    if not results_path.exists():
        return ""
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in results_path.rglob("*"):
            if f.is_file():
                tar.add(str(f), arcname=str(f.relative_to(results_path)))
                print("  {}  ({}KB)".format(f.relative_to(results_path), f.stat().st_size // 1024))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.local_entrypoint()
def main(collect_only: bool = False):
    if collect_only:
        print("Collecting results from volume...")
        result_tar = collect.remote()
        if not result_tar:
            print("No results yet.")
            return
    else:
        print("Running 135-feature cortical dynamics on Modal GPU...")
        print("Results save to volume in real-time (safe to close laptop)")
        print("Use --collect-only to download results later")
        result_tar = run.remote()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tar_bytes = base64.b64decode(result_tar)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(path=str(RESULTS_DIR), filter="data")

    files = [f for f in RESULTS_DIR.rglob("*") if f.is_file()]
    print("\nDownloaded {} files:".format(len(files)))
    for f in sorted(files):
        print("  {} ({}KB)".format(f.relative_to(RESULTS_DIR), f.stat().st_size // 1024))

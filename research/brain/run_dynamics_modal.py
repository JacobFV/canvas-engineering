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
)
def run():
    import subprocess, sys, os
    os.makedirs("/root/research/brain/results", exist_ok=True)

    proc = subprocess.Popen(
        [sys.executable, "-u", "/root/research/brain/run_dynamics_pipeline.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root/research/brain",
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    # Tar results
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in Path("/root/research/brain/results").rglob("*"):
            if f.is_file() and f.stat().st_size < 100_000_000:
                tar.add(str(f), arcname=str(f.relative_to("/root/research/brain/results")))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.local_entrypoint()
def main():
    print("Running 135-feature cortical dynamics on Modal GPU...")
    result_tar = run.remote()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tar_bytes = base64.b64decode(result_tar)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(path=str(RESULTS_DIR), filter="data")

    files = [f for f in RESULTS_DIR.rglob("*") if f.is_file()]
    print("\nDownloaded {} files:".format(len(files)))
    for f in sorted(files):
        print("  {} ({}KB)".format(f.relative_to(RESULTS_DIR), f.stat().st_size // 1024))

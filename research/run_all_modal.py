"""Run all three research experiments on Modal.

Usage:
    modal run research/run_all_modal.py                    # all three
    modal run research/run_all_modal.py --track brain      # just brain
    modal run research/run_all_modal.py --track browser     # just browser
    modal run research/run_all_modal.py --track robotics    # just robotics
"""

import modal
import base64
import io
import os
import sys
import tarfile
from pathlib import Path

app = modal.App("canvas-research")

# Image with all deps for all three tracks
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "canvas-engineering>=0.4.0",
        "torch", "numpy", "matplotlib", "scipy",
        "scikit-learn", "gymnasium",
    )
    .apt_install("ffmpeg")
    .add_local_dir("research", "/root/research", copy=True,
                    ignore=["*/results/*", "*/__pycache__/*", "*/.DS_Store"])
    .add_local_dir("canvas_engineering", "/root/canvas_engineering", copy=True,
                    ignore=["__pycache__", ".DS_Store"])
)

# Brain track needs TRIBE v2 + GPU — separate image
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
                    ignore=["results/*", "__pycache__", ".DS_Store"])
    .add_local_dir("canvas_engineering", "/root/canvas_engineering", copy=True,
                    ignore=["__pycache__", ".DS_Store"])
)

RESULTS_DIR = Path(__file__).parent


@app.function(image=image, timeout=1800, cpu=8, memory=32768)
def run_robotics():
    """Run the multi-robot fleet experiment."""
    import subprocess, sys, os
    os.makedirs("/root/research/robotics/results", exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "/root/research/robotics/run.py", "--fast", "--no-scaling"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    output = []
    for line in proc.stdout:
        print(f"[robotics] {line}", end="", flush=True)
        output.append(line)
    proc.wait()

    # Collect results
    results_tar = _tar_dir("/root/research/robotics/results")
    return {"status": "ok" if proc.returncode == 0 else "fail",
            "output": "".join(output[-50:]),
            "results_tar": results_tar,
            "returncode": proc.returncode}


@app.function(image=image, timeout=1800, cpu=8, memory=32768)
def run_browser():
    """Run the browser agent experiment."""
    import subprocess, sys, os
    os.makedirs("/root/research/browser/results", exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "/root/research/browser/run.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    output = []
    for line in proc.stdout:
        print(f"[browser] {line}", end="", flush=True)
        output.append(line)
    proc.wait()

    results_tar = _tar_dir("/root/research/browser/results")
    return {"status": "ok" if proc.returncode == 0 else "fail",
            "output": "".join(output[-50:]),
            "results_tar": results_tar,
            "returncode": proc.returncode}


@app.function(
    image=brain_image, gpu="A10G", timeout=3600, cpu=4, memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret", environment_name="test-20260327")],
)
def run_brain():
    """Run the cortical brain model experiment with TRIBE v2."""
    import subprocess, sys, os
    os.makedirs("/root/research/brain/results", exist_ok=True)

    # First generate synthetic data (fast, no GPU needed)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import sys; sys.path.insert(0, '/root/research/brain'); "
         "from data_pipeline import generate_synthetic_dataset; "
         "import numpy as np; "
         "d = generate_synthetic_dataset(); "
         "np.savez('/root/research/brain/results/synthetic_data.npz', "
         "region_activations=d['region_activations'], labels=d['labels']); "
         "print('Data generated:', d['region_activations'].shape)"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root", env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout:
        print(f"[brain-data] {line}", end="", flush=True)
    proc.wait()

    # Then train
    proc = subprocess.Popen(
        [sys.executable, "-u", "/root/research/brain/train.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root/research/brain",
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    output = []
    for line in proc.stdout:
        print(f"[brain] {line}", end="", flush=True)
        output.append(line)
    proc.wait()

    # Then evaluate
    proc2 = subprocess.Popen(
        [sys.executable, "-u", "/root/research/brain/evaluate.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/root/research/brain",
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
    )
    for line in proc2.stdout:
        print(f"[brain-eval] {line}", end="", flush=True)
        output.append(line)
    proc2.wait()

    results_tar = _tar_dir("/root/research/brain/results")
    return {"status": "ok" if proc.returncode == 0 and proc2.returncode == 0 else "fail",
            "output": "".join(output[-50:]),
            "results_tar": results_tar,
            "returncode": proc.returncode}


def _tar_dir(path):
    """Tar a directory and return base64-encoded bytes."""
    import tarfile, io, base64
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in Path(path).rglob("*"):
            if f.is_file() and f.stat().st_size < 50_000_000:  # skip >50MB
                tar.add(str(f), arcname=str(f.relative_to(path)))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _untar(b64_data, dest):
    """Untar base64-encoded data to destination."""
    import tarfile, io, base64
    tar_bytes = base64.b64decode(b64_data)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(path=dest, filter="data")


@app.local_entrypoint()
def main(track: str = "all"):
    tracks = []
    if track in ("all", "robotics"):
        tracks.append(("robotics", run_robotics))
    if track in ("all", "browser"):
        tracks.append(("browser", run_browser))
    if track in ("all", "brain"):
        tracks.append(("brain", run_brain))

    print(f"Running {len(tracks)} research track(s) on Modal...")

    # Launch all in parallel
    futures = {name: fn.spawn() for name, fn in tracks}

    for name, future in futures.items():
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {name}")
        print(f"{'=' * 60}")

        try:
            result = future.get()
            print(f"  Status: {result['status']}")
            print(f"  Return code: {result['returncode']}")

            if result["results_tar"]:
                dest = RESULTS_DIR / name / "results"
                dest.mkdir(parents=True, exist_ok=True)
                _untar(result["results_tar"], str(dest))
                files = list(dest.rglob("*"))
                file_list = [f for f in files if f.is_file()]
                print(f"  Downloaded {len(file_list)} result files to {dest}/")
                for f in sorted(file_list):
                    size = f.stat().st_size
                    print(f"    {f.relative_to(dest)} ({size // 1024}KB)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")

"""Run all research tracks on Modal.

Each track uses a proper script file — no inline Python, no escape issues.

Usage:
    modal run research/run_modal.py                    # all three
    modal run research/run_modal.py --track brain      # TRIBE v2 on GPU
    modal run research/run_modal.py --track robotics   # fleet simulation
    modal run research/run_modal.py --track browser    # browser agent
"""

import modal
import base64
import io
import tarfile
from pathlib import Path

app = modal.App("canvas-research")

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


def _run_script(script_path, label, env_extras=None):
    """Run a Python script, streaming stdout, return output."""
    import subprocess
    import sys
    import os

    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    if env_extras:
        env.update(env_extras)

    proc = subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(Path(script_path).parent),
        env=env,
    )
    output = []
    for line in proc.stdout:
        print("[{}] {}".format(label, line), end="", flush=True)
        output.append(line)
    proc.wait()
    return proc.returncode, "".join(output[-100:])


def _tar_dir(path):
    """Tar a directory, return base64."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in Path(path).rglob("*"):
            if f.is_file() and f.stat().st_size < 100_000_000:
                tar.add(str(f), arcname=str(f.relative_to(path)))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _untar(b64_data, dest):
    """Untar base64 data to dest."""
    tar_bytes = base64.b64decode(b64_data)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(path=dest, filter="data")


@app.function(
    image=brain_image, gpu="A10G", timeout=10800, cpu=4, memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret", environment_name="test-20260327")],
)
def run_brain():
    """Brain track: TRIBE v2 data generation + training + evaluation."""
    import os
    os.makedirs("/root/research/brain/results", exist_ok=True)
    code, output = _run_script("/root/research/brain/run_tribe_pipeline.py", "brain")
    return {"status": "ok" if code == 0 else "fail", "returncode": code,
            "results_tar": _tar_dir("/root/research/brain/results")}


@app.function(image=cpu_image, timeout=7200, cpu=8, memory=32768)
def run_robotics():
    """Robotics track: full training with scaling analysis."""
    import os
    os.makedirs("/root/research/robotics/results", exist_ok=True)
    code, output = _run_script("/root/research/robotics/run.py", "robotics")
    return {"status": "ok" if code == 0 else "fail", "returncode": code,
            "results_tar": _tar_dir("/root/research/robotics/results")}


@app.function(image=cpu_image, timeout=3600, cpu=8, memory=16384)
def run_browser():
    """Browser track: training + evaluation."""
    import os
    os.makedirs("/root/research/browser/results", exist_ok=True)
    code, output = _run_script("/root/research/browser/run.py", "browser")
    return {"status": "ok" if code == 0 else "fail", "returncode": code,
            "results_tar": _tar_dir("/root/research/browser/results")}


@app.local_entrypoint()
def main(track: str = "all"):
    tracks = []
    if track in ("all", "brain"):
        tracks.append(("brain", run_brain))
    if track in ("all", "robotics"):
        tracks.append(("robotics", run_robotics))
    if track in ("all", "browser"):
        tracks.append(("browser", run_browser))

    print("Launching {} track(s) on Modal...".format(len(tracks)))
    futures = {name: fn.spawn() for name, fn in tracks}

    for name, future in futures.items():
        print("\n" + "=" * 60)
        print("RESULTS: {}".format(name))
        print("=" * 60)
        try:
            result = future.get()
            print("  Status: {}".format(result["status"]))
            print("  Return code: {}".format(result["returncode"]))
            if result.get("results_tar"):
                dest = RESULTS_DIR / name / "results"
                dest.mkdir(parents=True, exist_ok=True)
                _untar(result["results_tar"], str(dest))
                files = [f for f in dest.rglob("*") if f.is_file()]
                print("  Downloaded {} files:".format(len(files)))
                for f in sorted(files):
                    print("    {} ({}KB)".format(
                        f.relative_to(dest), f.stat().st_size // 1024))
        except Exception as e:
            print("  ERROR: {}".format(e))

    print("\nDone.")

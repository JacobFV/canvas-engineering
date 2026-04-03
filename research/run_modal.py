"""Run research tracks on Modal with persistent result storage.

Results are saved to a Modal Volume so they survive detached runs.
Use --detach for long runs, then --collect to download results.

Usage:
    modal run --detach research/run_modal.py --track brain     # launch (detached)
    modal run --detach research/run_modal.py --track robotics  # launch (detached)
    modal run research/run_modal.py --collect                  # download results
"""

import modal
import base64
import io
import os
import tarfile
from pathlib import Path

app = modal.App("canvas-research")

# Persistent volume for results
results_vol = modal.Volume.from_name("canvas-research-results", create_if_missing=True)

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
    import subprocess, sys, os
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    if env_extras:
        env.update(env_extras)
    proc = subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(Path(script_path).parent), env=env,
    )
    output = []
    for line in proc.stdout:
        print("[{}] {}".format(label, line), end="", flush=True)
        output.append(line)
    proc.wait()
    return proc.returncode, "".join(output[-100:])


def _copy_to_volume(local_dir, vol_dir):
    """Copy local results to the Modal Volume."""
    import shutil
    for f in Path(local_dir).rglob("*"):
        if f.is_file():
            dest = Path(vol_dir) / f.relative_to(local_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(f), str(dest))
            print("  -> {}".format(dest))


@app.function(
    image=brain_image, gpu="A10G", timeout=28800, cpu=4, memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret", environment_name="test-20260327")],
    volumes={"/vol": results_vol},
)
def run_brain():
    """Brain track: TRIBE v2 data generation + training + evaluation."""
    import os, subprocess, shutil
    # Clear old results from volume and symlink
    if os.path.exists("/vol/brain"):
        shutil.rmtree("/vol/brain")
    os.makedirs("/vol/brain", exist_ok=True)
    if os.path.exists("/root/research/brain/results"):
        subprocess.run(["rm", "-rf", "/root/research/brain/results"])
    os.symlink("/vol/brain", "/root/research/brain/results")
    code, output = _run_script("/root/research/brain/run_dynamics_pipeline.py", "brain")
    results_vol.commit()
    return {"status": "ok" if code == 0 else "fail", "returncode": code}


@app.function(
    image=cpu_image, timeout=28800, cpu=8, memory=32768,
    volumes={"/vol": results_vol},
)
def run_robotics():
    """Robotics track: training with scaling. Saves to volume."""
    import os, subprocess
    os.makedirs("/vol/robotics", exist_ok=True)
    if os.path.exists("/root/research/robotics/results"):
        subprocess.run(["rm", "-rf", "/root/research/robotics/results"])
    os.symlink("/vol/robotics", "/root/research/robotics/results")
    code, output = _run_script("/root/research/robotics/run.py", "robotics")
    results_vol.commit()
    return {"status": "ok" if code == 0 else "fail", "returncode": code}


@app.function(
    image=cpu_image, timeout=7200, cpu=8, memory=16384,
    volumes={"/vol": results_vol},
)
def run_browser():
    """Browser track: training + evaluation. Saves to volume."""
    import os, subprocess
    os.makedirs("/vol/browser", exist_ok=True)
    if os.path.exists("/root/research/browser/results"):
        subprocess.run(["rm", "-rf", "/root/research/browser/results"])
    os.symlink("/vol/browser", "/root/research/browser/results")
    code, output = _run_script("/root/research/browser/run.py", "browser")
    results_vol.commit()
    return {"status": "ok" if code == 0 else "fail", "returncode": code}


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/vol": results_vol},
)
def collect_results():
    """Download results from the volume."""
    import tarfile, io, base64
    results = {}
    for track in ["brain", "robotics", "browser"]:
        track_dir = Path("/vol") / track
        if track_dir.exists():
            files = list(track_dir.rglob("*"))
            file_list = [f for f in files if f.is_file()]
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for f in file_list:
                    tar.add(str(f), arcname=str(f.relative_to(track_dir)))
            buf.seek(0)
            results[track] = {
                "tar": base64.b64encode(buf.read()).decode(),
                "files": [str(f.relative_to(track_dir)) for f in file_list],
            }
            print("{}: {} files".format(track, len(file_list)))
        else:
            print("{}: no results yet".format(track))
    return results


@app.local_entrypoint()
def main(track: str = "all", collect: bool = False):
    if collect:
        print("Collecting results from Modal Volume...")
        results = collect_results.remote()
        for track_name, data in results.items():
            dest = RESULTS_DIR / track_name / "results"
            dest.mkdir(parents=True, exist_ok=True)
            tar_bytes = base64.b64decode(data["tar"])
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(path=str(dest), filter="data")
            print("  {}: {} files downloaded to {}".format(
                track_name, len(data["files"]), dest))
            for f in sorted(data["files"]):
                fpath = dest / f
                if fpath.exists():
                    print("    {} ({}KB)".format(f, fpath.stat().st_size // 1024))
        return

    tracks = []
    if track in ("all", "brain"):
        tracks.append(("brain", run_brain))
    if track in ("all", "robotics"):
        tracks.append(("robotics", run_robotics))
    if track in ("all", "browser"):
        tracks.append(("browser", run_browser))

    print("Launching {} track(s) on Modal...".format(len(tracks)))
    print("Results will be saved to Modal Volume 'canvas-research-results'")
    print("Use --collect to download results later.")

    futures = {name: fn.spawn() for name, fn in tracks}

    for name, future in futures.items():
        print("\n" + "=" * 60)
        print("RESULTS: {}".format(name))
        print("=" * 60)
        try:
            result = future.get()
            print("  Status: {}".format(result["status"]))
        except Exception as e:
            print("  ERROR: {}".format(e))

    print("\nDone. Use --collect to download results.")

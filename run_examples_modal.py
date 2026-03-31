"""Run all canvas-engineering examples on Modal.

Usage:
    modal run run_examples_modal.py              # run all 13 examples
    modal run run_examples_modal.py --filter 04  # run only matching examples
"""

import modal
import os

app = modal.App("canvas-engineering-examples")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("canvas-engineering==0.4.0", "matplotlib", "numpy", "gymnasium", "scipy")
    .apt_install("ffmpeg")
    .add_local_dir("examples", "/root/examples", copy=True)
)

ALL_EXAMPLES = [
    "01_hello_canvas_types.py",
    "02_multi_frequency.py",
    "02_inheritance_and_arrays.py",
    "03_cartpole_control.py",
    "03_surgical_robot.py",
    "04_autonomous_vehicle_fleet.py",
    "05_protein_folding_complex.py",
    "06_air_traffic_control.py",
    "07_hospital_icu.py",
    "08_world_model_minecraft.py",
    "09_brain_computer_interface.py",
    "10_nuclear_fusion_reactor.py",
    "11_mars_colony.py",
]


@app.function(
    image=image,
    timeout=1800,
    cpu=8,
    memory=32768,
)
def run_example(filename: str) -> str:
    """Run a single example, streaming stdout, and capture result."""
    import subprocess
    import sys

    os.makedirs("/root/assets/examples", exist_ok=True)

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", f"/root/examples/{filename}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/root",
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
        )

        output_lines = []
        for line in proc.stdout:
            print(f"[{filename}] {line}", end="", flush=True)
            output_lines.append(line)

        proc.wait(timeout=1500)

        if proc.returncode != 0:
            return f"FAIL: {filename} (exit {proc.returncode})\n{''.join(output_lines[-20:])}"

        return f"OK: {filename}\n{''.join(output_lines[-10:])}"

    except subprocess.TimeoutExpired:
        proc.kill()
        return f"TIMEOUT: {filename} (>1500s)"
    except Exception as e:
        return f"ERROR: {filename}\n{e}"


@app.local_entrypoint()
def main(filter: str = ""):
    """Run examples in parallel on Modal. Use --filter to select specific ones."""
    if filter:
        example_files = [f for f in ALL_EXAMPLES if filter in f]
    else:
        example_files = ALL_EXAMPLES

    print(f"Running {len(example_files)} examples on Modal...")
    results = list(run_example.map(example_files))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    ok = 0
    fail = 0
    for r in results:
        if r.startswith("OK"):
            ok += 1
            print(f"  {r.split(chr(10))[0]}")
        else:
            fail += 1
            lines = r.split("\n")
            print(f"  {lines[0]}")
            for line in lines[1:8]:
                if line.strip():
                    print(f"    {line}")

    print(f"\n{ok}/{ok+fail} passed, {fail} failed")

"""Run all canvas-engineering examples on Modal."""

import modal
import os

app = modal.App("canvas-engineering-examples")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("canvas-engineering==0.4.0", "matplotlib", "numpy", "gymnasium")
    .apt_install("ffmpeg")
    .add_local_dir("examples", "/root/examples", copy=True)
)


@app.function(
    image=image,
    timeout=900,
    cpu=4,
    memory=16384,
)
def run_example(filename: str) -> str:
    """Run a single example and capture output."""
    import subprocess
    import sys

    os.makedirs("/root/assets/examples", exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, f"/root/examples/{filename}"],
            capture_output=True,
            text=True,
            timeout=800,
            cwd="/root",
            env={**os.environ, "MPLBACKEND": "Agg"},
        )

        output = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        if result.returncode != 0:
            error = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
            return f"FAIL: {filename}\n{error}"

        return f"OK: {filename}\n{output[-500:]}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {filename} (>800s)"
    except Exception as e:
        return f"ERROR: {filename}\n{e}"


@app.local_entrypoint()
def main():
    """Run all examples in parallel on Modal."""
    example_files = [
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

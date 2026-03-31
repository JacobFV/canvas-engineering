"""Main entry point: generate demonstrations, train all models, and evaluate.

Runs the complete browser agent experiment:
1. Generate expert demonstrations from the synthetic environment
2. Train three model variants:
   - Canvas: structured topology with scheduling and residual tracking
   - Dense: dense transformer baseline (fully connected)
   - Flat: flat MLP baseline (no attention structure)
3. Evaluate and generate comparison plots

Runnable locally on CPU (synthetic environment, no GPU needed).
Results saved to research/browser/results/.

Usage:
    python research/browser/run.py
    python research/browser/run.py --epochs 50 --d_model 64   # quick run
    python research/browser/run.py --skip_flat                 # skip flat baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is importable
_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

from research.browser.train import train, RESULTS_DIR
from research.browser.evaluate import plot_all


def main():
    parser = argparse.ArgumentParser(
        description="Browser Canvas Agent: full experiment pipeline",
    )
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs per model (default: 80)")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model dimension (default: 128)")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of attention/MLP layers (default: 4)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--n_demos", type=int, default=200,
                        help="Number of expert demonstrations (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip_flat", action="store_true",
                        help="Skip flat MLP baseline")
    parser.add_argument("--skip_dense", action="store_true",
                        help="Skip dense transformer baseline")
    parser.add_argument("--only", type=str, default=None,
                        choices=["canvas", "dense", "flat"],
                        help="Only train this specific model")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old logs
    for mode in ["canvas", "dense", "flat"]:
        log_path = RESULTS_DIR / "training_log_{}.jsonl".format(mode)
        if log_path.exists():
            log_path.unlink()

    print("=" * 70)
    print("BROWSER CANVAS AGENT EXPERIMENT")
    print("=" * 70)
    print("  Epochs:     {}".format(args.epochs))
    print("  d_model:    {}".format(args.d_model))
    print("  n_layers:   {}".format(args.n_layers))
    print("  n_demos:    {}".format(args.n_demos))
    print("  Seed:       {}".format(args.seed))
    print("  Results:    {}".format(RESULTS_DIR))
    print("=" * 70)
    print()

    all_start = time.time()
    models_trained = {}

    # Determine which models to train
    modes = []
    if args.only:
        modes = [args.only]
    else:
        modes.append("canvas")
        if not args.skip_dense:
            modes.append("dense")
        if not args.skip_flat:
            modes.append("flat")

    # Train each model variant
    for mode in modes:
        print()
        print("=" * 70)
        print("TRAINING: {} model".format(mode.upper()))
        print("=" * 70)

        model, log = train(
            mode=mode,
            epochs=args.epochs,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            lr=args.lr,
            batch_size=args.batch_size,
            n_demos=args.n_demos,
            seed=args.seed,
        )
        models_trained[mode] = {
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_epochs": len(log),
        }

    # Generate evaluation plots
    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    plot_all()

    # Print architecture summary for canvas model
    if "canvas" in modes:
        print()
        print("=" * 70)
        print("CANVAS ARCHITECTURE SUMMARY")
        print("=" * 70)
        from research.browser.browser_canvas import build_browser_program
        bound, program = build_browser_program(d_model=args.d_model)
        print(program.summary())
        print()
        print("Regions ({} total):".format(len(program.regions)))
        for name, rp in sorted(program.regions.items()):
            clock_str = ""
            if rp.clock:
                if rp.clock.mode == "periodic":
                    clock_str = " [every {} steps]".format(rp.clock.period)
                elif rp.clock.mode == "on_event":
                    clock_str = " [on {} > {}]".format(
                        rp.clock.event_source, rp.clock.event_threshold,
                    )
            print("  {:<35} family={:<12} carrier={:<15}{}".format(
                name, rp.family, rp.carrier, clock_str,
            ))

        print()
        if program.schema.topology:
            print("Connections: {}".format(len(program.schema.topology.connections)))
            op_counts = {}
            for conn in program.schema.topology.connections:
                op_counts[conn.operator] = op_counts.get(conn.operator, 0) + 1
            for op, count in sorted(op_counts.items()):
                print("  {}: {}".format(op, count))

    # Final summary
    total_time = time.time() - all_start
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("  Total time: {:.1f}s ({:.1f} min)".format(total_time, total_time / 60))
    print("  Models trained:")
    for mode, info in models_trained.items():
        print("    {}: {:,} params, {} epochs".format(
            mode, info["n_params"], info["n_epochs"],
        ))
    print("  Results in: {}".format(RESULTS_DIR))
    print()
    print("Generated plots:")
    for p in sorted(RESULTS_DIR.glob("*.png")):
        print("  {}".format(p.name))


if __name__ == "__main__":
    main()

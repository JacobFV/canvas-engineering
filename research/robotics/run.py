#!/usr/bin/env python
"""Multi-robot fleet control experiment via canvas-engineering.

End-to-end pipeline:
  1. Creates 2D multi-robot environment with obstacles
  2. Generates expert demonstrations (potential field controller)
  3. Trains three model variants:
     - Canvas fleet: structured topology with coarse-grained inter-robot comms
     - Dense fleet: fully connected (no bottleneck)
     - Independent: each robot isolated (no communication)
  4. Evaluates all models and generates comparison plots
  5. Saves everything to research/robotics/results/

Usage:
    python research/robotics/run.py
    python research/robotics/run.py --n_robots 8 --fast
    python research/robotics/run.py --no-scaling
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-robot fleet control experiment"
    )
    parser.add_argument("--n_robots", type=int, default=4,
                       help="Number of robots (default: 4)")
    parser.add_argument("--d_model", type=int, default=64,
                       help="Model dimension (default: 64)")
    parser.add_argument("--imitation_epochs", type=int, default=10,
                       help="Imitation learning epochs (default: 10)")
    parser.add_argument("--selfplay_episodes", type=int, default=10,
                       help="Self-play episodes (default: 10)")
    parser.add_argument("--eval_episodes", type=int, default=10,
                       help="Evaluation episodes (default: 10)")
    parser.add_argument("--no-scaling", action="store_true",
                       help="Skip scaling analysis")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode: fewer epochs, smaller models")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (default: cpu)")
    args = parser.parse_args()

    # Fast mode overrides
    if args.fast:
        args.imitation_epochs = 3
        args.selfplay_episodes = 5
        args.eval_episodes = 3

    try:
        from research.robotics.train import TrainConfig, train_all_models
        from research.robotics.evaluate import run_evaluation
    except ImportError:
        from train import TrainConfig, train_all_models
        from evaluate import run_evaluation

    config = TrainConfig(
        n_robots=args.n_robots,
        d_model=args.d_model,
        imitation_epochs=args.imitation_epochs,
        selfplay_episodes=args.selfplay_episodes,
        device=args.device,
    )

    if args.fast:
        config.n_expert_episodes = 10
        config.expert_max_steps = 50
        config.selfplay_steps = 30
        config.n_envs_train = 4
        config.batch_size = 16

    print("=" * 60)
    print("Canvas-Engineering Multi-Robot Fleet Experiment")
    print("=" * 60)
    print("  Robots:            {}".format(args.n_robots))
    print("  d_model:           {}".format(args.d_model))
    print("  Imitation epochs:  {}".format(args.imitation_epochs))
    print("  Self-play eps:     {}".format(args.selfplay_episodes))
    print("  Scaling analysis:  {}".format(not args.no_scaling))
    print("  Device:            {}".format(args.device))
    print()

    t0 = time.time()

    # Step 1-3: Train all models
    output = train_all_models(n_robots=args.n_robots, config=config)

    # Step 4: Evaluate and generate plots
    scaling_counts = [2, 4, 8] if not args.no_scaling else []
    if args.fast:
        scaling_counts = [2, 4] if not args.no_scaling else []

    run_evaluation(
        models=output["models"],
        env=output["env"],
        n_eval_episodes=args.eval_episodes,
        max_eval_steps=config.selfplay_steps if args.fast else 200,
        run_scaling=not args.no_scaling,
        scaling_counts=scaling_counts if scaling_counts else None,
    )

    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("Experiment complete in {:.1f}s ({:.1f}m)".format(
        total_time, total_time / 60))
    print("Results saved to: research/robotics/results/")
    print("=" * 60)

    # Print summary
    results_dir = Path(_CE_ROOT) / "research" / "robotics" / "results"
    summary_file = results_dir / "training_summary.json"
    if summary_file.exists():
        import json
        with open(summary_file) as f:
            summary = json.load(f)
        print("\nTraining Summary:")
        for mode, data in summary.items():
            print("  {}:".format(mode))
            for k, v in data.items():
                if isinstance(v, float):
                    print("    {}: {:.4f}".format(k, v))
                else:
                    print("    {}: {}".format(k, v))


if __name__ == "__main__":
    main()

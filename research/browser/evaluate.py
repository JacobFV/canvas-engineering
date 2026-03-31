"""Evaluation and visualization for the browser canvas agent experiment.

Generates:
1. Task completion rate by task type (bar chart)
2. Action prediction accuracy: top-1 and top-3 (line chart over epochs)
3. Planning frequency: how often does the planner fire? (line chart)
4. Page prediction accuracy (line chart)
5. Comparison: structured canvas agent vs flat vs dense baseline (multi-panel)
6. Loss component breakdown (stacked area chart)

All plots are saved to research/browser/results/.

Usage:
    python research/browser/evaluate.py
    python research/browser/evaluate.py --logs_dir research/browser/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is importable
_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

RESULTS_DIR = Path(_CE_ROOT) / "research" / "browser" / "results"


def load_training_log(mode: str, logs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load a JSONL training log for a given mode."""
    d = logs_dir or RESULTS_DIR
    path = d / "training_log_{}.jsonl".format(mode)
    if not path.exists():
        return []
    entries = []
    with open(str(path)) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_all_logs(logs_dir: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """Load training logs for all modes."""
    modes = ["canvas", "dense", "flat"]
    return {m: load_training_log(m, logs_dir) for m in modes if load_training_log(m, logs_dir)}


# ---- Metric extraction ----

def extract_metric(log: List[Dict], key: str) -> Tuple[List[int], List[float]]:
    """Extract (epochs, values) for a given metric key."""
    epochs = []
    values = []
    for entry in log:
        if key in entry:
            epochs.append(entry["epoch"])
            values.append(entry[key])
    return epochs, values


def extract_final_metrics(log: List[Dict]) -> Dict[str, float]:
    """Extract the last reported value for each metric."""
    final = {}
    for entry in log:
        for k, v in entry.items():
            if isinstance(v, (int, float)):
                final[k] = v
    return final


# ---- Plotting ----

def plot_all(logs_dir: Optional[Path] = None, save_dir: Optional[Path] = None):
    """Generate all evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plots.")
        return

    d = logs_dir or RESULTS_DIR
    save = save_dir or d
    save.mkdir(parents=True, exist_ok=True)

    all_logs = load_all_logs(d)
    if not all_logs:
        print("No training logs found in {}".format(d))
        return

    mode_colors = {"canvas": "#2196F3", "dense": "#FF9800", "flat": "#4CAF50"}
    mode_labels = {"canvas": "Canvas (structured)", "dense": "Dense transformer", "flat": "Flat MLP"}

    # ---- Plot 1: Task completion rate by task type (bar chart) ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    task_types = ["click", "type", "navigate", "form"]
    bar_width = 0.25
    x = np.arange(len(task_types))

    for i, (mode, log) in enumerate(all_logs.items()):
        final = extract_final_metrics(log)
        rates = [final.get("rollout_success_rate_{}".format(tt), 0.0) for tt in task_types]
        offset = (i - len(all_logs) / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, rates, bar_width,
            label=mode_labels.get(mode, mode),
            color=mode_colors.get(mode, "#999"),
            alpha=0.85,
        )

    ax.set_xlabel("Task Type", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Task Completion Rate by Type", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in task_types])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save / "task_completion_by_type.png"), dpi=150)
    plt.close(fig)
    print("  Saved task_completion_by_type.png")

    # ---- Plot 2: Action accuracy over epochs (top-1 and top-3) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mode, log in all_logs.items():
        color = mode_colors.get(mode, "#999")
        label = mode_labels.get(mode, mode)

        epochs, acc1 = extract_metric(log, "val_action_accuracy_top1")
        if epochs:
            ax1.plot(epochs, acc1, color=color, label=label, linewidth=2)

        epochs, acc3 = extract_metric(log, "val_action_accuracy_top3")
        if epochs:
            ax2.plot(epochs, acc3, color=color, label=label, linewidth=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Top-1 Accuracy")
    ax1.set_title("Action Prediction Accuracy (Top-1)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-3 Accuracy")
    ax2.set_title("Action Prediction Accuracy (Top-3)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save / "action_accuracy.png"), dpi=150)
    plt.close(fig)
    print("  Saved action_accuracy.png")

    # ---- Plot 3: Planning frequency ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for mode, log in all_logs.items():
        color = mode_colors.get(mode, "#999")
        label = mode_labels.get(mode, mode)
        epochs, rates = extract_metric(log, "train_plan_fire_rate")
        if epochs:
            ax.plot(epochs, rates, color=color, label=label, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Plan Fire Rate")
    ax.set_title("Planning Frequency (fraction of steps where planner fires)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save / "planning_frequency.png"), dpi=150)
    plt.close(fig)
    print("  Saved planning_frequency.png")

    # ---- Plot 4: Page prediction accuracy (SSL loss) ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for mode, log in all_logs.items():
        color = mode_colors.get(mode, "#999")
        label = mode_labels.get(mode, mode)

        epochs, ssl = extract_metric(log, "train_ssl_next_page")
        if epochs:
            ax.plot(epochs, ssl, color=color, label=label, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Next-Page Prediction Loss (lower = better page dynamics model)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save / "page_prediction.png"), dpi=150)
    plt.close(fig)
    print("  Saved page_prediction.png")

    # ---- Plot 5: Multi-panel comparison ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ("train_total", "Total Training Loss"),
        ("val_total", "Validation Loss"),
        ("val_action_accuracy_top1", "Action Accuracy (Top-1)"),
        ("rollout_success_rate", "Rollout Success Rate"),
        ("train_ssl_next_page", "Next-Page Prediction (SSL)"),
        ("train_bc_action", "Behavioral Cloning Loss"),
    ]

    for idx, (metric_key, title) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]

        for mode, log in all_logs.items():
            color = mode_colors.get(mode, "#999")
            label = mode_labels.get(mode, mode)
            epochs, vals = extract_metric(log, metric_key)
            if epochs:
                ax.plot(epochs, vals, color=color, label=label, linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Browser Agent: Canvas vs Baselines", fontsize=16)
    fig.tight_layout()
    fig.savefig(str(save / "comparison_multipanel.png"), dpi=150)
    plt.close(fig)
    print("  Saved comparison_multipanel.png")

    # ---- Plot 6: Loss component breakdown for canvas model ----
    if "canvas" in all_logs:
        log = all_logs["canvas"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        components = [
            ("train_bc_action", "BC: Action CE", "#2196F3"),
            ("train_bc_coord", "BC: Coordinate MSE", "#42A5F5"),
            ("train_bc_text", "BC: Text MSE", "#90CAF9"),
            ("train_ssl_next_page", "SSL: Next-Page", "#FF9800"),
            ("train_rl_reward", "RL: Reward-Weighted", "#4CAF50"),
            ("train_reward_pred", "Aux: Reward Pred", "#9E9E9E"),
        ]

        for key, label, color in components:
            epochs, vals = extract_metric(log, key)
            if epochs:
                ax.plot(epochs, vals, label=label, color=color, linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Component Breakdown (Canvas Model)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(save / "loss_breakdown.png"), dpi=150)
        plt.close(fig)
        print("  Saved loss_breakdown.png")

    # ---- Summary table ----
    print("\n{:=<70}".format(""))
    print("Final Metrics Summary")
    print("{:=<70}".format(""))
    mode_names = [mode_labels.get(m, m)[:12] for m in all_logs.keys()]
    header_parts = ["{:<20}".format("Metric")]
    for mn in mode_names:
        header_parts.append("{:>14}".format(mn))
    print("".join(header_parts))
    print("{:-<70}".format(""))

    summary_metrics = [
        ("val_total", "Val Loss"),
        ("val_action_accuracy_top1", "Top-1 Acc"),
        ("val_action_accuracy_top3", "Top-3 Acc"),
        ("rollout_success_rate", "Success Rate"),
        ("train_plan_fire_rate", "Plan Fire Rate"),
        ("train_ssl_next_page", "SSL Loss"),
    ]

    for key, name in summary_metrics:
        vals = []
        for mode, log in all_logs.items():
            final = extract_final_metrics(log)
            v = final.get(key, None)
            if v is not None:
                if "acc" in key.lower() or "rate" in key.lower():
                    vals.append("{:.1f}%".format(v * 100))
                else:
                    vals.append("{:.4f}".format(v))
            else:
                vals.append("-")
        print("{:<20} {}".format(name, "  ".join("{:>12}".format(v) for v in vals)))

    print("{:=<70}".format(""))

    # Save summary as JSON
    summary = {}
    for mode, log in all_logs.items():
        summary[mode] = extract_final_metrics(log)
    with open(str(save / "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to evaluation_summary.json")


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="Evaluate browser canvas agent")
    parser.add_argument("--logs_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    save_dir = Path(args.save_dir) if args.save_dir else None

    print("Generating evaluation plots...")
    plot_all(logs_dir, save_dir)


if __name__ == "__main__":
    main()

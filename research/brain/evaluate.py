"""Analysis and visualization for cortical canvas experiments.

Generates:
1. Connectivity matrix visualization (which regions attend to which)
2. Per-region prediction accuracy
3. Comparison: cortical vs dense vs flat baselines
4. Learned attention patterns vs known neuroscience
5. Category classification accuracy comparison

Usage:
    python research/brain/evaluate.py
    python research/brain/evaluate.py --checkpoints cortical dense flat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from research.brain.cortical_canvas import (
    CorticalBrain,
    CORTICAL_PATHWAYS,
    build_cortical_program,
    ROI_TO_CANVAS,
    CANVAS_TO_ROIS,
    get_region_names,
)
from research.brain.data_pipeline import generate_synthetic_dataset, load_dataset
from research.brain.train import (
    CorticalCanvasModel,
    CorticalDataset,
    TrainConfig,
    build_model,
    eval_epoch,
)

RESULTS_DIR = Path(_CE_ROOT) / "research" / "brain" / "results"


# ---- 1. Connectivity matrix visualization ----

def plot_connectivity_matrix(output_path: Optional[str] = None):
    """Visualize the cortical topology as a connectivity matrix.

    Shows which regions are connected, with connection weights as color
    intensity and operator type as annotations.
    """
    bound, program, _ = build_cortical_program(T=1, d_model=64)
    region_names = bound.field_names
    n = len(region_names)

    # Build adjacency matrix from topology
    adj = np.zeros((n, n))
    operator_matrix = [['' for _ in range(n)] for _ in range(n)]

    if program.schema.topology:
        name_to_idx = {name: i for i, name in enumerate(region_names)}
        for conn in program.schema.topology.connections:
            if conn.src in name_to_idx and conn.dst in name_to_idx:
                si = name_to_idx[conn.src]
                di = name_to_idx[conn.dst]
                adj[si, di] = max(adj[si, di], conn.weight)
                if conn.operator != "attend":
                    operator_matrix[si][di] = conn.operator[:3]

    # Shorten region names for display
    short_names = []
    for name in region_names:
        parts = name.split(".")
        if len(parts) >= 2:
            short_names.append(f"{parts[0][:3]}.{parts[1][:6]}")
        else:
            short_names.append(name[:10])

    # Color regions by network
    network_colors = {
        "visual": "#e74c3c",
        "auditory": "#3498db",
        "language": "#2ecc71",
        "frontal": "#f39c12",
        "default_mode": "#9b59b6",
        "subcortical": "#1abc9c",
        "prediction_error": "#95a5a6",
    }

    region_colors = []
    for name in region_names:
        network = name.split(".")[0] if "." in name else name
        region_colors.append(network_colors.get(network, "#95a5a6"))

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot the matrix
    im = ax.imshow(adj, cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=1)

    # Add operator annotations
    for i in range(n):
        for j in range(n):
            if operator_matrix[i][j]:
                ax.text(j, i, operator_matrix[i][j],
                        ha="center", va="center", fontsize=5, color="black")

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)

    # Color the tick labels by network
    for i, (label, color) in enumerate(zip(ax.get_yticklabels(), region_colors)):
        label.set_color(color)
        label.set_fontweight("bold")
    for i, (label, color) in enumerate(zip(ax.get_xticklabels(), region_colors)):
        label.set_color(color)
        label.set_fontweight("bold")

    ax.set_xlabel("Destination (keys/values)", fontsize=11)
    ax.set_ylabel("Source (queries)", fontsize=11)
    ax.set_title("Cortical Canvas Connectivity Matrix\n(Weight = color intensity, operator type = text)",
                 fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Connection weight")

    # Legend for networks
    patches = [mpatches.Patch(color=c, label=n) for n, c in network_colors.items()]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.15, 1),
              fontsize=8, title="Networks")

    fig.tight_layout()
    save_path = output_path or str(RESULTS_DIR / "connectivity_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Also save pathway stats
    n_self = int((adj.diagonal() > 0).sum())
    n_cross = int((adj > 0).sum()) - n_self
    print(f"Connectivity: {n_self} self-loops, {n_cross} cross-connections, "
          f"density = {(adj > 0).mean():.1%}")


# ---- 2. Per-region prediction accuracy ----

def plot_per_region_accuracy(
    region_names: List[str],
    per_region_mse: np.ndarray,
    mode: str = "cortical",
    output_path: Optional[str] = None,
):
    """Bar chart of prediction accuracy per brain region."""
    n = len(region_names)

    # Shorten names
    short_names = []
    for name in region_names:
        parts = name.split(".")
        if len(parts) >= 2:
            short_names.append(f"{parts[-1]}")
        else:
            short_names.append(name[:12])

    # Color by network
    network_colors = {
        "visual": "#e74c3c",
        "auditory": "#3498db",
        "language": "#2ecc71",
        "frontal": "#f39c12",
        "default_mode": "#9b59b6",
        "subcortical": "#1abc9c",
        "prediction": "#95a5a6",
    }

    colors = []
    for name in region_names:
        network = name.split(".")[0] if "." in name else "prediction"
        colors.append(network_colors.get(network, "#95a5a6"))

    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.barh(range(n), per_region_mse, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("MSE (lower = better)")
    ax.set_title(f"Per-Region Prediction MSE ({mode})", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, per_region_mse)):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)

    fig.tight_layout()
    save_path = output_path or str(RESULTS_DIR / f"per_region_mse_{mode}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---- 3. Comparison across architectures ----

def plot_training_comparison(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
):
    """Compare training curves across cortical, dense, and flat models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mode_colors = {
        "cortical": "#e74c3c",
        "dense": "#3498db",
        "flat": "#95a5a6",
    }

    # 1. Total validation loss
    ax = axes[0, 0]
    for mode, r in results.items():
        ax.plot(r["history"]["val_loss"], label=mode,
                color=mode_colors.get(mode, "black"), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Total Validation Loss", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Classification accuracy
    ax = axes[0, 1]
    for mode, r in results.items():
        ax.plot(r["history"]["val_cls_acc"], label=mode,
                color=mode_colors.get(mode, "black"), linewidth=2)
    n_cats = len(list(results.values())[0].get("category_names", []))
    if n_cats > 0:
        ax.axhline(1.0 / n_cats, color="gray", linestyle="--",
                    label=f"Chance ({1.0/n_cats:.0%})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Category Classification Accuracy", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Region prediction loss
    ax = axes[1, 0]
    for mode, r in results.items():
        ax.plot(r["history"]["val_region_loss"], label=mode,
                color=mode_colors.get(mode, "black"), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Region MSE")
    ax.set_title("Region Prediction MSE", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Final comparison bar chart
    ax = axes[1, 1]
    metrics = ["val_loss", "cls_acc", "region_loss"]
    labels = ["Val Loss", "Cls Acc", "Region MSE"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, (mode, r) in enumerate(results.items()):
        values = [
            r["final_val"]["loss"],
            r["final_val"]["cls_acc"],
            r["final_val"]["region_loss"],
        ]
        ax.bar(x + i * width, values, width, label=mode,
               color=mode_colors.get(mode, "black"))

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_title("Final Metrics Comparison", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Cortical Canvas vs Baselines", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_path = output_path or str(RESULTS_DIR / "training_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---- 4. Learned attention vs known neuroscience ----

def analyze_attention_patterns(
    model: CorticalCanvasModel,
    data: Dict,
    output_path: Optional[str] = None,
):
    """Compare learned attention weights to known cortical pathways.

    Extracts per-connection attention energy from the model and compares
    against the neuroscience-motivated CORTICAL_PATHWAYS.
    """
    region_names = data["region_names"]
    n = len(region_names)
    name_to_idx = {name: i for i, name in enumerate(region_names)}

    # Get model topology connections
    model.eval()

    # Build the known pathway matrix
    known_matrix = np.zeros((n, n))
    for src, dst, operator, weight in CORTICAL_PATHWAYS:
        if src in name_to_idx and dst in name_to_idx:
            si = name_to_idx[src]
            di = name_to_idx[dst]
            known_matrix[si, di] = weight

    # Extract learned connection strengths by running a probe batch
    dataset = CorticalDataset(data)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=False)
    activations, _ = next(iter(loader))
    device = next(model.parameters()).device
    activations = activations.to(device)

    # Hook to capture attention weights
    attention_energy = np.zeros((n, n))

    with torch.no_grad():
        B = activations.shape[0]
        x = model.input_proj(activations.unsqueeze(-1))
        region_ids = torch.arange(model.n_regions, device=device)
        x = x + model.region_embeddings(region_ids).unsqueeze(0)
        x = model.input_norm(x)

        for dispatcher, ln, ffn, ffn_ln in zip(
            model.layers, model.layer_norms, model.ffn_layers, model.ffn_norms
        ):
            # Compute attention energies per connection
            for src_name, dst_name, weight, fn_name in dispatcher._op_specs:
                if src_name in name_to_idx and dst_name in name_to_idx:
                    si = name_to_idx[src_name]
                    di = name_to_idx[dst_name]
                    # Measure information flow as mean src activation change
                    src_idx = dispatcher._get_device_idx(src_name, device)
                    queries = x[:, src_idx]
                    q_norm = queries.norm(dim=-1).mean().item()
                    attention_energy[si, di] += q_norm * weight

            attn_out = dispatcher(x)
            x = ln(x + attn_out)
            x = ffn_ln(x + ffn(x))

    # Normalize
    if attention_energy.max() > 0:
        attention_energy /= attention_energy.max()

    # Compare known vs learned
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Shorten names
    short_names = []
    for name in region_names:
        parts = name.split(".")
        if len(parts) >= 2:
            short_names.append(f"{parts[-1][:6]}")
        else:
            short_names.append(name[:8])

    # Known pathways
    ax = axes[0]
    im0 = ax.imshow(known_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_title("Known Cortical Pathways", fontweight="bold")
    plt.colorbar(im0, ax=ax, shrink=0.8)

    # Learned attention
    ax = axes[1]
    im1 = ax.imshow(attention_energy, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_title("Learned Attention Energy", fontweight="bold")
    plt.colorbar(im1, ax=ax, shrink=0.8)

    # Overlap (element-wise product)
    overlap = known_matrix * attention_energy
    ax = axes[2]
    im2 = ax.imshow(overlap, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_title("Overlap (Known * Learned)", fontweight="bold")
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Compute correlation
    known_flat = known_matrix.flatten()
    learned_flat = attention_energy.flatten()
    mask = known_flat > 0
    if mask.any():
        corr = np.corrcoef(known_flat[mask], learned_flat[mask])[0, 1]
    else:
        corr = 0.0

    fig.suptitle(
        f"Known vs Learned Connectivity (correlation on known edges: {corr:.3f})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_path = output_path or str(RESULTS_DIR / "attention_vs_neuroscience.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    print(f"Known-learned correlation on known edges: {corr:.3f}")

    return {"correlation": corr, "known_matrix": known_matrix,
            "learned_matrix": attention_energy}


# ---- 5. Category classification comparison ----

def plot_category_accuracy(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
):
    """Compare per-category classification accuracy across models."""
    # This uses the overall classification accuracy since per-category
    # requires re-running evaluation. We show final accuracy per model.
    categories = list(results.values())[0].get("category_names", [])
    modes = list(results.keys())

    mode_colors = {
        "cortical": "#e74c3c",
        "dense": "#3498db",
        "flat": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(modes))
    accs = [results[m]["final_val"]["cls_acc"] for m in modes]
    colors = [mode_colors.get(m, "gray") for m in modes]

    bars = ax.bar(x, accs, color=colors, edgecolor="white", linewidth=1.5)

    n_cats = len(categories) if categories else 1
    ax.axhline(1.0 / n_cats, color="gray", linestyle="--",
               label=f"Chance ({1.0/n_cats:.0%})")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", fontweight="bold", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modes], fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title("Category Classification: Cortical vs Baselines",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    save_path = output_path or str(RESULTS_DIR / "category_classification.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---- Main evaluation pipeline ----

def run_evaluation(
    data: Optional[Dict] = None,
    checkpoints: Optional[List[str]] = None,
    run_training: bool = True,
    epochs: int = 50,
    d_model: int = 128,
):
    """Run the full evaluation pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Connectivity matrix (requires no training)
    print("\n=== 1. Connectivity Matrix ===")
    plot_connectivity_matrix()

    # 2. Load or generate data
    if data is None:
        print("\n=== Generating synthetic data ===")
        data = generate_synthetic_dataset()

    # 3. Train all baselines if needed
    if run_training:
        print("\n=== 2-3. Training all models ===")
        from research.brain.train import run_all_baselines
        results = run_all_baselines(data, epochs=epochs, d_model=d_model)
    else:
        # Load from checkpoints
        results = {}
        modes = checkpoints or ["cortical", "dense", "flat"]
        for mode in modes:
            ckpt_path = RESULTS_DIR / f"checkpoint_{mode}.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                results[mode] = {
                    "history": ckpt["history"],
                    "final_val": {},
                    "n_params": ckpt["n_params"],
                    "config": ckpt["config"],
                    "region_names": ckpt["region_names"],
                    "category_names": ckpt["category_names"],
                }

                # Re-evaluate
                dataset = CorticalDataset(data)
                config = TrainConfig(**ckpt["config"])
                model = build_model(config, dataset)
                model.load_state_dict(ckpt["model_state"])
                loader = DataLoader(dataset, batch_size=16, shuffle=False)
                val_metrics = eval_epoch(model, loader, config, torch.device("cpu"))
                results[mode]["final_val"] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in val_metrics.items()
                }
                print(f"Loaded checkpoint: {mode} (val_cls_acc={val_metrics['cls_acc']:.1%})")
            else:
                print(f"WARNING: Checkpoint not found: {ckpt_path}")

    if not results:
        print("No results available. Run training first.")
        return

    # 4. Per-region accuracy plots
    print("\n=== 4. Per-Region Accuracy ===")
    for mode, r in results.items():
        if "per_region_mse" in r.get("final_val", {}):
            mse = r["final_val"]["per_region_mse"]
            if isinstance(mse, list):
                mse = np.array(mse)
            region_names = r.get("region_names", data["region_names"])
            plot_per_region_accuracy(region_names, mse, mode=mode)

    # 5. Training comparison
    print("\n=== 5. Training Comparison ===")
    plot_training_comparison(results)

    # 6. Category classification comparison
    print("\n=== 6. Category Classification ===")
    plot_category_accuracy(results)

    # 7. Attention analysis (cortical model only)
    if "cortical" in results:
        print("\n=== 7. Attention vs Neuroscience ===")
        ckpt_path = RESULTS_DIR / "checkpoint_cortical.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            dataset = CorticalDataset(data)
            config = TrainConfig(**ckpt["config"])
            model = build_model(config, dataset)
            model.load_state_dict(ckpt["model_state"])
            attention_results = analyze_attention_patterns(model, data)
        else:
            print("  Skipping (no cortical checkpoint)")

    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cortical canvas models")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz dataset")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="Checkpoint modes to load (e.g., cortical dense flat)")
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training, load from checkpoints")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--d-model", type=int, default=128)
    args = parser.parse_args()

    data = None
    if args.data:
        print(f"Loading dataset from {args.data}...")
        data = load_dataset(args.data)
    else:
        print("Using synthetic dataset")
        data = generate_synthetic_dataset()

    run_evaluation(
        data=data,
        checkpoints=args.checkpoints,
        run_training=not args.no_train,
        epochs=args.epochs,
        d_model=args.d_model,
    )

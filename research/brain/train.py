"""Train a cortical canvas model on brain activation data.

Three objectives:
1. Region prediction MSE: predict each region's activation from connected regions
2. Category classification CE: classify stimulus category from prefrontal readout
3. Prediction error calibration: residual region should track actual prediction error

Uses AttentionDispatcher with real cortical topology, ResidualAccumulator for
error tracking, and RegionScheduler with periodic clocks matching cortical
temporal dynamics.

Usage:
    # Train with synthetic data (no GPU needed):
    python research/brain/train.py --synthetic --epochs 50

    # Train with real data:
    python research/brain/train.py --data research/brain/results/cortical_dataset.npz

    # Train dense baseline for comparison:
    python research/brain/train.py --synthetic --mode dense --epochs 50

    # Train flat MLP baseline:
    python research/brain/train.py --synthetic --mode flat --epochs 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Ensure project root is importable
_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

from canvas_engineering import (
    AttentionDispatcher,
    CanvasTopology,
    Connection,
    RegionScheduler,
    ResidualAccumulator,
    ResidualSpec,
    ClockSpec,
)
from canvas_engineering.program import CanvasProgram, RegionProgram

from research.brain.cortical_canvas import (
    CorticalBrain,
    build_cortical_program,
    CORTICAL_PATHWAYS,
    get_region_names,
)
from research.brain.data_pipeline import (
    generate_synthetic_dataset,
    load_dataset,
)

RESULTS_DIR = Path(_CE_ROOT) / "research" / "brain" / "results"


# ---- Dataset ----

class CorticalDataset(Dataset):
    """Dataset of cortical region activations with category labels."""

    def __init__(self, data: Dict):
        self.activations = torch.tensor(
            data["region_activations"], dtype=torch.float32,
        )
        self.labels = torch.tensor(data["labels"], dtype=torch.long)
        self.region_names = data["region_names"]
        self.category_names = data["category_names"]
        self.n_regions = self.activations.shape[1]
        self.n_categories = len(self.category_names)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


# ---- Models ----

class CorticalCanvasModel(nn.Module):
    """Canvas-based model using real cortical topology.

    Architecture:
    1. Input embedding: project scalar ROI activations to d_model
    2. Positional/region embedding: learned per-region embedding
    3. N layers of AttentionDispatcher with cortical topology
    4. Region prediction heads: predict each region from its state
    5. Classification head: read from prefrontal region
    6. Residual tracking: ResidualAccumulator monitors prediction error
    """

    def __init__(
        self,
        n_regions: int,
        n_categories: int,
        region_names: List[str],
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        topology: Optional[CanvasTopology] = None,
        program: Optional[CanvasProgram] = None,
        use_scheduler: bool = True,
    ):
        super().__init__()
        self.n_regions = n_regions
        self.n_categories = n_categories
        self.d_model = d_model
        self.region_names = region_names
        self.use_scheduler = use_scheduler

        # Input projection: scalar activation -> d_model vector
        self.input_proj = nn.Linear(1, d_model)

        # Learned region embeddings
        self.region_embeddings = nn.Embedding(n_regions, d_model)

        # Layer norm
        self.input_norm = nn.LayerNorm(d_model)

        # Build topology-aware attention layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        # Use the cortical topology for attention routing
        if topology is not None:
            self.topology = topology
        else:
            # Fallback to dense self-attention
            self.topology = CanvasTopology.dense(region_names)

        # Build a simple layout for the dispatcher
        # Each region = 1 position (we work with region-level activations)
        from canvas_engineering.canvas import CanvasLayout, RegionSpec

        regions = {}
        for i, name in enumerate(region_names):
            regions[name] = RegionSpec(bounds=(0, 1, i, i + 1, 0, 1))

        self.layout = CanvasLayout(
            T=1, H=n_regions, W=1, d_model=d_model,
            regions=regions,
        )

        # Residual accumulator for prediction error tracking
        residual_regions = [name for name in region_names
                           if "prediction_error" in name or "residual" in name]
        self.residual_accumulator = ResidualAccumulator(
            residual_regions if residual_regions else ["_dummy"],
            ResidualSpec(kinds=("prediction", "novelty"), decay=0.9),
        ) if residual_regions else None

        for layer_idx in range(n_layers):
            dispatcher = AttentionDispatcher(
                topology=self.topology,
                layout=self.layout,
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                residual_accumulator=self.residual_accumulator if layer_idx == n_layers - 1 else None,
            )
            self.layers.append(dispatcher)
            self.layer_norms.append(nn.LayerNorm(d_model))

            ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            self.ffn_layers.append(ffn)
            self.ffn_norms.append(nn.LayerNorm(d_model))

        # Region prediction head: predict activation from hidden state
        self.region_pred_head = nn.Linear(d_model, 1)

        # Classification head: read from prefrontal region
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_categories),
        )

        # Prediction error head: predict actual error magnitude
        self.error_pred_head = nn.Linear(d_model, 1)

        # Region scheduler (optional)
        self.scheduler = None
        if use_scheduler and program is not None:
            self.scheduler = RegionScheduler(program)

        # Find prefrontal region index for classification readout
        self.prefrontal_idx = None
        for i, name in enumerate(region_names):
            if "prefrontal" in name:
                self.prefrontal_idx = i
                break
        if self.prefrontal_idx is None:
            self.prefrontal_idx = 0

        # Find prediction_error region index
        self.error_idx = None
        for i, name in enumerate(region_names):
            if "prediction_error" in name:
                self.error_idx = i
                break

    def forward(
        self,
        activations: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the cortical canvas.

        Args:
            activations: (B, n_regions) scalar activations per region
            step: current training step (for scheduler)

        Returns:
            region_preds: (B, n_regions) predicted activation per region
            cls_logits: (B, n_categories) classification logits
            error_pred: (B, 1) predicted error magnitude
        """
        B = activations.shape[0]

        # Project scalar activations to d_model
        x = self.input_proj(activations.unsqueeze(-1))  # (B, n_regions, d_model)

        # Add region embeddings
        region_ids = torch.arange(self.n_regions, device=x.device)
        x = x + self.region_embeddings(region_ids).unsqueeze(0)

        # Normalize
        x = self.input_norm(x)

        # Get active regions from scheduler
        active_regions = None
        if self.scheduler is not None and self.use_scheduler:
            active_regions = self.scheduler.step(step)

        # Run through attention layers with cortical topology
        for layer_idx, (dispatcher, ln, ffn, ffn_ln) in enumerate(
            zip(self.layers, self.layer_norms, self.ffn_layers, self.ffn_norms)
        ):
            # Attention with cortical routing
            attn_out = dispatcher(x, active_regions=active_regions)
            x = ln(x + attn_out)

            # FFN
            ffn_out = ffn(x)
            x = ffn_ln(x + ffn_out)

        # Region prediction: predict each region's activation
        region_preds = self.region_pred_head(x).squeeze(-1)  # (B, n_regions)

        # Classification: read from prefrontal
        prefrontal_state = x[:, self.prefrontal_idx]  # (B, d_model)
        cls_logits = self.cls_head(prefrontal_state)  # (B, n_categories)

        # Prediction error: read from error region
        error_pred = torch.zeros(B, 1, device=x.device)
        if self.error_idx is not None:
            error_state = x[:, self.error_idx]  # (B, d_model)
            error_pred = self.error_pred_head(error_state)

        return region_preds, cls_logits, error_pred


class DenseCanvasModel(CorticalCanvasModel):
    """Dense-topology baseline: every region attends to every other."""

    def __init__(self, n_regions, n_categories, region_names, **kwargs):
        # Override topology to dense
        topology = CanvasTopology.dense(region_names)
        super().__init__(
            n_regions, n_categories, region_names,
            topology=topology, use_scheduler=False, **kwargs,
        )


class FlatMLPModel(nn.Module):
    """Flat MLP baseline: no canvas structure, just a feed-forward network."""

    def __init__(
        self,
        n_regions: int,
        n_categories: int,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.n_regions = n_regions
        self.n_categories = n_categories

        layers = []
        in_dim = n_regions
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = d_model

        self.encoder = nn.Sequential(*layers)
        self.region_pred = nn.Linear(d_model, n_regions)
        self.cls_head = nn.Linear(d_model, n_categories)
        self.error_pred = nn.Linear(d_model, 1)

    def forward(self, activations, step=0):
        h = self.encoder(activations)
        region_preds = self.region_pred(h)
        cls_logits = self.cls_head(h)
        error_pred = self.error_pred(h)
        return region_preds, cls_logits, error_pred


# ---- Training loop ----

@dataclass
class TrainConfig:
    mode: str = "cortical"       # "cortical", "dense", "flat"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 16
    val_split: float = 0.2
    region_loss_weight: float = 1.0
    cls_loss_weight: float = 1.0
    error_loss_weight: float = 0.5
    seed: int = 42
    log_every: int = 5


def build_model(config: TrainConfig, dataset: CorticalDataset) -> nn.Module:
    """Build the appropriate model based on config.mode."""
    kwargs = dict(
        n_regions=dataset.n_regions,
        n_categories=dataset.n_categories,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )

    if config.mode == "flat":
        return FlatMLPModel(**kwargs)
    elif config.mode == "dense":
        return DenseCanvasModel(
            region_names=dataset.region_names, **kwargs,
        )
    else:
        # Build cortical topology
        bound, program, _ = build_cortical_program(T=1, d_model=config.d_model)
        return CorticalCanvasModel(
            region_names=dataset.region_names,
            topology=program.schema.topology,
            program=program,
            use_scheduler=True,
            **kwargs,
        )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_region_loss = 0.0
    total_cls_loss = 0.0
    total_error_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (activations, labels) in enumerate(loader):
        activations = activations.to(device)
        labels = labels.to(device)

        step = epoch * len(loader) + batch_idx

        # Forward
        region_preds, cls_logits, error_pred = model(activations, step=step)

        # Region prediction loss (MSE)
        region_loss = F.mse_loss(region_preds, activations)

        # Classification loss (CE)
        cls_loss = F.cross_entropy(cls_logits, labels)

        # Prediction error calibration loss
        # The error region should predict the actual prediction error magnitude
        actual_error = (region_preds - activations).abs().mean(dim=1, keepdim=True)
        error_loss = F.mse_loss(error_pred, actual_error.detach())

        # Combined loss
        loss = (
            config.region_loss_weight * region_loss
            + config.cls_loss_weight * cls_loss
            + config.error_loss_weight * error_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * activations.size(0)
        total_region_loss += region_loss.item() * activations.size(0)
        total_cls_loss += cls_loss.item() * activations.size(0)
        total_error_loss += error_loss.item() * activations.size(0)
        total_correct += (cls_logits.argmax(1) == labels).sum().item()
        total_samples += activations.size(0)

    n = total_samples
    return {
        "loss": total_loss / n,
        "region_loss": total_region_loss / n,
        "cls_loss": total_cls_loss / n,
        "error_loss": total_error_loss / n,
        "cls_acc": total_correct / n,
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_region_loss = 0.0
    total_cls_loss = 0.0
    total_error_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_cls_preds = []
    all_cls_targets = []

    for activations, labels in loader:
        activations = activations.to(device)
        labels = labels.to(device)

        region_preds, cls_logits, error_pred = model(activations)

        region_loss = F.mse_loss(region_preds, activations)
        cls_loss = F.cross_entropy(cls_logits, labels)
        actual_error = (region_preds - activations).abs().mean(dim=1, keepdim=True)
        error_loss = F.mse_loss(error_pred, actual_error)

        loss = (
            config.region_loss_weight * region_loss
            + config.cls_loss_weight * cls_loss
            + config.error_loss_weight * error_loss
        )

        total_loss += loss.item() * activations.size(0)
        total_region_loss += region_loss.item() * activations.size(0)
        total_cls_loss += cls_loss.item() * activations.size(0)
        total_error_loss += error_loss.item() * activations.size(0)
        total_correct += (cls_logits.argmax(1) == labels).sum().item()
        total_samples += activations.size(0)

        all_preds.append(region_preds.cpu())
        all_targets.append(activations.cpu())
        all_cls_preds.append(cls_logits.argmax(1).cpu())
        all_cls_targets.append(labels.cpu())

    n = total_samples
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Per-region MSE
    per_region_mse = ((preds - targets) ** 2).mean(dim=0)

    return {
        "loss": total_loss / n,
        "region_loss": total_region_loss / n,
        "cls_loss": total_cls_loss / n,
        "error_loss": total_error_loss / n,
        "cls_acc": total_correct / n,
        "per_region_mse": per_region_mse.numpy(),
    }


def train(
    config: TrainConfig,
    data: Dict,
    device: Optional[torch.device] = None,
) -> Dict:
    """Full training run.

    Returns:
        Dict with training history and final metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Build dataset
    dataset = CorticalDataset(data)
    n_val = max(1, int(len(dataset) * config.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
    )

    # Build model
    model = build_model(config, dataset).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.mode}, {n_params:,} parameters")
    print(f"Dataset: {n_train} train, {n_val} val, {dataset.n_categories} categories")
    print(f"Regions: {dataset.n_regions}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs,
    )

    # Training loop
    history = {
        "train_loss": [], "train_region_loss": [], "train_cls_loss": [],
        "train_error_loss": [], "train_cls_acc": [],
        "val_loss": [], "val_region_loss": [], "val_cls_loss": [],
        "val_error_loss": [], "val_cls_acc": [],
    }

    best_val_loss = float("inf")
    best_state = None
    log_file = RESULTS_DIR / f"training_{config.mode}.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w") as f:
        for epoch in range(config.epochs):
            t0 = time.time()
            train_metrics = train_epoch(
                model, train_loader, optimizer, config, epoch, device,
            )
            val_metrics = eval_epoch(model, val_loader, config, device)
            scheduler.step()
            elapsed = time.time() - t0

            # Record history
            for k, v in train_metrics.items():
                history[f"train_{k}"].append(v)
            for k, v in val_metrics.items():
                if k != "per_region_mse":
                    history[f"val_{k}"].append(v)

            # Log
            log_entry = {
                "epoch": epoch,
                "elapsed": elapsed,
                "mode": config.mode,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "per_region_mse"},
            }
            f.write(json.dumps(log_entry) + "\n")
            f.flush()

            if epoch % config.log_every == 0 or epoch == config.epochs - 1:
                print(
                    f"[{config.mode}] Epoch {epoch:3d}/{config.epochs} "
                    f"| train loss {train_metrics['loss']:.4f} "
                    f"| val loss {val_metrics['loss']:.4f} "
                    f"| val cls {val_metrics['cls_acc']:.1%} "
                    f"| {elapsed:.1f}s"
                )

            # Save best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save final checkpoint
    ckpt_path = RESULTS_DIR / f"checkpoint_{config.mode}.pt"
    torch.save({
        "model_state": best_state or model.state_dict(),
        "config": vars(config),
        "history": history,
        "region_names": dataset.region_names,
        "category_names": dataset.category_names,
        "n_params": n_params,
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Final evaluation with best model
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = eval_epoch(model, val_loader, config, device)

    return {
        "history": history,
        "final_val": {k: v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in final_val.items()},
        "n_params": n_params,
        "config": vars(config),
        "region_names": dataset.region_names,
        "category_names": dataset.category_names,
    }


def run_all_baselines(data: Dict, epochs: int = 100, d_model: int = 128) -> Dict[str, Dict]:
    """Run all three model variants for comparison."""
    results = {}
    for mode in ["cortical", "dense", "flat"]:
        print(f"\n{'='*60}")
        print(f"Training: {mode}")
        print(f"{'='*60}")
        config = TrainConfig(
            mode=mode, epochs=epochs, d_model=d_model,
        )
        results[mode] = train(config, data)

    # Save comparison summary
    summary = {}
    for mode, r in results.items():
        summary[mode] = {
            "n_params": r["n_params"],
            "final_val_loss": r["final_val"]["loss"],
            "final_val_cls_acc": r["final_val"]["cls_acc"],
            "final_val_region_loss": r["final_val"]["region_loss"],
        }

    summary_path = RESULTS_DIR / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nComparison summary saved: {summary_path}")
    print(json.dumps(summary, indent=2))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cortical canvas model")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz dataset")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic dataset")
    parser.add_argument("--mode", type=str, default="cortical",
                        choices=["cortical", "dense", "flat", "all"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load or generate data
    if args.data:
        print(f"Loading dataset from {args.data}...")
        data = load_dataset(args.data)
    elif args.synthetic:
        print("Generating synthetic dataset...")
        data = generate_synthetic_dataset()
    else:
        print("ERROR: Specify --data <path> or --synthetic")
        sys.exit(1)

    print(f"Dataset: {data['region_activations'].shape[0]} samples, "
          f"{data['region_activations'].shape[1]} regions, "
          f"{len(data['category_names'])} categories")

    if args.mode == "all":
        results = run_all_baselines(data, epochs=args.epochs, d_model=args.d_model)
    else:
        config = TrainConfig(
            mode=args.mode,
            epochs=args.epochs,
            d_model=args.d_model,
            n_layers=args.n_layers,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        result = train(config, data)

        print(f"\nFinal validation metrics ({args.mode}):")
        for k, v in result["final_val"].items():
            if isinstance(v, (list, np.ndarray)):
                continue
            print(f"  {k}: {v:.4f}")

"""Training harness for temporal fill integration tests.

Provides reusable infrastructure for training small models on synthetic
tasks that exercise temporal fill modes, PeriodEmbedding, and the
AttentionDispatcher's fill logic.

All results are logged to disk (JSONL + CSV + checkpoints) so they can
be revisited for analysis outside the test session.
"""

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from canvas_engineering import (
    CanvasLayout,
    RegionSpec,
    CanvasTopology,
    Connection,
    TemporalFill,
    SpatiotemporalCanvas,
)
from canvas_engineering.dispatch import AttentionDispatcher


RESULTS_DIR = Path(__file__).parent.parent / "test_results"


# ── Logging ──────────────────────────────────────────────────────────


class ResultLogger:
    """Logs per-step metrics to JSONL and saves checkpoints.

    Usage:
        logger = ResultLogger("stale_copy", "hold")
        for step in range(200):
            logger.log_step({"step": step, "loss": 0.1})
        logger.save_checkpoint(model, step=200)
        logger.write_summary({"final_loss": 0.05})
        logger.close()
    """

    def __init__(self, task_name: str, fill_mode: str, run_dir: Optional[Path] = None):
        if run_dir is None:
            ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            run_dir = RESULTS_DIR / "{}_{}".format(ts, task_name)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.task_name = task_name
        self.fill_mode = fill_mode

        # Subdirectory per fill mode within the run
        self.mode_dir = self.run_dir / fill_mode
        self.mode_dir.mkdir(exist_ok=True)

        self._metrics_path = self.mode_dir / "metrics.jsonl"
        self._metrics_file = open(self._metrics_path, "w")
        self._steps: List[dict] = []

    def log_config(self, config: dict):
        """Save the full configuration for reproducibility."""
        with open(self.mode_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_step(self, metrics: dict):
        """Log one training step. Written immediately to JSONL."""
        metrics["fill_mode"] = self.fill_mode
        metrics["task"] = self.task_name
        self._metrics_file.write(json.dumps(metrics) + "\n")
        self._metrics_file.flush()
        self._steps.append(metrics)

    def save_checkpoint(self, model: nn.Module, step: int):
        """Save model state dict."""
        path = self.mode_dir / "checkpoint_step_{:04d}.pt".format(step)
        torch.save(model.state_dict(), path)

    def save_learned_params(self, params: dict):
        """Save a snapshot of key learned parameter values."""
        # Convert tensors to lists for JSON serialization
        serializable = {}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                serializable[k] = v.detach().cpu().tolist()
            else:
                serializable[k] = v
        with open(self.mode_dir / "learned_params.json", "w") as f:
            json.dump(serializable, f, indent=2)

    def write_summary(self, summary: dict):
        """Write a summary dict (final metrics, etc.)."""
        summary["fill_mode"] = self.fill_mode
        summary["task"] = self.task_name
        with open(self.mode_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def close(self):
        self._metrics_file.close()

    @property
    def steps(self) -> List[dict]:
        return self._steps


# ── Model ────────────────────────────────────────────────────────────


class TemporalFillModel(nn.Module):
    """Minimal model exercising canvas temporal fill in a training loop.

    Architecture: input projection → canvas placement → N x (dispatcher + FFN) → readout.

    Uses AttentionDispatcher (not additive mask) so that _resolve_temporal_fill
    and TemporalFillModule.transform_keys are exercised on every forward pass.
    """

    def __init__(
        self,
        layout: CanvasLayout,
        topology: CanvasTopology,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.0,
        use_period_embedding: bool = True,
    ):
        super().__init__()
        self.layout = layout
        self.d_model = d_model
        self.use_period_embedding = use_period_embedding

        self.canvas = SpatiotemporalCanvas(layout)

        # Stack of dispatcher + FFN layers
        self.dispatchers = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(n_layers):
            self.dispatchers.append(
                AttentionDispatcher(
                    topology=topology,
                    layout=layout,
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            )
            self.ffns.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            ))

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        """Forward pass through dispatcher + FFN stack.

        Args:
            canvas: (B, N, d_model) canvas with data already placed.

        Returns:
            (B, N, d_model) updated canvas.
        """
        x = canvas
        for dispatcher, ffn in zip(self.dispatchers, self.ffns):
            x = dispatcher(x)
            x = x + ffn(x)  # residual
        return x


# ── Data generation ──────────────────────────────────────────────────


def generate_stale_copy_data(
    n_samples: int,
    d_model: int,
    n_fast_frames: int = 8,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Stale copy task: fast region should reconstruct slow region's value.

    The slow region has a value at t=0. The fast region's target at every
    timestep is that same value. Tests whether fill modes let the fast
    region access the slow region's value at non-aligned timesteps.

    Returns:
        dict with 'slow_data', 'fast_targets' tensors.
    """
    gen = torch.Generator().manual_seed(seed)
    slow_data = torch.randn(n_samples, 1, d_model, generator=gen)
    # Target: every fast frame should reconstruct the slow value
    fast_targets = slow_data.expand(n_samples, n_fast_frames, d_model).clone()
    return {"slow_data": slow_data, "fast_targets": fast_targets}


def generate_decay_relevance_data(
    n_samples: int,
    d_model: int,
    n_fast_frames: int = 8,
    tau: float = 3.0,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Decay relevance task: signal decays exponentially with staleness.

    The slow region holds a value v at t=0. The fast region's target at
    real time k is v * exp(-k/tau). A model with DECAY fill should learn
    to down-weight stale values, matching the ground-truth decay.

    Returns:
        dict with 'slow_data', 'fast_targets', 'tau'.
    """
    gen = torch.Generator().manual_seed(seed)
    slow_data = torch.randn(n_samples, 1, d_model, generator=gen)
    fast_targets = torch.zeros(n_samples, n_fast_frames, d_model)
    for k in range(n_fast_frames):
        decay = math.exp(-k / tau)
        fast_targets[:, k] = slow_data[:, 0] * decay
    return {"slow_data": slow_data, "fast_targets": fast_targets, "tau": tau}


def generate_drift_data(
    n_samples: int,
    d_model: int,
    n_fast_frames: int = 8,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Predictable drift task: slow region follows a linear trend.

    The slow region holds v0 at t=0. The fast region's target at time k
    is v0 + slope * k. The slope is large enough that holding v0 gives
    substantial error at later frames, rewarding learned extrapolation.

    Returns:
        dict with 'slow_data', 'fast_targets', 'slopes'.
    """
    gen = torch.Generator().manual_seed(seed)
    v0 = torch.randn(n_samples, 1, d_model, generator=gen) * 0.5
    slopes = torch.randn(n_samples, 1, d_model, generator=gen) * 0.3
    fast_targets = torch.zeros(n_samples, n_fast_frames, d_model)
    for k in range(n_fast_frames):
        fast_targets[:, k] = v0[:, 0] + slopes[:, 0] * k
    return {"slow_data": v0, "fast_targets": fast_targets, "slopes": slopes}


def generate_interpolation_data(
    n_samples: int,
    d_model: int,
    n_fast_frames: int = 8,
    slow_period: int = 4,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Interpolation task: fast region should lerp between slow region updates.

    The slow region updates at real times 0 and slow_period. Its value at
    real time 0 is v0 and at real time slow_period is v1. The fast region's
    target at real time k is lerp(v0, v1, k/slow_period) for k in [0, slow_period],
    clamped beyond that.

    Returns:
        dict with 'slow_data' (N, 2, d_model) and 'fast_targets'.
    """
    gen = torch.Generator().manual_seed(seed)
    v0 = torch.randn(n_samples, 1, d_model, generator=gen)
    v1 = torch.randn(n_samples, 1, d_model, generator=gen)
    # Slow region has 2 canvas frames: canvas t=0 → v0, canvas t=1 → v1
    slow_data = torch.cat([v0, v1], dim=1)  # (N, 2, d_model)
    # Fast targets: lerp between v0 and v1 across real time
    fast_targets = torch.zeros(n_samples, n_fast_frames, d_model)
    for k in range(n_fast_frames):
        alpha = min(k / slow_period, 1.0)
        fast_targets[:, k] = v0[:, 0] * (1 - alpha) + v1[:, 0] * alpha
    return {"slow_data": slow_data, "fast_targets": fast_targets}


# ── Training ─────────────────────────────────────────────────────────


def make_layout_and_topology(
    fill_mode: TemporalFill,
    d_model: int = 32,
    n_fast_frames: int = 8,
    slow_period: int = 1,
) -> Tuple[CanvasLayout, CanvasTopology]:
    """Create a layout with fast + slow regions and a fill-mode connection.

    Args:
        slow_period: Period of the slow region. When >1, the slow region's
            canvas frames map to non-adjacent real times, creating natural
            gaps that INTERPOLATE/DECAY can exploit.
            With slow_period=1, slow exists only at canvas t=0 (one frame).
            With slow_period=4, slow has 2 canvas frames mapping to real
            times 0 and 4, with gaps at real times 1-3 and 5-7.
    """
    if slow_period > 1:
        # Slow region spans 2 canvas frames, mapping to real times 0 and slow_period
        slow_n_frames = 2
        slow_spec = RegionSpec(bounds=(0, slow_n_frames, 1, 2, 0, 1), period=slow_period)
    else:
        slow_spec = RegionSpec(bounds=(0, 1, 1, 2, 0, 1), period=1)

    layout = CanvasLayout(
        T=n_fast_frames,
        H=2,
        W=1,
        d_model=d_model,
        regions={
            "fast": (0, n_fast_frames, 0, 1, 0, 1),
            "slow": slow_spec,
        },
    )
    topology = CanvasTopology(connections=[
        Connection(src="fast", dst="fast", t_src=0, t_dst=0),
        Connection(
            src="fast", dst="slow", t_src=0, t_dst=0,
            temporal_fill=fill_mode,
            decay_halflife=3.0,
        ),
    ])
    return layout, topology


def make_two_anchor_layout_and_topology(
    fill_mode: TemporalFill,
    d_model: int = 32,
    n_fast_frames: int = 8,
) -> Tuple[CanvasLayout, CanvasTopology]:
    """Layout with two slow anchors (start/end) and a fast region.

    slow_start at t=0, slow_end at t=(n_fast_frames-1).
    Fast queries both anchors with the given fill mode.
    INTERPOLATE can lerp between them for intermediate timesteps.
    """
    layout = CanvasLayout(
        T=n_fast_frames,
        H=3,
        W=1,
        d_model=d_model,
        regions={
            "fast": (0, n_fast_frames, 0, 1, 0, 1),
            "slow_start": (0, 1, 1, 2, 0, 1),                     # t=0 only
            "slow_end": (n_fast_frames - 1, n_fast_frames, 2, 3, 0, 1),  # t=last only
        },
    )
    topology = CanvasTopology(connections=[
        Connection(src="fast", dst="fast", t_src=0, t_dst=0),
        Connection(
            src="fast", dst="slow_start", t_src=0, t_dst=0,
            temporal_fill=fill_mode, decay_halflife=3.0,
        ),
        Connection(
            src="fast", dst="slow_end", t_src=0, t_dst=0,
            temporal_fill=fill_mode, decay_halflife=3.0,
        ),
    ])
    return layout, topology


def train_fill_mode(
    fill_mode: TemporalFill,
    data: Dict[str, torch.Tensor],
    n_steps: int = 200,
    d_model: int = 32,
    n_fast_frames: int = 8,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    logger: Optional[ResultLogger] = None,
    slow_period: int = 1,
) -> Tuple[nn.Module, List[float]]:
    """Train a TemporalFillModel on a synthetic task.

    Returns:
        (model, losses) — trained model and per-step loss list.
    """
    torch.manual_seed(seed)

    layout, topology = make_layout_and_topology(fill_mode, d_model, n_fast_frames, slow_period)
    model = TemporalFillModel(
        layout, topology, d_model=d_model, n_heads=2, n_layers=2, dropout=0.0,
    )

    # Input projection: raw d_model → canvas d_model
    input_proj = nn.Linear(d_model, d_model)
    # Readout: canvas d_model → target d_model
    readout = nn.Linear(d_model, d_model)

    params = list(model.parameters()) + list(input_proj.parameters()) + list(readout.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    slow_data = data["slow_data"]       # (N, 1, d_model)
    fast_targets = data["fast_targets"] # (N, n_fast_frames, d_model)
    n_samples = slow_data.shape[0]

    if logger:
        logger.log_config({
            "fill_mode": fill_mode.value,
            "d_model": d_model,
            "n_fast_frames": n_fast_frames,
            "n_steps": n_steps,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "n_samples": n_samples,
            "n_params": sum(p.numel() for p in params),
        })

    losses = []
    for step in range(n_steps):
        # Sample batch
        idx = torch.randint(0, n_samples, (batch_size,))
        slow_batch = slow_data[idx]       # (B, 1, d_model)
        target_batch = fast_targets[idx]  # (B, n_fast_frames, d_model)

        # Build canvas: place slow data at slow region
        canvas = model.canvas.create_empty(batch_size)
        slow_proj = input_proj(slow_batch)
        canvas = model.canvas.place(canvas, slow_proj, "slow")

        # Forward through dispatcher stack
        output = model(canvas)

        # Extract fast region and compute loss
        fast_out = model.canvas.extract(output, "fast")  # (B, n_fast_frames, d_model)
        predictions = readout(fast_out)
        loss = ((predictions - target_batch) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Per-frame losses for analysis
        with torch.no_grad():
            per_frame_losses = ((predictions - target_batch) ** 2).mean(dim=(0, 2))

        step_metrics = {
            "step": step,
            "loss": loss_val,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }
        # Add per-frame loss
        for f in range(n_fast_frames):
            step_metrics["loss_frame_{}".format(f)] = per_frame_losses[f].item()

        # Fill-mode-specific metrics
        if fill_mode == TemporalFill.DECAY:
            # Log effective decay weights at various staleness levels
            for s in [1, 2, 4]:
                w = math.exp(-s * math.log(2) / 3.0)
                step_metrics["decay_weight_staleness_{}".format(s)] = w

        if fill_mode == TemporalFill.PREDICT and model.dispatchers[0].fill_module is not None:
            fm = model.dispatchers[0].fill_module
            for key, head in fm.predict_heads.items():
                out_weight = head.net[-1].weight
                step_metrics["predict_head_{}_output_norm".format(key)] = out_weight.norm().item()

        # PeriodEmbedding norms
        pe = model.canvas.period_embedding
        for b in [0, 5, 10, 15]:
            if b < pe.n_buckets:
                step_metrics["period_emb_bucket_{}_norm".format(b)] = \
                    pe.embedding.weight[b].norm().item()

        if logger:
            logger.log_step(step_metrics)

        if logger and step in (0, n_steps // 2, n_steps - 1):
            logger.save_checkpoint(model, step)

    # Save final learned params
    if logger:
        learned = {}
        pe = model.canvas.period_embedding
        learned["period_embedding_weights"] = pe.embedding.weight
        if model.dispatchers[0].fill_module is not None:
            fm = model.dispatchers[0].fill_module
            for key, head in fm.predict_heads.items():
                learned["predict_head_{}_output_weight".format(key)] = head.net[-1].weight
                learned["predict_head_{}_output_bias".format(key)] = head.net[-1].bias
        logger.save_learned_params(learned)

        # Summary
        final_loss = sum(losses[-10:]) / 10 if len(losses) >= 10 else losses[-1]
        with torch.no_grad():
            canvas = model.canvas.create_empty(n_samples)
            slow_proj = input_proj(slow_data)
            canvas = model.canvas.place(canvas, slow_proj, "slow")
            output = model(canvas)
            fast_out = model.canvas.extract(output, "fast")
            predictions = readout(fast_out)
            eval_loss = ((predictions - fast_targets) ** 2).mean().item()
            per_frame = ((predictions - fast_targets) ** 2).mean(dim=(0, 2))

        summary = {
            "final_train_loss": final_loss,
            "eval_loss": eval_loss,
        }
        for f in range(n_fast_frames):
            summary["eval_loss_frame_{}".format(f)] = per_frame[f].item()
        logger.write_summary(summary)

    return model, losses


def train_two_anchor(
    fill_mode: TemporalFill,
    data: Dict[str, torch.Tensor],
    n_steps: int = 200,
    d_model: int = 32,
    n_fast_frames: int = 8,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    logger: Optional[ResultLogger] = None,
) -> Tuple[nn.Module, List[float]]:
    """Train on the two-anchor interpolation task."""
    torch.manual_seed(seed)

    layout, topology = make_two_anchor_layout_and_topology(fill_mode, d_model, n_fast_frames)
    model = TemporalFillModel(
        layout, topology, d_model=d_model, n_heads=2, n_layers=2, dropout=0.0,
    )

    input_proj_start = nn.Linear(d_model, d_model)
    input_proj_end = nn.Linear(d_model, d_model)
    readout = nn.Linear(d_model, d_model)

    params = (list(model.parameters()) + list(input_proj_start.parameters()) +
              list(input_proj_end.parameters()) + list(readout.parameters()))
    opt = torch.optim.Adam(params, lr=lr)

    slow_start = data["slow_start_data"]
    slow_end = data["slow_end_data"]
    fast_targets = data["fast_targets"]
    n_samples = slow_start.shape[0]

    if logger:
        logger.log_config({
            "fill_mode": fill_mode.value,
            "task": "two_anchor_interpolation",
            "d_model": d_model,
            "n_fast_frames": n_fast_frames,
            "n_steps": n_steps,
            "seed": seed,
            "n_params": sum(p.numel() for p in params),
        })

    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, n_samples, (batch_size,))

        canvas = model.canvas.create_empty(batch_size)
        canvas = model.canvas.place(canvas, input_proj_start(slow_start[idx]), "slow_start")
        canvas = model.canvas.place(canvas, input_proj_end(slow_end[idx]), "slow_end")

        output = model(canvas)
        fast_out = model.canvas.extract(output, "fast")
        predictions = readout(fast_out)
        loss = ((predictions - fast_targets[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if logger:
            logger.log_step({"step": step, "loss": loss_val})
            if step in (0, n_steps // 2, n_steps - 1):
                logger.save_checkpoint(model, step)

    if logger:
        final_loss = sum(losses[-10:]) / 10 if len(losses) >= 10 else losses[-1]
        logger.write_summary({"final_train_loss": final_loss})

    return model, losses


def run_comparison(
    task_name: str,
    data_fn,
    fill_modes: Optional[List[TemporalFill]] = None,
    n_steps: int = 200,
    d_model: int = 32,
    seed: int = 42,
    run_dir: Optional[Path] = None,
    slow_period: int = 1,
) -> Dict[str, Dict]:
    """Train all fill modes on a task and return results for comparison.

    Returns:
        {fill_mode_name: {"model": model, "losses": [...], "logger": logger}}
    """
    if fill_modes is None:
        fill_modes = list(TemporalFill)

    if run_dir is None:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = RESULTS_DIR / "{}_{}".format(ts, task_name)

    data = data_fn(n_samples=512, d_model=d_model, seed=seed)
    results = {}

    for mode in fill_modes:
        logger = ResultLogger(task_name, mode.value, run_dir=run_dir)
        model, losses = train_fill_mode(
            fill_mode=mode,
            data=data,
            n_steps=n_steps,
            d_model=d_model,
            seed=seed,
            logger=logger,
            slow_period=slow_period,
        )
        logger.close()
        results[mode.value] = {
            "model": model,
            "losses": losses,
            "logger": logger,
            "final_loss": sum(losses[-10:]) / 10 if len(losses) >= 10 else losses[-1],
        }

    # Write combined summary CSV
    import csv
    csv_path = run_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fill_mode", "final_loss", "task"])
        writer.writeheader()
        for mode_name, r in results.items():
            writer.writerow({
                "fill_mode": mode_name,
                "final_loss": r["final_loss"],
                "task": task_name,
            })

    return results

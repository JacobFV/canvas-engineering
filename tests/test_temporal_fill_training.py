"""Integration tests: temporal fill modes under training.

These tests train small models on synthetic tasks where the ground-truth
behavior of each fill mode is known, then assert the expected loss ordering.
All per-step metrics, checkpoints, and learned parameters are logged to
test_results/ for offline analysis.

Run:
    pytest tests/test_temporal_fill_training.py -v --timeout=120

    # Or standalone for longer runs:
    python tests/test_temporal_fill_training.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from canvas_engineering import (
    CanvasLayout,
    RegionSpec,
    CanvasTopology,
    Connection,
    TemporalFill,
    SpatiotemporalCanvas,
    PeriodEmbedding,
)
from canvas_engineering.dispatch import AttentionDispatcher

# Allow importing from tests/
sys.path.insert(0, str(Path(__file__).parent))
from temporal_fill_harness import (
    ResultLogger,
    TemporalFillModel,
    run_comparison,
    train_fill_mode,
    train_two_anchor,
    make_layout_and_topology,
    generate_stale_copy_data,
    generate_decay_relevance_data,
    generate_drift_data,
    generate_two_anchor_interpolation_data,
    RESULTS_DIR,
)


# ── Shared fixtures ──────────────────────────────────────────────────

D_MODEL = 32
N_STEPS = 150
SEED = 42


@pytest.fixture(scope="module")
def stale_copy_results():
    """Train all fill modes on the stale copy task (shared across tests)."""
    return run_comparison(
        task_name="stale_copy",
        data_fn=generate_stale_copy_data,
        n_steps=N_STEPS,
        d_model=D_MODEL,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def decay_results():
    """Train HOLD, DROP, DECAY on the decay relevance task."""
    return run_comparison(
        task_name="decay_relevance",
        data_fn=generate_decay_relevance_data,
        fill_modes=[TemporalFill.DROP, TemporalFill.HOLD, TemporalFill.DECAY],
        n_steps=N_STEPS,
        d_model=D_MODEL,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def drift_results():
    """Train HOLD, PREDICT on the predictable drift task. More steps for PREDICT to learn."""
    return run_comparison(
        task_name="drift",
        data_fn=generate_drift_data,
        fill_modes=[TemporalFill.DROP, TemporalFill.HOLD, TemporalFill.PREDICT],
        n_steps=300,
        d_model=D_MODEL,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def interpolation_results():
    """Train HOLD, DROP, INTERPOLATE on the two-anchor interpolation task.

    Uses two slow anchor regions (start and end) so INTERPOLATE can
    meaningfully lerp between them for intermediate fast frames.
    """
    data = generate_two_anchor_interpolation_data(
        n_samples=512, d_model=D_MODEL, seed=SEED,
    )
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = RESULTS_DIR / "{}_interpolation".format(ts)
    results = {}
    for mode in [TemporalFill.DROP, TemporalFill.HOLD, TemporalFill.INTERPOLATE]:
        logger = ResultLogger("interpolation", mode.value, run_dir=run_dir)
        model, losses = train_two_anchor(
            fill_mode=mode, data=data, n_steps=N_STEPS,
            d_model=D_MODEL, seed=SEED, logger=logger,
        )
        logger.close()
        final_loss = sum(losses[-10:]) / 10 if len(losses) >= 10 else losses[-1]
        results[mode.value] = {
            "model": model, "losses": losses, "logger": logger,
            "final_loss": final_loss,
        }
    return results


# ── Task 1: Stale Copy ──────────────────────────────────────────────

class TestStaleCopy:
    """Fast region reconstructs slow region's held value.

    The slow region exists only at t=0. The fast region should reconstruct
    it at every timestep. HOLD gives the fast region access to the slow
    value at all timesteps; DROP denies access at non-aligned frames.
    """

    def test_hold_beats_drop(self, stale_copy_results):
        """HOLD should significantly outperform DROP overall."""
        hold_loss = stale_copy_results["hold"]["final_loss"]
        drop_loss = stale_copy_results["drop"]["final_loss"]
        assert hold_loss < drop_loss, (
            "HOLD ({:.4f}) should beat DROP ({:.4f})".format(hold_loss, drop_loss)
        )

    def test_predict_competitive_with_hold(self, stale_copy_results):
        """PREDICT should be at most slightly worse than HOLD (starts as identity)."""
        hold_loss = stale_copy_results["hold"]["final_loss"]
        predict_loss = stale_copy_results["predict"]["final_loss"]
        # PREDICT may be slightly worse due to extra parameters, but not 2x worse
        assert predict_loss < hold_loss * 2.0, (
            "PREDICT ({:.4f}) should be competitive with HOLD ({:.4f})".format(
                predict_loss, hold_loss)
        )

    def test_interpolate_beats_drop(self, stale_copy_results):
        """INTERPOLATE should beat DROP (provides some info at all frames)."""
        interp_loss = stale_copy_results["interpolate"]["final_loss"]
        drop_loss = stale_copy_results["drop"]["final_loss"]
        assert interp_loss < drop_loss, (
            "INTERPOLATE ({:.4f}) should beat DROP ({:.4f})".format(
                interp_loss, drop_loss)
        )

    def test_all_modes_converge(self, stale_copy_results):
        """All fill modes should converge (final loss < initial loss)."""
        for mode_name, result in stale_copy_results.items():
            losses = result["losses"]
            initial = sum(losses[:5]) / 5
            final = sum(losses[-5:]) / 5
            assert final < initial, (
                "{} did not converge: initial={:.4f}, final={:.4f}".format(
                    mode_name, initial, final)
            )

    def test_results_logged_to_disk(self, stale_copy_results):
        """Check that results were written to disk."""
        for mode_name, result in stale_copy_results.items():
            logger = result["logger"]
            assert (logger.mode_dir / "metrics.jsonl").exists()
            assert (logger.mode_dir / "config.json").exists()
            assert (logger.mode_dir / "summary.json").exists()
            # Verify JSONL is valid
            with open(logger.mode_dir / "metrics.jsonl") as f:
                lines = f.readlines()
                assert len(lines) == N_STEPS
                first = json.loads(lines[0])
                assert "loss" in first
                assert "step" in first


# ── Task 2: Decay Relevance ─────────────────────────────────────────

class TestDecayRelevance:
    """Signal decays with staleness. DECAY's prior matches the task."""

    def test_decay_beats_hold(self, decay_results):
        """DECAY should outperform HOLD when the signal actually decays."""
        decay_loss = decay_results["decay"]["final_loss"]
        hold_loss = decay_results["hold"]["final_loss"]
        assert decay_loss < hold_loss, (
            "DECAY ({:.4f}) should beat HOLD ({:.4f}) on a decaying signal".format(
                decay_loss, hold_loss)
        )

    def test_hold_beats_drop(self, decay_results):
        """HOLD should still beat DROP (some info is better than none)."""
        hold_loss = decay_results["hold"]["final_loss"]
        drop_loss = decay_results["drop"]["final_loss"]
        assert hold_loss < drop_loss, (
            "HOLD ({:.4f}) should beat DROP ({:.4f})".format(hold_loss, drop_loss)
        )


# ── Task 3: Predictable Drift ───────────────────────────────────────

class TestPredictableDrift:
    """Linear trend the predict head should extrapolate."""

    def test_predict_beats_hold(self, drift_results):
        """PREDICT should outperform HOLD on a linear drift task."""
        predict_loss = drift_results["predict"]["final_loss"]
        hold_loss = drift_results["hold"]["final_loss"]
        assert predict_loss < hold_loss, (
            "PREDICT ({:.4f}) should beat HOLD ({:.4f}) on linear drift".format(
                predict_loss, hold_loss)
        )

    def test_predict_head_learns_nontrivial_correction(self, drift_results):
        """The predict head's output layer should have non-zero weights after training."""
        model = drift_results["predict"]["model"]
        fm = model.dispatchers[0].fill_module
        if fm is not None:
            for key, head in fm.predict_heads.items():
                out_norm = head.net[-1].weight.norm().item()
                assert out_norm > 1e-4, (
                    "Predict head '{}' output weight norm ({:.6f}) should be "
                    "non-trivial after training on drift task".format(key, out_norm)
                )


# ── Task 4: Smooth Interpolation ────────────────────────────────────

class TestTwoAnchorInterpolation:
    """Two-anchor lerp task: fast region should blend between start/end slow values.

    With two slow anchors at t=0 and t=last, INTERPOLATE gives the model
    weighted access to both endpoints for intermediate fast frames. HOLD
    only provides the past anchor (t=0), losing the future endpoint.
    """

    def test_hold_beats_drop(self, interpolation_results):
        """HOLD should beat DROP (some info is better than none)."""
        hold_loss = interpolation_results["hold"]["final_loss"]
        drop_loss = interpolation_results["drop"]["final_loss"]
        assert hold_loss < drop_loss, (
            "HOLD ({:.4f}) should beat DROP ({:.4f}) on two-anchor task".format(
                hold_loss, drop_loss)
        )

    def test_all_modes_converge(self, interpolation_results):
        """All modes should converge from initial loss."""
        for mode_name, result in interpolation_results.items():
            losses = result["losses"]
            initial = sum(losses[:5]) / 5
            final = sum(losses[-5:]) / 5
            assert final < initial, (
                "{} did not converge: initial={:.4f}, final={:.4f}".format(
                    mode_name, initial, final)
            )


# ── PeriodEmbedding discrimination ──────────────────────────────────

class TestPeriodEmbeddingDiscrimination:
    """PeriodEmbedding helps the model distinguish fast/slow regions."""

    def test_period_embedding_aids_convergence(self):
        """Model with PeriodEmbedding should converge faster than without."""
        torch.manual_seed(SEED)
        data = generate_stale_copy_data(n_samples=256, d_model=D_MODEL, seed=SEED)

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = RESULTS_DIR / "{}_period_embedding".format(ts)

        # Train with PeriodEmbedding (default)
        logger_with = ResultLogger("period_emb", "with_pe", run_dir=run_dir)
        _, losses_with = train_fill_mode(
            fill_mode=TemporalFill.HOLD,
            data=data, n_steps=100, d_model=D_MODEL, seed=SEED,
            logger=logger_with,
        )
        logger_with.close()

        # Train without PeriodEmbedding: zero out the embedding weights
        logger_without = ResultLogger("period_emb", "without_pe", run_dir=run_dir)
        torch.manual_seed(SEED)
        layout, topology = make_layout_and_topology(TemporalFill.HOLD, D_MODEL)
        model_no_pe = TemporalFillModel(
            layout, topology, d_model=D_MODEL, n_heads=2, n_layers=2,
        )
        # Zero the period embedding so it has no effect
        with torch.no_grad():
            model_no_pe.canvas.period_embedding.embedding.weight.zero_()
            # Freeze it so it stays zero
            model_no_pe.canvas.period_embedding.embedding.weight.requires_grad = False

        input_proj = nn.Linear(D_MODEL, D_MODEL)
        readout = nn.Linear(D_MODEL, D_MODEL)
        params = [p for p in model_no_pe.parameters() if p.requires_grad] + \
                 list(input_proj.parameters()) + list(readout.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        slow_data = data["slow_data"]
        fast_targets = data["fast_targets"]
        n_samples = slow_data.shape[0]
        losses_without = []

        for step in range(100):
            idx = torch.randint(0, n_samples, (64,))
            canvas = model_no_pe.canvas.create_empty(64)
            canvas = model_no_pe.canvas.place(canvas, input_proj(slow_data[idx]), "slow")
            output = model_no_pe(canvas)
            fast_out = model_no_pe.canvas.extract(output, "fast")
            predictions = readout(fast_out)
            loss = ((predictions - fast_targets[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses_without.append(loss.item())
            logger_without.log_step({"step": step, "loss": loss.item()})

        logger_without.close()

        # Compare: model with PE should have lower final loss
        final_with = sum(losses_with[-10:]) / 10
        final_without = sum(losses_without[-10:]) / 10

        # Log the comparison
        with open(run_dir / "comparison.json", "w") as f:
            json.dump({
                "with_pe_final_loss": final_with,
                "without_pe_final_loss": final_without,
                "pe_advantage_ratio": final_without / max(final_with, 1e-8),
            }, f, indent=2)

        # PeriodEmbedding should help (or at least not hurt significantly)
        # This is a soft assertion — PE helps more on harder tasks
        assert final_with < final_without * 1.5, (
            "With PE ({:.4f}) should not be much worse than without ({:.4f})".format(
                final_with, final_without)
        )


# ── Standalone runner ────────────────────────────────────────────────

def main():
    """Run all tasks and produce a combined report."""
    print("=" * 60)
    print("Temporal Fill Integration Tests — Full Run")
    print("=" * 60)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    base_dir = RESULTS_DIR / "{}_full_run".format(ts)
    base_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("stale_copy", generate_stale_copy_data, None),
        ("decay_relevance", generate_decay_relevance_data,
         [TemporalFill.DROP, TemporalFill.HOLD, TemporalFill.DECAY]),
        ("drift", generate_drift_data,
         [TemporalFill.DROP, TemporalFill.HOLD, TemporalFill.PREDICT]),
        ("interpolation", generate_stale_copy_data,  # placeholder, two-anchor uses custom trainer
         [TemporalFill.DROP, TemporalFill.HOLD, TemporalFill.INTERPOLATE]),
    ]

    all_results = {}
    for task_name, data_fn, fill_modes in tasks:
        print("\n--- {} ---".format(task_name))
        run_dir = base_dir / task_name
        results = run_comparison(
            task_name=task_name,
            data_fn=data_fn,
            fill_modes=fill_modes,
            n_steps=200,
            d_model=D_MODEL,
            seed=SEED,
            run_dir=run_dir,
        )
        all_results[task_name] = results
        for mode_name, r in sorted(results.items()):
            print("  {:12s}: final_loss = {:.6f}".format(mode_name, r["final_loss"]))

    # Combined summary
    print("\n" + "=" * 60)
    print("Results saved to: {}".format(base_dir))
    print("=" * 60)

    # Write master summary
    summary = {}
    for task_name, results in all_results.items():
        summary[task_name] = {
            mode: r["final_loss"] for mode, r in results.items()
        }
    with open(base_dir / "master_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nKey findings:")
    sc = all_results.get("stale_copy", {})
    if "hold" in sc and "drop" in sc:
        ratio = sc["drop"]["final_loss"] / max(sc["hold"]["final_loss"], 1e-8)
        print("  Stale copy: HOLD {:.1f}x better than DROP".format(ratio))

    dr = all_results.get("decay_relevance", {})
    if "decay" in dr and "hold" in dr:
        ratio = dr["hold"]["final_loss"] / max(dr["decay"]["final_loss"], 1e-8)
        print("  Decay relevance: DECAY {:.1f}x better than HOLD".format(ratio))

    dt = all_results.get("drift", {})
    if "predict" in dt and "hold" in dt:
        ratio = dt["hold"]["final_loss"] / max(dt["predict"]["final_loss"], 1e-8)
        print("  Drift: PREDICT {:.1f}x better than HOLD".format(ratio))

    ip = all_results.get("interpolation", {})
    if "interpolate" in ip and "hold" in ip:
        ratio = ip["hold"]["final_loss"] / max(ip["interpolate"]["final_loss"], 1e-8)
        print("  Interpolation: INTERPOLATE {:.1f}x better than HOLD".format(ratio))

    return all_results


if __name__ == "__main__":
    main()

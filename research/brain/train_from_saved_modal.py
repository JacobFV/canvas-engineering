"""Train 135-feature dynamics models from saved TRIBE v2 data.

Data already generated — just trains cortical/dense/flat with
activation snapshots. No TRIBE v2 inference needed (fast, no GPU required).

Usage:
    modal run --detach research/brain/train_from_saved_modal.py
    modal run research/brain/train_from_saved_modal.py --collect-only
"""

import modal
import base64
import io
import tarfile
from pathlib import Path

app = modal.App("brain-dynamics-train")

results_vol = modal.Volume.from_name("brain-dynamics-train-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("canvas-engineering>=0.4.0", "torch", "numpy", "matplotlib", "scipy", "scikit-learn")
    .add_local_dir("research/brain", "/root/research/brain", copy=True,
                    ignore=["__pycache__"])
    .add_local_dir("canvas_engineering", "/root/canvas_engineering", copy=True,
                    ignore=["__pycache__"])
)

LOCAL_RESULTS = Path(__file__).parent / "results"


@app.function(image=image, timeout=28800, cpu=8, memory=32768,
              volumes={"/vol": results_vol})
def train():
    """Train all 3 models from saved 135-feature data with activation snapshots."""
    import subprocess, sys, os, shutil, json, numpy as np

    # Setup: symlink results to volume
    if os.path.exists("/vol/train_results"):
        shutil.rmtree("/vol/train_results")
    os.makedirs("/vol/train_results", exist_ok=True)

    # Copy the saved data to the working directory
    src_data = "/root/research/brain/results/dynamics_data.npz"
    if not os.path.exists(src_data):
        print("ERROR: dynamics_data.npz not found in image")
        return

    # Load data
    sys.path.insert(0, "/root/research/brain")
    sys.path.insert(0, "/root")
    os.chdir("/root/research/brain")

    import torch
    import torch.nn as nn
    from canvas_engineering import CanvasTopology, CanvasLayout, RegionSpec, Connection
    from canvas_engineering.dispatch import AttentionDispatcher
    from cortical_canvas import build_cortical_program

    data = np.load(src_data, allow_pickle=True)
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]
    canvas_region_names = list(data["canvas_region_names"])
    feature_to_region = list(data["feature_to_region"])

    n_features = X_train.shape[2]
    window = X_train.shape[1]
    print("Data: {} train, {} val, {} features, window={}".format(
        len(X_train), len(X_val), n_features, window))

    # Save data to volume
    shutil.copy(src_data, "/vol/train_results/dynamics_data.npz")

    # Build topology
    bound, program, _ = build_cortical_program()
    topology = program.schema.topology

    # Group features by region
    region_features = {}
    for f_idx, rname in enumerate(feature_to_region):
        if rname not in region_features:
            region_features[rname] = []
        region_features[rname].append(f_idx)

    # Build connected pairs from topology
    connected_pairs = set()
    for c in topology.connections:
        connected_pairs.add((c.src, c.dst))

    d_model = 64
    n_heads = 4
    n_epochs = 200

    # Snapshot config
    snapshot_epochs = {0, 1, 2, 5, 10, 20, 50, 100, 150, 199}
    n_probe = min(10, len(X_val))

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val, dtype=torch.float32)
    Y_vl = torch.tensor(Y_val, dtype=torch.float32)
    probe_X = X_vl[:n_probe]

    all_results = {}

    for mode in ["dense", "flat"]:  # cortical already done
        print("\n" + "=" * 60)
        print("Training: {} ({} features)".format(mode, n_features))
        print("=" * 60)

        torch.manual_seed(42)

        if mode == "flat":
            hidden = max(d_model * 2, n_features * 2)
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(window * n_features, hidden),
                nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, n_features),
            )
        else:
            # Build per-feature dispatch layout
            dispatch_regions = {}
            for i in range(n_features):
                dispatch_regions["f{}".format(i)] = RegionSpec(
                    bounds=(0, 1, i, i + 1, 0, 1))
            dispatch_layout = CanvasLayout(
                T=1, H=n_features, W=1, d_model=d_model, regions=dispatch_regions)

            if mode == "cortical":
                feature_conns = []
                for r1, feats1 in region_features.items():
                    for r2, feats2 in region_features.items():
                        if (r1, r2) in connected_pairs:
                            for f1 in feats1:
                                for f2 in feats2:
                                    feature_conns.append(
                                        Connection(src="f{}".format(f1), dst="f{}".format(f2)))
                topo = CanvasTopology(connections=feature_conns)
            else:
                feature_names = ["f{}".format(i) for i in range(n_features)]
                topo = CanvasTopology.dense(feature_names)

            n_conns = len(topo.connections)
            print("  Topology: {} connections ({:.1%} density)".format(
                n_conns, n_conns / (n_features * n_features)))

            class DynamicsTransformer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.input_proj = nn.Linear(window, d_model)
                    self.layers = nn.ModuleList()
                    self.norms = nn.ModuleList()
                    self.ffns = nn.ModuleList()
                    for _ in range(3):
                        self.layers.append(AttentionDispatcher(
                            topo, dispatch_layout, d_model, n_heads, dropout=0.1))
                        self.norms.append(nn.LayerNorm(d_model))
                        self.ffns.append(nn.Sequential(
                            nn.Linear(d_model, d_model * 4), nn.GELU(),
                            nn.Dropout(0.1), nn.Linear(d_model * 4, d_model)))
                    self.output_proj = nn.Linear(d_model, 1)

                def forward(self, x, capture=False):
                    h = self.input_proj(x.transpose(1, 2))
                    layer_acts = [h.detach().cpu().numpy()] if capture else None
                    for layer, norm, ffn in zip(self.layers, self.norms, self.ffns):
                        h2 = layer(h)
                        h = norm(h + h2)
                        h = h + ffn(h)
                        if capture:
                            layer_acts.append(h.detach().cpu().numpy())
                    out = self.output_proj(h).squeeze(-1)
                    if capture:
                        self._captured = layer_acts
                    return out

            model = DynamicsTransformer()

        n_params = sum(p.numel() for p in model.parameters())
        print("  Parameters: {}".format(n_params))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        history = {"train_loss": [], "val_loss": [], "val_r2": []}
        activation_snapshots = {}
        batch_size = min(64, len(X_tr))

        for epoch in range(n_epochs):
            model.train()
            idx = torch.randperm(len(X_tr))[:batch_size]
            pred = model(X_tr[idx])
            loss = ((pred - Y_tr[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_vl)
                val_loss = ((val_pred - Y_vl) ** 2).mean().item()
                ss_res = ((val_pred - Y_vl) ** 2).sum().item()
                ss_tot = ((Y_vl - Y_vl.mean(dim=0)) ** 2).sum().item()
                r2 = 1 - ss_res / max(ss_tot, 1e-8)

                if epoch in snapshot_epochs and mode != "flat":
                    model(probe_X, capture=True)
                    activation_snapshots[epoch] = model._captured

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)
            history["val_r2"].append(r2)

            if epoch in snapshot_epochs:
                ckpt_path = "/vol/train_results/checkpoint_{}_{}.pt".format(mode, epoch)
                torch.save(model.state_dict(), ckpt_path)
                results_vol.commit()

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print("  [{}] Epoch {:3d}/{} | loss {:.4f} | val {:.4f} | R² {:.4f}".format(
                    mode, epoch, n_epochs, loss.item(), val_loss, r2), flush=True)

        # Save activation snapshots
        if activation_snapshots:
            snap = {}
            for ep, layers in activation_snapshots.items():
                for li, act in enumerate(layers):
                    snap["epoch{}_layer{}".format(ep, li)] = act
            snap["snapshot_epochs"] = np.array(sorted(activation_snapshots.keys()))
            snap["n_layers"] = len(next(iter(activation_snapshots.values())))
            np.savez_compressed("/vol/train_results/activations_{}.npz".format(mode), **snap)
            results_vol.commit()
            size = os.path.getsize("/vol/train_results/activations_{}.npz".format(mode))
            print("  Saved activations_{}.npz ({:.1f} MB)".format(mode, size / 1e6))

        # Save final checkpoint
        torch.save(model.state_dict(),
                   "/vol/train_results/checkpoint_{}_final.pt".format(mode))

        # Save per-region MSE
        model.eval()
        with torch.no_grad():
            vp = model(X_vl)
            per_region = ((vp - Y_vl) ** 2).mean(dim=0).numpy()

        all_results[mode] = {
            "n_params": n_params,
            "final_r2": history["val_r2"][-1],
            "peak_r2": max(history["val_r2"]),
            "final_val_loss": history["val_loss"][-1],
            "history": history,
            "per_region_mse": per_region.tolist(),
        }

        # Save training log
        with open("/vol/train_results/training_{}.jsonl".format(mode), "w") as f:
            for ep in range(len(history["train_loss"])):
                f.write(json.dumps({
                    "epoch": ep,
                    "train_loss": history["train_loss"][ep],
                    "val_loss": history["val_loss"][ep],
                    "val_r2": history["val_r2"][ep],
                }) + "\n")

        results_vol.commit()
        print("  Committed {} results to volume".format(mode))

    # Save comparison
    comparison = {mode: {k: v for k, v in r.items() if k != "history"}
                  for mode, r in all_results.items()}
    with open("/vol/train_results/comparison_135f.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Generate plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("135-Feature Cortical Dynamics (Real TRIBE v2)", fontsize=14, fontweight="bold")
    colors = {"cortical": "#E74C3C", "dense": "#3498DB", "flat": "#95A5A6"}

    for mode, r in all_results.items():
        axes[0].semilogy(r["history"]["val_loss"], color=colors[mode], lw=2, label=mode)
        axes[1].plot(r["history"]["val_r2"], color=colors[mode], lw=2, label=mode)
    axes[0].set_title("Validation Loss"); axes[0].legend(); axes[0].grid(alpha=0.2)
    axes[1].set_title("R²"); axes[1].legend(); axes[1].grid(alpha=0.2)

    modes = list(all_results.keys())
    peak_r2 = [all_results[m]["peak_r2"] for m in modes]
    axes[2].bar(modes, peak_r2, color=[colors[m] for m in modes])
    for i, v in enumerate(peak_r2):
        axes[2].text(i, v + 0.005, "{:.3f}".format(v), ha="center", fontweight="bold")
    axes[2].set_title("Peak R²"); axes[2].grid(alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig("/vol/train_results/comparison_135f.png", dpi=150, bbox_inches="tight")
    plt.close()
    results_vol.commit()

    print("\n" + "=" * 60)
    print("FINAL RESULTS (135 features, real TRIBE v2)")
    print("=" * 60)
    for mode, r in all_results.items():
        print("  {}: R²={:.4f} (peak {:.4f}), params={}".format(
            mode, r["final_r2"], r["peak_r2"], r["n_params"]))
    print("\nAll results saved to volume.")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/vol": results_vol},
)
def collect():
    """Download results from volume."""
    results_path = Path("/vol/train_results")
    if not results_path.exists():
        print("No results yet.")
        return ""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in results_path.rglob("*"):
            if f.is_file():
                tar.add(str(f), arcname=str(f.relative_to(results_path)))
                print("  {} ({}KB)".format(f.relative_to(results_path), f.stat().st_size // 1024))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.local_entrypoint()
def main(collect_only: bool = False):
    if collect_only:
        print("Collecting results...")
        result = collect.remote()
        if not result:
            return
        LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
        tar_bytes = base64.b64decode(result)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(path=str(LOCAL_RESULTS), filter="data")
        files = [f for f in LOCAL_RESULTS.rglob("*") if f.is_file()]
        print("\nDownloaded {} files to {}".format(len(files), LOCAL_RESULTS))
    else:
        print("Training 135-feature dynamics from saved data...")
        print("Results save to volume incrementally. Safe to disconnect.")
        print("Collect with: modal run research/brain/train_from_saved_modal.py --collect-only")
        train.remote()
        print("Done.")

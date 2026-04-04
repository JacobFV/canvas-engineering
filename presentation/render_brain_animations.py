"""Render brain activation animations from saved cortical dynamics data.

Generates:
1. Activation flow animation: regions lighting up in sequence as the model
   processes a stimulus (V1→V2→fusiform, A1→Wernicke→Broca)
2. Learning animation: how activation patterns change across training epochs
3. Per-region activation heatmap over time

Uses nilearn surface plotting on fsaverage5.
"""

import warnings
import numpy as np
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUTPUT = Path(__file__).parent / "animations"
OUTPUT.mkdir(parents=True, exist_ok=True)

RESULTS = Path(__file__).parent.parent / "research" / "brain" / "results"


# Region-to-ROI mapping (canvas region name → Destrieux atlas labels)
REGION_TO_ATLAS = {
    "visual.v1": ["S_calcarine", "G_cuneus"],
    "visual.v2_v4": ["G_occipital_sup", "G_occipital_middle", "Pole_occipital"],
    "visual.fusiform": ["G_oc-temp_lat-fusifor"],
    "auditory.a1": ["G_temp_sup-G_T_transv"],
    "auditory.wernicke": ["G_temp_sup-Lateral", "G_temp_sup-Plan_tempo"],
    "language.broca": ["G_front_inf-Opercular", "G_front_inf-Triangul"],
    "language.angular": ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    "language.temporal_mid": ["G_temporal_middle"],
    "frontal.prefrontal": ["G_front_sup", "G_orbital", "G_rectus"],
    "frontal.motor": ["G_precentral"],
    "frontal.premotor": ["G_front_middle"],
    "default_mode.precuneus": ["G_precuneus"],
    "default_mode.cingulate": ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant",
                                "G_cingul-Post-dorsal", "G_cingul-Post-ventral"],
    "default_mode.temporal_pole": ["Pole_temporal"],
    "subcortical.insula": ["G_insular_short", "G_Ins_lg_and_S_cent_ins"],
    "subcortical.somatosensory": ["G_postcentral"],
}

# Which regions have features (from the 135-feature mapping)
LEAF_REGIONS = [
    "visual.v1", "visual.v2_v4", "visual.fusiform",
    "auditory.a1", "auditory.wernicke",
    "language.broca", "language.angular", "language.temporal_mid",
    "frontal.prefrontal", "frontal.motor", "frontal.premotor",
    "default_mode.precuneus", "default_mode.cingulate", "default_mode.temporal_pole",
    "subcortical.insula", "subcortical.somatosensory",
]

# Network colors
NETWORK_COLORS = {
    "visual": "#FFEB3B",      # yellow
    "auditory": "#4CAF50",    # green
    "language": "#2196F3",    # blue
    "frontal": "#FF5722",     # red-orange
    "default_mode": "#9C27B0", # purple
    "subcortical": "#FF9800",  # orange
}


def load_atlas():
    """Load Destrieux atlas and build ROI → vertex mapping."""
    from nilearn import datasets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas = datasets.fetch_atlas_surf_destrieux()
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    lh = np.array(atlas["map_left"])
    rh = np.array(atlas["map_right"])
    labels = [str(l) for l in atlas["labels"]]

    return fsaverage, lh, rh, labels


def region_to_vertex_mask(region_name, lh_labels, rh_labels, atlas_labels):
    """Map a canvas region name to vertex masks for left and right hemispheres."""
    atlas_names = REGION_TO_ATLAS.get(region_name, [])
    lh_mask = np.zeros(len(lh_labels), dtype=bool)
    rh_mask = np.zeros(len(rh_labels), dtype=bool)

    for aname in atlas_names:
        if aname in atlas_labels:
            label_idx = atlas_labels.index(aname)
            lh_mask |= (lh_labels == label_idx)
            rh_mask |= (rh_labels == label_idx)

    return lh_mask, rh_mask


def render_brain_frame(fsaverage, lh_act, rh_act, title="", output_path=None):
    """Render one brain frame with activation overlay."""
    from nilearn import plotting

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              subplot_kw={"projection": "3d"})

    for ax, hemi, act, pial, sulc in [
        (axes[0], "left", lh_act, fsaverage["pial_left"], fsaverage["sulc_left"]),
        (axes[1], "right", rh_act, fsaverage["pial_right"], fsaverage["sulc_right"]),
    ]:
        plotting.plot_surf_stat_map(
            pial, act, hemi=hemi, view="lateral",
            colorbar=False, cmap="hot", threshold=0.05,
            bg_map=sulc, axes=ax,
            vmax=1.0,
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)

    if output_path:
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight",
                    facecolor="black")
    plt.close(fig)
    return fig


def animation_1_activation_flow():
    """Animate activation flowing through cortical pathways for one stimulus.

    Shows V1 lighting up first, then propagating through ventral stream
    to fusiform, and through language pathway to Broca's.
    """
    print("Animation 1: Cortical activation flow...")

    fsaverage, lh_labels, rh_labels, atlas_labels = load_atlas()

    # Load activation data
    data = np.load(str(RESULTS / "activations_cortical.npz"))
    canvas_names = np.load(str(RESULTS / "dynamics_data.npz"),
                           allow_pickle=True)["canvas_region_names"]
    feature_to_region = np.load(str(RESULTS / "dynamics_data.npz"),
                                 allow_pickle=True)["feature_to_region"]

    # Use epoch 199 (final trained model), stimulus 0
    # Layer activations: (10 stimuli, 135 features, 64 d_model)
    # We want per-region activation magnitude across layers
    stim_idx = 0

    # Build per-region activation per layer
    n_layers = int(data["n_layers"])
    region_activations = {}  # layer_idx -> {region: activation_magnitude}

    for layer in range(n_layers):
        key = "epoch199_layer{}".format(layer)
        if key not in data:
            continue
        act = data[key]  # (10, 135, 64)
        stim_act = act[stim_idx]  # (135, 64)

        region_act = {}
        for region in LEAF_REGIONS:
            # Get features for this region
            feat_idx = [i for i, r in enumerate(feature_to_region) if r == region]
            if feat_idx:
                # Activation magnitude = mean L2 norm across features
                region_act[region] = np.linalg.norm(stim_act[feat_idx], axis=1).mean()
            else:
                region_act[region] = 0.0

        # Normalize to [0, 1]
        max_act = max(region_act.values()) if region_act else 1.0
        if max_act > 0:
            region_act = {k: v / max_act for k, v in region_act.items()}
        region_activations[layer] = region_act

    # Build vertex activation maps for each layer
    frames_lh = []
    frames_rh = []
    frame_titles = []

    for layer in range(n_layers):
        lh_act = np.zeros(len(lh_labels), dtype=np.float32)
        rh_act = np.zeros(len(rh_labels), dtype=np.float32)

        if layer in region_activations:
            for region, magnitude in region_activations[layer].items():
                lh_mask, rh_mask = region_to_vertex_mask(
                    region, lh_labels, rh_labels, atlas_labels)
                lh_act[lh_mask] = magnitude
                rh_act[rh_mask] = magnitude

        frames_lh.append(lh_act)
        frames_rh.append(rh_act)
        layer_name = ["Input", "Layer 1", "Layer 2", "Layer 3"][layer] if layer < 4 else "Layer {}".format(layer)
        frame_titles.append("Activation Flow: {} (Trained Model)".format(layer_name))

    # Render frames
    from nilearn import plotting
    for i, (lh_act, rh_act, title) in enumerate(zip(frames_lh, frames_rh, frame_titles)):
        path = OUTPUT / "flow_frame_{}.png".format(i)
        plotting.plot_surf_stat_map(
            fsaverage["pial_left"], lh_act,
            hemi="left", view="lateral",
            colorbar=True, cmap="hot", threshold=0.05,
            bg_map=fsaverage["sulc_left"],
            title=title,
            output_file=str(path),
            vmax=1.0,
        )
        plt.close("all")
        print("  Saved: {}".format(path.name))

    # Make a GIF from the flow frames
    from PIL import Image
    gif_frames = []
    for i in range(len(frames_lh)):
        path = OUTPUT / "flow_frame_{}.png".format(i)
        if path.exists():
            gif_frames.append(Image.open(str(path)))
    if gif_frames:
        gif_path = OUTPUT / "activation_flow.gif"
        gif_frames[0].save(str(gif_path), save_all=True,
                           append_images=gif_frames[1:], duration=1000, loop=0)
        print("  Saved: {} ({} frames)".format(gif_path.name, len(gif_frames)))


def animation_2_learning_progression():
    """Show how brain activation patterns change across training epochs.

    At epoch 0: random activations (untrained)
    At epoch 50: starting to specialize
    At epoch 199: clean, structured activation
    """
    print("\nAnimation 2: Learning progression across epochs...")

    fsaverage, lh_labels, rh_labels, atlas_labels = load_atlas()

    data = np.load(str(RESULTS / "activations_cortical.npz"))
    feature_to_region = np.load(str(RESULTS / "dynamics_data.npz"),
                                 allow_pickle=True)["feature_to_region"]

    snapshot_epochs = data["snapshot_epochs"]
    stim_idx = 0
    layer_idx = 3  # final layer

    from nilearn import plotting

    # Render brain at each epoch
    epoch_frames = []
    for epoch in snapshot_epochs:
        key = "epoch{}_layer{}".format(epoch, layer_idx)
        if key not in data:
            continue

        act = data[key][stim_idx]  # (135, 64)

        # Per-region activation magnitude
        lh_act = np.zeros(len(lh_labels), dtype=np.float32)
        rh_act = np.zeros(len(rh_labels), dtype=np.float32)

        for region in LEAF_REGIONS:
            feat_idx = [i for i, r in enumerate(feature_to_region) if r == region]
            if feat_idx:
                magnitude = np.linalg.norm(act[feat_idx], axis=1).mean()
                lh_mask, rh_mask = region_to_vertex_mask(
                    region, lh_labels, rh_labels, atlas_labels)
                lh_act[lh_mask] = magnitude
                rh_act[rh_mask] = magnitude

        # Normalize
        max_val = max(lh_act.max(), rh_act.max(), 0.001)
        lh_act /= max_val
        rh_act /= max_val

        epoch_frames.append((epoch, lh_act, rh_act))

    # Save individual frames for animation
    for epoch, lh_act, _ in epoch_frames:
        path = OUTPUT / "learning_epoch_{:03d}.png".format(epoch)
        plotting.plot_surf_stat_map(
            fsaverage["pial_left"], lh_act,
            hemi="left", view="lateral",
            colorbar=True, cmap="hot", threshold=0.05,
            bg_map=fsaverage["sulc_left"],
            title="Epoch {} — R² = {:.3f}".format(epoch, 0),  # TODO: load actual R²
            output_file=str(path),
            vmax=1.0,
        )
        plt.close("all")
        print("  Saved: {}".format(path.name))

    # Make a GIF from the frames
    from PIL import Image
    gif_frames = []
    for epoch, _, _ in epoch_frames:
        path = OUTPUT / "learning_epoch_{:03d}.png".format(epoch)
        if path.exists():
            gif_frames.append(Image.open(str(path)))

    if gif_frames:
        gif_path = OUTPUT / "learning_progression.gif"
        gif_frames[0].save(
            str(gif_path), save_all=True, append_images=gif_frames[1:],
            duration=800, loop=0,
        )
        print("  Saved: {} ({} frames)".format(gif_path.name, len(gif_frames)))


def animation_3_region_heatmap():
    """Heatmap: per-region activation across layers and epochs."""
    print("\nAnimation 3: Region activation heatmap...")

    data = np.load(str(RESULTS / "activations_cortical.npz"))
    feature_to_region = np.load(str(RESULTS / "dynamics_data.npz"),
                                 allow_pickle=True)["feature_to_region"]

    snapshot_epochs = sorted(data["snapshot_epochs"])
    n_layers = int(data["n_layers"])
    stim_idx = 0

    # Build heatmap: (n_epochs, n_regions) for final layer
    heatmap = np.zeros((len(snapshot_epochs), len(LEAF_REGIONS)))

    for ei, epoch in enumerate(snapshot_epochs):
        key = "epoch{}_layer{}".format(epoch, n_layers - 1)
        if key not in data:
            continue
        act = data[key][stim_idx]  # (135, 64)

        for ri, region in enumerate(LEAF_REGIONS):
            feat_idx = [i for i, r in enumerate(feature_to_region) if r == region]
            if feat_idx:
                heatmap[ei, ri] = np.linalg.norm(act[feat_idx], axis=1).mean()

    # Normalize per epoch
    for ei in range(len(snapshot_epochs)):
        mx = heatmap[ei].max()
        if mx > 0:
            heatmap[ei] /= mx

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    im = ax.imshow(heatmap.T, aspect="auto", cmap="hot", interpolation="nearest",
                    vmin=0, vmax=1)

    ax.set_xticks(range(len(snapshot_epochs)))
    ax.set_xticklabels(["E{}".format(e) for e in snapshot_epochs], fontsize=8, color="white")
    ax.set_yticks(range(len(LEAF_REGIONS)))
    ax.set_yticklabels([r.split(".")[-1] for r in LEAF_REGIONS], fontsize=8, color="white")

    # Color-code region labels by network
    for i, region in enumerate(LEAF_REGIONS):
        network = region.split(".")[0]
        color = NETWORK_COLORS.get(network, "white")
        ax.get_yticklabels()[i].set_color(color)

    ax.set_xlabel("Training Epoch", fontsize=12, color="white")
    ax.set_ylabel("Brain Region", fontsize=12, color="white")
    ax.set_title("Region Activation Magnitude Across Training",
                 fontsize=14, fontweight="bold", color="white")
    ax.tick_params(colors="white")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Normalized activation", color="white")

    path = OUTPUT / "region_heatmap.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print("  Saved: {}".format(path.name))


if __name__ == "__main__":
    print("=" * 60)
    print("Rendering brain activation animations")
    print("=" * 60)

    animation_1_activation_flow()
    animation_2_learning_progression()
    animation_3_region_heatmap()

    print("\nAll animations saved to: {}".format(OUTPUT))
    print("Files:")
    for f in sorted(OUTPUT.glob("*")):
        print("  {} ({:.0f}KB)".format(f.name, f.stat().st_size / 1024))

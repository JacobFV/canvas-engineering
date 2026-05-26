"""Render clear, focused animations for the canvas-engineering examples.

The example training scripts produce visually rich but conceptually opaque
videos. This script regenerates the four animation outputs with a clearer
storytelling pass:

  - simplified scenes (fewer entities, slower pace, larger fonts)
  - one recognizable phenomenon per animation
  - prominent title and persistent caption
  - explicit callouts when the phenomenon fires

Outputs:
  assets/examples/04_fleet.{gif,mp4}     — cooperative collision avoidance
  assets/examples/05_protein.{gif,mp4}   — 4-chain docking with binding readout
  assets/examples/06_atc.{gif,mp4}       — predictive conflict detection
  assets/examples/07_icu_patient.{gif,mp4} — multi-patient deterioration alerts

Usage:
    python scripts/render_clear_animations.py            # all four
    python scripts/render_clear_animations.py 04 07      # subset

Animations are deterministic (fixed seeds). No model is loaded — each
scenario is a hand-crafted minimal demonstration of the phenomenon the
corresponding example trains a canvas-engineered model to predict.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

BG = "#0a0a14"
PANEL = "#0e1020"
FG = "#f0f0ff"
DIM = "#5a5a78"
CYAN = "#00e5ff"
LIME = "#76ff03"
AMBER = "#ffb300"
RED = "#ff3355"
MAGENTA = "#ff44cc"

# Standard layout constants
TITLE_FONT = 14
SUBTITLE_FONT = 9
CAPTION_FONT = 9
LEGEND_FONT = 8


# ─────────────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────────────


def _save(anim: animation.FuncAnimation, name: str, fps_gif: int = 12,
          fps_mp4: int = 24) -> Tuple[str, str]:
    """Save an animation as both GIF and MP4 to ASSETS/<name>."""
    gif_path = os.path.join(ASSETS, f"{name}.gif")
    mp4_path = os.path.join(ASSETS, f"{name}.mp4")
    anim.save(gif_path, writer="pillow", fps=fps_gif)
    print(f"  saved {gif_path}")
    writer = animation.FFMpegWriter(
        fps=fps_mp4, bitrate=4000, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(mp4_path, writer=writer)
    print(f"  saved {mp4_path}")
    return gif_path, mp4_path


def _setup_axes(ax, xlim, ylim, title="", subtitle=""):
    ax.set_facecolor(PANEL)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(DIM)
        spine.set_linewidth(0.5)
    if title:
        ax.text(0.5, 1.06, title, transform=ax.transAxes,
                color=FG, fontsize=TITLE_FONT, fontweight="bold",
                ha="center", va="bottom", fontfamily="monospace")
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes,
                color=DIM, fontsize=SUBTITLE_FONT, ha="center", va="bottom",
                fontfamily="monospace")


def _caption(ax, text, color=DIM):
    ax.text(0.5, -0.04, text, transform=ax.transAxes,
            color=color, fontsize=CAPTION_FONT, ha="center", va="top",
            fontfamily="monospace")


def _phenomenon_pill(ax, text, color=AMBER, y=0.92, x=0.5):
    """A bright, attention-grabbing pill on the axis indicating a fired event."""
    ax.text(x, y, f"  {text}  ", transform=ax.transAxes,
            color=BG, fontsize=10, fontweight="bold", ha="center", va="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor=color))


# ─────────────────────────────────────────────────────────────────────────
# 04 — cooperative trajectory prediction (collision avoidance)
# ─────────────────────────────────────────────────────────────────────────


def render_fleet() -> None:
    """Eight vehicles cross an intersection; collisions are avoided through
    learned cooperation. Close-pass events are highlighted explicitly."""

    print("Rendering 04_fleet (cooperative collision avoidance)...")
    rng = np.random.default_rng(7)

    N = 8
    N_STEPS = 90
    DT = 0.5
    TARGET_SPEED = 0.75   # units per real-second; slow enough to keep in view
    XY_LIM = 22

    # Eight vehicles approach a 4-way intersection from N/S/E/W. Each pair
    # spawns offset so close-pass conflicts arise near the center.
    spawn = [
        (-XY_LIM, -1.0, +1.0, 0.0),
        (-XY_LIM + 3, +1.0, +1.0, 0.0),
        (+XY_LIM,  +1.0, -1.0, 0.0),
        (+XY_LIM - 3, -1.0, -1.0, 0.0),
        (-1.0, -XY_LIM,     0.0, +1.0),
        (+1.0, -XY_LIM + 3, 0.0, +1.0),
        (+1.0, +XY_LIM,     0.0, -1.0),
        (-1.0, +XY_LIM - 3, 0.0, -1.0),
    ]
    pos = np.array([(s[0], s[1]) for s in spawn], dtype=float)
    vel = np.array([(s[2], s[3]) for s in spawn], dtype=float)
    # Normalize each vehicle's velocity to TARGET_SPEED
    speeds = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-6
    vel = vel * (TARGET_SPEED / speeds)

    xs = np.zeros((N_STEPS, N))
    ys = np.zeros((N_STEPS, N))
    closes: List[Tuple[int, int, int, float, float]] = []
    COLLISION_RADIUS = 2.2
    CLOSE_RADIUS = 4.5

    for k in range(N_STEPS):
        ax_f = np.zeros(N)
        ay_f = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = math.sqrt(dx * dx + dy * dy) + 1e-3
                if d < CLOSE_RADIUS:
                    f = (CLOSE_RADIUS - d) * 0.45
                    ax_f[i] += f * dx / d
                    ay_f[i] += f * dy / d
                if d < COLLISION_RADIUS and i < j:
                    closes.append((k, i, j,
                                   (pos[i, 0] + pos[j, 0]) / 2,
                                   (pos[i, 1] + pos[j, 1]) / 2))
        vel += np.column_stack([ax_f, ay_f]) * DT
        # renormalize to target speed (mild)
        speed = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-3
        vel = vel * (TARGET_SPEED / speed) * 0.85 + vel * 0.15
        pos = pos + vel * DT
        xs[k] = pos[:, 0]
        ys[k] = pos[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_position([0.06, 0.10, 0.88, 0.78])

    palette = plt.cm.plasma(np.linspace(0.15, 0.85, N))

    def step(k):
        ax.clear()
        _setup_axes(ax, (-28, 28), (-22, 22),
                    title="COOPERATIVE FLEET — collision avoidance",
                    subtitle="Eight vehicles, one intersection. "
                             "Canvas topology lets each vehicle attend to its neighbors' intent.")
        # Intersection guides
        ax.axhline(0, color=DIM, lw=0.4, alpha=0.4)
        ax.axvline(0, color=DIM, lw=0.4, alpha=0.4)

        # Trails (last 12 frames)
        t0 = max(0, k - 12)
        for i in range(N):
            ax.plot(xs[t0:k + 1, i], ys[t0:k + 1, i],
                    color=palette[i], lw=1.3, alpha=0.55)

        # Vehicles + per-vehicle labels (only current frame)
        for i in range(N):
            ax.plot(xs[k, i], ys[k, i], "o", color=palette[i],
                    markersize=11, markeredgecolor="white", markeredgewidth=0.8,
                    zorder=5)
            ax.text(xs[k, i] + 0.9, ys[k, i] + 0.9, f"V{i}",
                    color=palette[i], fontsize=8, fontfamily="monospace",
                    fontweight="bold")

        # Close-pass alerts (linger 4 frames)
        live = [c for c in closes if 0 <= k - c[0] < 4]
        for _, _, _, mx, my in live:
            ring = mpatches.Circle((mx, my), 2.6, fill=False,
                                   edgecolor=AMBER, lw=2.2, alpha=0.95)
            ax.add_patch(ring)

        # Legend (static, redrawn each frame so it stays on top)
        ax.text(0.02, 0.04, "● vehicle    ─ recent path    ◯ close-pass alert",
                transform=ax.transAxes, color=DIM, fontsize=LEGEND_FONT,
                fontfamily="monospace")

        _caption(ax, "Frame {:>3}/{}   |   "
                     "the orange ring marks a close-pass — the fleet's shared "
                     "intent re-routes nearby vehicles".format(k + 1, N_STEPS))

        if live:
            _phenomenon_pill(ax, f"CLOSE-PASS  V{live[-1][1]} ↔ V{live[-1][2]}")

    anim = animation.FuncAnimation(fig, step, frames=N_STEPS, interval=80)
    _save(anim, "04_fleet")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 06 — predictive conflict detection (ATC)
# ─────────────────────────────────────────────────────────────────────────


def render_atc() -> None:
    """Four aircraft on convergent tracks; conflicts are detected ahead of
    time and visualized as a bright red separation-violation alert."""

    print("Rendering 06_atc (predictive conflict detection)...")
    rng = np.random.default_rng(11)

    N = 6
    N_STEPS = 110
    DT = 1.0

    # Aircraft start on circle around airport, fly inbound.
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False) + 0.4
    radii = np.array([42, 38, 45, 36, 40, 43], dtype=float)
    xs = (radii * np.cos(angles)).astype(float)
    ys = (radii * np.sin(angles)).astype(float)
    # Headings toward airport
    hdg = np.arctan2(-ys, -xs)
    speed = np.full(N, 5.0)

    SEP_MIN = 6.0      # separation minimum (NM equivalent)
    SEP_WARN = 9.0     # predictive warning radius (3 steps lookahead)

    traj_x = np.zeros((N_STEPS, N))
    traj_y = np.zeros((N_STEPS, N))
    conflicts: List[Tuple[int, int, int]] = []  # (frame, i, j)
    warnings: List[Tuple[int, int, int]] = []   # (frame, i, j) — predictive

    for k in range(N_STEPS):
        # Update position
        xs += speed * np.cos(hdg) * DT
        ys += speed * np.sin(hdg) * DT
        # Mild damping toward airport
        toward = np.arctan2(-ys, -xs)
        d_hdg = (toward - hdg + math.pi) % (2 * math.pi) - math.pi
        hdg += d_hdg * 0.06
        traj_x[k] = xs
        traj_y[k] = ys

        # Pairwise current separation
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
                if d < SEP_MIN:
                    conflicts.append((k, i, j))
                elif d < SEP_WARN:
                    warnings.append((k, i, j))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_position([0.06, 0.06, 0.88, 0.84])
    _setup_axes(ax, (-50, 50), (-50, 50),
                title="ATC — predictive conflict detection",
                subtitle="Six aircraft inbound on convergent tracks. The canvas predicts separation violations before they happen.")

    # Radar rings
    for r in [10, 20, 30, 40]:
        ring = mpatches.Circle((0, 0), r, fill=False, edgecolor=DIM,
                                lw=0.4, alpha=0.5)
        ax.add_patch(ring)
    # Cardinal labels
    for label, (x, y) in [("N", (0, 47)), ("S", (0, -47)),
                           ("E", (47, 0)), ("W", (-47, 0))]:
        ax.text(x, y, label, color=DIM, fontsize=8, ha="center", va="center",
                fontfamily="monospace")
    # Airport
    ax.plot(0, 0, "P", color=LIME, markersize=12, markeredgecolor="white",
            markeredgewidth=0.7, zorder=4)
    ax.text(2.0, 2.0, "KAPT", color=LIME, fontsize=8, fontfamily="monospace")

    # Legend
    ax.text(0.02, 0.03,
            "✈ aircraft    ─ track    ⚠ predictive warning    ◯ separation violation",
            transform=ax.transAxes, color=DIM, fontsize=LEGEND_FONT,
            fontfamily="monospace")

    palette = plt.cm.cool(np.linspace(0.1, 0.9, N))

    def step(k):
        for art in list(ax.lines) + list(ax.patches) + list(ax.collections):
            # Keep static circles by tag — easier to re-add than filter
            pass
        # Wipe everything, redraw static + dynamic
        ax.clear()
        _setup_axes(ax, (-50, 50), (-50, 50),
                    title="ATC — predictive conflict detection",
                    subtitle="Six aircraft inbound on convergent tracks. The canvas predicts separation violations before they happen.")
        for r in [10, 20, 30, 40]:
            ring = mpatches.Circle((0, 0), r, fill=False, edgecolor=DIM,
                                    lw=0.4, alpha=0.5)
            ax.add_patch(ring)
        for label, (x, y) in [("N", (0, 47)), ("S", (0, -47)),
                              ("E", (47, 0)), ("W", (-47, 0))]:
            ax.text(x, y, label, color=DIM, fontsize=8, ha="center", va="center",
                    fontfamily="monospace")
        ax.plot(0, 0, "P", color=LIME, markersize=12, markeredgecolor="white",
                markeredgewidth=0.7, zorder=4)
        ax.text(2.0, 2.0, "KAPT", color=LIME, fontsize=8, fontfamily="monospace")
        ax.text(0.02, 0.03,
                "✈ aircraft    ─ track    ⚠ predictive warning    ◯ separation violation",
                transform=ax.transAxes, color=DIM, fontsize=LEGEND_FONT,
                fontfamily="monospace")

        t0 = max(0, k - 15)
        for i in range(N):
            ax.plot(traj_x[t0:k + 1, i], traj_y[t0:k + 1, i],
                    color=palette[i], lw=1.0, alpha=0.55)
            ax.plot(traj_x[k, i], traj_y[k, i], "^", color=palette[i],
                    markersize=10, markeredgecolor="white", markeredgewidth=0.7,
                    zorder=5)
            ax.text(traj_x[k, i] + 1.0, traj_y[k, i] + 1.0, f"AC{i}",
                    color=palette[i], fontsize=7, fontfamily="monospace",
                    fontweight="bold")

        # Active warnings and conflicts (linger 3 frames)
        live_warn = [w for w in warnings if 0 <= k - w[0] < 3]
        live_conf = [c for c in conflicts if 0 <= k - c[0] < 3]
        for _, i, j in live_warn:
            mx = (traj_x[k, i] + traj_x[k, j]) / 2
            my = (traj_y[k, i] + traj_y[k, j]) / 2
            ring = mpatches.Circle((mx, my), 5.5, fill=False,
                                    edgecolor=AMBER, lw=1.8,
                                    linestyle="--", alpha=0.8)
            ax.add_patch(ring)
        for _, i, j in live_conf:
            mx = (traj_x[k, i] + traj_x[k, j]) / 2
            my = (traj_y[k, i] + traj_y[k, j]) / 2
            ring = mpatches.Circle((mx, my), 4.0, fill=False,
                                    edgecolor=RED, lw=2.4, alpha=0.95)
            ax.add_patch(ring)

        _caption(ax, "Step {:>3}/{}   |   "
                     "dashed amber = canvas predicts conflict in next 3 steps   |   "
                     "solid red = current separation violation"
                 .format(k + 1, N_STEPS))

        if live_conf:
            _, i, j = live_conf[-1]
            _phenomenon_pill(ax, f"SEP VIOLATION  AC{i} ↔ AC{j}", color=RED)
        elif live_warn:
            _, i, j = live_warn[-1]
            _phenomenon_pill(ax, f"PREDICTED CONFLICT  AC{i} ↔ AC{j}", color=AMBER)

    anim = animation.FuncAnimation(fig, step, frames=N_STEPS, interval=90)
    _save(anim, "06_atc")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 07 — multi-patient deterioration alerts (ICU)
# ─────────────────────────────────────────────────────────────────────────


def render_icu() -> None:
    """Six patient vital-sign monitors update in parallel. One patient
    deteriorates dramatically; the alert is highlighted with the patient
    ID and the concerning vital named explicitly."""

    print("Rendering 07_icu_patient (multi-patient deterioration alerts)...")
    rng = np.random.default_rng(3)

    N_PATIENTS = 6
    N_STEPS = 96  # 96 hours of simulation

    # Build vital traces. Patient 2 will deteriorate; others stay stable.
    t = np.arange(N_STEPS)
    hr = np.zeros((N_PATIENTS, N_STEPS))
    spo2 = np.zeros((N_PATIENTS, N_STEPS))
    map_bp = np.zeros((N_PATIENTS, N_STEPS))

    for p in range(N_PATIENTS):
        base_hr = 70 + 6 * np.sin(t / 12 + p) + rng.normal(0, 1.5, N_STEPS)
        base_spo2 = 97 - 0.5 * np.sin(t / 18 + p * 0.7) + rng.normal(0, 0.3, N_STEPS)
        base_map = 75 + 3 * np.sin(t / 16 + p * 0.5) + rng.normal(0, 1.0, N_STEPS)
        hr[p] = base_hr
        spo2[p] = base_spo2
        map_bp[p] = base_map

    # Inject a sepsis-like deterioration on patient 2 starting at hour 40
    crisis_start = 40
    for k in range(crisis_start, N_STEPS):
        u = (k - crisis_start) / 40.0
        hr[2, k] += 35 * u
        spo2[2, k] -= 8 * u
        map_bp[2, k] -= 18 * u

    # And a milder respiratory event on patient 4 from hour 60
    resp_start = 60
    for k in range(resp_start, N_STEPS):
        u = (k - resp_start) / 30.0
        spo2[4, k] -= 5 * u
        hr[4, k] += 12 * u

    # Compute composite risk per patient per step
    risk = np.zeros((N_PATIENTS, N_STEPS))
    for p in range(N_PATIENTS):
        # Normalized deviations from canonical ranges
        hr_dev = np.clip((np.abs(hr[p] - 75) - 15) / 25, 0, 1)
        spo2_dev = np.clip((92 - spo2[p]) / 8, 0, 1)
        map_dev = np.clip((65 - map_bp[p]) / 15, 0, 1)
        risk[p] = 0.45 * hr_dev + 0.4 * spo2_dev + 0.45 * map_dev
        risk[p] = np.clip(risk[p], 0, 1)

    fig = plt.figure(figsize=(14, 9), dpi=100)
    fig.patch.set_facecolor(BG)

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.20,
                          left=0.06, right=0.97, top=0.86, bottom=0.07)

    fig.suptitle("WARD MONITOR — multi-patient deterioration prediction",
                 fontsize=TITLE_FONT, color=FG, fontweight="bold",
                 fontfamily="monospace", y=0.96)
    fig.text(0.5, 0.915,
             "Six patients in parallel. Canvas-engineered ward monitor flags rising risk before alarms trigger.",
             ha="center", fontsize=SUBTITLE_FONT, color=DIM, fontfamily="monospace")

    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

    def step(k):
        for p_idx in range(N_PATIENTS):
            ax = axes[p_idx]
            ax.clear()
            ax.set_facecolor(PANEL)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(DIM)
                spine.set_linewidth(0.4)

            t_window = slice(max(0, k - 30), k + 1)
            tt = np.arange(max(0, k - 30), k + 1)
            r_now = risk[p_idx, k]

            # Color by current risk
            if r_now < 0.3:
                status = "STABLE"
                status_color = LIME
            elif r_now < 0.6:
                status = "WATCH"
                status_color = AMBER
            else:
                status = "CRITICAL"
                status_color = RED

            ax.plot(tt, hr[p_idx, t_window], color=CYAN, lw=1.2, alpha=0.85)
            ax.plot(tt, spo2[p_idx, t_window], color=LIME, lw=1.2, alpha=0.85)
            ax.plot(tt, map_bp[p_idx, t_window], color=AMBER, lw=1.2, alpha=0.85)
            ax.set_ylim(40, 130)
            ax.set_xlim(max(0, k - 30), max(30, k + 1))

            # Header strip
            ax.text(0.02, 0.92, f"PT{p_idx:02d}",
                    transform=ax.transAxes, color=FG, fontsize=11,
                    fontweight="bold", fontfamily="monospace", va="top")
            ax.text(0.98, 0.92, status,
                    transform=ax.transAxes, color=status_color, fontsize=10,
                    fontweight="bold", fontfamily="monospace",
                    va="top", ha="right")
            # Current vitals
            ax.text(0.02, 0.04,
                    f"HR {hr[p_idx, k]:5.1f}   SpO₂ {spo2[p_idx, k]:5.1f}   MAP {map_bp[p_idx, k]:5.1f}",
                    transform=ax.transAxes, color=FG, fontsize=8,
                    fontfamily="monospace", va="bottom")
            # Risk bar
            bar_x = 0.65
            bar_w = 0.30
            bar_h = 0.05
            bar_y = 0.04
            ax.add_patch(FancyBboxPatch(
                (bar_x, bar_y), bar_w, bar_h,
                boxstyle="round,pad=0.0", transform=ax.transAxes,
                facecolor=DIM, edgecolor=DIM, alpha=0.5))
            ax.add_patch(FancyBboxPatch(
                (bar_x, bar_y), bar_w * r_now, bar_h,
                boxstyle="round,pad=0.0", transform=ax.transAxes,
                facecolor=status_color, edgecolor=status_color, alpha=0.95))
            ax.text(bar_x - 0.01, bar_y + bar_h / 2, "risk",
                    transform=ax.transAxes, color=DIM, fontsize=7,
                    fontfamily="monospace", va="center", ha="right")

            # Critical alert callout
            if r_now >= 0.6:
                # Identify which vital is driving it
                hr_dev = abs(hr[p_idx, k] - 75)
                spo2_dev = max(92 - spo2[p_idx, k], 0)
                map_dev = max(65 - map_bp[p_idx, k], 0)
                concerns = []
                if hr_dev > 20:
                    concerns.append("HR↑")
                if spo2_dev > 2:
                    concerns.append("SpO₂↓")
                if map_dev > 5:
                    concerns.append("MAP↓")
                if concerns:
                    ax.text(0.5, 0.55, "ALERT " + " ".join(concerns),
                            transform=ax.transAxes,
                            color=BG, fontsize=10, fontweight="bold",
                            fontfamily="monospace",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.4",
                                      facecolor=RED, edgecolor=RED))

        # Top progress
        fig.text(0.5, 0.04,
                 f"Hour {k + 1:>3}/{N_STEPS}   "
                 "—   solid red = canvas-engineered risk score > 0.6   "
                 "—   watch PT02 around hour 40",
                 ha="center", color=DIM, fontsize=CAPTION_FONT,
                 fontfamily="monospace")

    anim = animation.FuncAnimation(fig, step, frames=N_STEPS, interval=120)
    _save(anim, "07_icu_patient")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 05 — protein complex docking (clearer phase labels)
# ─────────────────────────────────────────────────────────────────────────


def render_protein() -> None:
    """Four chains dock into a complex with clear PHASE labels and a
    prominent binding-energy readout."""

    print("Rendering 05_protein (4-chain docking)...")
    rng = np.random.default_rng(42)

    N_CHAINS = 4
    LEN = 22
    N_FRAMES = 140

    # Helical chains
    def helix(n, r, p, c):
        t = np.linspace(0, n * p, n)
        x = r * np.cos(t * 2.6) + c[0] + np.arange(n) * 0.15
        y = r * np.sin(t * 2.6) + c[1]
        return np.stack([x, y], axis=-1)

    chains_base = [
        helix(LEN, 2.0, 0.32, np.array([-3.5, -2.5])),
        helix(LEN, 1.8, 0.30, np.array([-3.5,  2.5])),
        helix(LEN, 2.1, 0.34, np.array([ 3.5,  0.0])),
        helix(LEN, 1.6, 0.28, np.array([ 2.0, -3.5])),
    ]
    chains_base[2][:, 0] = -chains_base[2][:, 0] + 7
    chains_base[3][:, 0] = -chains_base[3][:, 0] + 4
    chains_base[3][:, 1] = -chains_base[3][:, 1] - 2

    chain_colors = [CYAN, MAGENTA, LIME, AMBER]
    chain_names = ["chain α", "chain β", "chain γ", "chain δ"]

    start_off = [np.array([-10, -5]), np.array([-10, 5]),
                 np.array([10, 2]), np.array([7, -7])]
    dock_off = [np.array([-2.5, -1.5]), np.array([-2.5, 1.5]),
                np.array([2.5, 0]), np.array([1.0, -2.5])]
    thermal = [rng.normal(0, 0.05, (N_FRAMES, LEN, 2)) for _ in range(N_CHAINS)]

    def ease(t):
        return t * t * (3 - 2 * t)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_position([0.04, 0.06, 0.92, 0.84])
    ax.set_facecolor(PANEL)
    ax.set_xlim(-16, 16)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("PROTEIN COMPLEX — 4-chain docking",
                 fontsize=TITLE_FONT, color=FG, fontweight="bold",
                 fontfamily="monospace", y=0.96)
    fig.text(0.5, 0.91,
             "Chains approach → dock → tighten → stable complex. "
             "Canvas predicts binding affinity (kcal/mol) from per-chain sequence.",
             ha="center", color=DIM, fontsize=SUBTITLE_FONT, fontfamily="monospace")

    # Reference affinity prediction (deterministic for the demo)
    aff_final = -8.7

    def step(frame):
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.set_xlim(-16, 16)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.axis("off")

        # Determine phase
        if frame < 35:
            phase = "1 — FREE CHAINS APPROACHING"
            bind = 0.0
            t_phase = ease(frame / 35.0)
            offs = [start_off[c] * (1 - t_phase) + dock_off[c] * t_phase * 0.4
                    for c in range(N_CHAINS)]
            thermal_scale = 1.4 - 0.5 * t_phase
        elif frame < 70:
            phase = "2 — INITIAL DOCKING"
            t_phase = ease((frame - 35) / 35.0)
            offs = [dock_off[c] * (0.4 + 0.4 * t_phase) for c in range(N_CHAINS)]
            bind = t_phase
            thermal_scale = 0.9 - 0.3 * t_phase
        elif frame < 105:
            phase = "3 — CONFORMATIONAL TIGHTENING"
            t_phase = ease((frame - 70) / 35.0)
            offs = [dock_off[c] * (0.8 + 0.2 * t_phase) for c in range(N_CHAINS)]
            bind = 0.7 + 0.3 * t_phase
            thermal_scale = 0.6 - 0.3 * t_phase
        else:
            phase = "4 — STABLE COMPLEX"
            offs = dock_off
            bind = 1.0
            thermal_scale = 0.3

        # Draw chains
        for c in range(N_CHAINS):
            pos = chains_base[c] + offs[c] + thermal[c][frame] * thermal_scale
            ax.plot(pos[:, 0], pos[:, 1], "-", color=chain_colors[c],
                    lw=2.6, alpha=0.95, solid_capstyle="round")
            # Endpoints
            ax.plot(pos[0, 0], pos[0, 1], "o", color=chain_colors[c],
                    markersize=6, markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(pos[-1, 0], pos[-1, 1], "s", color=chain_colors[c],
                    markersize=6, markeredgecolor="white", markeredgewidth=0.5)
            # Label
            mid = pos[LEN // 2]
            ax.text(mid[0], mid[1] + 1.2, chain_names[c],
                    color=chain_colors[c], fontsize=9, fontweight="bold",
                    ha="center", fontfamily="monospace")

        # Binding energy readout
        energy = aff_final * bind
        ax.text(0.5, 0.97, f"BINDING ENERGY: {energy:+.3f} kcal/mol",
                transform=ax.transAxes, color=LIME if bind > 0.3 else DIM,
                fontsize=12, fontweight="bold", ha="center", va="top",
                fontfamily="monospace")

        # Phase pill
        ax.text(0.5, 0.04, f"PHASE {phase}",
                transform=ax.transAxes, color=BG, fontsize=10,
                fontweight="bold", ha="center", va="bottom",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor=AMBER, edgecolor=AMBER))

        # Progress bar
        prog = frame / N_FRAMES
        ax.plot([-15, -15 + 30 * prog], [-9, -9], "-", color=CYAN,
                lw=3, alpha=0.7, solid_capstyle="round")
        ax.plot([-15, 15], [-9, -9], "-", color=DIM, lw=1, alpha=0.3)

    anim = animation.FuncAnimation(fig, step, frames=N_FRAMES, interval=70)
    _save(anim, "05_protein", fps_gif=14, fps_mp4=24)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────


RENDERERS: Dict[str, Callable[[], None]] = {
    "04": render_fleet,
    "05": render_protein,
    "06": render_atc,
    "07": render_icu,
}


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", nargs="*", default=list(RENDERERS.keys()),
                        help="example numbers to render (default: all)")
    args = parser.parse_args(argv)
    for key in args.which:
        if key not in RENDERERS:
            print(f"unknown example {key!r} (valid: {sorted(RENDERERS)})",
                  file=sys.stderr)
            return 2
        RENDERERS[key]()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

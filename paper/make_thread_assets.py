"""Generate image assets for the launch thread → paper/thread_assets/.

new renders:
  page1.png          : first page of the paper
  canvas_rotating.gif: rotating 3D (T,H,W) canvas with colored volume allocations
  attention_mask.png : the compiled attention mask for the Fig-1 layout+topology
  math_card.png      : the four core equations on one card
  schema_json.png    : serialized CanvasSchema snippet ("the schema is the ABI")
  results_table.png  : crop of Table 2 (loop x freeze grid) from the paper
plus copies of the paper figures and selected repo example art.
"""

import os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import numpy as np
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thread_assets")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 180})

C_VISUAL, C_ACTION, C_REWARD = "#7db8e8", "#f2a95c", "#8fd18f"

REGIONS = {  # the paper's Fig-1 layout
    "visual": ((0, 5, 0, 6, 0, 6), C_VISUAL),
    "action": ((0, 5, 6, 7, 0, 1), C_ACTION),
    "reward": ((2, 3, 7, 8, 0, 1), C_REWARD),
}
T, H, W = 5, 8, 8


def page1():
    doc = fitz.open(os.path.join(HERE, "canvas-engineering.pdf"))
    doc[0].get_pixmap(dpi=200).save(os.path.join(OUT, "page1.png"))
    print("wrote page1.png")


def results_table():
    doc = fitz.open(os.path.join(HERE, "canvas-engineering.pdf"))
    for page in doc:
        hits = page.search_for("Action loss (lower better)")
        if hits:
            r = hits[0]
            clip = fitz.Rect(r.x0 - 10, r.y0 - 8, r.x1 + 180, r.y1 + 130)
            page.get_pixmap(dpi=260, clip=clip).save(
                os.path.join(OUT, "results_table.png"))
            print("wrote results_table.png")
            return
    print("!! table not found")


def results_chart():
    # The walk-away receipt: at 3 loops, the FROZEN 350K-param model has the
    # lowest action loss — fewer trainable params than either alternative, yet
    # better actions. Single series (action loss); winner highlighted; params
    # annotated; direct value labels; recessive axis.
    labels = ["frozen\n350K params", "half-frozen\n3.7M params",
              "unfrozen\n11.7M params"]
    loss = [0.073, 0.107, 0.088]
    win = 0
    ink, muted = "#2a2f36", "#6b7580"
    bar_ctx, bar_win = "#9aa7b0", "#1f9e7a"

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    xs = np.arange(3)
    for i in range(3):
        ax.bar(xs[i], loss[i], width=0.62,
               color=bar_win if i == win else bar_ctx, zorder=3)
        ax.text(xs[i], loss[i] + 0.0022, f"{loss[i]:.3f}", ha="center",
                va="bottom", fontsize=12.5,
                fontweight="bold" if i == win else "normal",
                color=ink if i == win else muted)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10.5, color=ink)
    ax.set_ylim(0, 0.135)
    ax.set_ylabel("action loss  (lower is better)", fontsize=10.5, color=ink)
    ax.set_yticks([0, 0.05, 0.10])
    ax.tick_params(axis="y", labelcolor=muted, length=0)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cfd4d9")
    ax.grid(axis="y", color="#eef1f3", lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    # headline callout — land the arrow on the winner bar's left shoulder so it
    # never crosses the value label
    ax.annotate("33× fewer trainable params\nthan unfrozen — and the "
                "lowest loss", xy=(win - 0.31, loss[win] - 0.004),
                xytext=(0.55, 0.129), fontsize=10.5, color=bar_win,
                fontweight="bold", va="top", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=bar_win, lw=1.6,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_title("looped attention: recurrence beats scale\n"
                 "(3 loops on a frozen CogVideoX-2B backbone; "
                 "1.73× parameter efficiency, p<0.001)",
                 fontsize=11.5, color=ink, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "results_chart.png"),
                bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print("wrote results_chart.png")


def rotating_gif():
    # Only the allocated regions are filled (as isolated blocks, so no interior
    # faces get culled), and they're TRANSLUCENT — with alpha, every face of a
    # block renders and you see the back faces through the front ones. A
    # wireframe box marks the full (T,W,H) canvas extent.
    filled = np.zeros((T, W, H), dtype=bool)
    colors = np.zeros((T, W, H, 4))
    for name, ((t0, t1, h0, h1, w0, w1), c) in REGIONS.items():
        filled[t0:t1, w0:w1, H - h1:H - h0] = True
        colors[t0:t1, w0:w1, H - h1:H - h0] = matplotlib.colors.to_rgba(c, 0.42)

    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled, facecolors=colors,
              edgecolors=(0.15, 0.15, 0.15, 0.5), linewidth=0.4, shade=False)

    # wireframe of the full (T, W, H) canvas extent so the empty space reads
    corners = [(0, 0, 0), (T, 0, 0), (T, W, 0), (0, W, 0),
               (0, 0, H), (T, 0, H), (T, W, H), (0, W, H)]
    box_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in box_edges:
        xs, ys, zs = zip(corners[a], corners[b])
        ax.plot(xs, ys, zs, color="#aaaaaa", lw=0.8, zorder=0)

    ax.set_box_aspect((T, W, H))
    ax.set_xlabel("T (time)", fontsize=9, labelpad=2)
    ax.set_ylabel("W", fontsize=9, labelpad=2)
    ax.set_zlabel("H", fontsize=9, labelpad=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("the canvas: a typed (T, H, W) latent volume", fontsize=10.5)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=0.6)
               for c in (C_VISUAL, C_ACTION, C_REWARD)]
    ax.legend(handles, ["visual (180 pos)", "action (5 pos)", "reward (1 pos)"],
              loc="upper left", bbox_to_anchor=(-0.08, 0.98), fontsize=7.5,
              frameon=False)
    fig.tight_layout()

    def spin(i):
        ax.view_init(elev=18 + 6 * np.sin(2 * np.pi * i / 72), azim=i * 5)
        return ()

    anim = animation.FuncAnimation(fig, spin, frames=72, blit=False)
    anim.save(os.path.join(OUT, "canvas_rotating.gif"),
              writer=animation.PillowWriter(fps=14))
    plt.close(fig)
    print("wrote canvas_rotating.gif")


def attention_mask():
    # The REAL compiled mask, at true proportions. visual=180, action=5,
    # reward=1 positions, so the blocks are sized 180/5/1 — visual dominates
    # (bandwidth-proportional allocation), and the mask is asymmetric
    # (action queries visual, but visual does NOT query action).
    from canvas_engineering import CanvasLayout, CanvasTopology, Connection
    layout = CanvasLayout(
        T=T, H=H, W=W, d_model=32,
        regions={k: v[0] for k, v in REGIONS.items()}, t_current=2)
    topo = CanvasTopology(connections=[
        Connection(src="visual", dst="visual"),
        Connection(src="action", dst="visual"),
        Connection(src="action", dst="action"),
        Connection(src="reward", dst="visual"),
        Connection(src="reward", dst="action"),
        Connection(src="reward", dst="reward"),
    ])
    M = np.asarray(topo.to_attention_mask(layout), dtype=float)

    # keep only allocated positions, grouped by region (drop the 134 empty)
    order, bands, cursor = [], [], 0
    for name, col in (("visual", C_VISUAL), ("action", C_ACTION),
                      ("reward", C_REWARD)):
        idx = sorted(layout.region_indices(name))
        order += idx
        bands.append((name, col, cursor, cursor + len(idx)))
        cursor += len(idx)
    order = np.array(order)
    Mv = M[np.ix_(order, order)]
    n = len(order)

    names = [b[0] for b in bands]
    cols = [b[1] for b in bands]
    sizes = [b[3] - b[2] for b in bands]
    # edge = True where region i (src/query) attends region j (dst/key)
    edge = np.zeros((3, 3), bool)
    for i, ni in enumerate(names):
        ii = sorted(layout.region_indices(ni))
        for j, nj in enumerate(names):
            jj = sorted(layout.region_indices(nj))
            edge[i, j] = bool((M[np.ix_(ii, jj)] > 0).any())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 5.4),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- LEFT: the structure (who attends whom), equal cells for legibility
    for i in range(3):
        for j in range(3):
            on = edge[i, j]
            axL.add_patch(Rectangle((j, i), 1, 1,
                                    facecolor="#222222" if on else "#f0f0f0",
                                    edgecolor="#b03030", lw=1.0))
            if on:
                axL.text(j + 0.5, i + 0.5, f"{sizes[i]}×{sizes[j]}",
                         ha="center", va="center", fontsize=9,
                         color="white", family="monospace")
    for k, (name, col, sz) in enumerate(zip(names, cols, sizes)):
        axL.add_patch(Rectangle((k, -0.44), 1, 0.44, facecolor=col,
                                edgecolor="black", lw=0.5, clip_on=False))
        axL.add_patch(Rectangle((-0.44, k), 0.44, 1, facecolor=col,
                                edgecolor="black", lw=0.5, clip_on=False))
        axL.text(k + 0.5, -0.22, name, ha="center", va="center", fontsize=8,
                 family="monospace")
        axL.text(-0.22, k + 0.5, name, ha="center", va="center", fontsize=8,
                 family="monospace", rotation=90)
    axL.set_xlim(-0.44, 3); axL.set_ylim(3, -1.15)
    axL.set_aspect("equal"); axL.axis("off")
    axL.text(1.28, -0.92, "keys (dst) →", ha="center", fontsize=8.5)
    axL.text(-0.92, 1.5, "← queries (src)", va="center", rotation=90,
             fontsize=8.5)
    axL.set_title("the structure: which regions may attend\n"
                  "dark = edge; each cell is a full block of the labeled shape.\n"
                  "asymmetric — action→visual on, visual→action off",
                  fontsize=8.6, pad=6)

    # ---- RIGHT: the real mask, broken-axis so it stays linear per segment
    # and explicitly admits the skipped visual positions.
    Kv = 8                                   # visual positions kept each end
    vis = sorted(layout.region_indices("visual"))
    act = sorted(layout.region_indices("action"))
    rew = sorted(layout.region_indices("reward"))
    disp = vis[:Kv] + vis[-Kv:] + act + rew  # 8+8+5+1 = 22 shown
    Md = M[np.ix_(np.array(disp), np.array(disp))]
    m = len(disp)
    skip = len(vis) - 2 * Kv                  # 164 visual positions cut
    xb = Kv                                    # break location (both axes)
    vis_end, act_end = 2 * Kv, 2 * Kv + len(act)  # 16, 21

    axR.imshow(1 - Md, cmap="gray", vmin=0, vmax=1, interpolation="nearest",
               extent=[0, m, m, 0])
    # region dividers
    for d in (vis_end, act_end):
        axR.axhline(d, color="#b03030", lw=1.0)
        axR.axvline(d, color="#b03030", lw=1.0)

    # the cut-through: a white gap + diagonal break marks on both axes
    def break_marks(along_x):
        for edgepos in (0, m):
            if along_x:
                axR.plot([xb - 0.45, xb + 0.45],
                         [edgepos + 0.5, edgepos - 0.5],
                         color="#b03030", lw=1.3, clip_on=False, zorder=7)
            else:
                axR.plot([edgepos - 0.5, edgepos + 0.5],
                         [xb - 0.45, xb + 0.45],
                         color="#b03030", lw=1.3, clip_on=False, zorder=7)
    axR.plot([xb, xb], [0, m], color="white", lw=3.2, zorder=6)
    axR.plot([0, m], [xb, xb], color="white", lw=3.2, zorder=6)
    break_marks(along_x=True)
    break_marks(along_x=False)
    axR.text(xb + 0.35, m * 0.52, f"⁄⁄  {skip} visual positions skipped",
             rotation=90, va="center", ha="left", fontsize=7.5,
             color="#b03030",
             bbox=dict(fc="white", ec="none", pad=0.5))

    # colored region bars (display widths; per-cell scale is constant → linear)
    bar = m * 0.05
    for (name, col, sz, a, b) in [("visual", C_VISUAL, len(vis), 0, vis_end),
                                  ("action", C_ACTION, len(act), vis_end, act_end),
                                  ("reward", C_REWARD, len(rew), act_end, m)]:
        axR.add_patch(Rectangle((a, m), b - a, bar, facecolor=col,
                                edgecolor="black", lw=0.5, clip_on=False))
        axR.add_patch(Rectangle((-bar, a), bar, b - a, facecolor=col,
                                edgecolor="black", lw=0.5, clip_on=False))
        note = f"{name}·{sz}" + (f" ({skip} skipped)" if name == "visual" else "")
        axR.text((a + b) / 2, m + bar * 1.7, note, ha="center", va="bottom",
                 fontsize=7.5 if name == "visual" else 6.8, family="monospace",
                 rotation=0 if name == "visual" else 40)
    axR.set_xlim(-bar, m); axR.set_ylim(m + bar, -0.5)
    axR.set_xticks([]); axR.set_yticks([])
    axR.set_title("the real mask, broken-axis: linear within each segment,\n"
                  "the cut admits the skipped rows. 186 allocated of 320 "
                  "positions —\nvisual (180) dwarfs action (5) and reward (1)",
                  fontsize=8.6, pad=6)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(OUT, "attention_mask.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote attention_mask.png")


def math_card():
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axis("off")
    rows = [
        ("regions are index sets (struct-offset arithmetic)",
         r"$I_r=\{\,tHW{+}hW{+}w\;:\;t_0{\leq}t{<}t_1,\ h_0{\leq}h{<}h_1,"
         r"\ w_0{\leq}w{<}w_1\,\}$"),
        ("the topology compiles to a mask",
         r"$M_{ij}=\max_k\ w_k\,\mathbf{1}[i\in I_{r_k}]\,"
         r"\mathbf{1}[j\in I_{s_k}]\,A_{\tau_k}(t(i),t(j))$"),
        ("loss participation is a weight vector",
         r"$\mathcal{L}=\mathbb{E}_{x_0,\varepsilon,\sigma}\left[\frac"
         r"{\sum_i\omega_i\,\|\hat{\varepsilon}_\theta(x_\sigma)_i-"
         r"\varepsilon_i\|^2}{\sum_i\omega_i}\right]$"),
        ("reachability is the causal statement",
         r"$a\ \mathrm{influences}\ b\ \Leftrightarrow\ G\ \mathrm{has\ a\ "
         r"directed\ path}\ a\rightarrow b$  (else independence is exact)"),
    ]
    y = 0.97
    for title, eq in rows:
        ax.text(0.02, y, title, fontsize=10.5, fontweight="bold",
                va="top", color="#333333")
        ax.text(0.06, y - 0.085, eq, fontsize=12.5, va="top")
        y -= 0.245
    ax.text(0.02, 0.01, "four pieces of arithmetic carry the whole construction",
            fontsize=9, color="#777777", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "math_card.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote math_card.png")


def schema_json():
    lines = [
        ('// robot_v1.json — the complete type signature', "#7a7a7a"),
        ('{', "#222222"),
        ('  "layout": {', "#222222"),
        ('    "T": 5, "H": 8, "W": 8, "d_model": 256,', "#222222"),
        ('    "regions": {', "#222222"),
        ('      "visual": {"bounds": [0,5, 0,6, 0,6],', "#1a5fa8"),
        ('                 "semantic_type": "RGB video 224x224"},', "#1a5fa8"),
        ('      "action": {"bounds": [0,5, 6,7, 0,1],', "#b06010"),
        ('                 "loss_weight": 2.0,', "#b06010"),
        ('                 "semantic_type": "6-DOF EE + gripper"},', "#b06010"),
        ('      "reward": {"bounds": [2,3, 7,8, 0,1]}', "#1e7a1e"),
        ('    }', "#222222"),
        ('  },', "#222222"),
        ('  "topology": [["visual","visual"], ["action","visual"],', "#222222"),
        ('               ["action","action"], ["reward","visual"], ...],', "#222222"),
        ('  "metadata": {"model": "CogVideoX-2B", "data": "bridge_v2"}', "#222222"),
        ('}', "#222222"),
        ('', "#222222"),
        ('// two models sharing this file can exchange latent state', "#7a7a7a"),
        ('// directly. no tokenization. no re-encoding.', "#7a7a7a"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(len(lines) + 0.5, -1.2)
    ax.add_patch(Rectangle((0, -0.9), 1, len(lines) + 1.1,
                           facecolor="#f7f7f7", edgecolor="#cccccc", lw=0.8))
    for i, (txt, col) in enumerate(lines):
        ax.text(0.025, i, txt, fontsize=10.5, family="monospace",
                va="center", color=col)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "schema_json.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote schema_json.png")


def code_to_canvas():
    """Declarations on the left; each region line points to its block, and each
    Connection(...) line points to its arrow overlaid on the canvas."""
    from matplotlib.patches import ConnectionPatch, FancyArrowPatch

    C_AV, C_RV, C_RA = "#8e44ad", "#c0392b", "#8a6d00"
    CODE = [
        ("layout = CanvasLayout(", "#222222", None),
        ("    T=5, H=8, W=8, d_model=256,", "#222222", None),
        ("    regions={", "#222222", None),
        ('        "visual": (0,5, 0,6, 0,6),', "#1a5fa8", "visual"),
        ('        "action": (0,5, 6,7, 0,1),', "#b06010", "action"),
        ('        "reward": (2,3, 7,8, 0,1),', "#1e7a1e", "reward"),
        ("    },", "#222222", None),
        ("    t_current=2,", "#222222", None),
        (")", "#222222", None),
        ("", "#222222", None),
        ("topology = CanvasTopology(connections=[", "#222222", None),
        ('    Connection(src="visual", dst="visual"),', "#777777", "loop_v"),
        ('    Connection(src="action", dst="visual"),', C_AV, "conn_av"),
        ('    Connection(src="action", dst="action"),', "#777777", "loop_a"),
        ('    Connection(src="reward", dst="visual"),', C_RV, "conn_rv"),
        ('    Connection(src="reward", dst="action"),', C_RA, "conn_ra"),
        ("])", "#222222", None),
    ]
    n = len(CODE)

    fig = plt.figure(figsize=(9.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.14,
                          left=0.02, right=0.98, top=0.90, bottom=0.05)
    axC = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1])

    # ---- code panel ----
    axC.set_xlim(0, 1); axC.set_ylim(n + 0.5, -1.0); axC.axis("off")
    axC.add_patch(Rectangle((0, -0.7), 1, n + 0.9, facecolor="#f7f7f7",
                            edgecolor="#cccccc", lw=0.8))
    tags = {}
    for i, (txt, col, tag) in enumerate(CODE):
        axC.text(0.03, i, txt, fontsize=8.2, family="monospace",
                 va="center", color=col,
                 fontweight="bold" if tag and tag.startswith(("conn", "visual",
                                                              "action", "reward"))
                 else "normal")
        if tag:
            tags[tag] = i
    axC.set_title("you declare it", fontsize=11)

    # ---- canvas panel (slice t=2) ----
    ax.set_xlim(-0.4, 8.4); ax.set_ylim(8.6, -0.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("the compiler wires it  (slice $t=2$)", fontsize=11)
    for h in range(8):
        for w in range(8):
            ax.add_patch(Rectangle((w, h), 1, 1, facecolor="#f2f2f2",
                                   edgecolor="#dddddd", lw=0.4))
    blocks = {"visual": ((0, 6, 0, 6), C_VISUAL),
              "action": ((6, 7, 0, 1), C_ACTION),
              "reward": ((7, 8, 0, 1), C_REWARD)}
    for name, ((h0, h1, w0, w1), c) in blocks.items():
        ax.add_patch(Rectangle((w0, h0), w1 - w0, h1 - h0, facecolor=c,
                               edgecolor="black", lw=1.1, zorder=3))
    ax.text(3, 3, "visual", ha="center", va="center", fontsize=10,
            family="monospace", zorder=4)

    # information-flow arrows for the declared connections
    def arrow(p, q, color, rad=0.0, lw=1.8):
        a = FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}",
                            arrowstyle="-|>", mutation_scale=13,
                            color=color, lw=lw, zorder=6)
        ax.add_patch(a)
        return a

    # targets descend with the code lines so pointer lines never cross
    mid = {}
    arrow((1.6, 5.5), (0.78, 6.26), C_AV, rad=-0.12)   # visual -> action
    mid["conn_av"] = (1.28, 5.82)
    arrow((2.6, 5.7), (1.02, 7.42), C_RV, rad=0.20)    # visual -> reward
    mid["conn_rv"] = (2.05, 6.9)
    arrow((0.5, 7.02), (0.5, 7.34), C_RA, lw=1.6)      # action -> reward
    mid["conn_ra"] = (0.58, 7.22)
    # self-attention loops
    for tag, (x, y) in [("loop_v", (0.55, 2.6)), ("loop_a", (1.45, 6.5))]:
        th = np.linspace(0.3 * np.pi, 2.1 * np.pi, 40)
        ax.plot(x + 0.30 * np.cos(th), y + 0.28 * np.sin(th),
                lw=1.2, color="#777777", zorder=6)
        mid[tag] = (x - 0.30, y)
    ax.text(6.9, 5.0, "arrows =\ninformation\nflow", fontsize=7.5,
            color="#555555", ha="center")

    # ---- pointer lines: code line -> its object on the canvas ----
    def pointer(line_i, target, color, ls, rad=0.0, lw=1.0):
        fig.add_artist(ConnectionPatch(
            xyA=(1.0, line_i), coordsA=axC.transData,
            xyB=target, coordsB=ax.transData,
            arrowstyle="-|>", mutation_scale=9, lw=lw,
            linestyle=ls, color=color, alpha=0.85, zorder=9,
            connectionstyle=f"arc3,rad={rad}"))

    # region pointers bow outward (down-left) so they read as one layer;
    # connection pointers run straight as a second layer
    pointer(tags["visual"], (0.10, 1.6), "#1a5fa8", (0, (5, 2)), rad=-0.10, lw=1.3)
    pointer(tags["action"], (-0.05, 6.45), "#b06010", (0, (5, 2)), rad=0.30, lw=1.3)
    pointer(tags["reward"], (-0.05, 7.80), "#1e7a1e", (0, (5, 2)), rad=0.32, lw=1.3)
    for tag in ("conn_av", "conn_rv", "conn_ra", "loop_v", "loop_a"):
        col = dict(conn_av=C_AV, conn_rv=C_RV, conn_ra=C_RA,
                   loop_v="#999999", loop_a="#999999")[tag]
        pointer(tags[tag], mid[tag], col, (0, (1.5, 1.8)))

    fig.savefig(os.path.join(OUT, "code_to_canvas.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote code_to_canvas.png")


# ---------------------------------------------------------------- stagger canvas
# A reusable renderer: canvas slices staggered over time (oblique/orthographic),
# region blocks drawn per frame, with within-frame and cross-frame (temporal)
# information-flow edges. One edge is highlighted and annotated with the causal
# reason it was declared. Same connectivity, attention drawn to a different edge
# per image.
from matplotlib.patches import FancyArrowPatch  # noqa: E402

SX, SY = None, None  # set per-render


def _center(reg, f, sx, sy):
    c0, c1, r0, r1 = reg["c0"], reg["c1"], reg["r0"], reg["r1"]
    return ((c0 + c1) / 2 + f * sx, (r0 + r1) / 2 + f * sy)


def render_stagger(regions, edges, highlight, annotation, title, outfile,
                   Wc, Hc, frames=3, figsize=(9.0, 4.8)):
    sx, sy = Wc + 2.4, -1.15
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal"); ax.axis("off")

    # frame extents + faint slice backdrops, back (f=0) to front
    for f in range(frames):
        x0, y0 = f * sx, f * sy
        ax.add_patch(Rectangle((x0 - 0.4, y0 - 0.4), Wc + 0.8, Hc + 0.8,
                               facecolor="#fbfbfb", edgecolor="#d8d8d8",
                               lw=0.8, zorder=1 + f * 3))
        tlab = ["$t{-}1$", "$t$", "$t{+}1$"][f] if frames == 3 else f"$t{{+}}{f}$"
        ax.text(x0 + Wc / 2, y0 + Hc + 0.9, tlab, ha="center", va="top",
                fontsize=13, color="#555555")

    hl = next((e for e in edges if e["id"] == highlight), None)

    def draw_edge(e, f_from, f_to, strong):
        p = _center(regions[e["src"]], f_from, sx, sy)
        q = _center(regions[e["dst"]], f_to, sx, sy)
        col = e.get("color", "#cf3a3a") if strong else "#bcbcbc"
        lw = 2.6 if strong else 1.0
        rad = e.get("rad", 0.12)
        z = 40 if strong else 8
        ax.add_patch(FancyArrowPatch(
            p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=15 if strong else 10, color=col, lw=lw,
            shrinkA=13, shrinkB=13, zorder=z,
            alpha=1.0 if strong else 0.7))
        return p, q

    # draw non-highlight edges first (context), then blocks, then highlight
    def emit(edge_set, strong):
        for e in edge_set:
            if e["kind"] == "within":
                for f in range(frames):
                    draw_edge(e, f, f, strong)
            else:  # temporal: f -> f+1
                for f in range(frames - 1):
                    draw_edge(e, f, f + 1, strong)

    emit([e for e in edges if e is not hl], strong=False)

    # region blocks per frame (labels on the front frame only)
    for f in range(frames):
        for name, reg in regions.items():
            x0, y0 = reg["c0"] + f * sx, reg["r0"] + f * sy
            involved = hl is not None and name in (hl["src"], hl["dst"])
            ax.add_patch(Rectangle(
                (x0, y0), reg["c1"] - reg["c0"], reg["r1"] - reg["r0"],
                facecolor=reg["color"],
                alpha=1.0 if (involved or hl is None) else 0.45,
                edgecolor="black", lw=1.4 if involved else 0.8,
                zorder=20 + f * 3))
            if f == frames - 1:
                cx, cy = _center(reg, f, sx, sy)
                ax.text(cx, cy, reg["label"], ha="center", va="center",
                        fontsize=11, family="monospace", zorder=60, weight="bold",
                        color="white" if reg.get("dark") else "black")

    hmid = None
    if hl is not None:
        if hl["kind"] == "within":
            hmid = draw_edge(hl, frames - 1, frames - 1, strong=True)
            for f in range(frames - 1):
                draw_edge(hl, f, f, strong=True)
        else:
            for f in range(frames - 1):
                p, q = draw_edge(hl, f, f + 1, strong=True)
                if f == 0:
                    hmid = (p, q)

    # annotation box with a leader to the highlighted edge midpoint. wrap the
    # text to a narrow width and use a large font so it stays legible when X
    # tiles four images into a 2x2 grid.
    if annotation and hmid is not None:
        import textwrap
        (px, py), (qx, qy) = hmid
        mx, my = (px + qx) / 2, (py + qy) / 2
        wrapped = "\n".join(textwrap.wrap(annotation, width=42))
        ax.annotate(
            wrapped, xy=(mx, my),
            xytext=(0.5, -0.16), textcoords="axes fraction",
            ha="center", va="top", fontsize=13, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.6", fc="#fff6e6",
                      ec=hl.get("color", "#cf3a3a"), lw=1.6),
            arrowprops=dict(arrowstyle="-|>", lw=1.6,
                            color=hl.get("color", "#cf3a3a"),
                            connectionstyle="arc3,rad=0.15"))

    ax.set_title(title, fontsize=13.5, pad=8, weight="bold")
    ax.autoscale_view()
    ax.margins(0.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile), bbox_inches="tight",
                facecolor="white", dpi=200)
    plt.close(fig)
    print("wrote", outfile)


# ---- schema A: diffusion policy (the two-node canvas) --------------------
def stagger_diffusion_policy():
    regions = {
        "obs":    dict(c0=0, c1=5, r0=0, r1=3, color="#7db8e8", label="observation"),
        "action": dict(c0=1, c1=4, r0=4, r1=6, color="#f2a95c", label="action"),
    }
    edges = [
        dict(id="cond", src="obs", dst="action", kind="within", color="#b03060",
             rad=-0.25),
        dict(id="cont", src="action", dst="action", kind="temporal",
             color="#3a7abf", rad=0.55),
    ]
    render_stagger(
        regions, edges, highlight="cond",
        annotation="diffusion policy is the two-node canvas: the observation "
                   "region conditions the action region — the base case every "
                   "richer topology composes from.",
        title="diffusion policy — observation $\\rightarrow$ action",
        outfile="stagger_diffusion_policy.png", Wc=5, Hc=6)


# ---- schema B: multi-agent (same primitive composed) ---------------------
def stagger_multi_agent():
    regions = {
        "a_obs": dict(c0=0, c1=3, r0=0, r1=2, color="#7db8e8", label="A.obs"),
        "a_act": dict(c0=0, c1=3, r0=4, r1=6, color="#6fa8dc", label="A.act"),
        "task":  dict(c0=4, c1=7, r0=2, r1=4, color="#8e7cc3", label="shared\ntask",
                      dark=True),
        "b_obs": dict(c0=8, c1=11, r0=0, r1=2, color="#93c47d", label="B.obs"),
        "b_act": dict(c0=8, c1=11, r0=4, r1=6, color="#7bbf63", label="B.act"),
    }
    edges = [
        dict(id="a_pol", src="a_obs", dst="a_act", kind="within", rad=-0.3),
        dict(id="b_pol", src="b_obs", dst="b_act", kind="within", rad=0.3),
        dict(id="a_hub", src="a_act", dst="task", kind="within", rad=0.1),
        dict(id="b_hub", src="b_act", dst="task", kind="within", rad=-0.1),
        dict(id="hub_a", src="task", dst="a_obs", kind="within", color="#8e5cc3",
             rad=0.1),
        dict(id="hub_b", src="task", dst="b_obs", kind="within", color="#8e5cc3",
             rad=-0.1),
    ]
    render_stagger(
        regions, edges, highlight="a_hub",
        annotation="two agents = the same primitive composed. they never read "
                   "each other directly — coordination is forced through the "
                   "shared-task region we declared between them.",
        title="multi-agent — coordination only through the shared task",
        outfile="stagger_multi_agent.png", Wc=11, Hc=6, figsize=(10.6, 4.6))


# ---- schema C: mini-ICU (one connectivity, different edges highlighted) --
_ICU_REGIONS = {
    "monitor": dict(c0=0, c1=3, r0=0, r1=2, color="#7db8e8", label="monitor\n(vitals)"),
    "state":   dict(c0=4, c1=8, r0=0, r1=3, color="#5aa0c0", label="patient\nstate",
                    dark=True),
    "risk":    dict(c0=4, c1=8, r0=4, r1=6, color="#d98a8a", label="deterior.\nrisk"),
    "nurse":   dict(c0=0, c1=3, r0=4, r1=6, color="#93c47d", label="nurse\naction"),
}
_ICU_EDGES = [
    dict(id="obs",     src="monitor", dst="state", kind="within", rad=0.12),
    dict(id="persist", src="state",   dst="state", kind="temporal", rad=0.5),
    dict(id="nurse",   src="nurse",   dst="state", kind="temporal", rad=0.28,
         color="#1e7a1e"),
    dict(id="risk",    src="state",   dst="risk",  kind="within", rad=-0.2,
         color="#c0392b"),
    dict(id="prompt",  src="risk",    dst="nurse", kind="within", rad=-0.2),
]


def stagger_icu(highlight, annotation, outfile):
    render_stagger(
        dict(_ICU_REGIONS), [dict(e) for e in _ICU_EDGES],
        highlight=highlight, annotation=annotation,
        title="one hospital-ward connectivity — declared, not learned",
        outfile=outfile, Wc=8, Hc=6, figsize=(9.2, 5.0))


def stagger_icu_all():
    stagger_icu(
        "nurse",
        "we enable nurse$\\rightarrow$patient across frames because a nurse's "
        "actions materially change the patient's physiology on the next step. "
        "the edge encodes a known causal effect.",
        "stagger_icu_nurse.png")
    stagger_icu(
        "risk",
        "deterioration risk may only read from physiological state — so the "
        "model is forced to route through the real pathway instead of latching "
        "onto a shortcut feature.",
        "stagger_icu_risk.png")
    stagger_icu(
        "persist",
        "physiology is continuous, so patient state attends to its own previous "
        "frame: temporal self-attention wires in persistence rather than hoping "
        "the model discovers it.",
        "stagger_icu_persist.png")


def real_vs_neural():
    """Same causal structure, two depictions: the physical scene (robots
    observing and acting in a field) and the neural canvas the compiler builds
    from the declared schema, with matching obs->act and coordination edges."""
    from matplotlib.patches import FancyBboxPatch, Wedge, FancyArrowPatch
    R = ["#3a7abf", "#4f9d4f", "#d98a2b", "#8e6fc0"]      # robot colors
    RL = [f"r{i}" for i in range(4)]

    fig = plt.figure(figsize=(12.4, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.5],
                          width_ratios=[1.0, 1.0], hspace=0.12, wspace=0.10,
                          left=0.015, right=0.985, top=0.93, bottom=0.02)
    axW = fig.add_subplot(gs[0, 0])   # real world
    axN = fig.add_subplot(gs[0, 1])   # neural canvas
    axC = fig.add_subplot(gs[1, :])   # code, full width

    # ---------- REAL WORLD ----------
    axW.set_xlim(0, 10); axW.set_ylim(0, 7.4); axW.set_aspect("equal")
    axW.axis("off")
    axW.add_patch(Rectangle((0, 0), 10, 7.4, facecolor="#eef4ea",
                            edgecolor="#c8d6bf", lw=1.0))
    axW.set_title("the real world: what the causal structure IS",
                  fontsize=11, pad=6)
    target = (5.0, 3.9)
    axW.plot(*target, marker="*", ms=26, color="#d4af37",
             markeredgecolor="#8a6d00", mew=0.8, zorder=5)
    axW.text(target[0], target[1] - 0.75, "shared task", ha="center",
             fontsize=8.5, color="#7a5c00")
    robots = [(2.0, 1.8), (8.0, 1.9), (2.2, 5.5), (7.8, 5.4)]
    for i, (rx, ry) in enumerate(robots):
        ang = np.degrees(np.arctan2(target[1] - ry, target[0] - rx))
        # vision cone (observation)
        axW.add_patch(Wedge((rx, ry), 2.4, ang - 26, ang + 26,
                            facecolor=R[i], alpha=0.16, edgecolor="none",
                            zorder=2))
        # action arrow (motion toward the task)
        dx, dy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        axW.add_patch(FancyArrowPatch((rx, ry), (rx + 1.5 * dx, ry + 1.5 * dy),
                                      arrowstyle="-|>", mutation_scale=16,
                                      color=R[i], lw=2.4, zorder=4))
        # robot body
        axW.add_patch(FancyBboxPatch((rx - 0.42, ry - 0.32), 0.84, 0.64,
                                     boxstyle="round,pad=0.02,rounding_size=0.12",
                                     facecolor=R[i], edgecolor="black", lw=1.0,
                                     zorder=6))
        axW.text(rx, ry, RL[i], ha="center", va="center", color="white",
                 fontsize=8.5, fontweight="bold", family="monospace", zorder=7)
    # coordination ring (robots share the task, not each other directly)
    for a, b in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        axW.add_patch(FancyArrowPatch(robots[a], robots[b], arrowstyle="-",
                                      color="#9a9a9a", lw=1.0, ls=(0, (4, 3)),
                                      zorder=1))
    # exemplar labels on r0
    axW.annotate("observation\n(vision cone)", xy=(2.9, 3.0), xytext=(0.2, 6.7),
                 fontsize=8, color=R[0], ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.8, color=R[0]))
    axW.annotate("action\n(how it moves)", xy=(2.9, 2.4), xytext=(3.4, 0.35),
                 fontsize=8, color=R[0], ha="center",
                 arrowprops=dict(arrowstyle="-", lw=0.8, color=R[0]))

    # ---------- NEURAL CANVAS ----------
    axN.set_xlim(0, 10); axN.set_ylim(0, 7.4); axN.set_aspect("equal")
    axN.axis("off")
    axN.add_patch(Rectangle((0, 0), 10, 7.4, facecolor="#fbfbfb",
                            edgecolor="#d8d8d8", lw=1.0))
    axN.set_title("the neural canvas: what the compiler builds",
                  fontsize=11, pad=6)
    # dispatch (shared task) in the middle
    disp = (5.0, 3.7)
    axN.add_patch(Rectangle((disp[0] - 0.9, disp[1] - 0.7), 1.8, 1.4,
                            facecolor="#8e7cc3", edgecolor="black", lw=1.0,
                            zorder=6))
    axN.text(*disp, "dispatch", ha="center", va="center", color="white",
             fontsize=7.6, family="monospace", zorder=7)
    # robot region groups, positions mirroring the field
    grp = [(1.9, 1.6), (8.1, 1.6), (1.9, 5.6), (8.1, 5.6)]
    for i, (gx, gy) in enumerate(grp):
        # obs sub-block (light) over act sub-block (dark)
        axN.add_patch(Rectangle((gx - 1.0, gy + 0.05), 2.0, 0.95,
                                facecolor=R[i], alpha=0.35, edgecolor="black",
                                lw=0.8, zorder=5))
        axN.add_patch(Rectangle((gx - 1.0, gy - 1.0), 2.0, 0.95,
                                facecolor=R[i], edgecolor="black", lw=0.8,
                                zorder=5))
        axN.text(gx, gy + 0.52, f"{RL[i]}.obs", ha="center", va="center",
                 fontsize=7, family="monospace", zorder=7)
        axN.text(gx, gy - 0.52, f"{RL[i]}.act", ha="center", va="center",
                 fontsize=7, family="monospace", color="white", zorder=7)
        obs_c = (gx, gy + 0.52); act_c = (gx, gy - 0.52)
        # obs -> act (the policy), within the robot
        axN.add_patch(FancyArrowPatch(obs_c, act_c, arrowstyle="-|>",
                                      mutation_scale=11, color=R[i], lw=1.8,
                                      connectionstyle="arc3,rad=0.55", zorder=8))
        # act -> dispatch (report), dispatch -> obs (coordinate)
        axN.add_patch(FancyArrowPatch(act_c, disp, arrowstyle="-|>",
                                      mutation_scale=9, color="#9a9a9a", lw=1.1,
                                      connectionstyle="arc3,rad=0.1", zorder=3))
        axN.add_patch(FancyArrowPatch(disp, obs_c, arrowstyle="-|>",
                                      mutation_scale=9, color="#8e7cc3", lw=1.1,
                                      connectionstyle="arc3,rad=0.1", zorder=3))
    axN.annotate("obs$\\rightarrow$act\n(same as the\nvision$\\rightarrow$motion\n"
                 "on the left)", xy=(1.9, 0.9), xytext=(4.0, 0.9),
                 fontsize=7.4, va="center", color=R[0],
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color=R[0]))

    # correspondence bridge between the panels
    fig.text(0.5, 0.60, "$\\equiv$", ha="center", va="center", fontsize=22,
             color="#555555")
    fig.text(0.5, 0.55, "same\ncausal\nstructure", ha="center", va="center",
             fontsize=7.5, color="#555555")

    # ---------- CODE ----------
    axC.axis("off"); axC.set_xlim(0, 1); axC.set_ylim(0, 1)
    axC.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f7f7f7",
                            edgecolor="#cccccc", lw=0.8))
    left = [
        ("# declare the geometry", "#7a7a7a"),
        ("layout = CanvasLayout(T=4, H=16, W=16, d_model=256,", "#222"),
        ('    regions={', "#222"),
        ('        "r0.obs": (0,4, 0,6, 0,6),  "r0.act": (0,4, 6,8, 0,4),',
         "#1a5fa8"),
        ('        "r1.obs": (0,4, 0,6, 10,16), "r1.act": (0,4, 6,8, 12,16),',
         "#2e7d32"),
        ('        # ... r2, r3 ...', "#7a7a7a"),
        ('        "dispatch": (0,4, 7,9, 7,9),', "#6a4fa0"),
        ('    })', "#222"),
    ]
    right = [
        ("# declare who may influence whom", "#7a7a7a"),
        ("topology = CanvasTopology(connections=[", "#222"),
        ('  # each robot\'s policy: observation conditions action', "#7a7a7a"),
        ('  Connection(src="r0.act", dst="r0.obs"),   # (r1..r3 alike)', "#1a5fa8"),
        ('  # report up to the shared task, coordinate back down', "#7a7a7a"),
        ('  Connection(src="r0.act",  dst="dispatch"),', "#444"),
        ('  Connection(src="r0.obs",  dst="dispatch"),', "#6a4fa0"),
        ('])', "#222"),
    ]
    # tight, code-like line spacing (was airy); block sits centered in the panel
    dy, y0 = 0.082, 0.80
    for col, lines, x0 in ((0, left, 0.02), (1, right, 0.52)):
        for i, (txt, c) in enumerate(lines):
            axC.text(x0, y0 - i * dy, txt, fontsize=9.2,
                     family="monospace", va="top", color=c)
    axC.axvline(0.505, color="#dddddd", lw=0.8)

    fig.suptitle("you declare the causal structure once; it lives in the world "
                 "and in the canvas the same way", fontsize=12.5, y=0.985)
    fig.savefig(os.path.join(OUT, "real_vs_neural.png"),
                bbox_inches="tight", facecolor="white", dpi=190)
    plt.close(fig)
    print("wrote real_vs_neural.png")


def copies():
    fig = os.path.join(HERE, "figures")
    ass = os.path.join(HERE, "..", "assets")
    exa = os.path.join(ass, "examples")
    for src, dst in [
        (os.path.join(fig, "fig_layout_example.png"), "fig_layout_example.png"),
        (os.path.join(fig, "fig_type_system.png"), "fig_type_system.png"),
        (os.path.join(fig, "fig_icu_allocation.png"), "fig_icu_allocation.png"),
        # transfer_distance parked (needs the representation-stability caveat);
        # kept available for a possible interop follow-up thread.
        (os.path.join(fig, "transfer_distance.png"), "transfer_distance.png"),
        (os.path.join(exa, "09b_bci_tribe.png"), "bci_tribe.png"),
        (os.path.join(exa, "03_cartpole.png"), "cartpole.png"),
    ]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, dst))
        else:
            print("!! missing", src)
    print("copied existing figures")


if __name__ == "__main__":
    page1()
    results_chart()
    rotating_gif()
    attention_mask()
    math_card()
    schema_json()
    code_to_canvas()
    stagger_diffusion_policy()
    stagger_multi_agent()
    stagger_icu_all()
    real_vs_neural()
    copies()
    print("\nassets in", OUT)

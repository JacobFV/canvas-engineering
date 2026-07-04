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


def rotating_gif():
    filled = np.zeros((T, W, H), dtype=bool)
    colors = np.zeros((T, W, H, 4))
    # faint shell for unallocated cells
    filled[:] = True
    colors[..., :] = (0.93, 0.93, 0.93, 0.08)
    for name, ((t0, t1, h0, h1, w0, w1), c) in REGIONS.items():
        rgba = matplotlib.colors.to_rgba(c, alpha=0.95)
        colors[t0:t1, w0:w1, H - h1:H - h0] = rgba
        # note: flip h so "top" of the 2D layout renders up

    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled, facecolors=colors,
              edgecolors=(0.35, 0.35, 0.35, 0.18), linewidth=0.3)
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
    # one cell per (region, frame): visual t0..t4, action t0..t4, reward t2.
    # same topology as the paper example; dense in time.
    cells = ([f"v{t}" for t in range(5)] + [f"a{t}" for t in range(5)] + ["r2"])
    grp = {c: c[0] for c in cells}
    edges = {("v", "v"), ("a", "v"), ("a", "a"),
             ("r", "v"), ("r", "a"), ("r", "r")}
    n = len(cells)
    M = np.zeros((n, n))
    for i, ci in enumerate(cells):
        for j, cj in enumerate(cells):
            if (grp[ci], grp[cj]) in edges:
                M[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(5.9, 5.9))
    ax.imshow(1 - M, cmap="gray", vmin=-0.25, vmax=1, interpolation="nearest")
    for k in (4.5, 9.5):
        ax.axhline(k, color="#b03030", lw=1.4)
        ax.axvline(k, color="#b03030", lw=1.4)
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="#ffffff", lw=0.6, zorder=1)
        ax.axvline(k - 0.5, color="#ffffff", lw=0.6, zorder=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cells, fontsize=9, family="monospace")
    ax.set_yticklabels(cells, fontsize=9, family="monospace")
    ax.set_xlabel("keys (dst)  —  visual | action | reward", fontsize=9.5)
    ax.set_ylabel("reward | action | visual  —  queries (src)", fontsize=9.5)
    ax.set_title("the topology, compiled to an attention mask\n"
                 "(one cell per region-frame; dark = attention allowed)",
                 fontsize=10.5)
    ax.tick_params(length=0)
    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(len(lines) + 0.5, -1.2)
    ax.add_patch(Rectangle((0, -0.9), 1, len(lines) + 1.1,
                           facecolor="#f7f7f7", edgecolor="#cccccc", lw=0.8))
    for i, (txt, col) in enumerate(lines):
        ax.text(0.025, i, txt, fontsize=8.6, family="monospace",
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


def copies():
    fig = os.path.join(HERE, "figures")
    ass = os.path.join(HERE, "..", "assets")
    exa = os.path.join(ass, "examples")
    for src, dst in [
        (os.path.join(fig, "fig_layout_example.png"), "fig_layout_example.png"),
        (os.path.join(fig, "fig_topology.png"), "fig_topology.png"),
        (os.path.join(fig, "fig_type_system.png"), "fig_type_system.png"),
        (os.path.join(fig, "fig_icu_allocation.png"), "fig_icu_allocation.png"),
        (os.path.join(fig, "transfer_distance.png"), "transfer_distance.png"),
        (os.path.join(ass, "looped_attention.png"), "looped_attention.png"),
        (os.path.join(exa, "07_icu_patient.gif"), "icu_ward_monitor.gif"),
        (os.path.join(exa, "04_fleet.gif"), "vehicle_fleet.gif"),
        (os.path.join(exa, "06_atc.png"), "air_traffic.png"),
        (os.path.join(exa, "08_world_model_minecraft.png"), "minecraft_world_model.png"),
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
    results_table()
    rotating_gif()
    attention_mask()
    math_card()
    schema_json()
    code_to_canvas()
    copies()
    print("\nassets in", OUT)

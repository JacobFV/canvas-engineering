"""Generate the paper's bespoke figures.

fig_layout_example.png : renders EXACTLY the CanvasLayout from Section 2.1's
    code block as five (H, W) time slices, with a legend instead of in-figure
    callouts so nothing overlaps.
fig_icu_allocation.png : compiles the ICU ward schema from
    examples/07_hospital_icu.py onto a 26x26 canvas and renders the compiler's
    block allocation, colored by owning entity.
fig_topology.png       : the five topology constructors, drawn as clean node
    graphs (replaces the imported raster with unreadable text).
fig_type_system.png    : C struct layout <-> canvas schema, drawn natively.
"""

import sys, os, re, colorsys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 200,
})

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")

C_VISUAL, C_ACTION, C_REWARD = "#7db8e8", "#f2a95c", "#8fd18f"
C_EMPTY, C_EDGE = "#f2f2f2", "#dddddd"

# ---------------------------------------------------------------- Figure 1
def fig_layout_example():
    T, H, W = 5, 8, 8
    t_current = 2
    regions = {  # exactly the paper's code block
        "visual": ((0, 5, 0, 6, 0, 6), C_VISUAL),
        "action": ((0, 5, 6, 7, 0, 1), C_ACTION),
        "reward": ((2, 3, 7, 8, 0, 1), C_REWARD),
    }

    fig, axes = plt.subplots(1, T, figsize=(8.6, 2.15))
    for t, ax in enumerate(axes):
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.set_xticks(range(W + 1)); ax.set_yticks(range(H + 1))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(length=0)
        for h in range(H):
            for w in range(W):
                ax.add_patch(Rectangle((w, h), 1, 1, facecolor=C_EMPTY,
                                       edgecolor=C_EDGE, lw=0.4))
        for name, ((t0, t1, h0, h1, w0, w1), color) in regions.items():
            if t0 <= t < t1:
                ax.add_patch(Rectangle((w0, h0), w1 - w0, h1 - h0,
                                       facecolor=color, edgecolor="black",
                                       lw=1.0, zorder=3))
        future = t >= t_current
        if future:
            tag = "(future)"
        elif t == t_current - 1:
            tag = "(context, present)"
        else:
            tag = "(context, past)"
        ax.set_title(f"$t={t}$  {tag}", fontsize=8.5,
                     color="#b03030" if future else "#333333")
        for spine in ax.spines.values():
            spine.set_edgecolor("#b03030" if future else "#999999")
            spine.set_linewidth(1.6 if future else 0.8)
    axes[0].set_ylabel("H = 8", fontsize=8)
    axes[0].set_xlabel("W = 8", fontsize=8)

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=0.8)
               for c in (C_VISUAL, C_ACTION, C_REWARD, C_EMPTY)]
    labels = ["visual (0,5, 0,6, 0,6) — 180 pos",
              "action (0,5, 6,7, 0,1) — 5 pos",
              "reward (2,3, 7,8, 0,1) — 1 pos",
              "unallocated — 134 pos"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               ncol=4, fontsize=7.6, frameon=False,
               handlelength=1.1, columnspacing=1.2)
    fig.suptitle(r"CanvasLayout(T=5, H=8, W=8), $t_{\rm current}=2$",
                 fontsize=9.5, y=1.06)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_layout_example.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig_layout_example.png")


# ---------------------------------------------------------------- Figure 2
NODE_COLORS = ["#6fa8dc", "#93c47d", "#f2a95c", "#c27ba0", "#9a9a9a"]

def _node(ax, x, y, color, label=None, r=0.13):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="black",
                        lw=0.9, zorder=4))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                zorder=5, color="white", fontweight="bold")

def _edge(ax, p, q, rad=0.0, bidir=False, color="#444444", lw=1.0):
    style = "<|-|>" if bidir else "-|>"
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle=style, mutation_scale=9,
                                 shrinkA=8.5, shrinkB=8.5, color=color,
                                 lw=lw, zorder=2))

def _selfloop(ax, x, y, r=0.13, color="#888888"):
    th = np.linspace(0.25 * np.pi, 2.25 * np.pi, 40)
    lx = x + 0.16 * np.cos(th)
    ly = y + r + 0.10 + 0.13 * np.sin(th) * 0.8
    ax.plot(lx, ly, lw=0.9, color=color, zorder=1)

def fig_topology():
    fig, axes = plt.subplots(1, 5, figsize=(8.6, 1.85))
    for ax in axes:
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.0, 1.35)
        ax.set_aspect("equal"); ax.axis("off")

    # dense: 4 nodes, all pairs both directions
    ax = axes[0]
    pts = [(np.cos(a), np.sin(a) * 0.75 - 0.05)
           for a in np.pi / 4 + np.arange(4) * np.pi / 2]
    pts = [(0.72 * x, 0.72 * y) for x, y in pts]
    for i, p in enumerate(pts):
        _node(ax, *p, NODE_COLORS[i])
        _selfloop(ax, *p)
    for i in range(4):
        for j in range(i + 1, 4):
            _edge(ax, pts[i], pts[j], bidir=True)
    ax.set_title("dense", fontsize=8.5, family="monospace")

    # isolated: self-loops only
    ax = axes[1]
    for i, p in enumerate(pts):
        _node(ax, *p, NODE_COLORS[i])
        _selfloop(ax, *p)
    ax.set_title("isolated", fontsize=8.5, family="monospace")

    # hub_spoke
    ax = axes[2]
    hub = (0.0, 0.05)
    _node(ax, *hub, "#8e7cc3", "h")
    for i, a in enumerate(np.pi / 4 + np.arange(4) * np.pi / 2):
        p = (0.78 * np.cos(a), 0.62 * np.sin(a) + 0.05)
        _node(ax, *p, NODE_COLORS[i])
        _edge(ax, p, hub, bidir=True)
    ax.set_title("hub_spoke", fontsize=8.5, family="monospace")

    # causal_chain: obs -> plan -> act
    ax = axes[3]
    chain = [(-0.72, 0.05), (0.0, 0.05), (0.72, 0.05)]
    labs = ["A", "B", "C"]
    for p, lb, c in zip(chain, labs, NODE_COLORS):
        _node(ax, *p, c, lb)
        _selfloop(ax, *p)
    _edge(ax, chain[0], chain[1])   # arrows show information flow
    _edge(ax, chain[1], chain[2])
    ax.text(0, -0.62, "info flows A$\\to$B$\\to$C", ha="center", fontsize=7)
    ax.set_title("causal_chain", fontsize=8.5, family="monospace")

    # causal_temporal: two frames, same-frame self + prev-frame cross
    ax = axes[4]
    o0, a0, o1, a1 = (-0.62, 0.42), (-0.62, -0.38), (0.62, 0.42), (0.62, -0.38)
    for p, lb, c in [(o0, "o", NODE_COLORS[0]), (a0, "a", NODE_COLORS[2]),
                     (o1, "o", NODE_COLORS[0]), (a1, "a", NODE_COLORS[2])]:
        _node(ax, *p, c, lb)
    for p in (o0, a0, o1, a1):
        _selfloop(ax, *p)
    _edge(ax, o0, o1); _edge(ax, o0, a1)
    _edge(ax, a0, o1); _edge(ax, a0, a1)
    ax.text(-0.62, -0.80, "$t-1$", ha="center", fontsize=7)
    ax.text(0.62, -0.80, "$t$", ha="center", fontsize=7)
    ax.set_title("causal_temporal", fontsize=8.5, family="monospace")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_topology.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig_topology.png")


# ---------------------------------------------------------------- Figure 3
def fig_type_system():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.7),
                                   gridspec_kw={"width_ratios": [1.0, 1.05]})

    # left: C struct byte layout
    axL.set_xlim(0, 10); axL.set_ylim(10.6, -1.1); axL.axis("off")
    fields = [  # (label, rel_height, color)
        ("float visual[180]  @ 0", 5.6, C_VISUAL),
        ("float action[5]  @ 720", 1.6, C_ACTION),
        ("float reward  @ 740", 0.9, C_REWARD),
        ("(padding)", 0.9, C_EMPTY),
    ]
    axL.text(5, -0.55, "C struct layout", ha="center", fontsize=9,
             fontweight="bold")
    y = 0.35
    for label, h, color in fields:
        axL.add_patch(Rectangle((1.4, y), 7.4, h, facecolor=color,
                                edgecolor="black", lw=0.9))
        axL.text(5.1, y + h / 2, label, ha="center", va="center",
                 fontsize=7.4, family="monospace")
        y += h

    # right: canvas schema (one slice of the Fig-1 layout, t=2)
    H, W = 8, 8
    axR.set_xlim(-0.6, W + 0.4); axR.set_ylim(H + 1.15, -1.35)
    axR.set_aspect("equal"); axR.axis("off")
    axR.text(W / 2, -0.75, "canvas schema (slice $t=2$)", ha="center",
             fontsize=9, fontweight="bold")
    for h in range(H):
        for w in range(W):
            axR.add_patch(Rectangle((w, h), 1, 1, facecolor=C_EMPTY,
                                    edgecolor=C_EDGE, lw=0.4))
    for (h0, h1, w0, w1), color, label in [
            ((0, 6, 0, 6), C_VISUAL, "visual"),
            ((6, 7, 0, 1), C_ACTION, None),
            ((7, 8, 0, 1), C_REWARD, None)]:
        axR.add_patch(Rectangle((w0, h0), w1 - w0, h1 - h0, facecolor=color,
                                edgecolor="black", lw=1.0, zorder=3))
        if label:
            axR.text((w0 + w1) / 2, (h0 + h1) / 2, label, ha="center",
                     va="center", fontsize=7.6, family="monospace")
    axR.annotate("action", xy=(0.5, 6.5), xytext=(2.3, 6.8), fontsize=6.8,
                 family="monospace", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#444444"))
    axR.annotate("reward", xy=(0.5, 7.5), xytext=(2.3, 8.35), fontsize=6.8,
                 family="monospace", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#444444"))

    # arrow between the two views
    fig.text(0.49, 0.52, "$\\leftrightarrow$", ha="center", va="center",
             fontsize=16)
    fig.text(0.49, 0.40, "same\nidea", ha="center", va="center", fontsize=7,
             color="#555555")

    fig.tight_layout(w_pad=2.2)
    fig.savefig(os.path.join(FIGDIR, "fig_type_system.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig_type_system.png")


# ---------------------------------------------------------------- Figure 4
GROUP_COLORS = {
    "ward":         "#9a9a9a",
    "bureaucratic": "#e8b04c",
    "patient":      "#6fa8dc",
    "nurse":        "#93c47d",
    "family":       "#c27ba0",
}

def group_of(name):
    if name.startswith("patients["):
        return "patient", int(re.match(r"patients\[(\d+)\]", name).group(1))
    if name.startswith("nurses["):
        return "nurse", int(re.match(r"nurses\[(\d+)\]", name).group(1))
    if name.startswith("families["):
        return "family", int(re.match(r"families\[(\d+)\]", name).group(1))
    if name.startswith("bureaucratic"):
        return "bureaucratic", 0
    return "ward", 0

def shade(hex_color, k, n):
    r, g, b = (int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l2 = min(0.92, max(0.35, l * (0.72 + 0.5 * (k / max(1, n - 1)))))
    return colorsys.hls_to_rgb(h, l2, s)

def fig_icu_allocation():
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "examples", "07_hospital_icu.py")).read()
    cut = src.index('print(f"ICU Ward canvas')
    ns = {"__file__": "examples/07_hospital_icu.py"}
    exec(src[:cut], ns)
    # recompile on the tightest grid that packs (32x32 in the example is
    # roomy; 26x26 fits with 68% utilization)
    bound = ns["compile_schema"](
        ns["ward"], T=1, H=26, W=26, d_model=32,
        connectivity=ns["ConnectivityPolicy"](
            intra="dense", array_element="ring", temporal="dense"),
    )
    lay = bound.layout
    H, W = lay.H, lay.W
    n_conn = len(bound.topology.connections)
    alloc = sum((s.bounds[3] - s.bounds[2]) * (s.bounds[5] - s.bounds[4])
                for s in lay.regions.values())

    h_max = max((s.bounds if hasattr(s, "bounds") else s)[3]
                for s in lay.regions.values())
    fig = plt.figure(figsize=(10.4, 6.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.66, 1.0], wspace=0.10,
                          left=0.02, right=0.90, top=0.93, bottom=0.13)
    axC = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, W); ax.set_ylim(h_max + 1.2, 0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- left panel: the schema source that produced the allocation ----
    CODE = [
        ("@dataclass", "kw", None),
        ("class Patient:", "hl", "patient"),
        ("    cardiovascular: CardiovascularSystem", "plain", None),
        ("    respiratory:    RespiratorySystem", "plain", None),
        ("    renal:          RenalSystem", "plain", None),
        ("    neurological:   NeurologicalSystem", "plain", None),
        ("    psychological:  PsychologicalState", "plain", None),
        ("    deterioration_risk: Field = \\", "plain", None),
        ("        Field(2, 4, loss_weight=8.0)", "plain", None),
        ("    organ_failure_risk: Field = \\", "plain", None),
        ("        Field(1, 6, loss_weight=5.0)", "plain", None),
        ("", "plain", None),
        ("@dataclass", "kw", None),
        ("class Nurse:", "hl", "nurse"),
        ("    workload:   Field = Field(1, 2)", "plain", None),
        ("    fatigue:    Field = Field(1, 2,", "plain", None),
        ("                      loss_weight=2.0)", "plain", None),
        ("    stress:     Field = Field(1, 2,", "plain", None),
        ("                      loss_weight=2.0)", "plain", None),
        ("    competence: Field = Field(1, 2,", "plain", None),
        ("                      is_output=False)", "plain", None),
        ("    rapport:    Field = Field(1, 2)", "plain", None),
        ("", "plain", None),
        ("ward = ICUWard(", "plain", None),
        ("    patients=[Patient() for _ in range(6)],", "hl2", "patient"),
        ("    nurses=[Nurse() for _ in range(4)],", "hl2", "nurse"),
        ("    families=[FamilyUnit() for _ in range(6)],", "plain", None),
        (")", "plain", None),
        ("bound = compile_schema(ward, H=26, W=26)", "kw", None),
    ]
    n = len(CODE)
    axC.set_xlim(0, 1); axC.set_ylim(n + 0.5, -1.0)
    axC.axis("off")
    axC.add_patch(Rectangle((0, -0.8), 1, n + 1.0, facecolor="#f7f7f7",
                            edgecolor="#cccccc", lw=0.7))
    HL = {"patient": "#cfe2f3", "nurse": "#d9ead3"}
    HL_EDGE = {"patient": "#1a3f6f", "nurse": "#1e5c1e"}
    anchors = {}
    for i, (text, kind, ent) in enumerate(CODE):
        if kind in ("hl", "hl2") and ent:
            axC.add_patch(Rectangle((0.015, i - 0.42), 0.97, 0.86,
                                    facecolor=HL[ent],
                                    edgecolor=HL_EDGE[ent], lw=0.8, zorder=2))
            if kind == "hl2":
                anchors[ent] = i
        color = "#7a2f8f" if kind == "kw" else "#222222"
        axC.text(0.035, i, text, fontsize=6.6, family="monospace",
                 va="center", color=color, zorder=3)
    axC.set_title("the schema (examples/07_hospital_icu.py)", fontsize=9)

    counts = {"patient": 6, "nurse": 4, "family": 6,
              "bureaucratic": 1, "ward": 1}
    for name, spec in lay.regions.items():
        t0, t1, h0, h1, w0, w1 = spec.bounds if hasattr(spec, "bounds") else spec
        grp, idx = group_of(name)
        color = shade(GROUP_COLORS[grp], idx, counts[grp])
        is_coarse = re.fullmatch(
            r"(patients|nurses|families)\[\d+\]"
            r"(\.(cardiovascular|respiratory|renal|neurological|psychological))?"
            r"|bureaucratic", name) is not None
        ax.add_patch(Rectangle((w0, h0), w1 - w0, h1 - h0,
                               facecolor=color,
                               edgecolor="black" if is_coarse else "#555555",
                               lw=1.3 if is_coarse else 0.45,
                               hatch="////" if is_coarse else None,
                               zorder=3 if is_coarse else 2))
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="#f5f5f5",
                           edgecolor="none", zorder=1))
    ax.text(W / 2, h_max + 0.62,
            f"{H * W - alloc} positions free ({alloc}/{H * W} used)",
            ha="center", va="center", fontsize=7.2, color="#777777")

    # --- pointability: outline the full extent of two entities -----------
    def entity_outline(prefix, edge_color, label, label_xy):
        occ = np.zeros((H, W), dtype=bool)
        for name, spec in lay.regions.items():
            if name == prefix or name.startswith(prefix + "."):
                _, _, h0, h1, w0, w1 = (spec.bounds if hasattr(spec, "bounds")
                                        else spec)
                occ[h0:h1, w0:w1] = True
        # draw boundary segments of the occupied set
        for h in range(H):
            for w in range(W):
                if not occ[h, w]:
                    continue
                if h == 0 or not occ[h - 1, w]:
                    ax.plot([w, w + 1], [h, h], color=edge_color, lw=2.4, zorder=6)
                if h == H - 1 or not occ[h + 1, w]:
                    ax.plot([w, w + 1], [h + 1, h + 1], color=edge_color, lw=2.4, zorder=6)
                if w == 0 or not occ[h, w - 1]:
                    ax.plot([w, w], [h, h + 1], color=edge_color, lw=2.4, zorder=6)
                if w == W - 1 or not occ[h, w + 1]:
                    ax.plot([w + 1, w + 1], [h, h + 1], color=edge_color, lw=2.4, zorder=6)
        hs, ws = np.where(occ)
        ax.annotate(label, xy=(ws.mean() + 0.5, hs.mean() + 0.5),
                    xytext=label_xy, fontsize=8.4, fontweight="bold",
                    color=edge_color, ha="left", va="center",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color=edge_color))
        return hs, ws

    from matplotlib.patches import ConnectionPatch
    extents = {}
    extents["patient"] = entity_outline("patients[2]", "#1a3f6f",
                                        "this is patients[2]", (W + 0.7, 6.0))
    extents["nurse"] = entity_outline("nurses[1]", "#1e5c1e",
                                      "this is nurses[1]", (W + 0.7, h_max - 2.0))
    # dashed connectors: instance line in the schema -> its blocks on canvas
    for ent, line_i in anchors.items():
        hs, ws = extents[ent]
        row = hs.min()
        target = (float(ws[hs == row].min()), float(row) + 0.5)
        fig.add_artist(ConnectionPatch(
            xyA=(0.985, line_i), coordsA=axC.transData,
            xyB=target, coordsB=ax.transData,
            arrowstyle="-|>", mutation_scale=10, lw=1.2,
            linestyle=(0, (4, 2)), color=HL_EDGE[ent], zorder=8))
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS["patient"], edgecolor="#555"),
        Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS["nurse"], edgecolor="#555"),
        Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS["family"], edgecolor="#555"),
        Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS["bureaucratic"], edgecolor="#555"),
        Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS["ward"], edgecolor="#555"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                  hatch="////", lw=1.2),
    ]
    labels = ["patients[0..5] (organ systems, risks)",
              "nurses[0..3] (workload, fatigue)",
              "families[0..5]",
              "bureaucratic (insurance, beds, staffing)",
              "ward globals (acuity, resources)",
              "coarse-grained field (auto-inserted)"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, fontsize=7.4, frameon=False)
    ax.set_title(f"compile_schema(ward, H=26, W=26) $\\rightarrow$ "
                 f"{len(lay.regions)} regions, {n_conn} connections",
                 fontsize=9)
    fig.savefig(os.path.join(FIGDIR, "fig_icu_allocation.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote fig_icu_allocation.png ({len(lay.regions)} regions, "
          f"{alloc}/{H*W} used, {n_conn} connections)")


if __name__ == "__main__":
    fig_layout_example()
    fig_topology()
    fig_type_system()
    fig_icu_allocation()

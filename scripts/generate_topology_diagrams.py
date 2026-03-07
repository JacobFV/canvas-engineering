"""Generate topology convenience constructor diagrams for canvas-engineering.

Generates assets/topology_constructors.png — a panel of 5 topology patterns
with LaTeX-style mathematical node labels showing the graph structure each
convenience constructor produces.

Patterns:
  1. dense(a, b, c)           — fully connected
  2. isolated(a, b, c)        — block-diagonal
  3. hub_spoke(h, [a, b, c])  — star topology
  4. causal_chain(a, b, c)    — A → B → C
  5. causal_temporal(x, y)    — same-frame self + prev-frame cross

Usage:
    python scripts/generate_topology_diagrams.py

All diagrams are deterministic and can be regenerated from this script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects
import os

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS, exist_ok=True)

# Color palette
COLORS = {
    'node_a': '#4A90D9',
    'node_b': '#E8734A',
    'node_c': '#5CB85C',
    'hub': '#9B59B6',
    'bg': '#FAFAFA',
    'edge': '#2C3E50',
    'self_edge': '#7F8C8D',
    'text': '#2C3E50',
    'temporal': '#E74C3C',
    'title': '#1a1a2e',
}

NODE_COLORS = ['#4A90D9', '#E8734A', '#5CB85C', '#9B59B6', '#F5A623']


def _draw_node(ax, x, y, label, color, radius=0.32):
    """Draw a node circle with mathematical label."""
    circle = plt.Circle((x, y), radius, facecolor=color, edgecolor='white',
                         linewidth=2.5, alpha=0.9, zorder=10)
    ax.add_patch(circle)
    txt = ax.text(x, y, label, fontsize=13, ha='center', va='center',
                  color='white', fontweight='bold', zorder=11,
                  fontfamily='serif')
    txt.set_path_effects([
        path_effects.withStroke(linewidth=1, foreground='black', alpha=0.3)
    ])


def _draw_arrow(ax, x0, y0, x1, y1, color=None, style='->', lw=1.8,
                curved=0, radius=0.32, label=None, label_color=None):
    """Draw an arrow between nodes, stopping at the node boundary."""
    if color is None:
        color = COLORS['edge']

    # Shorten arrow to stop at node boundaries
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    x0s = x0 + ux * radius
    y0s = y0 + uy * radius
    x1s = x1 - ux * radius
    y1s = y1 - uy * radius

    if curved != 0:
        conn = f"arc3,rad={curved}"
    else:
        conn = "arc3,rad=0"

    arrow = FancyArrowPatch(
        (x0s, y0s), (x1s, y1s),
        connectionstyle=conn,
        arrowstyle=style, color=color,
        lw=lw, mutation_scale=14, zorder=5,
    )
    ax.add_patch(arrow)

    if label:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        if curved > 0:
            mx += curved * 0.3
            my += abs(curved) * 0.15
        elif curved < 0:
            mx += curved * 0.3
            my -= abs(curved) * 0.15
        ax.text(mx, my, label, fontsize=8, ha='center', va='center',
                color=label_color or '#7F8C8D', fontfamily='serif',
                fontstyle='italic', zorder=15,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))


def _draw_self_loop(ax, x, y, color=None, direction='top', radius=0.32):
    """Draw a self-loop arc above/below a node."""
    if color is None:
        color = COLORS['self_edge']

    if direction == 'top':
        loop_y = y + radius
        arc = FancyArrowPatch(
            (x - 0.15, loop_y), (x + 0.15, loop_y),
            connectionstyle="arc3,rad=-1.2",
            arrowstyle='->', color=color, lw=1.5, mutation_scale=10, zorder=5,
        )
    elif direction == 'bottom':
        loop_y = y - radius
        arc = FancyArrowPatch(
            (x + 0.15, loop_y), (x - 0.15, loop_y),
            connectionstyle="arc3,rad=-1.2",
            arrowstyle='->', color=color, lw=1.5, mutation_scale=10, zorder=5,
        )
    elif direction == 'left':
        loop_x = x - radius
        arc = FancyArrowPatch(
            (loop_x, y + 0.15), (loop_x, y - 0.15),
            connectionstyle="arc3,rad=-1.2",
            arrowstyle='->', color=color, lw=1.5, mutation_scale=10, zorder=5,
        )
    else:  # right
        loop_x = x + radius
        arc = FancyArrowPatch(
            (loop_x, y - 0.15), (loop_x, y + 0.15),
            connectionstyle="arc3,rad=-1.2",
            arrowstyle='->', color=color, lw=1.5, mutation_scale=10, zorder=5,
        )
    ax.add_patch(arc)


def _setup_ax(ax, title, subtitle=None):
    """Configure axis for topology diagram."""
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold',
                 color=COLORS['title'], pad=12, fontfamily='serif')
    if subtitle:
        ax.text(0.5, -0.08, subtitle, fontsize=9, ha='center', va='top',
                color='#7F8C8D', transform=ax.transAxes, fontfamily='monospace')


def draw_dense(ax):
    """dense(["a", "b", "c"]) — fully connected."""
    _setup_ax(ax, 'Dense', 'CanvasTopology.dense(["a","b","c"])')

    # Triangle layout
    nodes = [
        (1.0, 2.2, r'$a$', NODE_COLORS[0]),
        (0.0, 0.5, r'$b$', NODE_COLORS[1]),
        (2.0, 0.5, r'$c$', NODE_COLORS[2]),
    ]
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(-0.2, 3.0)

    # All pairs, bidirectional
    pairs = [(0,1), (0,2), (1,2)]
    for i, j in pairs:
        x0, y0 = nodes[i][0], nodes[i][1]
        x1, y1 = nodes[j][0], nodes[j][1]
        _draw_arrow(ax, x0, y0, x1, y1, curved=0.15)
        _draw_arrow(ax, x1, y1, x0, y0, curved=0.15)

    # Self-loops
    _draw_self_loop(ax, 1.0, 2.2, direction='top')
    _draw_self_loop(ax, 0.0, 0.5, direction='left')
    _draw_self_loop(ax, 2.0, 0.5, direction='right')

    for x, y, label, color in nodes:
        _draw_node(ax, x, y, label, color)


def draw_isolated(ax):
    """isolated(["a", "b", "c"]) — block-diagonal."""
    _setup_ax(ax, 'Isolated', 'CanvasTopology.isolated(["a","b","c"])')

    nodes = [
        (0.0, 1.5, r'$a$', NODE_COLORS[0]),
        (1.0, 1.5, r'$b$', NODE_COLORS[1]),
        (2.0, 1.5, r'$c$', NODE_COLORS[2]),
    ]
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(0.0, 3.0)

    # Self-loops only
    for x, y, _, _ in nodes:
        _draw_self_loop(ax, x, y, direction='top')

    for x, y, label, color in nodes:
        _draw_node(ax, x, y, label, color)


def draw_hub_spoke(ax):
    """hub_spoke("h", ["a", "b", "c"]) — star topology."""
    _setup_ax(ax, 'Hub-Spoke', 'CanvasTopology.hub_spoke("h",["a","b","c"])')

    hub = (1.0, 1.3, r'$h$', COLORS['hub'])
    spokes = [
        (1.0, 2.7, r'$a$', NODE_COLORS[0]),
        (-0.2, 0.3, r'$b$', NODE_COLORS[1]),
        (2.2, 0.3, r'$c$', NODE_COLORS[2]),
    ]
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(-0.5, 3.5)

    # Hub ↔ each spoke (bidirectional)
    for sx, sy, _, _ in spokes:
        _draw_arrow(ax, hub[0], hub[1], sx, sy, curved=0.12,
                    color=COLORS['hub'])
        _draw_arrow(ax, sx, sy, hub[0], hub[1], curved=0.12,
                    color=COLORS['hub'])

    # Self-loops
    _draw_self_loop(ax, hub[0], hub[1], direction='left', color=COLORS['hub'])
    _draw_self_loop(ax, spokes[0][0], spokes[0][1], direction='top')
    _draw_self_loop(ax, spokes[1][0], spokes[1][1], direction='left')
    _draw_self_loop(ax, spokes[2][0], spokes[2][1], direction='right')

    _draw_node(ax, *hub)
    for s in spokes:
        _draw_node(ax, *s)


def draw_causal_chain(ax):
    """causal_chain(["obs", "plan", "act"]) — A → B → C."""
    _setup_ax(ax, 'Causal Chain', 'CanvasTopology.causal_chain(["obs","plan","act"])')

    nodes = [
        (0.0, 1.3, r'$obs$', NODE_COLORS[0]),
        (1.2, 1.3, r'$plan$', NODE_COLORS[1]),
        (2.4, 1.3, r'$act$', NODE_COLORS[2]),
    ]
    ax.set_xlim(-0.8, 3.2)
    ax.set_ylim(0.0, 2.8)

    # Causal: plan ← obs, act ← plan
    _draw_arrow(ax, 1.2, 1.3, 0.0, 1.3, lw=2.2, radius=0.36)
    _draw_arrow(ax, 2.4, 1.3, 1.2, 1.3, lw=2.2, radius=0.36)

    # Self-loops
    _draw_self_loop(ax, 0.0, 1.3, direction='top')
    _draw_self_loop(ax, 1.2, 1.3, direction='top')
    _draw_self_loop(ax, 2.4, 1.3, direction='top')

    for x, y, label, color in nodes:
        _draw_node(ax, x, y, label, color, radius=0.36)


def draw_causal_temporal(ax):
    """causal_temporal(["x", "y"]) — same-frame self + prev-frame cross."""
    _setup_ax(ax, 'Causal Temporal', 'CanvasTopology.causal_temporal(["x","y"])')

    ax.set_xlim(-0.8, 4.5)
    ax.set_ylim(-0.5, 3.5)

    # Two timesteps, two regions each
    # t-1 column (left), t column (right)
    r = 0.32

    # t-1 nodes (faded)
    x_tm1 = (0.5, 2.3)  # x at t-1
    y_tm1 = (0.5, 0.7)  # y at t-1
    # t nodes
    x_t = (3.0, 2.3)    # x at t
    y_t = (3.0, 0.7)    # y at t

    # Time labels
    ax.text(0.5, 3.2, r'$t\!-\!1$', fontsize=14, ha='center', va='center',
            color='#7F8C8D', fontfamily='serif', fontstyle='italic')
    ax.text(3.0, 3.2, r'$t$', fontsize=14, ha='center', va='center',
            color=COLORS['text'], fontfamily='serif', fontstyle='italic')

    # Dashed box around each timestep
    for bx in [(-0.2, -0.1, 1.4, 3.0), (2.2, -0.1, 1.6, 3.0)]:
        rect = plt.Rectangle((bx[0], bx[1]), bx[2], bx[3],
                               linewidth=1, edgecolor='#BDC3C7',
                               facecolor='none', linestyle='--', zorder=1)
        ax.add_patch(rect)

    # Same-frame self-attention at t (solid)
    _draw_self_loop(ax, x_t[0], x_t[1], direction='right', color=NODE_COLORS[0])
    _draw_self_loop(ax, y_t[0], y_t[1], direction='right', color=NODE_COLORS[1])

    # Prev-frame cross-attention: x_t ← x_{t-1}
    _draw_arrow(ax, x_t[0], x_t[1], x_tm1[0], x_tm1[1],
                color=COLORS['temporal'], lw=1.8, curved=0.1,
                label=r'$t\!-\!1$', label_color=COLORS['temporal'])
    # Prev-frame cross-attention: y_t ← y_{t-1}
    _draw_arrow(ax, y_t[0], y_t[1], y_tm1[0], y_tm1[1],
                color=COLORS['temporal'], lw=1.8, curved=-0.1,
                label=r'$t\!-\!1$', label_color=COLORS['temporal'])
    # Prev-frame cross-attention: x_t ← y_{t-1}
    _draw_arrow(ax, x_t[0], x_t[1], y_tm1[0], y_tm1[1],
                color=COLORS['temporal'], lw=1.3, curved=0.15,
                style='->')
    # Prev-frame cross-attention: y_t ← x_{t-1}
    _draw_arrow(ax, y_t[0], y_t[1], x_tm1[0], x_tm1[1],
                color=COLORS['temporal'], lw=1.3, curved=-0.15,
                style='->')

    # Draw nodes (t-1 faded, t solid)
    for pos, label, color in [(x_tm1, r'$x$', NODE_COLORS[0]),
                               (y_tm1, r'$y$', NODE_COLORS[1])]:
        circle = plt.Circle(pos, r, facecolor=color, edgecolor='white',
                             linewidth=2, alpha=0.4, zorder=10)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, fontsize=13, ha='center', va='center',
                color='white', fontweight='bold', zorder=11, alpha=0.5,
                fontfamily='serif')

    _draw_node(ax, x_t[0], x_t[1], r'$x$', NODE_COLORS[0])
    _draw_node(ax, y_t[0], y_t[1], r'$y$', NODE_COLORS[1])


def generate_all():
    """Generate the combined topology panel."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=150)
    fig.patch.set_facecolor('white')
    fig.suptitle('Topology Convenience Constructors', fontsize=16,
                 fontweight='bold', color=COLORS['title'], y=1.02,
                 fontfamily='serif')

    draw_dense(axes[0])
    draw_isolated(axes[1])
    draw_hub_spoke(axes[2])
    draw_causal_chain(axes[3])
    draw_causal_temporal(axes[4])

    plt.tight_layout()
    path = os.path.join(ASSETS, "topology_constructors.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150, pad_inches=0.3)
    plt.close()
    print(f"  saved {path}")


if __name__ == "__main__":
    print("Generating topology diagrams...")
    generate_all()
    print("Done.")

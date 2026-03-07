#!/usr/bin/env python3
"""Generate orthographic 3D renderings of canvas allocations as PNGs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'monospace',
    'font.size': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.facecolor': '#0d0d0f',
    'figure.facecolor': '#0d0d0f',
    'text.color': '#d4d4d8',
    'axes.edgecolor': '#27272a',
    'axes.labelcolor': '#a1a1aa',
    'xtick.color': '#71717a',
    'ytick.color': '#71717a',
})

REGION_COLORS = {
    'visual':   '#60a5fa',
    'screen':   '#60a5fa',
    'action':   '#a3e635',
    'proprio':  '#f59e0b',
    'reward':   '#ef4444',
    'mouse':    '#a3e635',
    'keyboard': '#f472b6',
    'llm':      '#a78bfa',
    'robot1_cam':    '#60a5fa',
    'robot1_action': '#a3e635',
    'robot2_cam':    '#38bdf8',
    'robot2_action': '#4ade80',
    'shared_task':   '#f59e0b',
}


def draw_canvas_3d(ax, T, H, W, regions, title, elev=25, azim=-50):
    """Draw a 3D orthographic rendering of canvas block allocation."""
    ax.set_xlim(0, T)
    ax.set_ylim(0, H)
    ax.set_zlim(0, W)
    ax.set_xlabel('T (time)', fontsize=8, labelpad=1)
    ax.set_ylabel('H', fontsize=8, labelpad=1)
    ax.set_zlabel('W', fontsize=8, labelpad=1)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='#e4e4e7')
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#27272a')
    ax.yaxis.pane.set_edgecolor('#27272a')
    ax.zaxis.pane.set_edgecolor('#27272a')
    ax.grid(True, alpha=0.1, color='#71717a')

    legend_handles = []
    seen_names = set()

    for name, (t0, t1, h0, h1, w0, w1) in regions.items():
        color = REGION_COLORS.get(name, '#71717a')
        # Draw filled block
        verts = _box_verts(t0, t1, h0, h1, w0, w1)
        poly = Poly3DCollection(verts, alpha=0.25, facecolor=color, edgecolor=color, linewidth=0.6)
        ax.add_collection3d(poly)

        # Wireframe edges (brighter)
        for face in verts:
            xs = [v[0] for v in face] + [face[0][0]]
            ys = [v[1] for v in face] + [face[0][1]]
            zs = [v[2] for v in face] + [face[0][2]]
            ax.plot(xs, ys, zs, color=color, alpha=0.6, linewidth=0.4)

        if name not in seen_names:
            n_pos = (t1-t0) * (h1-h0) * (w1-w0)
            pct = n_pos / (T * H * W) * 100
            legend_handles.append(mpatches.Patch(color=color, alpha=0.5, label=f'{name} ({n_pos}, {pct:.0f}%)'))
            seen_names.add(name)

    ax.legend(handles=legend_handles, loc='upper left', fontsize=6.5, frameon=True,
              facecolor='#111113', edgecolor='#27272a', labelcolor='#d4d4d8',
              borderpad=0.6, handlelength=1.2)


def _box_verts(x0, x1, y0, y1, z0, z1):
    """Return 6 faces of a rectangular prism as vertex lists."""
    return [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],  # bottom
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # top
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],  # front
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],  # back
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],  # left
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],  # right
    ]


# ── Robot Manipulation Canvas ────────────────────────────────────────────────
def fig_robot():
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    regions = {
        'visual':  (0, 5, 0, 6, 0, 6),
        'action':  (0, 5, 6, 7, 0, 1),
        'proprio': (0, 3, 6, 7, 1, 3),
        'reward':  (2, 3, 7, 8, 0, 2),
    }
    draw_canvas_3d(ax, T=5, H=8, W=8, regions=regions,
                   title='Robot Manipulation Canvas (320 positions)')
    fig.savefig(os.path.join(OUT_DIR, 'canvas_robot.png'))
    plt.close(fig)
    print('  Saved canvas_robot.png')


# ── Computer Use Agent Canvas ────────────────────────────────────────────────
def fig_computer():
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    # Scaled down for visualization (actual is 16×32×32)
    T, H, W = 8, 16, 16
    regions = {
        'screen':   (0, 8, 0, 12, 0, 12),
        'mouse':    (0, 8, 12, 13, 0, 2),
        'keyboard': (0, 8, 13, 14, 0, 2),
        'llm':      (0, 8, 14, 16, 0, 4),
    }
    draw_canvas_3d(ax, T=T, H=H, W=W, regions=regions,
                   title='Computer Use Agent Canvas (2,048 positions, scaled)')
    fig.savefig(os.path.join(OUT_DIR, 'canvas_computer.png'))
    plt.close(fig)
    print('  Saved canvas_computer.png')


# ── Multi-Robot Control Canvas ───────────────────────────────────────────────
def fig_multi_robot():
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    T, H, W = 8, 8, 8
    regions = {
        'robot1_cam':    (0, 8, 0, 4, 0, 4),
        'robot1_action': (0, 8, 0, 4, 4, 5),
        'robot2_cam':    (0, 8, 4, 8, 0, 4),
        'robot2_action': (0, 8, 4, 8, 4, 5),
        'shared_task':   (0, 1, 0, 8, 5, 8),
    }
    draw_canvas_3d(ax, T=T, H=H, W=W, regions=regions,
                   title='Multi-Robot Control Canvas (512 positions, scaled)')
    fig.savefig(os.path.join(OUT_DIR, 'canvas_multi_robot.png'))
    plt.close(fig)
    print('  Saved canvas_multi_robot.png')


# ── Combined figure ──────────────────────────────────────────────────────────
def fig_combined():
    fig = plt.figure(figsize=(18, 5))

    # Robot
    ax1 = fig.add_subplot(131, projection='3d')
    draw_canvas_3d(ax1, 5, 8, 8, {
        'visual': (0,5,0,6,0,6), 'action': (0,5,6,7,0,1),
        'proprio': (0,3,6,7,1,3), 'reward': (2,3,7,8,0,2),
    }, 'Robot Manipulation')

    # Computer
    ax2 = fig.add_subplot(132, projection='3d')
    draw_canvas_3d(ax2, 8, 16, 16, {
        'screen': (0,8,0,12,0,12), 'mouse': (0,8,12,13,0,2),
        'keyboard': (0,8,13,14,0,2), 'llm': (0,8,14,16,0,4),
    }, 'Computer Use Agent')

    # Multi-robot
    ax3 = fig.add_subplot(133, projection='3d')
    draw_canvas_3d(ax3, 8, 8, 8, {
        'robot1_cam': (0,8,0,4,0,4), 'robot1_action': (0,8,0,4,4,5),
        'robot2_cam': (0,8,4,8,0,4), 'robot2_action': (0,8,4,8,4,5),
        'shared_task': (0,1,0,8,5,8),
    }, 'Multi-Robot Control')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'canvas_layouts_combined.png'))
    plt.close(fig)
    print('  Saved canvas_layouts_combined.png')


if __name__ == '__main__':
    print('Generating canvas allocation figures...')
    fig_robot()
    fig_computer()
    fig_multi_robot()
    fig_combined()
    print('Done.')

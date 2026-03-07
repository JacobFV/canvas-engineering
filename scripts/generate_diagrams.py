"""Generate README diagrams for canvas-engineering.

Generates:
  assets/looped_attention.png  — looped attention block diagram
  assets/canvas_type_system.png — type system analogy diagram
  assets/canvas_robot_3d.gif  — animated 3D rotating canvas allocation

Usage:
  python scripts/generate_diagrams.py

All diagrams are deterministic and can be regenerated from this script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import io
import os

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS, exist_ok=True)

# Color palette
C = {
    "visual": "#4A90D9",
    "action": "#E8734A",
    "reward": "#5CB85C",
    "thought": "#9B59B6",
    "mouse": "#F5A623",
    "keyboard": "#1ABC9C",
    "llm": "#E74C3C",
    "prompt": "#95A5A6",
    "bg": "#FAFAFA",
    "border": "#2C3E50",
    "arrow": "#2C3E50",
    "text": "#2C3E50",
    "grid": "#ECF0F1",
}


def generate_looped_attention():
    """Looped attention block diagram showing the iteration loop."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Input arrow
    ax.annotate('', xy=(1.5, 2), xytext=(0.3, 2),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2))
    ax.text(0.1, 2.3, 'h', fontsize=16, fontweight='bold', color=C['text'], style='italic')

    # + e_loop label
    ax.text(0.5, 1.3, '+ e_loop', fontsize=10, color=C['thought'], fontstyle='italic')

    # Frozen DiT block
    block = FancyBboxPatch((1.5, 1.0), 3.5, 2.0,
                            boxstyle="round,pad=0.15",
                            facecolor='#EBF5FB', edgecolor=C['visual'], linewidth=2.5)
    ax.add_patch(block)
    ax.text(3.25, 2.3, 'Frozen DiT Block', fontsize=13, fontweight='bold',
            color=C['text'], ha='center')
    ax.text(3.25, 1.7, '(pretrained weights)', fontsize=10,
            color='#7F8C8D', ha='center')

    # Loop arrow (curved, going around the block)
    loop_arrow = FancyArrowPatch((5.0, 2.7), (1.5, 2.7),
                                  connectionstyle="arc3,rad=-0.5",
                                  arrowstyle='->', color=C['thought'],
                                  lw=2.5, mutation_scale=15)
    ax.add_patch(loop_arrow)
    ax.text(3.25, 3.7, '\u00d7 3 iterations', fontsize=12,
            color=C['thought'], ha='center', fontweight='bold')

    # Output arrow
    ax.annotate('', xy=(6.5, 2), xytext=(5.0, 2),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2))

    # Learned gate box
    gate = FancyBboxPatch((6.5, 1.2), 2.5, 1.6,
                           boxstyle="round,pad=0.12",
                           facecolor='#FDEDEC', edgecolor=C['action'], linewidth=2)
    ax.add_patch(gate)
    ax.text(7.75, 2.25, 'Learned Gate', fontsize=12, fontweight='bold',
            color=C['text'], ha='center')
    ax.text(7.75, 1.7, '\u03b1 \u00b7 residual', fontsize=11,
            color='#7F8C8D', ha='center')

    # Output
    ax.annotate('', xy=(9.8, 2), xytext=(9.0, 2),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2))
    ax.text(9.85, 2.3, "h'", fontsize=16, fontweight='bold', color=C['text'], style='italic')

    # Bottom labels
    ax.text(3.25, 0.4, '350K trainable params', fontsize=11,
            color=C['action'], ha='center', fontweight='bold')
    ax.text(3.25, 0.05, '(loop_emb + loop_gate)', fontsize=9,
            color='#7F8C8D', ha='center')

    plt.tight_layout()
    path = os.path.join(ASSETS, "looped_attention.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print(f"  saved {path}")


def generate_type_system():
    """Type system analogy: struct layout ↔ canvas schema."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor('white')

    def draw_struct(ax):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('C struct layout', fontsize=14, fontweight='bold',
                     color=C['text'], pad=15)

        # Memory blocks
        fields = [
            ("int32 x", 0, 1, C['visual']),
            ("int32 y", 1, 2, C['visual']),
            ("float32 value", 2, 4, C['action']),
            ("uint8 flags", 4, 5, C['reward']),
            ("padding", 5, 6, C['grid']),
            ("float64 data", 6, 8, C['thought']),
        ]
        for label, y0, y1, color in fields:
            rect = FancyBboxPatch((1, y0 * 0.85 + 0.5), 6, (y1 - y0) * 0.85 - 0.05,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, alpha=0.3,
                                   edgecolor=color, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(4, (y0 + y1) / 2 * 0.85 + 0.5, label,
                    fontsize=10, ha='center', va='center', color=C['text'],
                    fontfamily='monospace')
            # offset labels
            ax.text(0.5, y0 * 0.85 + 0.5 + 0.15, f'0x{y0 * 4:02X}',
                    fontsize=8, ha='right', va='center', color='#7F8C8D',
                    fontfamily='monospace')

        ax.text(4, 7.8, 'offset + size = field location',
                fontsize=9, ha='center', color='#7F8C8D', style='italic')

    def draw_canvas(ax):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Canvas schema', fontsize=14, fontweight='bold',
                     color=C['text'], pad=15)

        regions = [
            ('visual (0,5, 0,6, 0,6)', 0, 4.5, C['visual']),
            ('action (0,5, 6,7, 0,1)', 4.5, 5.5, C['action']),
            ('reward (2,3, 7,8, 0,1)', 5.5, 6.3, C['reward']),
            ('thought (0,2, 7,8, 0,4)', 6.3, 7.5, C['thought']),
        ]
        for label, y0, y1, color in regions:
            rect = FancyBboxPatch((1, y0 * 0.85 + 0.5), 6, (y1 - y0) * 0.85 - 0.05,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, alpha=0.3,
                                   edgecolor=color, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(4, (y0 + y1) / 2 * 0.85 + 0.5, label,
                    fontsize=9, ha='center', va='center', color=C['text'],
                    fontfamily='monospace')

        ax.text(4, 7.8, 'bounds + period = region location',
                fontsize=9, ha='center', color='#7F8C8D', style='italic')

    draw_struct(ax1)
    draw_canvas(ax2)

    # Arrow between
    fig.text(0.5, 0.5, '\u2194', fontsize=40, ha='center', va='center',
             color=C['arrow'], fontweight='bold')

    plt.tight_layout()
    path = os.path.join(ASSETS, "canvas_type_system.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print(f"  saved {path}")


def _draw_3d_canvas(ax, elev, azim):
    """Draw a 3D canvas with colored region blocks."""
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(0, 5)
    ax.set_xlabel('W', fontsize=10, labelpad=5)
    ax.set_ylabel('H', fontsize=10, labelpad=5)
    ax.set_zlabel('T', fontsize=10, labelpad=5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#DDDDDD')
    ax.yaxis.pane.set_edgecolor('#DDDDDD')
    ax.zaxis.pane.set_edgecolor('#DDDDDD')
    ax.grid(True, alpha=0.2)

    def box(x0, x1, y0, y1, z0, z1, color, alpha=0.25):
        """Draw a 3D box."""
        verts = [
            [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
            [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
            [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
            [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
        ]
        collection = Poly3DCollection(verts, alpha=alpha,
                                       facecolor=color, edgecolor=color,
                                       linewidth=0.5)
        ax.add_collection3d(collection)

    # visual: (0,5, 0,6, 0,6) → T=0..5, H=0..6, W=0..6
    box(0, 6, 0, 6, 0, 5, C['visual'], alpha=0.2)
    # action: (0,5, 6,7, 0,1)
    box(0, 1, 6, 7, 0, 5, C['action'], alpha=0.35)
    # reward: (2,3, 7,8, 0,1)
    box(0, 1, 7, 8, 2, 3, C['reward'], alpha=0.45)
    # thought: (0,2, 7,8, 1,5)
    box(1, 5, 7, 8, 0, 2, C['thought'], alpha=0.35)


def generate_3d_gif():
    """Animated rotating 3D canvas allocation."""
    frames = []
    n_frames = 60

    for i in range(n_frames):
        azim = -60 + (360 * i / n_frames)
        elev = 25 + 10 * np.sin(2 * np.pi * i / n_frames)

        fig = plt.figure(figsize=(6, 5), dpi=100)
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111, projection='3d')
        _draw_3d_canvas(ax, elev, azim)

        ax.set_title('Robot manipulation canvas', fontsize=13,
                     fontweight='bold', color=C['text'], pad=10)

        # Legend
        legend_patches = [
            mpatches.Patch(color=C['visual'], alpha=0.4, label='visual (180 pos)'),
            mpatches.Patch(color=C['action'], alpha=0.5, label='action (5 pos)'),
            mpatches.Patch(color=C['reward'], alpha=0.6, label='reward (1 pos)'),
            mpatches.Patch(color=C['thought'], alpha=0.5, label='thought (8 pos)'),
        ]
        ax.legend(handles=legend_patches, loc='upper left', fontsize=8,
                  framealpha=0.9)

        plt.tight_layout()

        # Render to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close()

    path = os.path.join(ASSETS, "canvas_robot_3d.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=True)
    print(f"  saved {path} ({len(frames)} frames)")


def generate_3d_static():
    """Static 3D canvas for fallback."""
    fig = plt.figure(figsize=(7, 5.5), dpi=150)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    _draw_3d_canvas(ax, elev=25, azim=-45)

    ax.set_title('Robot manipulation canvas — 3D region allocation',
                 fontsize=13, fontweight='bold', color=C['text'], pad=10)

    legend_patches = [
        mpatches.Patch(color=C['visual'], alpha=0.4, label='visual (T=0:5, H=0:6, W=0:6)'),
        mpatches.Patch(color=C['action'], alpha=0.5, label='action (T=0:5, H=6:7, W=0:1)'),
        mpatches.Patch(color=C['reward'], alpha=0.6, label='reward (T=2:3, H=7:8, W=0:1)'),
        mpatches.Patch(color=C['thought'], alpha=0.5, label='thought (T=0:2, H=7:8, W=1:5)'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(ASSETS, "canvas_robot_3d.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print(f"  saved {path}")


if __name__ == "__main__":
    print("Generating diagrams...")
    generate_looped_attention()
    generate_type_system()
    generate_3d_static()
    generate_3d_gif()
    print("Done.")

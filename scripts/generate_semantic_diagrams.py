"""Generate semantic type / transfer distance diagrams for canvas-engineering.

Generates:
  assets/transfer_distance.png   — modality embedding space with distance labels
  assets/schema_alignment.png    — cross-schema region alignment between two agents

Usage:
    python scripts/generate_semantic_diagrams.py

All diagrams are deterministic and can be regenerated from this script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import matplotlib.patheffects as path_effects
import os

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS, exist_ok=True)

# Palette
C = {
    'visual': '#4A90D9',
    'depth': '#5DADE2',
    'flow': '#2E86C1',
    'action': '#E8734A',
    'joints': '#DC7633',
    'language': '#9B59B6',
    'reward': '#5CB85C',
    'thought': '#8E44AD',
    'proprio': '#F39C12',
    'memory': '#1ABC9C',
    'policy': '#E74C3C',
    'goal': '#F5A623',
    'text': '#2C3E50',
    'title': '#1a1a2e',
    'line_close': '#27AE60',
    'line_mid': '#F39C12',
    'line_far': '#E74C3C',
    'bg_box': '#F8F9FA',
}


def generate_transfer_distance():
    """Modality embedding space with pairwise distances.

    Shows realistic modalities positioned by semantic similarity,
    with distance annotations showing transfer cost estimation.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Semantic Embedding Space — Transfer Distance',
                 fontsize=15, fontweight='bold', color=C['title'],
                 fontfamily='serif', pad=18)

    # Modality positions (hand-placed to reflect semantic clustering)
    # Vision cluster (upper left)
    mods = {
        'RGB video\n224×224 30fps':        (2.0, 7.5, C['visual'], 0.48),
        'Depth map\n224×224':              (3.8, 7.0, C['depth'], 0.42),
        'Optical flow\n224×224':           (3.2, 8.5, C['flow'], 0.40),
        # Embodiment cluster (lower left)
        '6-DOF EE\ndelta pose':           (1.5, 3.0, C['action'], 0.44),
        '7-DOF joint\nangles 30Hz':        (3.2, 2.0, C['joints'], 0.42),
        'Proprioception\n(force-torque)':  (2.0, 1.0, C['proprio'], 0.44),
        # Language/abstract (right)
        'Natural language\n(English)':     (8.0, 6.5, C['language'], 0.46),
        'Natural language\n(French)':      (9.2, 5.5, C['language'], 0.42),
        'Scalar reward':                   (5.5, 4.5, C['reward'], 0.38),
        # Cognitive (upper right)
        'Latent thought\nbuffer':          (7.0, 8.5, C['thought'], 0.42),
    }

    # Draw cluster backgrounds
    for (cx, cy, w, h, label) in [
        (2.8, 7.7, 4.5, 3.5, 'Visual modalities'),
        (2.2, 2.0, 3.5, 4.0, 'Embodiment'),
        (8.3, 6.0, 3.2, 2.5, 'Language'),
    ]:
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                               boxstyle="round,pad=0.3",
                               facecolor=C['bg_box'], edgecolor='#D5D8DC',
                               linewidth=1, linestyle='--', zorder=0)
        ax.add_patch(rect)
        ax.text(cx, cy + h/2 - 0.25, label, fontsize=8, ha='center',
                color='#95A5A6', fontfamily='serif', fontstyle='italic')

    # Draw nodes
    for label, (x, y, color, r) in mods.items():
        circle = plt.Circle((x, y), r, facecolor=color, edgecolor='white',
                             linewidth=2, alpha=0.85, zorder=10)
        ax.add_patch(circle)
        txt = ax.text(x, y, label, fontsize=7.5, ha='center', va='center',
                      color='white', fontweight='bold', zorder=11,
                      fontfamily='serif', linespacing=1.2)
        txt.set_path_effects([
            path_effects.withStroke(linewidth=1.5, foreground='black', alpha=0.25)
        ])

    # Distance annotations with lines
    distances = [
        # (from, to, distance, cost_label, color)
        ('RGB video\n224×224 30fps', 'Depth map\n224×224',
         0.12, '1-2 layers', C['line_close']),
        ('RGB video\n224×224 30fps', 'Optical flow\n224×224',
         0.18, '2-3 layers', C['line_close']),
        ('6-DOF EE\ndelta pose', '7-DOF joint\nangles 30Hz',
         0.15, '1-2 layers', C['line_close']),
        ('Natural language\n(English)', 'Natural language\n(French)',
         0.08, 'thin adapter', C['line_close']),
        ('RGB video\n224×224 30fps', '6-DOF EE\ndelta pose',
         0.65, 'full MLP', C['line_far']),
        ('RGB video\n224×224 30fps', 'Natural language\n(English)',
         0.78, 'deep adapter', C['line_far']),
        ('Scalar reward', '6-DOF EE\ndelta pose',
         0.42, 'small MLP', C['line_mid']),
    ]

    for from_label, to_label, dist, cost, color in distances:
        x0, y0 = mods[from_label][0], mods[from_label][1]
        x1, y1 = mods[to_label][0], mods[to_label][1]
        r0, r1 = mods[from_label][3], mods[to_label][3]

        # Shorten line to node edges
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / length, dy / length
        xs, ys = x0 + ux * r0, y0 + uy * r0
        xe, ye = x1 - ux * r1, y1 - uy * r1

        ax.plot([xs, xe], [ys, ye], color=color, lw=1.5, alpha=0.6,
                linestyle='-', zorder=5)

        # Label at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        # Offset perpendicular to the line
        px, py = -uy * 0.35, ux * 0.35
        ax.text(mx + px, my + py,
                f'd={dist:.2f}\n{cost}',
                fontsize=7, ha='center', va='center', color=color,
                fontfamily='serif', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.9, linewidth=0.8),
                zorder=15)

    # Legend
    ax.text(0.0, -0.2,
            'distance = 1 − cos(embed_a, embed_b)    |    '
            'closer = cheaper adapter    |    '
            'embedding_model: openai/text-embedding-3-small (fixed)',
            fontsize=8, color='#7F8C8D', fontfamily='serif',
            fontstyle='italic')

    plt.tight_layout()
    path = os.path.join(ASSETS, "transfer_distance.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print(f"  saved {path}")


def generate_schema_alignment():
    """Cross-schema region alignment between a robot agent and a computer agent.

    Shows two canvas schemas side-by-side with dotted lines connecting
    semantically compatible regions found by compatible_regions().
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 8), dpi=150,
                                             gridspec_kw={'wspace': 0.35})
    fig.patch.set_facecolor('white')
    fig.suptitle('Cross-Schema Region Alignment via Semantic Types',
                 fontsize=15, fontweight='bold', color=C['title'],
                 fontfamily='serif', y=0.97)

    def draw_schema(ax, title, regions, subtitle):
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, len(regions) * 1.6 + 1.0)
        ax.set_aspect('auto')
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold',
                     color=C['title'], fontfamily='serif', pad=10)
        ax.text(3.0, len(regions) * 1.6 + 0.5, subtitle,
                fontsize=8, ha='center', color='#95A5A6',
                fontfamily='monospace')

        boxes = {}
        for i, (name, spec_text, semantic, color, period) in enumerate(reversed(regions)):
            y = i * 1.6 + 0.3
            # Region box
            rect = FancyBboxPatch((0.1, y), 5.8, 1.2,
                                   boxstyle="round,pad=0.12",
                                   facecolor=color, edgecolor=color,
                                   linewidth=1.5, alpha=0.15, zorder=2)
            ax.add_patch(rect)
            # Color bar on left
            bar = plt.Rectangle((0.1, y), 0.25, 1.2, facecolor=color,
                                 alpha=0.7, zorder=3)
            ax.add_patch(bar)
            # Region name
            ax.text(0.6, y + 0.85, name, fontsize=11, fontweight='bold',
                    color=C['text'], fontfamily='serif', va='center')
            # Spec
            ax.text(0.6, y + 0.45, spec_text, fontsize=7.5,
                    color='#7F8C8D', fontfamily='monospace', va='center')
            # Semantic type
            ax.text(0.6, y + 0.15, f'semantic: "{semantic}"',
                    fontsize=7, color=color, fontfamily='serif',
                    fontstyle='italic', va='center', alpha=0.85)
            # Period badge
            if period != 1:
                ax.text(5.5, y + 0.85, f'period={period}',
                        fontsize=7, color='#7F8C8D', fontfamily='monospace',
                        ha='right', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='#F0F0F0',
                                  edgecolor='#D0D0D0', linewidth=0.5))

            boxes[name] = (y + 0.6)  # center y for alignment lines

        return boxes

    # Robot manipulation agent
    robot_regions = [
        ('perception.visual', 'bounds=(0,16, 0,24, 0,24), loss_weight=1.0',
         'RGB video 224×224 30fps front monocular camera', C['visual'], 1),
        ('perception.proprio', 'bounds=(0,16, 24,26, 0,4), loss_weight=2.0',
         '7-DOF joint positions + velocities 30Hz', C['proprio'], 1),
        ('memory.episodic', 'bounds=(0,4, 26,28, 0,8), is_output=False',
         'episodic memory buffer, compressed observations', C['memory'], 4),
        ('policy.active', 'bounds=(0,16, 28,30, 0,4), loss_weight=2.0',
         '6-DOF end-effector delta pose + gripper', C['action'], 1),
        ('goal.current', 'bounds=(0,1, 30,32, 0,8), is_output=False',
         'natural language task instruction', C['language'], 16),
    ]

    # Computer use agent
    computer_regions = [
        ('perception.screen', 'bounds=(0,16, 0,24, 0,24), loss_weight=1.0',
         'RGB video 224×224 30fps screen capture', C['visual'], 1),
        ('perception.mouse', 'bounds=(0,16, 24,26, 0,2), loss_weight=2.0',
         'mouse position + button state 30Hz', C['proprio'], 1),
        ('memory.working', 'bounds=(0,2, 26,28, 0,8), is_output=False',
         'working memory, recent context window', C['memory'], 8),
        ('policy.active', 'bounds=(0,16, 28,30, 0,4), loss_weight=2.0',
         'mouse click/move + keyboard action sequence', C['action'], 1),
        ('goal.current', 'bounds=(0,1, 30,32, 0,8), is_output=False',
         'natural language user instruction', C['language'], 16),
    ]

    robot_boxes = draw_schema(ax_left, 'Robot Manipulation Agent',
                               robot_regions, 'CanvasSchema "robot_v1"')
    computer_boxes = draw_schema(ax_right, 'Computer Use Agent',
                                  computer_regions, 'CanvasSchema "computer_v1"')

    # Draw alignment lines between compatible regions
    alignments = [
        # (robot_region, computer_region, distance, color)
        ('perception.visual', 'perception.screen', 0.04, C['line_close']),
        ('perception.proprio', 'perception.mouse', 0.22, C['line_mid']),
        ('memory.episodic', 'memory.working', 0.11, C['line_close']),
        ('policy.active', 'policy.active', 0.35, C['line_mid']),
        ('goal.current', 'goal.current', 0.02, C['line_close']),
    ]

    for robot_name, comp_name, dist, color in alignments:
        ry = robot_boxes[robot_name]
        cy = computer_boxes[comp_name]

        # Use ConnectionPatch for cross-axes lines
        con = ConnectionPatch(
            xyA=(6.3, ry), coordsA=ax_left.transData,
            xyB=(-0.3, cy), coordsB=ax_right.transData,
            arrowstyle='-', color=color, lw=2.0,
            linestyle='--' if dist > 0.3 else '-',
            alpha=0.7, zorder=20,
        )
        fig.add_artist(con)

        # Distance label in the middle
        # We'll place it using figure coordinates
        mid_x = 0.5
        # Approximate y from data coords
        ry_fig = (ry - (-0.5)) / (len(robot_regions) * 1.6 + 1.5)
        cy_fig = (cy - (-0.5)) / (len(computer_regions) * 1.6 + 1.5)
        mid_y_data = (ry + cy) / 2

        fig.text(0.5, 0.15 + 0.65 * (mid_y_data - 0) / (len(robot_regions) * 1.6 + 1.0),
                 f'd={dist:.2f}',
                 fontsize=8, ha='center', va='center', color=color,
                 fontweight='bold', fontfamily='serif',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                           edgecolor=color, linewidth=0.8, alpha=0.95),
                 zorder=25)

    # Legend at bottom
    fig.text(0.5, 0.02,
             'Solid lines: d ≤ 0.3 (direct latent transfer possible)    '
             '|    Dashed lines: d > 0.3 (adapter layers needed)    '
             '|    compatible_regions(threshold=0.4)',
             fontsize=8, ha='center', color='#7F8C8D',
             fontfamily='serif', fontstyle='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = os.path.join(ASSETS, "schema_alignment.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print(f"  saved {path}")


if __name__ == "__main__":
    print("Generating semantic diagrams...")
    generate_transfer_distance()
    generate_schema_alignment()
    print("Done.")

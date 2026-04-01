"""Visualize browser agent rollout as an animated GIF.

Renders the agent's interaction with the synthetic browser environment
step-by-step, showing the page state, agent's action, and task progress.

Usage:
    python research/browser/visualize_rollout.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from environment import BrowserEnvironment, generate_demonstrations
from browser_canvas import build_browser_program
from train import BrowserCanvasModel

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTION_NAMES = ["click", "scroll", "type", "navigate", "wait", "done"]
ELEMENT_TYPES = ["button", "input", "link", "label", "div"]
ELEMENT_COLORS = {
    "button": "#3498DB",
    "input": "#2ECC71",
    "link": "#9B59B6",
    "label": "#95A5A6",
    "div": "#BDC3C7",
}


def render_page(ax, env, step_info=None):
    """Render the current page state on a matplotlib axis."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Background
    ax.set_facecolor("#F8F9FA")

    # Draw elements from the environment's current page
    page = env.current_page
    elements = page.elements if hasattr(page, "elements") else []

    for elem in elements:
        etype = getattr(elem, "type", "div") if not isinstance(elem, dict) else elem.get("type", "div")
        x = getattr(elem, "x", 0.5) if not isinstance(elem, dict) else elem.get("x", 0.5)
        y = getattr(elem, "y", 0.5) if not isinstance(elem, dict) else elem.get("y", 0.5)
        w = getattr(elem, "width", 0.15) if not isinstance(elem, dict) else elem.get("w", 0.15)
        h = getattr(elem, "height", 0.06) if not isinstance(elem, dict) else elem.get("h", 0.06)
        text = getattr(elem, "text", etype) if not isinstance(elem, dict) else elem.get("text", etype)
        visible = getattr(elem, "visible", True) if not isinstance(elem, dict) else elem.get("visible", True)

        if not visible:
            continue

        color = ELEMENT_COLORS.get(etype, "#BDC3C7")
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")

    # Draw action if provided
    if step_info:
        action = step_info.get("action_name", "")
        target_x = step_info.get("target_x", 0.5)
        target_y = step_info.get("target_y", 0.5)

        if action == "click":
            ax.plot(target_x, target_y, "rx", markersize=15, markeredgewidth=3)
            circle = plt.Circle((target_x, target_y), 0.03,
                                fill=False, color="red", linewidth=2, linestyle="--")
            ax.add_patch(circle)
        elif action == "type":
            ax.annotate(f'type: "{step_info.get("text", "")}"',
                        (target_x, target_y), fontsize=8, color="green",
                        fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        elif action == "scroll":
            ax.annotate("scroll", (0.5, 0.5), fontsize=10, color="blue",
                        fontweight="bold", ha="center")

    ax.set_title(step_info.get("title", "Browser") if step_info else "Browser",
                 fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def run_rollout_with_recording(model, env, task_type=None, max_steps=8):
    """Run a rollout and record each step for visualization."""
    model.eval()
    frames = []

    obs, task = env.reset(task_type=task_type)

    task_desc = getattr(task, 'description', str(task_type or 'random'))

    frames.append({
        "title": "Task: {}".format(task_desc),
        "action_name": "start",
        "target_x": 0.5,
        "target_y": 0.5,
        "reward": 0,
        "step": 0,
        "done": False,
    })

    instruction = torch.zeros(1, 32)
    type_idx = ACTION_NAMES.index(task_type) if task_type in ACTION_NAMES else 0
    instruction[0, type_idx] = 1.0

    total_reward = 0
    for step in range(max_steps):
        with torch.no_grad():
            screen_t = torch.tensor(obs.screen, dtype=torch.float32).unsqueeze(0) if hasattr(obs, 'screen') else torch.randn(1, 4, 14, 14)
            dom_elem_t = torch.tensor(obs.dom_elements, dtype=torch.float32).unsqueeze(0) if hasattr(obs, 'dom_elements') else torch.randn(1, 16, 12)
            dom_layout_t = torch.tensor(obs.dom_layout, dtype=torch.float32).unsqueeze(0) if hasattr(obs, 'dom_layout') else torch.randn(1, 16, 4)

            outputs = model(screen_t, dom_elem_t, dom_layout_t, instruction, external_t=step)

        action_logits = outputs["action_logits"]
        action_idx = action_logits.argmax(dim=-1).item()
        action_name = ACTION_NAMES[min(action_idx, len(ACTION_NAMES) - 1)]

        coord_pred = outputs.get("coord_pred", torch.tensor([[0.5, 0.5]]))
        target_x = float(coord_pred[0, 0].clamp(0, 1))
        target_y = float(coord_pred[0, 1].clamp(0, 1))

        text = "hello" if action_name == "type" else ""

        action = {"type": action_idx, "x": target_x, "y": target_y, "text": text}
        obs, reward, done, info = env.step(action)
        total_reward += reward

        frames.append({
            "title": "Step {}: {} ({:.2f}, {:.2f})".format(step+1, action_name, target_x, target_y),
            "action_name": action_name,
            "target_x": target_x,
            "target_y": target_y,
            "text": text,
            "reward": reward,
            "total_reward": total_reward,
            "step": step + 1,
            "done": done,
        })

        if done:
            break

    return frames, total_reward


def create_rollout_gif(model, n_episodes=6, output_path=None):
    """Create an animated GIF showing the agent's rollouts."""
    if output_path is None:
        output_path = RESULTS_DIR / "browser_rollout.gif"

    env = BrowserEnvironment()
    task_types = ["click", "type", "navigate", "click", "type", "navigate"]

    all_rollouts = []
    for i, ttype in enumerate(task_types[:n_episodes]):
        frames, reward = run_rollout_with_recording(model, env, task_type=ttype)
        all_rollouts.append({
            "frames": frames,
            "reward": reward,
            "task_type": ttype,
            "task": task,
        })

    # Create figure with grid of rollouts
    n_cols = min(3, n_episodes)
    n_rows = (n_episodes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle("Browser Agent Rollouts", fontsize=16, fontweight="bold")

    # Find max number of frames across all rollouts
    max_frames = max(len(r["frames"]) for r in all_rollouts)

    def update(frame_idx):
        for i, rollout in enumerate(all_rollouts):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            frames = rollout["frames"]
            fidx = min(frame_idx, len(frames) - 1)
            frame = frames[fidx]

            render_page(ax, env, step_info=frame)

            # Add reward info
            if "total_reward" in frame:
                reward_text = f"R={frame['total_reward']:.2f}"
                color = "green" if frame["total_reward"] > 0 else "red"
                ax.text(0.98, 0.02, reward_text, transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=9, fontweight="bold",
                        color=color, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            if frame.get("done"):
                ax.text(0.5, 0.5, "DONE", transform=ax.transAxes,
                        ha="center", va="center", fontsize=20, fontweight="bold",
                        color="green" if frame.get("total_reward", 0) > 0 else "gray",
                        alpha=0.5)

    # Create animation
    anim = FuncAnimation(fig, update, frames=max_frames, interval=1000, repeat=True)
    anim.save(str(output_path), writer=PillowWriter(fps=1))
    plt.close(fig)
    print(f"Saved rollout GIF: {output_path}")
    return output_path


def main():
    print("Building canvas model...")
    bound, program = build_browser_program()
    model = BrowserCanvasModel(d_model=128, n_layers=4)

    # Try to load checkpoint
    ckpt_path = RESULTS_DIR / "checkpoint_canvas.pt"
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    else:
        print("No checkpoint found — using random weights for visualization")

    print("Generating rollout GIF...")
    create_rollout_gif(model, n_episodes=6)
    print("Done!")


if __name__ == "__main__":
    main()

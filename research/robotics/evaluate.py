"""Evaluation and visualization for multi-robot fleet experiments.

Generates:
  1. Formation error over time (all three models)
  2. Collision rate comparison
  3. Communication emergence analysis
  4. Attention weight visualization
  5. Animated trajectory GIF
  6. Scaling analysis: 2, 4, 8, 16 robots

Usage:
    from research.robotics.evaluate import run_evaluation
    run_evaluation(models, env, results_dir="research/robotics/results/")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

import torch
import torch.nn.functional as F

RESULTS_DIR = Path(_CE_ROOT) / "research" / "robotics" / "results"


# ---- Evaluation rollouts -------------------------------------------


def evaluate_model(
    model,
    env,
    n_episodes: int = 10,
    max_steps: int = 200,
    record_trajectories: bool = False,
) -> Dict:
    """Run evaluation episodes and collect metrics.

    Returns dict with:
      formation_errors: (n_episodes, max_steps) distance to goal
      collision_counts: (n_episodes,) total collisions per episode
      total_rewards: (n_episodes,) cumulative rewards
      messages: (n_episodes, max_steps, n_robots, 4) broadcast messages
      trajectories: optional (n_episodes, max_steps, n_robots, 2) positions
    """
    from research.robotics.environment import MultiRobotEnv, EnvConfig

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()

    n_robots = env.cfg.n_robots
    single_cfg = EnvConfig(
        n_robots=n_robots, n_envs=1, task=env.cfg.task,
        max_steps=max_steps, n_obstacles=env.cfg.n_obstacles,
    )
    eval_env = MultiRobotEnv(single_cfg)
    eval_env.obstacle_pos = env.obstacle_pos.copy()
    eval_env.obstacle_radii = env.obstacle_radii.copy()

    all_formation_errors = []
    all_collisions = []
    all_rewards = []
    all_messages = []
    all_trajectories = []

    with torch.no_grad():
        for ep in range(n_episodes):
            obs = eval_env.reset()
            ep_errors = []
            ep_collisions = 0.0
            ep_reward = 0.0
            ep_messages = []
            ep_traj = []

            for t in range(max_steps):
                lidar_t = torch.tensor(obs["lidar"], dtype=torch.float32, device=device)
                pos_t = torch.tensor(obs["positions"], dtype=torch.float32, device=device)
                vel_t = torch.tensor(obs["velocities"], dtype=torch.float32, device=device)
                goal_t = torch.tensor(obs["goal"], dtype=torch.float32, device=device)
                form_t = torch.tensor(obs["formation"], dtype=torch.float32, device=device)

                vel_cmds, messages = model(
                    lidar_t, pos_t, vel_t, goal_t, form_t, step=t,
                )

                actions_np = vel_cmds.cpu().numpy()
                obs, rewards, dones, info = eval_env.step(actions_np)

                # Track formation error
                goal_dist = info.get("mean_goal_dist", 0.0)
                ep_errors.append(float(goal_dist))
                ep_collisions += info.get("collisions", 0)
                ep_reward += float(rewards.mean())
                ep_messages.append(messages[0].cpu().numpy())

                if record_trajectories:
                    ep_traj.append(eval_env.positions[0].copy())

                if dones[0]:
                    # Pad remaining steps
                    remaining = max_steps - t - 1
                    ep_errors.extend([ep_errors[-1]] * remaining)
                    for _ in range(remaining):
                        ep_messages.append(ep_messages[-1])
                        if record_trajectories:
                            ep_traj.append(ep_traj[-1])
                    break

            all_formation_errors.append(ep_errors)
            all_collisions.append(ep_collisions)
            all_rewards.append(ep_reward)
            all_messages.append(np.array(ep_messages))
            if record_trajectories:
                all_trajectories.append(np.array(ep_traj))

    result = {
        "formation_errors": np.array(all_formation_errors),
        "collision_counts": np.array(all_collisions),
        "total_rewards": np.array(all_rewards),
        "messages": np.array(all_messages),
    }
    if record_trajectories:
        result["trajectories"] = np.array(all_trajectories)
    return result


# ---- Plot generation ------------------------------------------------


def _ensure_matplotlib():
    """Import matplotlib with non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_formation_error(
    eval_results: Dict[str, Dict],
    save_path: str,
):
    """Plot 1: Formation error over time for all three models."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {"canvas": "#2196F3", "dense": "#FF9800", "independent": "#4CAF50"}
    labels = {"canvas": "Canvas Fleet", "dense": "Dense Fleet", "independent": "Independent"}

    for mode, results in eval_results.items():
        errors = results["formation_errors"]
        mean_err = errors.mean(axis=0)
        std_err = errors.std(axis=0)
        steps = np.arange(len(mean_err))

        color = colors.get(mode, "#999")
        ax.plot(steps, mean_err, color=color, label=labels.get(mode, mode), linewidth=2)
        ax.fill_between(steps, mean_err - std_err, mean_err + std_err,
                        color=color, alpha=0.2)

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Mean Distance to Goal (m)", fontsize=12)
    ax.set_title("Formation Error Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print("  Saved: {}".format(save_path))


def plot_collision_comparison(
    eval_results: Dict[str, Dict],
    save_path: str,
):
    """Plot 2: Collision rate comparison bar chart."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    modes = list(eval_results.keys())
    means = [eval_results[m]["collision_counts"].mean() for m in modes]
    stds = [eval_results[m]["collision_counts"].std() for m in modes]

    colors = ["#2196F3", "#FF9800", "#4CAF50"][:len(modes)]
    labels = {"canvas": "Canvas Fleet", "dense": "Dense Fleet", "independent": "Independent"}

    bars = ax.bar(
        [labels.get(m, m) for m in modes], means,
        yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5,
    )
    ax.set_ylabel("Total Collisions per Episode", fontsize=12)
    ax.set_title("Collision Rate Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                "{:.1f}".format(mean), ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print("  Saved: {}".format(save_path))


def plot_communication_analysis(
    eval_results: Dict[str, Dict],
    save_path: str,
):
    """Plot 3: Communication emergence analysis.

    Shows:
      - Message variance over time (are messages becoming more informative?)
      - PCA of message space (do robots learn distinct message types?)
    """
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: message variance over time for canvas model
    if "canvas" in eval_results:
        messages = eval_results["canvas"]["messages"]  # (n_ep, steps, n_robots, 4)
        # Message variance across robots at each timestep
        msg_var = messages.var(axis=2).mean(axis=(0, 2))  # (steps,)
        axes[0].plot(msg_var, color="#2196F3", linewidth=2)
        axes[0].set_xlabel("Timestep", fontsize=12)
        axes[0].set_ylabel("Inter-Robot Message Variance", fontsize=12)
        axes[0].set_title("Communication Diversity Over Time", fontsize=13)
        axes[0].grid(True, alpha=0.3)

    # Right: PCA of message embeddings (canvas vs dense)
    for mode, color, marker in [("canvas", "#2196F3", "o"), ("dense", "#FF9800", "s")]:
        if mode not in eval_results:
            continue
        messages = eval_results[mode]["messages"]
        # Flatten: (n_ep * steps * n_robots, 4)
        flat_msgs = messages.reshape(-1, messages.shape[-1])

        # Simple PCA via SVD
        centered = flat_msgs - flat_msgs.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            pc = centered @ Vt[:2].T  # project onto top 2 PCs

            # Subsample for plot clarity
            n_plot = min(1000, len(pc))
            idx = np.random.choice(len(pc), n_plot, replace=False)
            labels = {"canvas": "Canvas Fleet", "dense": "Dense Fleet"}
            axes[1].scatter(
                pc[idx, 0], pc[idx, 1],
                c=color, alpha=0.3, s=8, marker=marker,
                label=labels.get(mode, mode),
            )
        except np.linalg.LinAlgError:
            pass

    axes[1].set_xlabel("PC1", fontsize=12)
    axes[1].set_ylabel("PC2", fontsize=12)
    axes[1].set_title("Message Space (PCA)", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print("  Saved: {}".format(save_path))


def plot_attention_weights(
    models: Dict[str, object],
    save_path: str,
):
    """Plot 4: Attention weight heatmap for the canvas model.

    Shows which regions attend to which, revealing the learned
    information flow topology.
    """
    plt = _ensure_matplotlib()

    if "canvas" not in models:
        print("  Skipping attention visualization (no canvas model)")
        return

    model = models["canvas"]

    # Get the attention topology structure
    topology = model.bound.schema.topology
    layout = model.bound.schema.layout

    # Build adjacency matrix from topology
    region_names = sorted(layout.regions.keys())
    n_regions = len(region_names)
    name_to_idx = {name: i for i, name in enumerate(region_names)}

    adj = np.zeros((n_regions, n_regions))
    for conn in topology.connections:
        if conn.src in name_to_idx and conn.dst in name_to_idx:
            i = name_to_idx[conn.src]
            j = name_to_idx[conn.dst]
            adj[i, j] = max(adj[i, j], conn.weight)

    # Shorten names for display
    short_names = []
    for name in region_names:
        parts = name.split(".")
        if len(parts) > 2:
            short = "..".join([parts[-2][:3], parts[-1][:6]])
        elif len(parts) > 1:
            short = "..".join([p[:6] for p in parts[-2:]])
        else:
            short = name[:10]
        short_names.append(short)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(adj, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_xlabel("Key/Value Region (dst)", fontsize=11)
    ax.set_ylabel("Query Region (src)", fontsize=11)
    ax.set_title("Canvas Fleet Attention Topology", fontsize=14)
    fig.colorbar(im, ax=ax, label="Connection Weight")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print("  Saved: {}".format(save_path))


def create_trajectory_gif(
    trajectories: np.ndarray,
    obstacle_pos: np.ndarray,
    obstacle_radii: np.ndarray,
    goal_positions: np.ndarray,
    world_size: float,
    save_path: str,
    fps: int = 10,
    max_frames: int = 100,
):
    """Plot 5: Animated trajectory GIF showing robots moving in formation.

    Args:
        trajectories: (n_steps, n_robots, 2) positions
        obstacle_pos: (n_obstacles, 2)
        obstacle_radii: (n_obstacles,)
        goal_positions: (n_robots, 2)
        world_size: float
        save_path: output GIF path
    """
    plt = _ensure_matplotlib()
    from matplotlib.patches import Circle
    import matplotlib.animation as animation

    n_steps = min(len(trajectories), max_frames)
    n_robots = trajectories.shape[1]

    robot_colors = plt.cm.Set1(np.linspace(0, 1, max(n_robots, 2)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    def draw_frame(t):
        ax.clear()
        ax.set_xlim(-0.5, world_size + 0.5)
        ax.set_ylim(-0.5, world_size + 0.5)
        ax.set_aspect("equal")
        ax.set_title("Robot Fleet Formation (t={})".format(t), fontsize=14)
        ax.grid(True, alpha=0.2)

        # Draw obstacles
        for o_idx in range(len(obstacle_pos)):
            circle = Circle(
                obstacle_pos[o_idx], obstacle_radii[o_idx],
                color="gray", alpha=0.5, zorder=1,
            )
            ax.add_patch(circle)

        # Draw goal positions
        for i in range(n_robots):
            ax.plot(goal_positions[i, 0], goal_positions[i, 1],
                    "x", color=robot_colors[i], markersize=12, markeredgewidth=2,
                    zorder=2)

        # Draw trajectories (trail)
        trail_start = max(0, t - 20)
        for i in range(n_robots):
            trail = trajectories[trail_start:t+1, i]
            ax.plot(trail[:, 0], trail[:, 1],
                    "-", color=robot_colors[i], alpha=0.4, linewidth=1, zorder=3)

        # Draw current robot positions
        for i in range(n_robots):
            pos = trajectories[t, i]
            circle = Circle(pos, 0.3, color=robot_colors[i], alpha=0.8, zorder=4)
            ax.add_patch(circle)
            ax.annotate("R{}".format(i), pos, ha="center", va="center",
                       fontsize=8, fontweight="bold", color="white", zorder=5)

    # Create animation
    anim = animation.FuncAnimation(
        fig, draw_frame, frames=n_steps, interval=1000 // fps,
    )

    try:
        anim.save(save_path, writer="pillow", fps=fps)
        print("  Saved: {}".format(save_path))
    except Exception as e:
        # Fallback: save key frames as static image
        print("  GIF save failed ({}), saving key frames instead".format(e))
        fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
        key_frames = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
        for ax2, frame_idx in zip(axes2, key_frames):
            ax2.set_xlim(-0.5, world_size + 0.5)
            ax2.set_ylim(-0.5, world_size + 0.5)
            ax2.set_aspect("equal")
            ax2.set_title("t={}".format(frame_idx))
            ax2.grid(True, alpha=0.2)

            for o_idx in range(len(obstacle_pos)):
                circle = Circle(
                    obstacle_pos[o_idx], obstacle_radii[o_idx],
                    color="gray", alpha=0.5,
                )
                ax2.add_patch(circle)

            for i in range(n_robots):
                ax2.plot(goal_positions[i, 0], goal_positions[i, 1],
                        "x", color=robot_colors[i], markersize=10, markeredgewidth=2)

                trail_start = max(0, frame_idx - 20)
                trail = trajectories[trail_start:frame_idx+1, i]
                ax2.plot(trail[:, 0], trail[:, 1],
                        "-", color=robot_colors[i], alpha=0.4, linewidth=1)

                pos = trajectories[frame_idx, i]
                circle = Circle(pos, 0.3, color=robot_colors[i], alpha=0.8)
                ax2.add_patch(circle)

        fig2.suptitle("Robot Fleet Trajectory Key Frames", fontsize=14)
        fig2.tight_layout()
        static_path = save_path.replace(".gif", "_keyframes.png")
        fig2.savefig(static_path, dpi=150)
        plt.close(fig2)
        print("  Saved: {}".format(static_path))

    plt.close(fig)


def plot_scaling_analysis(
    scaling_results: Dict[int, Dict[str, Dict]],
    save_path: str,
):
    """Plot 6: Scaling analysis showing performance vs number of robots.

    Shows how formation error and collision rate change as fleet size grows.
    """
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"canvas": "#2196F3", "dense": "#FF9800", "independent": "#4CAF50"}
    labels = {"canvas": "Canvas Fleet", "dense": "Dense Fleet", "independent": "Independent"}

    robot_counts = sorted(scaling_results.keys())

    for mode in ["canvas", "dense", "independent"]:
        final_errors = []
        collision_rates = []

        for n in robot_counts:
            if mode in scaling_results[n]:
                res = scaling_results[n][mode]
                # Final formation error (mean of last 20% of steps)
                errors = res["formation_errors"]
                tail_len = max(1, errors.shape[1] // 5)
                final_err = errors[:, -tail_len:].mean()
                final_errors.append(final_err)
                collision_rates.append(res["collision_counts"].mean())
            else:
                final_errors.append(np.nan)
                collision_rates.append(np.nan)

        color = colors.get(mode, "#999")
        label = labels.get(mode, mode)

        axes[0].plot(robot_counts, final_errors, "o-", color=color,
                    label=label, linewidth=2, markersize=8)
        axes[1].plot(robot_counts, collision_rates, "o-", color=color,
                    label=label, linewidth=2, markersize=8)

    axes[0].set_xlabel("Number of Robots", fontsize=12)
    axes[0].set_ylabel("Final Formation Error (m)", fontsize=12)
    axes[0].set_title("Formation Error vs Fleet Size", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Number of Robots", fontsize=12)
    axes[1].set_ylabel("Collisions per Episode", fontsize=12)
    axes[1].set_title("Collision Rate vs Fleet Size", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print("  Saved: {}".format(save_path))


# ---- Scaling experiment ---------------------------------------------


def run_scaling_experiment(
    robot_counts: List[int] = [2, 4, 8, 16],
    eval_episodes: int = 5,
    max_eval_steps: int = 100,
    d_model: int = 64,
    imitation_epochs: int = 10,
    selfplay_episodes: int = 15,
    n_expert_episodes: int = 30,
) -> Dict[int, Dict[str, Dict]]:
    """Run scaling analysis: train and evaluate for different fleet sizes.

    Returns nested dict: {n_robots: {mode: eval_results}}.
    """
    from research.robotics.train import TrainConfig, train_single_model
    from research.robotics.environment import (
        MultiRobotEnv, EnvConfig, PotentialFieldController, generate_expert_demos,
    )

    scaling_results = {}

    for n in robot_counts:
        print("\n=== Scaling: {} robots ===".format(n))
        scaling_results[n] = {}

        env_cfg = EnvConfig(n_robots=n, n_envs=1, task="formation", max_steps=max_eval_steps)
        env = MultiRobotEnv(env_cfg)

        controller = PotentialFieldController()
        demos = generate_expert_demos(
            env, controller, n_episodes=n_expert_episodes, max_steps=max_eval_steps,
        )

        config = TrainConfig(
            n_robots=n, d_model=d_model,
            imitation_epochs=imitation_epochs,
            selfplay_episodes=selfplay_episodes,
            n_expert_episodes=n_expert_episodes,
            selfplay_steps=max_eval_steps,
        )

        for mode in ["canvas", "dense", "independent"]:
            print("  Training {} with {} robots...".format(mode, n))
            try:
                model, _ = train_single_model(mode, demos, env, config)
                eval_res = evaluate_model(
                    model, env,
                    n_episodes=eval_episodes,
                    max_steps=max_eval_steps,
                )
                scaling_results[n][mode] = eval_res
            except Exception as e:
                print("    Failed: {}".format(e))

    return scaling_results


# ---- Main evaluation entry point ------------------------------------


def run_evaluation(
    models: Dict[str, object],
    env,
    results_dir: Optional[str] = None,
    n_eval_episodes: int = 10,
    max_eval_steps: int = 200,
    run_scaling: bool = True,
    scaling_counts: Optional[List[int]] = None,
):
    """Run full evaluation pipeline and generate all plots.

    Args:
        models: dict mapping mode name to FleetModel
        env: MultiRobotEnv instance
        results_dir: output directory
        n_eval_episodes: episodes per model for evaluation
        max_eval_steps: max steps per episode
        run_scaling: whether to run scaling experiment
        scaling_counts: robot counts for scaling experiment
    """
    if results_dir is None:
        results_dir = str(RESULTS_DIR)
    rdir = Path(results_dir)
    rdir.mkdir(parents=True, exist_ok=True)

    if scaling_counts is None:
        scaling_counts = [2, 4, 8]

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    # Evaluate all models
    eval_results = {}
    for mode, model in models.items():
        print("\nEvaluating {} model...".format(mode))
        eval_results[mode] = evaluate_model(
            model, env,
            n_episodes=n_eval_episodes,
            max_steps=max_eval_steps,
            record_trajectories=(mode == "canvas"),
        )
        print("  Mean formation error: {:.3f}".format(
            eval_results[mode]["formation_errors"][:, -1].mean()))
        print("  Mean collisions: {:.1f}".format(
            eval_results[mode]["collision_counts"].mean()))
        print("  Mean reward: {:.1f}".format(
            eval_results[mode]["total_rewards"].mean()))

    # Plot 1: Formation error
    print("\nGenerating plots...")
    plot_formation_error(eval_results, str(rdir / "formation_error.png"))

    # Plot 2: Collision comparison
    plot_collision_comparison(eval_results, str(rdir / "collision_comparison.png"))

    # Plot 3: Communication analysis
    plot_communication_analysis(eval_results, str(rdir / "communication_analysis.png"))

    # Plot 4: Attention weights
    plot_attention_weights(models, str(rdir / "attention_topology.png"))

    # Plot 5: Trajectory GIF
    if "canvas" in eval_results and "trajectories" in eval_results["canvas"]:
        traj = eval_results["canvas"]["trajectories"]
        if len(traj) > 0:
            # Use first episode
            from research.robotics.environment import MultiRobotEnv, EnvConfig
            single_cfg = EnvConfig(n_robots=env.cfg.n_robots, n_envs=1, task=env.cfg.task)
            single_env = MultiRobotEnv(single_cfg)
            single_env.obstacle_pos = env.obstacle_pos.copy()
            single_env.obstacle_radii = env.obstacle_radii.copy()
            single_env.reset()

            create_trajectory_gif(
                trajectories=traj[0],
                obstacle_pos=env.obstacle_pos,
                obstacle_radii=env.obstacle_radii,
                goal_positions=single_env.goal_positions[0],
                world_size=env.cfg.world_size,
                save_path=str(rdir / "trajectory.gif"),
                fps=10,
                max_frames=min(max_eval_steps, 100),
            )

    # Plot 6: Scaling analysis
    if run_scaling:
        print("\nRunning scaling analysis with {} robots...".format(scaling_counts))
        scaling_results = run_scaling_experiment(
            robot_counts=scaling_counts,
            eval_episodes=5,
            max_eval_steps=100,
            imitation_epochs=8,
            selfplay_episodes=10,
            n_expert_episodes=20,
        )
        plot_scaling_analysis(scaling_results, str(rdir / "scaling_analysis.png"))

        # Save scaling data
        scaling_data = {}
        for n_robots, mode_results in scaling_results.items():
            scaling_data[str(n_robots)] = {}
            for mode, res in mode_results.items():
                scaling_data[str(n_robots)][mode] = {
                    "final_formation_error": float(res["formation_errors"][:, -1].mean()),
                    "mean_collisions": float(res["collision_counts"].mean()),
                    "mean_reward": float(res["total_rewards"].mean()),
                }
        with open(rdir / "scaling_data.json", "w") as f:
            json.dump(scaling_data, f, indent=2)

    # Save evaluation summary
    eval_summary = {}
    for mode, res in eval_results.items():
        eval_summary[mode] = {
            "final_formation_error": float(res["formation_errors"][:, -1].mean()),
            "mean_collisions": float(res["collision_counts"].mean()),
            "mean_reward": float(res["total_rewards"].mean()),
        }
    with open(rdir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("\nEvaluation complete. Results saved to {}".format(rdir))
    return eval_results

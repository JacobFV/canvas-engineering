"""Train multi-robot fleet models with three objectives.

Training pipeline:
  1. Supervised imitation from potential field expert demonstrations
  2. Self-play with environment reward (REINFORCE-style policy gradient)
  3. Communication emergence through the 'communicate' action field

Three model comparisons:
  1. Canvas fleet: structured topology with coarse-grained inter-robot comms
  2. Dense fleet: fully connected (no bottleneck)
  3. Independent: each robot acts alone

Uses compile_program() with families for auto-wired operators.
Uses RegionScheduler: plan only updates every 4 steps, sensor every step.
Logs JSONL metrics per step.

Usage:
    from research.robotics.train import train_all_models
    results = train_all_models(n_robots=4, n_epochs=30)
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

from canvas_engineering import (
    RegionScheduler,
    ClockSpec,
    RegionProgram,
)

from research.robotics.robot_canvas import (
    build_fleet_program,
    FleetModel,
)
from research.robotics.environment import (
    MultiRobotEnv,
    EnvConfig,
    PotentialFieldController,
    generate_expert_demos,
)

RESULTS_DIR = Path(_CE_ROOT) / "research" / "robotics" / "results"


# ---- Dataset -------------------------------------------------------


class ExpertDataset(Dataset):
    """Dataset of expert demonstrations."""

    def __init__(self, demos: Dict[str, np.ndarray]):
        self.lidar = torch.tensor(demos["lidar"], dtype=torch.float32)
        self.positions = torch.tensor(demos["positions"], dtype=torch.float32)
        self.velocities = torch.tensor(demos["velocities"], dtype=torch.float32)
        self.goal = torch.tensor(demos["goal"], dtype=torch.float32)
        self.formation = torch.tensor(demos["formation"], dtype=torch.float32)
        self.actions = torch.tensor(demos["actions"], dtype=torch.float32)

    def __len__(self):
        return len(self.lidar)

    def __getitem__(self, idx):
        return {
            "lidar": self.lidar[idx],
            "positions": self.positions[idx],
            "velocities": self.velocities[idx],
            "goal": self.goal[idx],
            "formation": self.formation[idx],
            "actions": self.actions[idx],
        }


# ---- Training loop --------------------------------------------------


@dataclass
class TrainConfig:
    """Training configuration."""
    n_robots: int = 4
    d_model: int = 64
    n_heads: int = 4
    batch_size: int = 32
    lr: float = 1e-3
    imitation_epochs: int = 20
    selfplay_episodes: int = 30
    selfplay_steps: int = 100
    n_expert_episodes: int = 50
    expert_max_steps: int = 200
    n_envs_train: int = 16
    comm_loss_weight: float = 0.1
    device: str = "cpu"
    log_file: Optional[str] = None


def _add_scheduling_clocks(program):
    """Add scheduling clocks to the CanvasProgram.

    Plan regions update every 4 steps (slow planning loop).
    Sensor regions update every step (fast perception loop).
    Other regions update every step.
    """
    from canvas_engineering.program import CanvasProgram, RegionProgram, ClockSpec

    new_regions = {}
    for name, rp in program.regions.items():
        if "plan" in name.lower():
            new_regions[name] = RegionProgram(
                family=rp.family,
                tags=rp.tags,
                carrier=rp.carrier,
                clock=ClockSpec(period=4),
                learning=rp.learning,
                compile_mode=rp.compile_mode,
            )
        else:
            new_regions[name] = rp

    return CanvasProgram(
        schema=program.schema,
        regions=new_regions,
        connections=program.connections,
        version=program.version,
    )


def _log_metric(log_file, step, phase, metrics):
    """Append a JSONL metric line."""
    if log_file is None:
        return
    entry = {"step": step, "phase": phase, "timestamp": time.time()}
    entry.update(metrics)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def train_imitation(
    model: FleetModel,
    demos: Dict[str, np.ndarray],
    config: TrainConfig,
    mode_name: str,
    log_file: Optional[str] = None,
) -> List[Dict]:
    """Phase 1: Supervised imitation learning from expert demos.

    Loss: MSE between predicted and expert velocity commands.
    """
    dataset = ExpertDataset(demos)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.imitation_epochs * len(loader),
    )

    device = config.device
    model = model.to(device)
    model.train()

    history = []
    global_step = 0

    for epoch in range(config.imitation_epochs):
        epoch_loss = 0.0
        epoch_vel_loss = 0.0
        epoch_comm_reg = 0.0
        n_batches = 0

        for batch in loader:
            lidar = batch["lidar"].to(device)
            positions = batch["positions"].to(device)
            velocities = batch["velocities"].to(device)
            goal = batch["goal"].to(device)
            formation = batch["formation"].to(device)
            expert_actions = batch["actions"].to(device)

            vel_cmds, messages = model(
                lidar, positions, velocities, goal, formation,
                step=global_step,
            )

            # Velocity imitation loss
            vel_loss = F.mse_loss(vel_cmds, expert_actions)

            # Communication regularization: encourage diverse messages
            msg_var = messages.var(dim=1).mean()
            comm_reg = -config.comm_loss_weight * msg_var

            loss = vel_loss + comm_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_vel_loss += vel_loss.item()
            epoch_comm_reg += comm_reg.item()
            n_batches += 1
            global_step += 1

        metrics = {
            "epoch": epoch,
            "mode": mode_name,
            "loss": epoch_loss / max(n_batches, 1),
            "vel_loss": epoch_vel_loss / max(n_batches, 1),
            "comm_reg": epoch_comm_reg / max(n_batches, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)
        _log_metric(log_file, global_step, "imitation", metrics)

    return history


def train_selfplay(
    model: FleetModel,
    env: MultiRobotEnv,
    config: TrainConfig,
    mode_name: str,
    log_file: Optional[str] = None,
) -> List[Dict]:
    """Phase 2: Self-play with environment reward.

    Uses a simple policy gradient (REINFORCE) with baseline.
    """
    device = config.device
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr * 0.1)

    history = []
    baseline_reward = 0.0
    global_step = 0

    for episode in range(config.selfplay_episodes):
        obs = env.reset()
        episode_rewards = []
        log_probs_list = []
        episode_collisions = 0.0
        episode_goal_dist = 0.0

        for t in range(config.selfplay_steps):
            # Convert obs to torch
            lidar_t = torch.tensor(obs["lidar"], dtype=torch.float32, device=device)
            pos_t = torch.tensor(obs["positions"], dtype=torch.float32, device=device)
            vel_t = torch.tensor(obs["velocities"], dtype=torch.float32, device=device)
            goal_t = torch.tensor(obs["goal"], dtype=torch.float32, device=device)
            form_t = torch.tensor(obs["formation"], dtype=torch.float32, device=device)

            vel_cmds, messages = model(
                lidar_t, pos_t, vel_t, goal_t, form_t, step=t,
            )

            # Add exploration noise
            noise = torch.randn_like(vel_cmds) * 0.1
            noisy_cmds = torch.clamp(vel_cmds + noise, -1, 1)

            # Log probability (Gaussian policy)
            diff = noisy_cmds - vel_cmds
            log_prob = -0.5 * (diff ** 2 / (0.1 ** 2 + 1e-8)).sum(dim=-1).mean()
            log_probs_list.append(log_prob)

            # Step environment
            actions_np = noisy_cmds.detach().cpu().numpy()
            obs, rewards, dones, info = env.step(actions_np)

            episode_rewards.append(rewards.mean())
            episode_collisions += info.get("collisions", 0)
            episode_goal_dist += info.get("mean_goal_dist", 0)

        # REINFORCE update
        total_reward = sum(episode_rewards)
        advantage = total_reward - baseline_reward
        baseline_reward = 0.9 * baseline_reward + 0.1 * total_reward

        policy_loss = torch.tensor(0.0, device=device)
        for lp in log_probs_list:
            policy_loss -= lp * advantage

        if len(log_probs_list) > 0:
            policy_loss = policy_loss / len(log_probs_list)

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        metrics = {
            "episode": episode,
            "mode": mode_name,
            "total_reward": float(total_reward),
            "mean_reward": float(total_reward / max(config.selfplay_steps, 1)),
            "collisions": float(episode_collisions),
            "mean_goal_dist": float(episode_goal_dist / max(config.selfplay_steps, 1)),
            "policy_loss": float(policy_loss.item()),
            "baseline": float(baseline_reward),
        }
        history.append(metrics)
        _log_metric(log_file, global_step, "selfplay", metrics)

    return history


def train_single_model(
    mode: str,
    demos: Dict[str, np.ndarray],
    env: MultiRobotEnv,
    config: TrainConfig,
    log_file: Optional[str] = None,
) -> Tuple[FleetModel, Dict]:
    """Train a single model through both phases.

    Returns:
        (model, results_dict)
    """
    print("  Building {} model...".format(mode))
    bound, program = build_fleet_program(
        n_robots=config.n_robots, mode=mode, d_model=config.d_model,
    )

    # Add scheduling clocks
    program = _add_scheduling_clocks(program)

    print("    Schema: {} regions, {} connections".format(
        len(bound.field_names),
        len(bound.schema.topology.connections) if bound.schema.topology else 0,
    ))
    print("    Program: {}".format(program.summary()))

    model = FleetModel(
        bound=bound, program=program,
        n_robots=config.n_robots, d_model=config.d_model,
        n_heads=config.n_heads,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print("    Parameters: {:,}".format(n_params))

    # Phase 1: Imitation learning
    print("  Phase 1: Imitation learning ({} epochs)...".format(config.imitation_epochs))
    imitation_history = train_imitation(model, demos, config, mode, log_file)

    # Phase 2: Self-play
    print("  Phase 2: Self-play ({} episodes)...".format(config.selfplay_episodes))
    selfplay_env = MultiRobotEnv(EnvConfig(
        n_robots=config.n_robots,
        n_envs=config.n_envs_train,
        task=env.cfg.task,
        max_steps=config.selfplay_steps,
    ))
    selfplay_env.obstacle_pos = env.obstacle_pos.copy()
    selfplay_env.obstacle_radii = env.obstacle_radii.copy()

    selfplay_history = train_selfplay(model, selfplay_env, config, mode, log_file)

    results = {
        "mode": mode,
        "n_params": n_params,
        "n_regions": len(bound.field_names),
        "n_connections": len(bound.schema.topology.connections) if bound.schema.topology else 0,
        "imitation_history": imitation_history,
        "selfplay_history": selfplay_history,
    }

    return model, results


def train_all_models(
    n_robots: int = 4,
    config: Optional[TrainConfig] = None,
) -> Dict:
    """Train all three model variants and return results.

    Returns dict with keys: canvas, dense, independent.
    Each value is a dict with model and training history.
    """
    if config is None:
        config = TrainConfig(n_robots=n_robots)
    else:
        config.n_robots = n_robots

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = str(RESULTS_DIR / "training_log.jsonl")

    # Clear old log
    open(log_file, "w").close()

    print("=" * 60)
    print("Multi-Robot Fleet Training")
    print("  Robots: {}".format(n_robots))
    print("  d_model: {}".format(config.d_model))
    print("  Device: {}".format(config.device))
    print("=" * 60)

    # Create environment
    env_cfg = EnvConfig(n_robots=n_robots, n_envs=1, task="formation")
    env = MultiRobotEnv(env_cfg)

    # Generate expert demonstrations
    print("\nGenerating expert demonstrations...")
    controller = PotentialFieldController()
    demos = generate_expert_demos(
        env, controller,
        n_episodes=config.n_expert_episodes,
        max_steps=config.expert_max_steps,
    )
    print("  Collected {} demo steps".format(len(demos["lidar"])))

    all_results = {}
    models = {}

    for mode in ["canvas", "dense", "independent"]:
        print("\n--- Training {} model ---".format(mode.upper()))
        t0 = time.time()
        model, results = train_single_model(mode, demos, env, config, log_file)
        results["train_time"] = time.time() - t0
        print("  Done in {:.1f}s".format(results["train_time"]))

        all_results[mode] = results
        models[mode] = model

    # Save results summary
    summary = {}
    for mode, res in all_results.items():
        summary[mode] = {
            "n_params": res["n_params"],
            "n_regions": res["n_regions"],
            "n_connections": res["n_connections"],
            "train_time": res["train_time"],
            "final_imitation_loss": res["imitation_history"][-1]["vel_loss"] if res["imitation_history"] else None,
            "final_selfplay_reward": res["selfplay_history"][-1]["total_reward"] if res["selfplay_history"] else None,
            "final_collisions": res["selfplay_history"][-1]["collisions"] if res["selfplay_history"] else None,
        }

    with open(RESULTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {"results": all_results, "models": models, "demos": demos, "env": env}

"""Vectorized 2D multi-robot physics simulation.

A fast, numpy-based 2D environment for cooperative multi-robot control.
Supports batched parallel simulation for efficient training.

Features:
  - N robots in a bounded 10x10m world
  - Each robot: position (x,y), velocity (vx,vy), heading
  - Static circular obstacles
  - 16-beam lidar with ray-casting against obstacles and other robots
  - Velocity-based control (commanded vx, vy)
  - Collision detection and response
  - Three task modes: formation, tracking, coverage

Physics: simple Euler integration with velocity damping and collision forces.
Lidar: vectorized ray-obstacle intersection using analytical circle-ray tests.

Usage:
    env = MultiRobotEnv(n_robots=4, n_envs=32)
    obs = env.reset()
    for t in range(100):
        actions = policy(obs)  # (n_envs, n_robots, 2)
        obs, reward, done, info = env.step(actions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    """Environment configuration."""
    n_robots: int = 4
    n_envs: int = 32
    world_size: float = 10.0
    dt: float = 0.1
    max_speed: float = 2.0
    robot_radius: float = 0.3
    lidar_n_beams: int = 16
    lidar_max_range: float = 5.0
    n_obstacles: int = 5
    obstacle_radius_range: Tuple[float, float] = (0.3, 0.8)
    collision_penalty: float = 5.0
    energy_penalty: float = 0.1
    goal_reward_scale: float = 10.0
    velocity_damping: float = 0.9
    task: str = "formation"  # formation, tracking, coverage
    max_steps: int = 200


class MultiRobotEnv:
    """Vectorized 2D multi-robot environment.

    All state is stored as (n_envs, ...) numpy arrays for fast batched
    simulation. No Python loops over environments.

    Observation per robot:
      - lidar: (16, 2) = 16 beams * (range, intensity), flattened to (32,)
      - position: (2,)
      - velocity: (2,)
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(42)
        self._step_count = np.zeros(self.cfg.n_envs, dtype=np.int32)

        # Obstacle positions and radii (shared across envs for simplicity)
        self._init_obstacles()

        # Robot state: (n_envs, n_robots, dim)
        self.positions = np.zeros((self.cfg.n_envs, self.cfg.n_robots, 2))
        self.velocities = np.zeros((self.cfg.n_envs, self.cfg.n_robots, 2))
        self.headings = np.zeros((self.cfg.n_envs, self.cfg.n_robots))

        # Goal state
        self.goal_positions = np.zeros((self.cfg.n_envs, self.cfg.n_robots, 2))

        # Tracking target (for tracking task)
        self.target_pos = np.zeros((self.cfg.n_envs, 2))
        self.target_vel = np.zeros((self.cfg.n_envs, 2))

    def _init_obstacles(self):
        """Place static circular obstacles."""
        n = self.cfg.n_obstacles
        ws = self.cfg.world_size
        margin = 1.5
        self.obstacle_pos = self.rng.uniform(margin, ws - margin, (n, 2))
        self.obstacle_radii = self.rng.uniform(
            self.cfg.obstacle_radius_range[0],
            self.cfg.obstacle_radius_range[1],
            (n,),
        )

    def reset(self, env_ids: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Reset environments and return initial observations.

        Args:
            env_ids: Optional subset of environment indices to reset.
                     If None, resets all environments.

        Returns:
            Dict with 'lidar', 'positions', 'velocities', 'goal', 'formation'.
        """
        if env_ids is None:
            env_ids = np.arange(self.cfg.n_envs)

        n = len(env_ids)
        ws = self.cfg.world_size
        margin = 1.0

        # Spawn robots with some spacing
        for i in range(self.cfg.n_robots):
            angle = 2 * np.pi * i / self.cfg.n_robots
            radius = 2.0
            cx, cy = ws / 2, ws / 2
            self.positions[env_ids, i, 0] = cx + radius * np.cos(angle) + self.rng.uniform(-0.5, 0.5, n)
            self.positions[env_ids, i, 1] = cy + radius * np.sin(angle) + self.rng.uniform(-0.5, 0.5, n)

        self.velocities[env_ids] = 0.0
        self.headings[env_ids] = self.rng.uniform(0, 2 * np.pi, (n, self.cfg.n_robots))
        self._step_count[env_ids] = 0

        # Set goals based on task
        self._set_goals(env_ids)

        return self._get_obs()

    def _set_goals(self, env_ids: np.ndarray):
        """Set goal positions based on task type."""
        n = len(env_ids)
        ws = self.cfg.world_size
        cx, cy = ws / 2, ws / 2

        if self.cfg.task == "formation":
            # Square formation centered in world
            for i in range(self.cfg.n_robots):
                angle = 2 * np.pi * i / self.cfg.n_robots + np.pi / 4
                radius = 2.5
                self.goal_positions[env_ids, i, 0] = cx + radius * np.cos(angle)
                self.goal_positions[env_ids, i, 1] = cy + radius * np.sin(angle)

        elif self.cfg.task == "tracking":
            # All robots should converge on a moving target
            self.target_pos[env_ids] = self.rng.uniform(2, ws - 2, (n, 2))
            self.target_vel[env_ids] = self.rng.uniform(-0.3, 0.3, (n, 2))
            for i in range(self.cfg.n_robots):
                angle = 2 * np.pi * i / self.cfg.n_robots
                self.goal_positions[env_ids, i, 0] = self.target_pos[env_ids, 0] + 1.0 * np.cos(angle)
                self.goal_positions[env_ids, i, 1] = self.target_pos[env_ids, 1] + 1.0 * np.sin(angle)

        elif self.cfg.task == "coverage":
            # Spread out to cover the area
            grid_n = int(np.ceil(np.sqrt(self.cfg.n_robots)))
            spacing = ws / (grid_n + 1)
            for i in range(self.cfg.n_robots):
                row = i // grid_n
                col = i % grid_n
                self.goal_positions[env_ids, i, 0] = spacing * (col + 1)
                self.goal_positions[env_ids, i, 1] = spacing * (row + 1)

    def step(
        self, actions: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
        """Step the simulation forward.

        Args:
            actions: (n_envs, n_robots, 2) velocity commands in [-1, 1]

        Returns:
            obs: observation dict
            rewards: (n_envs,) scalar rewards
            dones: (n_envs,) boolean done flags
            info: dict with diagnostics
        """
        cfg = self.cfg
        actions = np.clip(actions, -1.0, 1.0) * cfg.max_speed

        # Apply velocity commands
        self.velocities = cfg.velocity_damping * self.velocities + (1 - cfg.velocity_damping) * actions

        # Integrate positions
        self.positions = self.positions + self.velocities * cfg.dt

        # Update headings from velocity direction
        speed = np.linalg.norm(self.velocities, axis=-1, keepdims=True)
        moving = speed.squeeze(-1) > 0.01
        heading_update = np.arctan2(self.velocities[..., 1], self.velocities[..., 0])
        self.headings = np.where(moving, heading_update, self.headings)

        # Collision detection and response
        collisions = self._resolve_collisions()

        # Wall bouncing
        self._resolve_walls()

        # Update tracking target
        if cfg.task == "tracking":
            self._update_tracking_target()

        # Compute rewards
        rewards, info = self._compute_rewards(actions, collisions)

        self._step_count += 1
        dones = self._step_count >= cfg.max_steps

        # Auto-reset done environments
        done_ids = np.where(dones)[0]
        if len(done_ids) > 0:
            self.reset(done_ids)

        obs = self._get_obs()
        return obs, rewards, dones, info

    def _resolve_collisions(self) -> np.ndarray:
        """Detect and resolve robot-robot and robot-obstacle collisions.

        Returns:
            collisions: (n_envs,) count of collisions per environment
        """
        cfg = self.cfg
        collisions = np.zeros(cfg.n_envs)

        # Robot-robot collisions
        for i in range(cfg.n_robots):
            for j in range(i + 1, cfg.n_robots):
                diff = self.positions[:, i] - self.positions[:, j]  # (n_envs, 2)
                dist = np.linalg.norm(diff, axis=-1)  # (n_envs,)
                min_dist = 2 * cfg.robot_radius
                overlap = min_dist - dist
                colliding = overlap > 0

                if np.any(colliding):
                    # Normalize direction
                    safe_dist = np.maximum(dist, 1e-6)
                    direction = diff / safe_dist[:, None]
                    push = direction * (overlap[:, None] / 2) * colliding[:, None]
                    self.positions[:, i] += push
                    self.positions[:, j] -= push

                    # Dampen velocities on collision
                    self.velocities[colliding, i] *= 0.5
                    self.velocities[colliding, j] *= 0.5
                    collisions += colliding.astype(float)

        # Robot-obstacle collisions
        for i in range(cfg.n_robots):
            for o_idx in range(len(self.obstacle_pos)):
                obs_pos = self.obstacle_pos[o_idx]
                obs_r = self.obstacle_radii[o_idx]
                diff = self.positions[:, i] - obs_pos[None, :]
                dist = np.linalg.norm(diff, axis=-1)
                min_dist = cfg.robot_radius + obs_r
                overlap = min_dist - dist
                colliding = overlap > 0

                if np.any(colliding):
                    safe_dist = np.maximum(dist, 1e-6)
                    direction = diff / safe_dist[:, None]
                    push = direction * overlap[:, None] * colliding[:, None]
                    self.positions[:, i] += push
                    self.velocities[colliding, i] *= 0.3
                    collisions += colliding.astype(float)

        return collisions

    def _resolve_walls(self):
        """Keep robots within world bounds."""
        ws = self.cfg.world_size
        r = self.cfg.robot_radius
        low = r
        high = ws - r

        # Clamp positions
        below = self.positions < low
        above = self.positions > high
        self.positions = np.clip(self.positions, low, high)

        # Reverse velocity on wall hit
        self.velocities = np.where(below | above, -0.5 * self.velocities, self.velocities)

    def _update_tracking_target(self):
        """Move tracking target and update goal positions."""
        ws = self.cfg.world_size
        self.target_pos += self.target_vel * self.cfg.dt

        # Bounce off walls
        bounce_low = self.target_pos < 2.0
        bounce_high = self.target_pos > ws - 2.0
        self.target_vel = np.where(bounce_low | bounce_high, -self.target_vel, self.target_vel)
        self.target_pos = np.clip(self.target_pos, 2.0, ws - 2.0)

        for i in range(self.cfg.n_robots):
            angle = 2 * np.pi * i / self.cfg.n_robots
            self.goal_positions[:, i, 0] = self.target_pos[:, 0] + 1.0 * np.cos(angle)
            self.goal_positions[:, i, 1] = self.target_pos[:, 1] + 1.0 * np.sin(angle)

    def _compute_rewards(
        self, actions: np.ndarray, collisions: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """Compute reward signal.

        Components:
          - goal_reward: negative distance to goal positions
          - collision_penalty: per collision
          - energy_penalty: penalize large velocities
        """
        cfg = self.cfg

        # Distance to goal
        goal_dists = np.linalg.norm(
            self.positions - self.goal_positions, axis=-1,
        )  # (n_envs, n_robots)
        mean_goal_dist = goal_dists.mean(axis=-1)  # (n_envs,)
        goal_reward = -mean_goal_dist * cfg.goal_reward_scale

        # Collision penalty
        collision_pen = -collisions * cfg.collision_penalty

        # Energy penalty
        energy = np.linalg.norm(actions, axis=-1).mean(axis=-1)
        energy_pen = -energy * cfg.energy_penalty

        rewards = goal_reward + collision_pen + energy_pen

        info = {
            "mean_goal_dist": mean_goal_dist.mean(),
            "collisions": collisions.sum(),
            "energy": energy.mean(),
            "per_robot_goal_dist": goal_dists.mean(axis=0),
        }
        return rewards, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Compute observations for all robots in all environments.

        Returns dict:
          lidar: (n_envs, n_robots, 32)
          positions: (n_envs, n_robots, 2)
          velocities: (n_envs, n_robots, 2)
          goal: (n_envs, n_robots * 2) -- flattened goal positions
          formation: (n_envs, 4) -- formation shape encoding
        """
        lidar = self._compute_lidar()

        # Normalize positions to [-1, 1]
        pos_norm = self.positions / self.cfg.world_size * 2 - 1
        vel_norm = self.velocities / self.cfg.max_speed

        # Goal encoding: flattened target positions
        goal_norm = self.goal_positions.reshape(self.cfg.n_envs, -1) / self.cfg.world_size * 2 - 1
        # Truncate/pad to n_robots * 2
        goal_size = self.cfg.n_robots * 2
        if goal_norm.shape[-1] >= goal_size:
            goal_norm = goal_norm[:, :goal_size]
        else:
            goal_norm = np.pad(goal_norm, ((0, 0), (0, goal_size - goal_norm.shape[-1])))

        # Formation encoding: mean position, spread, aspect ratio, rotation
        centroid = self.positions.mean(axis=1)  # (n_envs, 2)
        spread = np.std(self.positions, axis=1).mean(axis=-1, keepdims=True)
        diffs = self.positions - centroid[:, None, :]
        cov_xx = (diffs[..., 0] ** 2).mean(axis=-1, keepdims=True)
        cov_yy = (diffs[..., 1] ** 2).mean(axis=-1, keepdims=True)
        formation = np.concatenate([
            centroid / self.cfg.world_size * 2 - 1,
            spread / self.cfg.world_size,
            (cov_xx - cov_yy) / (cov_xx + cov_yy + 1e-6),
        ], axis=-1)  # (n_envs, 4)

        return {
            "lidar": lidar.astype(np.float32),
            "positions": pos_norm.astype(np.float32),
            "velocities": vel_norm.astype(np.float32),
            "goal": goal_norm.astype(np.float32),
            "formation": formation.astype(np.float32),
        }

    def _compute_lidar(self) -> np.ndarray:
        """Vectorized 16-beam lidar for all robots in all environments.

        Returns: (n_envs, n_robots, 32) -- 16 beams * (range, intensity)
        """
        cfg = self.cfg
        n_beams = cfg.lidar_n_beams
        max_range = cfg.lidar_max_range
        n_envs = cfg.n_envs
        n_robots = cfg.n_robots

        # Beam angles relative to heading
        beam_angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)

        # Output: (n_envs, n_robots, n_beams, 2)
        lidar = np.zeros((n_envs, n_robots, n_beams, 2))
        lidar[:, :, :, 0] = 1.0  # default: max range (normalized)

        for r in range(n_robots):
            robot_pos = self.positions[:, r]  # (n_envs, 2)
            robot_heading = self.headings[:, r]  # (n_envs,)

            for b in range(n_beams):
                # Ray direction
                angle = robot_heading + beam_angles[b]  # (n_envs,)
                ray_dir = np.stack([np.cos(angle), np.sin(angle)], axis=-1)  # (n_envs, 2)

                min_t = np.full(n_envs, max_range)
                hit_intensity = np.zeros(n_envs)

                # Check against obstacles
                for o_idx in range(len(self.obstacle_pos)):
                    t_hit = self._ray_circle_intersect(
                        robot_pos, ray_dir,
                        self.obstacle_pos[o_idx], self.obstacle_radii[o_idx],
                    )
                    closer = (t_hit > 0) & (t_hit < min_t)
                    min_t = np.where(closer, t_hit, min_t)
                    hit_intensity = np.where(closer, 0.8, hit_intensity)

                # Check against other robots
                for other_r in range(n_robots):
                    if other_r == r:
                        continue
                    other_pos = self.positions[:, other_r]
                    t_hit = self._ray_circle_intersect(
                        robot_pos, ray_dir,
                        other_pos, cfg.robot_radius,
                    )
                    closer = (t_hit > 0) & (t_hit < min_t)
                    min_t = np.where(closer, t_hit, min_t)
                    hit_intensity = np.where(closer, 1.0, hit_intensity)

                # Check against walls
                t_walls = self._ray_wall_intersect(robot_pos, ray_dir, cfg.world_size)
                closer = (t_walls > 0) & (t_walls < min_t)
                min_t = np.where(closer, t_walls, min_t)
                hit_intensity = np.where(closer, 0.5, hit_intensity)

                lidar[:, r, b, 0] = min_t / max_range  # normalized range
                lidar[:, r, b, 1] = hit_intensity

        return lidar.reshape(n_envs, n_robots, n_beams * 2)

    @staticmethod
    def _ray_circle_intersect(
        origin: np.ndarray,
        direction: np.ndarray,
        center,
        radius: float,
    ) -> np.ndarray:
        """Analytical ray-circle intersection.

        Args:
            origin: (n_envs, 2) ray origins
            direction: (n_envs, 2) unit ray directions
            center: (2,) or (n_envs, 2) circle center
            radius: circle radius

        Returns:
            t: (n_envs,) distance to intersection, inf if no hit
        """
        if isinstance(center, np.ndarray) and center.ndim == 1:
            oc = origin - center[None, :]
        else:
            oc = origin - center

        a = np.sum(direction * direction, axis=-1)  # should be ~1.0
        b = 2.0 * np.sum(oc * direction, axis=-1)
        c = np.sum(oc * oc, axis=-1) - radius * radius

        discriminant = b * b - 4 * a * c
        has_hit = discriminant >= 0

        sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
        t1 = (-b - sqrt_disc) / (2 * a + 1e-12)
        t2 = (-b + sqrt_disc) / (2 * a + 1e-12)

        # Take nearest positive intersection
        t = np.where(t1 > 0.01, t1, t2)
        t = np.where(has_hit & (t > 0.01), t, np.inf)
        return t

    @staticmethod
    def _ray_wall_intersect(
        origin: np.ndarray,
        direction: np.ndarray,
        world_size: float,
    ) -> np.ndarray:
        """Ray intersection with axis-aligned walls (box [0, world_size]^2).

        Returns nearest positive t.
        """
        dx = direction[:, 0]
        dy = direction[:, 1]
        ox = origin[:, 0]
        oy = origin[:, 1]

        big = np.float64(1e12)

        # t for each wall
        t_left = np.where(np.abs(dx) > 1e-12, -ox / dx, big)
        t_right = np.where(np.abs(dx) > 1e-12, (world_size - ox) / dx, big)
        t_bottom = np.where(np.abs(dy) > 1e-12, -oy / dy, big)
        t_top = np.where(np.abs(dy) > 1e-12, (world_size - oy) / dy, big)

        # Only positive t
        candidates = np.stack([t_left, t_right, t_bottom, t_top], axis=-1)
        candidates = np.where(candidates > 0.01, candidates, big)
        return candidates.min(axis=-1)


# ---- Expert controller (potential fields) ----------------------------


class PotentialFieldController:
    """Rule-based expert using artificial potential fields.

    Each robot is attracted to its goal and repelled by obstacles and
    other robots. Simple, fast, and provides reasonable demonstrations
    for imitation learning.
    """

    def __init__(
        self,
        attract_gain: float = 1.0,
        repel_gain: float = 2.0,
        repel_range: float = 2.0,
        obstacle_repel_gain: float = 3.0,
        max_speed: float = 2.0,
    ):
        self.attract_gain = attract_gain
        self.repel_gain = repel_gain
        self.repel_range = repel_range
        self.obstacle_repel_gain = obstacle_repel_gain
        self.max_speed = max_speed

    def act(
        self,
        positions: np.ndarray,
        goal_positions: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_radii: np.ndarray,
    ) -> np.ndarray:
        """Compute velocity commands using potential fields.

        Args:
            positions: (n_envs, n_robots, 2)
            goal_positions: (n_envs, n_robots, 2)
            obstacle_pos: (n_obstacles, 2)
            obstacle_radii: (n_obstacles,)

        Returns:
            actions: (n_envs, n_robots, 2) in [-1, 1]
        """
        n_envs, n_robots = positions.shape[:2]

        # Attractive force toward goal
        goal_diff = goal_positions - positions
        goal_dist = np.linalg.norm(goal_diff, axis=-1, keepdims=True)
        attract = self.attract_gain * goal_diff / (goal_dist + 0.1)

        # Repulsive force from other robots
        repel = np.zeros_like(positions)
        for i in range(n_robots):
            for j in range(n_robots):
                if i == j:
                    continue
                diff = positions[:, i] - positions[:, j]
                dist = np.linalg.norm(diff, axis=-1, keepdims=True)
                in_range = dist < self.repel_range
                force = self.repel_gain * diff / (dist ** 2 + 0.01) * in_range
                repel[:, i] += force.squeeze(-1) if force.ndim > 2 else force

        # Repulsive force from obstacles
        obs_repel = np.zeros_like(positions)
        for i in range(n_robots):
            for o_idx in range(len(obstacle_pos)):
                diff = positions[:, i] - obstacle_pos[o_idx]
                dist = np.linalg.norm(diff, axis=-1, keepdims=True)
                effective_range = self.repel_range + obstacle_radii[o_idx]
                in_range = dist < effective_range
                force = self.obstacle_repel_gain * diff / (dist ** 2 + 0.01) * in_range
                obs_repel[:, i] += force.squeeze(-1) if force.ndim > 2 else force

        # Combine forces
        total = attract + repel + obs_repel

        # Normalize to [-1, 1]
        speed = np.linalg.norm(total, axis=-1, keepdims=True)
        actions = total / (speed + 0.01)
        actions = np.clip(actions, -1.0, 1.0)

        return actions


def generate_expert_demos(
    env: MultiRobotEnv,
    controller: PotentialFieldController,
    n_episodes: int = 50,
    max_steps: int = 200,
) -> Dict[str, np.ndarray]:
    """Generate expert demonstrations.

    Returns dict with:
      lidar: (total_steps, n_robots, 32)
      positions: (total_steps, n_robots, 2)
      velocities: (total_steps, n_robots, 2)
      goal: (total_steps, n_robots*2)
      formation: (total_steps, 4)
      actions: (total_steps, n_robots, 2)
    """
    all_lidar = []
    all_pos = []
    all_vel = []
    all_goal = []
    all_form = []
    all_actions = []

    # Use single-env for expert data collection
    single_cfg = EnvConfig(
        n_robots=env.cfg.n_robots,
        n_envs=1,
        n_obstacles=env.cfg.n_obstacles,
        task=env.cfg.task,
        max_steps=max_steps,
    )
    single_env = MultiRobotEnv(single_cfg)
    single_env.obstacle_pos = env.obstacle_pos.copy()
    single_env.obstacle_radii = env.obstacle_radii.copy()

    for ep in range(n_episodes):
        obs = single_env.reset()

        for t in range(max_steps):
            actions = controller.act(
                single_env.positions,
                single_env.goal_positions,
                single_env.obstacle_pos,
                single_env.obstacle_radii,
            )

            all_lidar.append(obs["lidar"][0])
            all_pos.append(obs["positions"][0])
            all_vel.append(obs["velocities"][0])
            all_goal.append(obs["goal"][0])
            all_form.append(obs["formation"][0])
            all_actions.append(actions[0])

            obs, _, done, _ = single_env.step(actions)
            if done[0]:
                break

    return {
        "lidar": np.array(all_lidar, dtype=np.float32),
        "positions": np.array(all_pos, dtype=np.float32),
        "velocities": np.array(all_vel, dtype=np.float32),
        "goal": np.array(all_goal, dtype=np.float32),
        "formation": np.array(all_form, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
    }

"""Multi-robot fleet architecture as canvas types.

Defines a compositional type hierarchy:
  Sensor -> RobotState -> RobotAction -> Robot -> RobotFleet

Each Robot is a nested type with __coarse__ = Field(2,2) so the fleet
sees each robot as a compact 2x2 summary. Inter-robot communication
bottlenecks through these coarse-grained fields, encouraging robots to
learn compressed broadcasts rather than leaking full internal state.

Three compile variants:
  1. canvas_fleet  -- structured topology with coarse-grained bottleneck
  2. dense_fleet   -- all regions fully connected (no bottleneck)
  3. independent   -- each robot isolated (no communication)

Usage:
    from research.robotics.robot_canvas import build_fleet_program

    bound, program = build_fleet_program(n_robots=4, mode="canvas")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

from canvas_engineering import (
    Field,
    Connection,
    CanvasTopology,
    ConnectivityPolicy,
    compile_program,
    compile_schema,
    CanvasProgram,
    RegionProgram,
    ConnectionProgram,
    ClockSpec,
    RegionScheduler,
    AttentionDispatcher,
    ResidualAccumulator,
    ResidualSpec,
    BoundSchema,
)


# ---- Type hierarchy ------------------------------------------------


@dataclass
class Sensor:
    """Robot sensor suite."""
    __coarse__ = Field(2, 2)
    lidar: Field = Field(4, 4, family="observation",
                         semantic_type="2D lidar scan 16 beams range+intensity")
    position: Field = Field(1, 2, family="observation",
                            semantic_type="robot position x y in world frame")
    velocity: Field = Field(1, 2, family="observation",
                            semantic_type="robot velocity vx vy")


@dataclass
class RobotState:
    """Internal robot representations."""
    belief: Field = Field(3, 3, family="state", tags=("belief",), carrier="filter",
                          semantic_type="belief about local environment and obstacles")
    plan: Field = Field(2, 3, family="state", tags=("planning",),
                        semantic_type="trajectory plan next 5 timesteps")
    prediction_error: Field = Field(1, 2, family="residual",
                                    semantic_type="environment prediction error")


@dataclass
class RobotAction:
    """Robot control outputs."""
    velocity_cmd: Field = Field(1, 2, family="action", loss_weight=3.0,
                                semantic_type="commanded velocity vx vy")
    communicate: Field = Field(1, 4, family="action", loss_weight=1.0,
                               semantic_type="message to broadcast to fleet")


@dataclass
class Robot:
    """A single robot agent."""
    __coarse__ = Field(2, 2)
    sensor: Sensor = dc_field(default_factory=Sensor)
    state: RobotState = dc_field(default_factory=RobotState)
    action: RobotAction = dc_field(default_factory=RobotAction)


@dataclass
class FleetCoordination:
    """Fleet-level coordination signals."""
    shared_goal: Field = Field(2, 4, family="state", tags=("goal",), is_output=False,
                               semantic_type="shared fleet objective and target positions")
    formation: Field = Field(2, 2, family="state", tags=("belief",),
                             semantic_type="current formation state and desired shape")


@dataclass
class RobotFleet:
    """Multi-robot fleet with shared coordination."""
    coordination: FleetCoordination = dc_field(default_factory=FleetCoordination)
    robots: list = dc_field(default_factory=list)


# ---- Flat variants for dense/independent (no __coarse__ bottleneck) --


@dataclass
class FlatSensor:
    """Sensor without coarse-grained bottleneck."""
    lidar: Field = Field(4, 4, family="observation",
                         semantic_type="2D lidar scan 16 beams range+intensity")
    position: Field = Field(1, 2, family="observation",
                            semantic_type="robot position x y in world frame")
    velocity: Field = Field(1, 2, family="observation",
                            semantic_type="robot velocity vx vy")


@dataclass
class FlatRobot:
    """Robot without coarse-grained bottleneck -- all fields exposed."""
    sensor: FlatSensor = dc_field(default_factory=FlatSensor)
    state: RobotState = dc_field(default_factory=RobotState)
    action: RobotAction = dc_field(default_factory=RobotAction)


@dataclass
class FlatRobotFleet:
    """Fleet without coarse-grained bottleneck."""
    coordination: FleetCoordination = dc_field(default_factory=FleetCoordination)
    robots: list = dc_field(default_factory=list)


# ---- Build helpers --------------------------------------------------


def _make_fleet(n_robots: int) -> RobotFleet:
    """Create a RobotFleet instance with n_robots."""
    fleet = RobotFleet()
    fleet.robots = [Robot() for _ in range(n_robots)]
    return fleet


def _make_flat_fleet(n_robots: int) -> FlatRobotFleet:
    """Create a flat fleet (no coarse-grained bottleneck)."""
    fleet = FlatRobotFleet()
    fleet.robots = [FlatRobot() for _ in range(n_robots)]
    return fleet


def build_fleet_program(
    n_robots: int = 4,
    mode: str = "canvas",
    d_model: int = 64,
    T: int = 1,
) -> Tuple[BoundSchema, CanvasProgram]:
    """Compile a multi-robot fleet into a BoundSchema + CanvasProgram.

    Args:
        n_robots: Number of robots in the fleet.
        mode: One of "canvas" (structured with coarse-grained bottleneck),
              "dense" (fully connected, no bottleneck),
              "independent" (each robot isolated).
        d_model: Latent dimensionality.
        T: Temporal extent.

    Returns:
        (BoundSchema, CanvasProgram) tuple.
    """
    if mode == "canvas":
        # Structured topology: coarse-grained inter-robot communication
        fleet = _make_fleet(n_robots)
        policy = ConnectivityPolicy(
            intra="dense",
            array_element="matched_fields",
            temporal="dense",
        )
    elif mode == "dense":
        # Flat types: all robot fields directly connected to each other
        fleet = _make_flat_fleet(n_robots)
        policy = ConnectivityPolicy(
            intra="dense",
            array_element="dense",
            temporal="dense",
        )
    elif mode == "independent":
        # Flat types: no inter-robot connections at all
        fleet = _make_flat_fleet(n_robots)
        policy = ConnectivityPolicy(
            intra="dense",
            array_element="isolated",
            temporal="dense",
        )
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    bound, program = compile_program(
        fleet, T=T, d_model=d_model, connectivity=policy,
    )

    # For dense mode: add direct inter-robot field connections
    # (bypassing the coarse-grained bottleneck)
    # We connect matched leaf fields across robots (same semantic role)
    # so belief<->belief, plan<->plan, lidar<->lidar, etc.
    if mode == "dense":
        extra_conns = []
        robot_leaf_fields = {}  # robot_idx -> {local_name: full_path}
        for name in bound.field_names:
            for i in range(n_robots):
                prefix = "robots[{}]".format(i)
                if name.startswith(prefix) and name != prefix:
                    # Extract the local field name (after robot prefix)
                    local = name[len(prefix) + 1:]  # e.g. "sensor.lidar"
                    robot_leaf_fields.setdefault(i, {})[local] = name

        # Connect matched fields across robots (dense matched)
        for i in range(n_robots):
            for j in range(n_robots):
                if i == j:
                    continue
                fields_i = robot_leaf_fields.get(i, {})
                fields_j = robot_leaf_fields.get(j, {})
                for local_name in fields_i:
                    if local_name in fields_j:
                        extra_conns.append(Connection(
                            src=fields_i[local_name],
                            dst=fields_j[local_name],
                        ))

        if extra_conns:
            all_conns = list(bound.schema.topology.connections) + extra_conns
            new_topology = CanvasTopology(connections=all_conns)
            from canvas_engineering.schema import CanvasSchema
            new_schema = CanvasSchema(
                layout=bound.schema.layout,
                topology=new_topology,
                version=bound.schema.version,
                metadata=bound.schema.metadata,
            )
            bound.schema = new_schema
            program = CanvasProgram(
                schema=new_schema,
                regions=program.regions,
                connections=program.connections,
                version=program.version,
            )

    return bound, program


def get_region_names(bound: BoundSchema) -> Dict[str, List[str]]:
    """Categorize compiled regions by type.

    Returns dict with keys: sensor, state, action, residual, coordination, coarse.
    """
    cats = {
        "sensor": [], "state": [], "action": [],
        "residual": [], "coordination": [], "coarse": [],
    }
    for name in bound.field_names:
        lower = name.lower()
        if "coordination" in lower:
            cats["coordination"].append(name)
        elif "sensor" in lower or "lidar" in lower or "position" in lower or "velocity" in lower:
            if "velocity_cmd" in lower or "communicate" in lower:
                cats["action"].append(name)
            else:
                cats["sensor"].append(name)
        elif "belief" in lower or "plan" in lower:
            cats["state"].append(name)
        elif "prediction_error" in lower:
            cats["residual"].append(name)
        elif "velocity_cmd" in lower or "communicate" in lower:
            cats["action"].append(name)
        elif "action" in lower:
            cats["action"].append(name)
        elif "state" in lower:
            cats["state"].append(name)
        else:
            cats["coarse"].append(name)
    return cats


# ---- Canvas-based Fleet Model ---------------------------------------


class FleetModel(nn.Module):
    """Neural network model for multi-robot fleet control.

    Wraps AttentionDispatcher with input/output projections that interface
    with the 2D physics environment. Supports three topology modes.

    Forward pass:
      1. Encode raw observations into canvas regions
      2. Set coordination signals (goal, formation)
      3. Run AttentionDispatcher (topology-constrained attention)
      4. Decode action regions to velocity commands and messages
    """

    def __init__(
        self,
        bound: BoundSchema,
        program: CanvasProgram,
        n_robots: int,
        d_model: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.bound = bound
        self.program = program
        self.n_robots = n_robots
        self.d_model = d_model

        layout = bound.schema.layout
        topology = bound.schema.topology

        # Residual accumulator for prediction error tracking
        residual_regions = [
            name for name in bound.field_names
            if "prediction_error" in name
        ]
        self.residual_acc = ResidualAccumulator(
            residual_regions, ResidualSpec(kinds=("prediction",), decay=0.95)
        ) if residual_regions else None

        # Attention dispatcher
        self.dispatcher = AttentionDispatcher(
            topology=topology,
            layout=layout,
            d_model=d_model,
            n_heads=n_heads,
            residual_accumulator=self.residual_acc,
        )

        # Region scheduler with clocks
        self.scheduler = RegionScheduler(program)

        # Input projections (raw obs -> d_model)
        self.lidar_proj = nn.Linear(32, d_model)  # 16 beams * 2 (range+intensity)
        self.pos_proj = nn.Linear(2, d_model)
        self.vel_proj = nn.Linear(2, d_model)
        self.goal_proj = nn.Linear(8, d_model)  # target positions (4 robots * 2D)
        self.formation_proj = nn.Linear(4, d_model)  # formation shape encoding

        # Output decoders
        self.velocity_decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Tanh(),  # velocities in [-1, 1]
        )
        self.message_decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_model)

        # Region name cache
        self._cache_region_names()

    def _cache_region_names(self):
        """Cache region name lookups for fast forward pass."""
        self._lidar_regions = []
        self._pos_regions = []
        self._vel_regions = []
        self._vel_cmd_regions = []
        self._comm_regions = []
        self._belief_regions = []
        self._plan_regions = []
        self._pred_err_regions = []
        self._goal_region = None
        self._formation_region = None

        for name in self.bound.field_names:
            lower = name.lower()
            if "lidar" in lower:
                self._lidar_regions.append(name)
            elif "velocity_cmd" in lower:
                self._vel_cmd_regions.append(name)
            elif "communicate" in lower:
                self._comm_regions.append(name)
            elif ".sensor.position" in name or "sensor.position" in name:
                self._pos_regions.append(name)
            elif ".sensor.velocity" in name or "sensor.velocity" in name:
                self._vel_regions.append(name)
            elif "belief" in lower and "coordination" not in lower:
                self._belief_regions.append(name)
            elif "plan" in lower:
                self._plan_regions.append(name)
            elif "prediction_error" in lower:
                self._pred_err_regions.append(name)
            elif "shared_goal" in lower:
                self._goal_region = name
            elif "formation" in lower and "coordination" in lower:
                self._formation_region = name

    def forward(
        self,
        lidar: torch.Tensor,        # (B, n_robots, 32)
        positions: torch.Tensor,     # (B, n_robots, 2)
        velocities: torch.Tensor,    # (B, n_robots, 2)
        goal: torch.Tensor,          # (B, 8) or (B, n_robots*2)
        formation: torch.Tensor,     # (B, 4)
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: observations -> (velocity_cmds, messages).

        Args:
            lidar: (B, n_robots, 32) lidar readings
            positions: (B, n_robots, 2) robot positions
            velocities: (B, n_robots, 2) robot velocities
            goal: (B, n_robots*2) target positions for formation
            formation: (B, 4) formation shape encoding
            step: current timestep (for scheduling)

        Returns:
            velocity_cmds: (B, n_robots, 2) velocity commands
            messages: (B, n_robots, 4) broadcast messages
        """
        B = lidar.shape[0]
        layout = self.bound.schema.layout
        N = layout.num_positions

        # Create canvas tensor
        canvas = torch.zeros(B, N, self.d_model, device=lidar.device)

        # Encode observations into canvas regions
        for i, region_name in enumerate(self._lidar_regions):
            if i >= self.n_robots:
                break
            indices = layout.region_indices(region_name)
            lidar_emb = self.lidar_proj(lidar[:, i])  # (B, d_model)
            # Broadcast to all positions in this region
            for idx in indices:
                canvas[:, idx] = lidar_emb

        for i, region_name in enumerate(self._pos_regions):
            if i >= self.n_robots:
                break
            indices = layout.region_indices(region_name)
            pos_emb = self.pos_proj(positions[:, i])  # (B, d_model)
            for idx in indices:
                canvas[:, idx] = pos_emb

        for i, region_name in enumerate(self._vel_regions):
            if i >= self.n_robots:
                break
            indices = layout.region_indices(region_name)
            vel_emb = self.vel_proj(velocities[:, i])  # (B, d_model)
            for idx in indices:
                canvas[:, idx] = vel_emb

        # Encode coordination signals
        if self._goal_region is not None:
            indices = layout.region_indices(self._goal_region)
            # Pad goal to expected size
            goal_padded = goal
            if goal.shape[-1] < 8:
                goal_padded = torch.nn.functional.pad(goal, (0, 8 - goal.shape[-1]))
            goal_emb = self.goal_proj(goal_padded[:, :8])
            for idx in indices:
                canvas[:, idx] = goal_emb

        if self._formation_region is not None:
            indices = layout.region_indices(self._formation_region)
            form_emb = self.formation_proj(formation)
            for idx in indices:
                canvas[:, idx] = form_emb

        # Run attention dispatcher
        canvas = self.layer_norm(canvas)
        canvas = self.dispatcher(canvas)

        # Decode velocity commands
        vel_cmds = torch.zeros(B, self.n_robots, 2, device=lidar.device)
        for i, region_name in enumerate(self._vel_cmd_regions):
            if i >= self.n_robots:
                break
            indices = layout.region_indices(region_name)
            if indices:
                region_out = canvas[:, indices[0]]  # (B, d_model)
                vel_cmds[:, i] = self.velocity_decoder(region_out)

        # Decode messages
        messages = torch.zeros(B, self.n_robots, 4, device=lidar.device)
        for i, region_name in enumerate(self._comm_regions):
            if i >= self.n_robots:
                break
            indices = layout.region_indices(region_name)
            if indices:
                region_out = canvas[:, indices[0]]
                messages[:, i] = self.message_decoder(region_out)

        return vel_cmds, messages

    def get_summaries(self) -> Optional[Dict]:
        """Get residual summaries for scheduling."""
        if self.residual_acc is not None:
            return self.residual_acc.summaries()
        return None

"""Non-Euclidean canvas connectivity: declarative block-DAG attention topology.

Define which block-to-block attention operations are performed per forward step.
Each connection specifies a cross-attention op: src tokens query against dst tokens.
Self-connections = self-attention within a block.

The topology is the compute graph of attention operations, not a soft mask.

Temporal connections:
    Connections can optionally constrain which timesteps participate.
    t_src/t_dst are relative offsets from a shared reference frame.

    Connection(src="cam", dst="cam")                       # all timesteps (default)
    Connection(src="cam", dst="action", t_src=0, t_dst=0)  # same-frame only
    Connection(src="action", dst="obs", t_src=0, t_dst=-1) # action queries prev obs
    Connection(src="thought", dst="thought")               # full temporal self-attn

Example:
    topology = CanvasTopology(
        connections=[
            # Self-attention within each region
            Connection(src="robot1_cam", dst="robot1_cam"),
            Connection(src="robot1_action", dst="robot1_action"),
            Connection(src="shared_task", dst="shared_task"),

            # Causal: each robot's camera informs its actions
            Connection(src="robot1_action", dst="robot1_cam"),

            # Temporal: action at frame t queries obs at frame t-1
            Connection(src="robot1_action", dst="robot1_cam", t_src=0, t_dst=-1),

            # Coordination via hub
            Connection(src="shared_task", dst="robot1_cam"),
            Connection(src="robot1_action", dst="shared_task"),
        ]
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from canvas_engineering.canvas import CanvasLayout


@dataclass(frozen=True)
class Connection:
    """A single block-to-block attention operation.

    src queries attend to dst keys/values. This is one cross-attention op
    in the per-step compute DAG.

    Args:
        src: Region whose tokens are the queries.
        dst: Region whose tokens are the keys/values.
        weight: Attention weight scaling (1.0 = full attention).
        t_src: Temporal offset for src (query) positions. None = all timesteps.
            When int, constrains src to positions at reference_frame + t_src.
        t_dst: Temporal offset for dst (key/value) positions. None = all timesteps.
            When int, constrains dst to positions at reference_frame + t_dst.
        fn: Attention function type for this connection. None = use the src
            region's default_attn. See ATTENTION_TYPES in canvas.py for the
            full registry. The schema declares intent; execution is
            backend-dependent.
    """
    src: str
    dst: str
    weight: float = 1.0
    t_src: Optional[int] = None
    t_dst: Optional[int] = None
    fn: Optional[str] = None


@dataclass
class CanvasTopology:
    """Declarative specification of block-to-block attention connectivity.

    Each Connection defines a cross-attention operation performed per step.
    Self-connections (src == dst) = self-attention within a block.
    The full set of connections is the compute DAG.

    Temporal semantics:
        When t_src and t_dst are both None (default), all positions in src
        attend to all positions in dst regardless of timestep (dense in time).

        When t_src and/or t_dst are ints, they define relative temporal offsets.
        The mask iterates over reference frames and pairs:
            src positions at (ref + t_src) with dst positions at (ref + t_dst).

        Mixed (one None, one int): the None side includes ALL its timesteps,
        the int side is constrained to (ref + offset) per reference frame.

    Special cases:
        - Every region self-connected = block-diagonal self-attention
        - Every pair connected = dense attention (standard transformer)
        - DAG structure = structured information flow
        - t_src=0, t_dst=0 = same-frame only (no temporal leakage)
        - t_src=0, t_dst=-1 = previous-frame cross-attention
    """
    connections: List[Connection] = field(default_factory=list)

    # ── Convenience constructors ─────────────────────────────────────────

    @staticmethod
    def dense(regions: List[str]) -> "CanvasTopology":
        """Fully connected: every region attends to every other."""
        return CanvasTopology(
            connections=[Connection(src=s, dst=d) for s in regions for d in regions]
        )

    @staticmethod
    def isolated(regions: List[str]) -> "CanvasTopology":
        """Block-diagonal: each region only attends to itself."""
        return CanvasTopology(
            connections=[Connection(src=r, dst=r) for r in regions]
        )

    @staticmethod
    def hub_spoke(hub: str, spokes: List[str], bidirectional: bool = True) -> "CanvasTopology":
        """Hub-and-spoke: hub reads from all spokes, spokes read from hub."""
        conns = [Connection(src=r, dst=r) for r in [hub] + spokes]  # self-loops
        for spoke in spokes:
            conns.append(Connection(src=hub, dst=spoke))
            if bidirectional:
                conns.append(Connection(src=spoke, dst=hub))
        return CanvasTopology(connections=conns)

    @staticmethod
    def causal_chain(regions: List[str]) -> "CanvasTopology":
        """Causal chain: A → B → C (each attends to self + previous)."""
        conns = [Connection(src=r, dst=r) for r in regions]  # self-loops
        for i in range(1, len(regions)):
            conns.append(Connection(src=regions[i], dst=regions[i - 1]))
        return CanvasTopology(connections=conns)

    @staticmethod
    def causal_temporal(regions: List[str]) -> "CanvasTopology":
        """Temporal causal: same-frame self-attn + prev-frame cross-attn.

        Each region attends to itself at the same frame, and to all other
        regions at the previous frame. No future leakage.
        """
        conns = []
        for r in regions:
            # Same-frame self-attention
            conns.append(Connection(src=r, dst=r, t_src=0, t_dst=0))
            # Previous-frame self-attention (temporal context)
            conns.append(Connection(src=r, dst=r, t_src=0, t_dst=-1))
        # Cross-region: each region queries all others at previous frame
        for s in regions:
            for d in regions:
                if s != d:
                    conns.append(Connection(src=s, dst=d, t_src=0, t_dst=-1))
        return CanvasTopology(connections=conns)

    # ── Query methods ────────────────────────────────────────────────────

    @property
    def regions(self) -> Set[str]:
        """All region names referenced in connections."""
        return {c.src for c in self.connections} | {c.dst for c in self.connections}

    @property
    def has_temporal_constraints(self) -> bool:
        """Whether any connection has temporal offsets."""
        return any(c.t_src is not None or c.t_dst is not None for c in self.connections)

    def resolve_fn(self, connection: Connection,
                   layout: Optional[CanvasLayout] = None) -> str:
        """Resolve the attention function type for a connection.

        Resolution order:
            1. connection.fn if explicitly set
            2. layout.region_spec(connection.src).default_attn if layout provided
            3. "cross_attention" (global default)
        """
        if connection.fn is not None:
            return connection.fn
        if layout is not None:
            try:
                return layout.region_spec(connection.src).default_attn
            except KeyError:
                pass
        return "cross_attention"

    def attention_ops(self, layout: Optional[CanvasLayout] = None,
                      ) -> List[Tuple[str, str, float, str]]:
        """List of (src, dst, weight, fn) attention operations per step.

        If layout is provided, connection fn is resolved via region defaults.
        """
        return [(c.src, c.dst, c.weight, self.resolve_fn(c, layout))
                for c in self.connections]

    def neighbors_of(self, region: str) -> List[str]:
        """Which regions does `region` attend to (as queries)?"""
        return [c.dst for c in self.connections if c.src == region]

    def attended_by(self, region: str) -> List[str]:
        """Which regions query against `region` (as keys/values)?"""
        return [c.src for c in self.connections if c.dst == region]

    # ── Mask generation ──────────────────────────────────────────────────

    def to_attention_mask(self, layout: CanvasLayout, device: str = "cpu") -> torch.Tensor:
        """Generate (N, N) attention mask from topology.

        mask[i, j] > 0 means token i (query) attends to token j (key).
        Value is the connection weight.

        For temporal connections (t_src/t_dst specified), iterates over
        reference frames and only connects positions at the appropriate
        absolute timesteps.
        """
        N = layout.num_positions
        mask = torch.zeros(N, N, device=device)

        # Cache region indices (all timesteps)
        idx_cache: Dict[str, List[int]] = {}
        for name in layout.regions:
            idx_cache[name] = layout.region_indices(name)

        # Cache per-timestep indices
        t_idx_cache: Dict[str, Dict[int, List[int]]] = {}
        for name in layout.regions:
            t_idx_cache[name] = {}
            for t in layout.region_timesteps(name):
                t_idx_cache[name][t] = layout.region_indices_at_t(name, t)

        for conn in self.connections:
            if conn.src not in idx_cache or conn.dst not in idx_cache:
                continue

            if conn.t_src is None and conn.t_dst is None:
                # Dense in time: all src positions attend to all dst positions
                src_idx = idx_cache[conn.src]
                dst_idx = idx_cache[conn.dst]
                for si in src_idx:
                    for di in dst_idx:
                        mask[si, di] = max(mask[si, di].item(), conn.weight)

            elif conn.t_src is not None and conn.t_dst is not None:
                # Both specified: iterate reference frames, pair by offset
                for ref in range(layout.T):
                    abs_src = ref + conn.t_src
                    abs_dst = ref + conn.t_dst
                    src_at_t = t_idx_cache[conn.src].get(abs_src, [])
                    dst_at_t = t_idx_cache[conn.dst].get(abs_dst, [])
                    for si in src_at_t:
                        for di in dst_at_t:
                            mask[si, di] = max(mask[si, di].item(), conn.weight)

            elif conn.t_src is not None:
                # src constrained, dst sees all its timesteps
                dst_idx = idx_cache[conn.dst]
                for ref in range(layout.T):
                    abs_src = ref + conn.t_src
                    src_at_t = t_idx_cache[conn.src].get(abs_src, [])
                    for si in src_at_t:
                        for di in dst_idx:
                            mask[si, di] = max(mask[si, di].item(), conn.weight)

            else:
                # dst constrained, src sees all its timesteps
                src_idx = idx_cache[conn.src]
                for ref in range(layout.T):
                    abs_dst = ref + conn.t_dst
                    dst_at_t = t_idx_cache[conn.dst].get(abs_dst, [])
                    for si in src_idx:
                        for di in dst_at_t:
                            mask[si, di] = max(mask[si, di].item(), conn.weight)

        return mask

    def to_additive_mask(self, layout: CanvasLayout, device: str = "cpu") -> torch.Tensor:
        """Generate (N, N) additive attention mask for use with nn.Transformer.

        Returns a float mask where 0.0 = attend and -inf = block.
        Unused positions (not in any region) get self-attention to avoid
        NaN in softmax. Use with: transformer(x, mask=topology.to_additive_mask(layout))
        """
        raw = self.to_attention_mask(layout, device=device)
        additive = torch.full_like(raw, float('-inf'))
        additive[raw > 0] = 0.0
        # Unused positions: self-attend to avoid NaN
        for i in range(layout.num_positions):
            if additive[i].eq(float('-inf')).all():
                additive[i, i] = 0.0
        return additive

    def to_block_adjacency(self) -> Dict[Tuple[str, str], float]:
        """Region-level adjacency dict: (src, dst) → weight."""
        adj: Dict[Tuple[str, str], float] = {}
        for c in self.connections:
            key = (c.src, c.dst)
            adj[key] = max(adj.get(key, 0.0), c.weight)
        return adj

    def summary(self) -> str:
        """Human-readable summary."""
        adj = self.to_block_adjacency()
        lines = [f"CanvasTopology ({len(self.connections)} ops, {len(self.regions)} regions):"]
        for (src, dst), w in sorted(adj.items()):
            if src == dst:
                lines.append(f"  {src} ↺ self  (w={w:.2f})")
            else:
                arrow = "↔" if (dst, src) in adj else "→"
                lines.append(f"  {src} {arrow} {dst}  (w={w:.2f})")

        if self.has_temporal_constraints:
            temporal = [c for c in self.connections if c.t_src is not None or c.t_dst is not None]
            lines.append(f"  [{len(temporal)} temporal constraints]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CanvasTopology(connections={len(self.connections)}, regions={sorted(self.regions)})"

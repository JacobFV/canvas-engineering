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

Temporal fill modes:
    When a connection has temporal constraints and the dst region has no
    positions at the requested timestep (e.g. a slow-updating field between
    updates), the temporal_fill mode determines behavior:

    TemporalFill.DROP         — skip (no connection).
    TemporalFill.HOLD         — use most recent available value. Default.
    TemporalFill.INTERPOLATE  — weighted interpolation between surrounding values.

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

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from canvas_engineering.canvas import CanvasLayout


class TemporalFill(str, Enum):
    """How to handle temporal mismatches when dst has no value at the requested timestep.

    When regions update at different frequencies, a fast region (period=1)
    attending to a slow region (period=576) will often find no dst positions
    at the exact requested timestep. This enum controls what happens.

    DROP:        No connection. The fast region gets nothing from the slow region
                 at misaligned frames. Information is lost.
    HOLD:        Zero-order hold. Use the most recent available value from the
                 slow region. This is how the real world works — you trade on
                 last quarter's GDP until this quarter's is released.
    INTERPOLATE: Weighted interpolation between surrounding available values.
                 For order=1 (default): linear lerp between nearest past and
                 future endpoints (falls back to HOLD if only past is available).
                 For order=N: inverse-distance weighting (IDW) over up to N+1
                 nearest anchor points with weights proportional to 1/dist^N,
                 non-negative and normalized. Best for smooth signals during
                 training where both endpoints are known.
    """
    DROP = "drop"
    HOLD = "hold"
    INTERPOLATE = "interpolate"


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
        temporal_fill: How to handle missing dst positions at the requested
            timestep. Only applies when t_dst is not None. Default HOLD.
        interpolation_order: For TemporalFill.INTERPOLATE, the IDW order.
            order=1 (default) gives linear lerp between nearest past and future.
            order=N uses up to N+1 anchor points weighted by 1/dist^N, normalized.
            Ignored for DROP and HOLD fill modes.
    """
    src: str
    dst: str
    weight: float = 1.0
    t_src: Optional[int] = None
    t_dst: Optional[int] = None
    fn: Optional[str] = None
    temporal_fill: TemporalFill = TemporalFill.HOLD
    interpolation_order: int = 1


# ── Temporal fill helpers ────────────────────────────────────────────

def _find_nearest_past(
    t_cache: Dict[int, List[int]], target_t: int,
) -> Optional[Tuple[int, int]]:
    """Find the most recent populated timestep before target_t.

    Returns (timestep, staleness) or None.
    """
    for t in range(target_t - 1, -1, -1):
        if t_cache.get(t):
            return (t, target_t - t)
    return None


def _find_nearest_future(
    t_cache: Dict[int, List[int]], target_t: int, max_T: int,
) -> Optional[Tuple[int, int]]:
    """Find the nearest populated timestep after target_t.

    Returns (timestep, distance) or None.
    """
    for t in range(target_t + 1, max_T):
        if t_cache.get(t):
            return (t, t - target_t)
    return None


def _resolve_temporal_fill(
    conn: "Connection",
    dst_t_cache: Dict[int, List[int]],
    abs_dst: int,
    max_T: int,
) -> List[Tuple[List[int], float]]:
    """Resolve dst indices and weights for a temporal connection with fill.

    When the dst has no positions at abs_dst, applies the connection's
    temporal_fill mode to find alternative positions and compute weights.

    Returns:
        List of (dst_indices, weight) pairs. Empty list = no connection.
        Multiple pairs possible for INTERPOLATE (past + future endpoints).
    """
    dst_at_t = dst_t_cache.get(abs_dst, [])
    if dst_at_t:
        return [(dst_at_t, conn.weight)]

    fill = conn.temporal_fill

    if fill == TemporalFill.DROP:
        return []

    if fill == TemporalFill.HOLD:
        past = _find_nearest_past(dst_t_cache, abs_dst)
        if past is not None:
            return [(dst_t_cache[past[0]], conn.weight)]
        return []

    if fill == TemporalFill.INTERPOLATE:
        order = conn.interpolation_order

        if order <= 1:
            # Linear lerp: use nearest past and future
            past = _find_nearest_past(dst_t_cache, abs_dst)
            future = _find_nearest_future(dst_t_cache, abs_dst, max_T)
            if past is not None and future is not None:
                past_t, past_d = past
                future_t, future_d = future
                span = past_d + future_d
                w_past = (future_d / span) * conn.weight
                w_future = (past_d / span) * conn.weight
                return [
                    (dst_t_cache[past_t], w_past),
                    (dst_t_cache[future_t], w_future),
                ]
            elif past is not None:
                # Only past available — fall back to hold
                return [(dst_t_cache[past[0]], conn.weight)]
            elif future is not None:
                # Only future available — use it
                return [(dst_t_cache[future[0]], conn.weight)]
            return []
        else:
            # Higher-order IDW: collect up to order+1 nearest anchor points
            # Gather all populated timesteps and their distances
            anchors: List[Tuple[int, int]] = []  # (timestep, distance)
            for t in sorted(dst_t_cache.keys()):
                if dst_t_cache.get(t):
                    dist = abs(t - abs_dst)
                    if dist > 0:
                        anchors.append((t, dist))

            if not anchors:
                return []

            # Sort by distance and take the closest order+1 anchors
            anchors.sort(key=lambda x: x[1])
            anchors = anchors[: order + 1]

            # Compute IDW weights: w_i = 1/dist^order, normalized
            raw_weights = [1.0 / (d ** order) for _, d in anchors]
            total = sum(raw_weights)
            if total < 1e-12:
                return []

            result = []
            for (t, _), rw in zip(anchors, raw_weights):
                w = (rw / total) * conn.weight
                result.append((dst_t_cache[t], w))
            return result

    return []


# ── Topology ─────────────────────────────────────────────────────────

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

        When dst has no positions at the requested timestep, the connection's
        temporal_fill mode determines behavior (see TemporalFill enum).

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
    def causal_temporal(
        regions: List[str],
        layout: Optional[CanvasLayout] = None,
        temporal_fill: TemporalFill = TemporalFill.HOLD,
    ) -> "CanvasTopology":
        """Temporal causal: same-frame self-attn + prev-frame cross-attn.

        Each region attends to itself at the same frame, and to all other
        regions at the previous frame. No future leakage.

        All temporal connections use the specified fill mode (default HOLD),
        ensuring fast regions can always read from slow regions' most recent
        values even when update frequencies don't align.

        Args:
            regions: Region names.
            layout: Optional layout (reserved for future period-aware logic).
            temporal_fill: Fill mode for all temporal connections.
        """
        conns = []
        for r in regions:
            # Same-frame self-attention
            conns.append(Connection(src=r, dst=r, t_src=0, t_dst=0,
                                    temporal_fill=temporal_fill))
            # Previous-frame self-attention (temporal context)
            conns.append(Connection(src=r, dst=r, t_src=0, t_dst=-1,
                                    temporal_fill=temporal_fill))
        # Cross-region: each region queries all others at previous frame
        for s in regions:
            for d in regions:
                if s != d:
                    conns.append(Connection(src=s, dst=d, t_src=0, t_dst=-1,
                                            temporal_fill=temporal_fill))
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

        The resolved fn string is dispatched by AttentionDispatcher
        (see canvas_engineering.dispatch) using ATTENTION_REGISTRY
        (see canvas_engineering.attention). Each fn maps to an nn.Module
        with forward(queries, keys, values) -> output.
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
        Value is the connection weight (may be fractional for INTERPOLATE).

        Temporal fill modes are applied when dst has no positions at the
        requested timestep:
            DROP:        No connection (mask stays 0).
            HOLD:        Connect to most recent available dst positions.
            INTERPOLATE: Connect to surrounding dst positions with IDW weights.
        """
        N = layout.num_positions
        mask = torch.zeros(N, N, device=device)

        # Cache region indices (all timesteps)
        idx_cache: Dict[str, List[int]] = {}
        for name in layout.regions:
            idx_cache[name] = layout.region_indices(name)

        # Cache per-timestep indices (canvas frame space)
        t_idx_cache: Dict[str, Dict[int, List[int]]] = {}
        for name in layout.regions:
            t_idx_cache[name] = {}
            for t in layout.region_timesteps(name):
                t_idx_cache[name][t] = layout.region_indices_at_t(name, t)

        # Cache per-region real-time indices for period-aware fill resolution.
        # Maps real_time → position_indices, exposing natural gaps between
        # updates of slow-period regions (e.g. period=4 → real times {0,4,8,...}).
        real_t_idx_cache: Dict[str, Dict[int, List[int]]] = {}
        max_real_T = 0
        for name in layout.regions:
            spec = layout.region_spec(name)
            real_t_idx_cache[name] = {}
            for canvas_t in layout.region_timesteps(name):
                real_t = canvas_t * spec.period
                real_t_idx_cache[name][real_t] = layout.region_indices_at_t(name, canvas_t)
                if real_t + 1 > max_real_T:
                    max_real_T = real_t + 1
        if max_real_T == 0:
            max_real_T = layout.T

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

            elif conn.t_dst is not None:
                # dst is temporally constrained — temporal fill applies.
                # Resolve in real-time space so period-mismatched regions
                # expose their natural inter-update gaps to INTERPOLATE.
                dst_real = real_t_idx_cache.get(conn.dst, {})
                for ref in range(layout.T):
                    # Resolve src positions (canvas frame space)
                    if conn.t_src is not None:
                        abs_src = ref + conn.t_src
                        src_at_t = t_idx_cache[conn.src].get(abs_src, [])
                        src_period = layout.region_spec(conn.src).period
                    else:
                        src_at_t = idx_cache[conn.src]
                        src_period = 1

                    if not src_at_t:
                        continue

                    # Convert target to real time for period-aware fill
                    target_real = (ref + conn.t_dst) * src_period
                    resolved = _resolve_temporal_fill(
                        conn, dst_real, target_real, max_real_T,
                    )
                    for dst_indices, w in resolved:
                        for si in src_at_t:
                            for di in dst_indices:
                                mask[si, di] = max(mask[si, di].item(), w)

            else:
                # t_src is not None, t_dst is None
                # src constrained, dst sees all its timesteps — no fill needed
                dst_idx = idx_cache[conn.dst]
                for ref in range(layout.T):
                    abs_src = ref + conn.t_src
                    src_at_t = t_idx_cache[conn.src].get(abs_src, [])
                    for si in src_at_t:
                        for di in dst_idx:
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
        lines = ["CanvasTopology ({} ops, {} regions):".format(
            len(self.connections), len(self.regions))]
        for (src, dst), w in sorted(adj.items()):
            if src == dst:
                lines.append("  {} ↺ self  (w={:.2f})".format(src, w))
            else:
                arrow = "↔" if (dst, src) in adj else "→"
                lines.append("  {} {} {}  (w={:.2f})".format(src, arrow, dst, w))

        if self.has_temporal_constraints:
            temporal = [c for c in self.connections if c.t_src is not None or c.t_dst is not None]
            fill_counts: Dict[str, int] = {}
            for c in temporal:
                fill_counts[c.temporal_fill.value] = fill_counts.get(c.temporal_fill.value, 0) + 1
            fill_summary = ", ".join("{}={}".format(f, n) for f, n in sorted(fill_counts.items()))
            lines.append("  [{} temporal constraints: {}]".format(len(temporal), fill_summary))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "CanvasTopology(connections={}, regions={})".format(
            len(self.connections), sorted(self.regions))

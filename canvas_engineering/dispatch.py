"""Per-connection attention dispatch.

Routes each topology connection to its resolved attention function,
extracting src/dst positions from the layout. Replaces monolithic
masked attention with heterogeneous per-edge computation.

Temporal fill modes are fully integrated: when a dst region has no
positions at the requested timestep, the connection's temporal_fill
mode determines how to resolve the dst (HOLD, INTERPOLATE, or DROP).

Usage:
    from canvas_engineering import CanvasTopology, CanvasLayout
    from canvas_engineering.dispatch import AttentionDispatcher

    dispatcher = AttentionDispatcher(
        topology=topology,
        layout=layout,
        d_model=256,
        n_heads=4,
    )

    # In forward pass:
    output = dispatcher(hidden_states)  # (B, N, d_model)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from canvas_engineering.attention import create_attention
from canvas_engineering.canvas import CanvasLayout
from canvas_engineering.connectivity import (
    CanvasTopology,
    Connection,
    TemporalFill,
    _resolve_temporal_fill,
)


class AttentionDispatcher(nn.Module):
    """Routes attention operations per topology connection.

    For each connection in topology.attention_ops(layout):
      1. Extract src/dst position indices from layout
      2. Gather queries from src positions, keys/values from dst positions
      3. Apply temporal fill logic for cross-frequency connections
      4. Run the resolved attention function
      5. Scale by connection weight
      6. Scatter-add results back to output

    Multiple connections writing to the same src region accumulate
    additively and are normalized by the total incoming weight.

    Args:
        topology: CanvasTopology declaring connections.
        layout: CanvasLayout declaring region positions.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        skip_temporal: If True, ignore temporal constraints (t_src/t_dst)
            and treat all connections as dense-in-time.
        residual_accumulator: Optional ResidualAccumulator for tracking
            error summaries on residual-carrier regions. Updated as a
            side effect during forward(). Access via .summaries property.
    """

    def __init__(
        self,
        topology: CanvasTopology,
        layout: CanvasLayout,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        skip_temporal: bool = False,
        residual_accumulator=None,
    ):
        super().__init__()
        self.topology = topology
        self.layout = layout
        self.d_model = d_model
        self.skip_temporal = skip_temporal
        self.residual_accumulator = residual_accumulator

        # Resolve all operations: (src, dst, weight, fn_name)
        ops = topology.attention_ops(layout)

        # Deduplicate attention module types — share parameters for same fn
        # within a dispatcher (each dispatcher = one layer).
        fn_modules: Dict[str, nn.Module] = {}
        self._op_specs: List[Tuple[str, str, float, str]] = []

        for src, dst, weight, fn_name in ops:
            if src not in layout.regions or dst not in layout.regions:
                continue
            self._op_specs.append((src, dst, weight, fn_name))
            if fn_name not in fn_modules:
                fn_modules[fn_name] = create_attention(
                    fn_name, d_model, n_heads, dropout
                )

        self.fn_modules = nn.ModuleDict(fn_modules)

        # Pre-compute position indices for each region (all timesteps)
        self._region_idx: Dict[str, torch.Tensor] = {}
        for name in layout.regions:
            idx = layout.region_indices(name)
            self._region_idx[name] = torch.tensor(idx, dtype=torch.long)

        # Pre-compute per-timestep indices for temporal connections
        self._region_t_idx: Dict[str, Dict[int, torch.Tensor]] = {}
        self._region_t_idx_lists: Dict[str, Dict[int, List[int]]] = {}
        # Real-time indexed cache for period-aware fill resolution
        self._region_real_t_lists: Dict[str, Dict[int, List[int]]] = {}
        self._max_real_T = layout.T
        if topology.has_temporal_constraints and not skip_temporal:
            for name in layout.regions:
                spec = layout.region_spec(name)
                self._region_t_idx[name] = {}
                self._region_t_idx_lists[name] = {}
                self._region_real_t_lists[name] = {}
                for t in layout.region_timesteps(name):
                    idx = layout.region_indices_at_t(name, t)
                    if idx:
                        self._region_t_idx[name][t] = torch.tensor(
                            idx, dtype=torch.long
                        )
                        self._region_t_idx_lists[name][t] = idx
                        real_t = t * spec.period
                        self._region_real_t_lists[name][real_t] = idx
                        if real_t + 1 > self._max_real_T:
                            self._max_real_T = real_t + 1

        # Pre-compute incoming weight sum per region for normalization
        self._incoming_weight: Dict[str, float] = {}
        for src, dst, weight, fn_name in self._op_specs:
            self._incoming_weight[src] = (
                self._incoming_weight.get(src, 0.0) + weight
            )


    def _get_device_idx(
        self, name: str, device: torch.device
    ) -> torch.Tensor:
        idx = self._region_idx[name]
        if idx.device != device:
            idx = idx.to(device)
            self._region_idx[name] = idx
        return idx

    def forward(self, x: torch.Tensor, active_regions: Optional[Set[str]] = None) -> torch.Tensor:
        """Run dispatched attention over all topology connections.

        Args:
            x: (B, N, d_model) hidden states.
            active_regions: Optional set of region names that should update.
                If None, all regions fire (backward compatible). If provided,
                connections where src is not in active_regions are skipped
                and those positions pass through from x unchanged.

        Returns:
            (B, N, d_model) updated hidden states. Positions not in any
            src region (or in inactive regions) are passed through unchanged.
        """
        B, N, D = x.shape
        device = x.device

        # Accumulate updates per position
        output = torch.zeros_like(x)
        weight_map = torch.zeros(N, device=device)

        for src, dst, weight, fn_name in self._op_specs:
            # Skip connections where src is inactive
            if active_regions is not None and src not in active_regions:
                continue
            conn_obj = None
            for c in self.topology.connections:
                if c.src == src and c.dst == dst and self.topology.resolve_fn(c, self.layout) == fn_name:
                    conn_obj = c
                    break

            has_temporal = (
                conn_obj is not None
                and (conn_obj.t_src is not None or conn_obj.t_dst is not None)
                and not self.skip_temporal
            )

            attn_fn = self.fn_modules[fn_name]

            if not has_temporal:
                # Dense in time: all src positions attend to all dst positions
                src_idx = self._get_device_idx(src, device)
                dst_idx = self._get_device_idx(dst, device)

                queries = x[:, src_idx]  # (B, N_src, D)
                keys = x[:, dst_idx]  # (B, N_dst, D)
                values = x[:, dst_idx]

                attended = attn_fn(queries, keys, values)  # (B, N_src, D)
                output[:, src_idx] += attended * weight
                weight_map[src_idx] += weight

                # Track residual if dst is a residual-carrier region
                if self.residual_accumulator is not None:
                    try:
                        if self.layout.region_spec(dst).carrier == "residual":
                            self.residual_accumulator.update(dst, (attended - queries).detach())
                    except (KeyError, AttributeError):
                        pass
            else:
                # Temporal: iterate reference frames with fill logic
                t_src_off = conn_obj.t_src
                t_dst_off = conn_obj.t_dst

                for ref in range(self.layout.T):
                    # Resolve src indices
                    if t_src_off is not None:
                        abs_src = ref + t_src_off
                        src_t_dict = self._region_t_idx.get(src, {})
                        if abs_src not in src_t_dict:
                            continue
                        s_idx = src_t_dict[abs_src].to(device)
                    else:
                        s_idx = self._get_device_idx(src, device)

                    if len(s_idx) == 0:
                        continue

                    # Resolve dst indices with temporal fill (real-time space)
                    if t_dst_off is not None:
                        if t_src_off is not None:
                            src_period = self.layout.region_spec(src).period
                        else:
                            src_period = 1
                        target_real = (ref + t_dst_off) * src_period
                        dst_real_lists = self._region_real_t_lists.get(dst, {})
                        resolved = _resolve_temporal_fill(
                            conn_obj, dst_real_lists, target_real,
                            self._max_real_T,
                        )

                        for dst_indices, w in resolved:
                            d_idx = torch.tensor(
                                dst_indices, dtype=torch.long, device=device,
                            )

                            queries = x[:, s_idx]
                            keys = x[:, d_idx]
                            values = x[:, d_idx]

                            attended = attn_fn(queries, keys, values)
                            output[:, s_idx] += attended * w
                            weight_map[s_idx] += w
                    else:
                        # dst is dense (t_dst=None), no fill needed
                        d_idx = self._get_device_idx(dst, device)
                        if len(d_idx) == 0:
                            continue

                        queries = x[:, s_idx]
                        keys = x[:, d_idx]
                        values = x[:, d_idx]

                        attended = attn_fn(queries, keys, values)
                        output[:, s_idx] += attended * weight
                        weight_map[s_idx] += weight

        # Normalize by total incoming weight (avoid divide-by-zero)
        nonzero = weight_map > 0
        if nonzero.any():
            output[:, nonzero] = output[:, nonzero] / weight_map[nonzero].unsqueeze(0).unsqueeze(-1)

        # Pass through positions not in any src region
        not_attended = ~nonzero
        if not_attended.any():
            output[:, not_attended] = x[:, not_attended]

        return output

    @property
    def summaries(self):
        """Current residual summaries, or None if no accumulator."""
        if self.residual_accumulator is None:
            return None
        return self.residual_accumulator.summaries()

    def __repr__(self) -> str:
        fn_counts: Dict[str, int] = {}
        for _, _, _, fn in self._op_specs:
            fn_counts[fn] = fn_counts.get(fn, 0) + 1
        fn_summary = ", ".join(f"{fn}={n}" for fn, n in sorted(fn_counts.items()))
        return (
            f"AttentionDispatcher("
            f"{len(self._op_specs)} ops, "
            f"{len(self.fn_modules)} fn types: {fn_summary})"
        )

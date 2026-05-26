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
from canvas_engineering.program import (
    CanvasProgram,
    ConnectionProgram,
    evaluate_trigger,
    resolve_operator_defaults,
)
from canvas_engineering.cortex import CortexRegistry
from canvas_engineering.identity import IdentitySpec, SlotBindingModule
from canvas_engineering.masks import MaskSpec, mask_to_index_pairs


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
        program: Optional[CanvasProgram] = None,
        cortex_registry: Optional[CortexRegistry] = None,
    ):
        super().__init__()
        self.topology = topology
        self.layout = layout
        self.d_model = d_model
        self.n_heads = n_heads
        self.skip_temporal = skip_temporal
        self.residual_accumulator = residual_accumulator
        self.program = program
        self.cortex_registry = cortex_registry

        # Resolve effective per-edge (fn, write_mode, trigger). Operator
        # defaults from ``OPERATOR_DEFAULTS`` apply when the
        # ConnectionProgram leaves a field at its bare default. The cortex
        # registry overrides ``fn`` for intra-cortex edges with the
        # cortex's declared ``local_backend``.
        ops = topology.attention_ops(layout)

        fn_modules: Dict[str, nn.Module] = {}
        # _op_specs entries: (src, dst, weight, fn_name, write_mode, trigger)
        self._op_specs: List[Tuple[str, str, float, str, str, Optional[str]]] = []

        for src, dst, weight, fn_name in ops:
            if src not in layout.regions or dst not in layout.regions:
                continue
            cp = program.connections.get((src, dst)) if program is not None else None
            eff_fn, eff_write_mode, eff_trigger = resolve_operator_defaults(cp, fn_name)
            # Cortex local-backend override for intra-cortex edges.
            if cortex_registry is not None and cortex_registry.same_cortex(src, dst):
                cortex = cortex_registry.cortex_for(src)
                if cortex is not None:
                    eff_fn = cortex.local_backend
            self._op_specs.append((src, dst, weight, eff_fn, eff_write_mode, eff_trigger))
            if eff_fn not in fn_modules:
                fn_modules[eff_fn] = create_attention(
                    eff_fn, d_model, n_heads, dropout
                )

        self.fn_modules = nn.ModuleDict(fn_modules)

        # Learned gate scalars per (src, dst, fn) for write_mode="gate".
        self._gate_params = nn.ParameterDict()
        for src, dst, _w, fn_name, write_mode, _trig in self._op_specs:
            if write_mode == "gate":
                key = self._gate_key(src, dst, fn_name)
                if key not in self._gate_params:
                    self._gate_params[key] = nn.Parameter(torch.zeros(1))

        # SlotBindingModule per region with IdentitySpec. These run on the
        # *values* gathered from a dst region before the attention reduces
        # them, projecting variable-count observations into a fixed-size
        # slot bank.
        self._slot_modules = nn.ModuleDict()
        if program is not None:
            for name, rp in program.regions.items():
                if rp.identity is None or name not in layout.regions:
                    continue
                self._slot_modules[name] = SlotBindingModule(
                    d_model=d_model,
                    n_slots=rp.identity.slot_capacity,
                    n_heads=n_heads,
                    birth_death=rp.identity.birth_death,
                )

        # Pre-computed mask-derived index pairs per edge. When a
        # ConnectionProgram declares a ``mask_spec``, the dispatcher
        # iterates these pairs instead of attending dense src↔dst.
        self._mask_pairs: Dict[Tuple[str, str], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        if program is not None:
            for (src, dst), cp in program.connections.items():
                if cp.mask_spec is None:
                    continue
                if src not in layout.regions or dst not in layout.regions:
                    continue
                pairs = mask_to_index_pairs(cp.mask_spec, layout, src, dst)
                tensor_pairs = [
                    (
                        torch.tensor(s_idx, dtype=torch.long),
                        torch.tensor(d_idx, dtype=torch.long),
                    )
                    for s_idx, d_idx in pairs
                ]
                self._mask_pairs[(src, dst)] = tensor_pairs

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
        for src, _dst, weight, _fn, _wm, _trig in self._op_specs:
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

    @staticmethod
    def _gate_key(src: str, dst: str, fn_name: str) -> str:
        # ParameterDict keys cannot contain '.'; sanitize names.
        s = src.replace(".", "_")
        d = dst.replace(".", "_")
        f = fn_name.replace(".", "_")
        return f"{s}__{d}__{f}"

    def _resolve_connection_program(
        self, src: str, dst: str
    ) -> Optional[ConnectionProgram]:
        if self.program is None:
            return None
        return self.program.connections.get((src, dst))

    def forward(
        self,
        x: torch.Tensor,
        active_regions: Optional[Set[str]] = None,
        summaries: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> torch.Tensor:
        """Run dispatched attention over all topology connections.

        Args:
            x: (B, N, d_model) hidden states.
            active_regions: Optional set of region names that should update.
                If None, all regions fire (backward compatible). If provided,
                connections where src is not in active_regions are skipped
                and those positions pass through from x unchanged.
            summaries: Optional residual summaries used to evaluate per-edge
                ``ConnectionProgram.trigger`` expressions when a program is
                attached. If None and the dispatcher has a residual
                accumulator, the accumulator's current summaries are used.

        Returns:
            (B, N, d_model) updated hidden states. Positions not in any
            src region (or in inactive regions) are passed through unchanged.

        Write modes (from ``ConnectionProgram.write_mode``):
            - ``add`` (default): contributions are weighted and the resulting
              accumulator is normalized by total incoming weight.
            - ``replace``: contribution overwrites the destination position
              entirely (last-writer-wins among replace edges).
            - ``gate``: contribution is scaled by a learned per-edge sigmoid
              gate and summed (no normalization). May coexist with ``add``
              contributions at the same position; the gated sum is added
              to the normalized additive value.
        """
        B, N, D = x.shape
        device = x.device

        # Resolve summaries source for trigger evaluation.
        eval_summaries = summaries
        if eval_summaries is None and self.residual_accumulator is not None:
            try:
                eval_summaries = self.residual_accumulator.summaries()
            except Exception:
                eval_summaries = None

        # Per-mode accumulators.
        add_output = torch.zeros_like(x)
        add_weight = torch.zeros(N, device=device)
        gate_output = torch.zeros_like(x)
        gate_mask = torch.zeros(N, dtype=torch.bool, device=device)
        replace_output = torch.zeros_like(x)
        replace_mask = torch.zeros(N, dtype=torch.bool, device=device)

        for src, dst, weight, fn_name, write_mode, eff_trigger in self._op_specs:
            # Skip connections where src is inactive
            if active_regions is not None and src not in active_regions:
                continue

            # Evaluate the effective trigger (cp.trigger > operator default).
            if eff_trigger is not None and not evaluate_trigger(eff_trigger, eval_summaries):
                continue

            # Find the underlying Connection (for temporal info). The
            # original Connection.fn (or region default) may differ from
            # the dispatcher's effective fn after operator/cortex
            # overrides, so match on (src, dst) only — temporal info is
            # carried on the Connection itself regardless of fn.
            conn_obj = None
            for c in self.topology.connections:
                if c.src == src and c.dst == dst:
                    conn_obj = c
                    break

            has_temporal = (
                conn_obj is not None
                and (conn_obj.t_src is not None or conn_obj.t_dst is not None)
                and not self.skip_temporal
            )

            attn_fn = self.fn_modules[fn_name]
            gate_key = self._gate_key(src, dst, fn_name) if write_mode == "gate" else None

            # SlotBindingModule wrapper for dst values (when the dst
            # region declares an IdentitySpec). Bind the gathered
            # observations to a fixed-size slot bank before attention.
            slot_module = self._slot_modules[dst] if dst in self._slot_modules else None
            mask_pairs = self._mask_pairs.get((src, dst))

            def _commit(s_idx: torch.Tensor, attended: torch.Tensor, w: float) -> None:
                if write_mode == "replace":
                    replace_output[:, s_idx] = attended
                    replace_mask[s_idx] = True
                elif write_mode == "gate":
                    g = torch.sigmoid(self._gate_params[gate_key])
                    gate_output[:, s_idx] = gate_output[:, s_idx] + attended * g
                    gate_mask[s_idx] = True
                else:  # add
                    add_output[:, s_idx] = add_output[:, s_idx] + attended * w
                    add_weight[s_idx] = add_weight[s_idx] + w

            if not has_temporal:
                # Dense in time. Iterate mask-derived (src_idx, dst_idx)
                # pairs if a MaskSpec was attached to this edge; otherwise
                # one pair covering all of src ↔ all of dst.
                if mask_pairs is not None:
                    pair_iter = [
                        (s_idx.to(device), d_idx.to(device))
                        for s_idx, d_idx in mask_pairs
                    ]
                else:
                    pair_iter = [(
                        self._get_device_idx(src, device),
                        self._get_device_idx(dst, device),
                    )]

                for src_idx, dst_idx in pair_iter:
                    if len(src_idx) == 0 or len(dst_idx) == 0:
                        continue
                    queries = x[:, src_idx]
                    keys = x[:, dst_idx]
                    values = x[:, dst_idx]
                    if slot_module is not None:
                        # Bind dst observations to slot bank.
                        bound = slot_module(values)
                        keys = bound
                        values = bound

                    attended = attn_fn(queries, keys, values)
                    _commit(src_idx, attended, weight)

                    if self.residual_accumulator is not None:
                        try:
                            if self.layout.region_spec(dst).carrier == "residual":
                                self.residual_accumulator.update(
                                    dst, (attended - queries).detach()
                                )
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
                            if slot_module is not None:
                                bound = slot_module(values)
                                keys = bound
                                values = bound

                            attended = attn_fn(queries, keys, values)
                            _commit(s_idx, attended, w)
                    else:
                        # dst is dense (t_dst=None), no fill needed
                        d_idx = self._get_device_idx(dst, device)
                        if len(d_idx) == 0:
                            continue

                        queries = x[:, s_idx]
                        keys = x[:, d_idx]
                        values = x[:, d_idx]
                        if slot_module is not None:
                            bound = slot_module(values)
                            keys = bound
                            values = bound

                        attended = attn_fn(queries, keys, values)
                        _commit(s_idx, attended, weight)

        # Combine accumulators.
        # Precedence: replace > (add normalized + gate sum) > pass-through.
        output = torch.zeros_like(x)

        # Additive: normalize by weight.
        add_active = add_weight > 0
        if add_active.any():
            output[:, add_active] = (
                add_output[:, add_active]
                / add_weight[add_active].unsqueeze(0).unsqueeze(-1)
            )

        # Gate contributions stack on top of additive at the same positions
        # (or stand alone where no additive edges write).
        if gate_mask.any():
            output[:, gate_mask] = output[:, gate_mask] + gate_output[:, gate_mask]

        # Replace overrides whatever else wrote those positions.
        if replace_mask.any():
            output[:, replace_mask] = replace_output[:, replace_mask]

        # Pass through positions not touched by any mode.
        attended_mask = add_active | gate_mask | replace_mask
        not_attended = ~attended_mask
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
        for _src, _dst, _w, fn, _wm, _trig in self._op_specs:
            fn_counts[fn] = fn_counts.get(fn, 0) + 1
        fn_summary = ", ".join(f"{fn}={n}" for fn, n in sorted(fn_counts.items()))
        extras = []
        if self.cortex_registry is not None:
            extras.append(f"cortices={self.cortex_registry.n_cortices}")
        if len(self._slot_modules):
            extras.append(f"identity_slots={len(self._slot_modules)}")
        if self._mask_pairs:
            extras.append(f"masked_edges={len(self._mask_pairs)}")
        extra_str = (", " + ", ".join(extras)) if extras else ""
        return (
            f"AttentionDispatcher("
            f"{len(self._op_specs)} ops, "
            f"{len(self.fn_modules)} fn types: {fn_summary}{extra_str})"
        )

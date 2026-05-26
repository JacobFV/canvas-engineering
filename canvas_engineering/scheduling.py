"""Region scheduling: clock-driven firing decisions.

RegionScheduler evaluates ClockSpec rules to decide which regions
should update each step. Regions without clocks always fire. Regions
with periodic clocks fire on their period. Event-triggered regions
fire when a residual summary exceeds a threshold. Boundary regions
fire on named lifecycle events.

Usage:
    from canvas_engineering import CanvasProgram
    from canvas_engineering.scheduling import RegionScheduler

    scheduler = RegionScheduler(program)

    for t in range(100):
        active = scheduler.step(
            external_t=t,
            summaries=dispatcher.summaries,
            boundary="episode_end" if t == 99 else None,
        )
        output = dispatcher(canvas, active_regions=active)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from canvas_engineering.program import CanvasProgram, ClockSpec
from canvas_engineering.clock_ir import ClockContext


class RegionScheduler:
    """Evaluates clock rules to decide which regions fire each step.

    Each region's ClockSpec determines when it fires:
    - periodic: fires when external_t % period == 0
    - on_event: fires when a residual summary exceeds a threshold
    - boundary: fires on a named lifecycle event

    Regions without a ClockSpec always fire (backward compatible).

    Cooldown suppresses re-firing for N steps after a fire.
    max_silence forces firing after N steps of silence regardless.

    Internal microsteps:
        Regions with ``clock.domain == "internal"`` and ``clock.max_inner_steps``
        produce additional inner steps beyond the main external step.
        Use ``step_plan()`` to get a list of active-region sets:
        the first set is the normal external step, subsequent sets are
        internal microsteps for regions with internal clocks.
    """

    def __init__(self, program: CanvasProgram):
        self._program = program
        self._last_fired: Dict[str, int] = {}
        self._cooldown_until: Dict[str, int] = {}

    def step(
        self,
        external_t: int,
        summaries: Optional[Dict[str, Dict[str, float]]] = None,
        boundary: Optional[str] = None,
    ) -> Set[str]:
        """Determine which regions should fire this step.

        Args:
            external_t: Current external timestep.
            summaries: Residual summaries from ResidualAccumulator.summaries().
                Format: {region_name: {kind_name: float_value}}.
            boundary: Optional named boundary event (e.g., "episode_end").

        Returns:
            Set of region names that should update this step.
        """
        active: Set[str] = set()
        for name, rp in self._program.regions.items():
            clock = rp.clock
            if clock is None:
                active.add(name)
                continue
            if self._should_fire(name, clock, external_t, summaries, boundary):
                active.add(name)
                self._last_fired[name] = external_t
                self._cooldown_until[name] = external_t + clock.cooldown
        return active

    def step_plan(
        self,
        external_t: int,
        summaries: Optional[Dict[str, Dict[str, float]]] = None,
        boundary: Optional[str] = None,
    ) -> List[Set[str]]:
        """Returns a list of active-region sets, one per inner step.

        The first set is the external step (always present, same as step()).
        Subsequent sets are internal microsteps for regions with
        ``clock.domain == "internal"`` and ``clock.max_inner_steps``.

        Usage:
            plan = scheduler.step_plan(t, summaries)
            for active in plan:
                canvas = dispatcher(canvas, active_regions=active)
        """
        # First: normal external step
        external_active = self.step(external_t, summaries, boundary)
        plan: List[Set[str]] = [external_active]

        # Collect internal-domain regions that fired in the external step
        # and have max_inner_steps > 0
        internal_regions: List[str] = []
        max_inner = 0
        for name in external_active:
            rp = self._program.regions.get(name)
            if rp is None or rp.clock is None:
                continue
            clock = rp.clock
            if clock.domain == "internal" and clock.max_inner_steps is not None and clock.max_inner_steps > 0:
                internal_regions.append(name)
                max_inner = max(max_inner, clock.max_inner_steps)

        # Generate internal microsteps
        for inner_step in range(max_inner):
            inner_active: Set[str] = set()
            for name in internal_regions:
                rp = self._program.regions[name]
                clock = rp.clock
                if clock is not None and clock.max_inner_steps is not None:
                    if inner_step < clock.max_inner_steps:
                        inner_active.add(name)
            if inner_active:
                plan.append(inner_active)

        return plan

    def _should_fire(
        self,
        region: str,
        clock: ClockSpec,
        external_t: int,
        summaries: Optional[Dict[str, Dict[str, float]]],
        boundary: Optional[str],
    ) -> bool:
        """Evaluate one region's clock rule."""
        # Check cooldown
        if external_t < self._cooldown_until.get(region, 0):
            return False

        # ClockExpr IR takes precedence over flat mode-based fields.
        # Combinators (And/Or/Not) and decorators (Cooldown/MaxSilence) on
        # the expression itself govern firing entirely.
        if clock.expr is not None:
            ctx = ClockContext(
                external_t=external_t,
                summaries=summaries,
                boundary=boundary,
                last_fired=self._last_fired.get(region, -1),
                cooldown_until=self._cooldown_until.get(region, 0),
            )
            return bool(clock.expr.evaluate(ctx))

        # Check max_silence: force fire if silent too long
        if clock.max_silence is not None:
            last = self._last_fired.get(region, -1)
            if last < 0 or (external_t - last) >= clock.max_silence:
                return True

        # Mode-specific firing
        if clock.mode == "periodic":
            return external_t % max(clock.period, 1) == 0

        elif clock.mode == "on_event":
            if summaries is None or clock.event_source is None:
                return False
            # event_source format: "region_name.kind_name"
            parts = clock.event_source.rsplit(".", 1)
            if len(parts) != 2:
                return False
            src_region, kind = parts
            val = summaries.get(src_region, {}).get(kind, 0.0)
            return val > clock.event_threshold

        elif clock.mode == "boundary":
            return boundary is not None and boundary == clock.event_source

        # Unknown mode: always fire (safe default)
        return True

    def reset(self) -> None:
        """Reset scheduler state (e.g., between episodes)."""
        self._last_fired.clear()
        self._cooldown_until.clear()

    def __repr__(self) -> str:
        n_clocked = sum(1 for rp in self._program.regions.values() if rp.clock is not None)
        return "RegionScheduler(regions={}, clocked={})".format(
            len(self._program.regions), n_clocked)


# ── Learned scheduling ──────────────────────────────────────────────


class LearnedScheduler(nn.Module):
    """Learns which regions to activate based on residual summaries.

    Uses a small MLP to score regions and selects top-k via
    straight-through Gumbel top-k for differentiable discrete
    selection.

    Args:
        n_regions: Number of regions to score.
        summary_dim: Dimension of the flattened summary input.
        max_active: Maximum number of regions to activate per step.
        temperature: Gumbel softmax temperature. Lower = more discrete.

    Usage:
        scheduler = LearnedScheduler(n_regions=5, summary_dim=10, max_active=3)
        summaries_flat = torch.randn(1, 10)  # flattened residual summaries
        active_indices, log_probs = scheduler(summaries_flat)
    """

    def __init__(
        self,
        n_regions: int,
        summary_dim: int,
        max_active: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_regions = n_regions
        self.max_active = min(max_active, n_regions)
        self.temperature = temperature
        self.scorer = nn.Sequential(
            nn.Linear(summary_dim, n_regions * 2),
            nn.GELU(),
            nn.Linear(n_regions * 2, n_regions),
        )

    def forward(self, summaries: torch.Tensor) -> Tuple[Set[int], torch.Tensor]:
        """Score regions, select top-k.

        Args:
            summaries: (summary_dim,) or (B, summary_dim) flattened residual
                summaries. If batched, uses the first item for selection
                (scheduling is per-step, not per-sample).

        Returns:
            Tuple of:
            - active_indices: Set of int region indices to activate.
            - log_probs: (n_regions,) tensor of log-probabilities for each
              region being selected (for REINFORCE-style training).
        """
        if summaries.dim() == 1:
            summaries = summaries.unsqueeze(0)

        # Score regions
        scores = self.scorer(summaries[0:1]).squeeze(0)  # (n_regions,)

        # Straight-through Gumbel top-k
        if self.training:
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(scores).clamp(min=1e-8)
            ))
            perturbed = (scores + gumbel_noise) / max(self.temperature, 1e-8)
        else:
            perturbed = scores

        # Select top-k
        k = min(self.max_active, self.n_regions)
        topk_vals, topk_idx = torch.topk(perturbed, k)
        active_indices = set(topk_idx.tolist())

        # Compute log-probabilities for all regions
        log_probs = torch.log_softmax(scores, dim=0)

        return active_indices, log_probs

    def __repr__(self) -> str:
        return "LearnedScheduler(n_regions={}, max_active={}, temperature={})".format(
            self.n_regions, self.max_active, self.temperature)


# ── Hybrid scheduling ───────────────────────────────────────────────


class HybridScheduler(nn.Module):
    """Compose ``RegionScheduler`` (rule-based) with ``LearnedScheduler``.

    Some regions fire according to declarative ``ClockSpec`` rules; others
    are gated by a small learned MLP that scores them from residual
    summaries. ``HybridScheduler`` produces a unified ``Set[str]`` of
    region names to fire per step.

    Args:
        program: The ``CanvasProgram`` providing rule-based clocks.
        learned_regions: Names of regions whose firing decisions are
            taken by ``LearnedScheduler`` instead of their declared
            ``ClockSpec``. If empty, behaves like ``RegionScheduler``.
        summary_dim: Dimension of the flattened summary input that the
            learned scheduler consumes.
        max_active: Maximum number of learned-region activations per step.
        temperature: Gumbel temperature for the learned scheduler.

    Usage:
        sched = HybridScheduler(program, learned_regions=["belief"],
                                summary_dim=16, max_active=1)
        active = sched.step(t, summaries={"err": {"prediction": 0.4}})
        # -> active is a set of region names (rule-based ∪ learned)
    """

    def __init__(
        self,
        program: CanvasProgram,
        learned_regions: Optional[List[str]] = None,
        summary_dim: int = 0,
        max_active: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.region_scheduler = RegionScheduler(program)
        self._learned_names: List[str] = list(learned_regions or [])
        self._learned_set: Set[str] = set(self._learned_names)
        self._idx_to_name: Dict[int, str] = {
            i: name for i, name in enumerate(self._learned_names)
        }
        if self._learned_names:
            if summary_dim <= 0:
                raise ValueError(
                    "HybridScheduler requires summary_dim > 0 when "
                    "learned_regions is non-empty"
                )
            self.learned_scheduler: Optional[LearnedScheduler] = LearnedScheduler(
                n_regions=len(self._learned_names),
                summary_dim=summary_dim,
                max_active=max_active,
                temperature=temperature,
            )
        else:
            self.learned_scheduler = None

    def step(
        self,
        external_t: int,
        summaries: Optional[Dict[str, Dict[str, float]]] = None,
        boundary: Optional[str] = None,
        summary_tensor: Optional[torch.Tensor] = None,
    ) -> Set[str]:
        """Return the set of region names that fire this step.

        Args:
            external_t: Current external timestep.
            summaries: Residual summaries (forwarded to RegionScheduler).
            boundary: Optional boundary event name.
            summary_tensor: Flat ``(summary_dim,)`` tensor consumed by the
                learned scheduler. Required when ``learned_regions`` is
                non-empty. If omitted while learned regions exist, the
                learned regions simply don't fire this step.
        """
        rule_active = self.region_scheduler.step(external_t, summaries, boundary)
        # Strip learned-controlled regions from the rule-based result —
        # the learned scheduler has sole authority over those.
        rule_active = rule_active - self._learned_set

        if self.learned_scheduler is None or summary_tensor is None:
            return rule_active

        learned_idx, _logp = self.learned_scheduler(summary_tensor)
        learned_active = {self._idx_to_name[i] for i in learned_idx
                          if i in self._idx_to_name}
        return rule_active | learned_active

    def reset(self) -> None:
        """Reset internal scheduling state (e.g., between episodes)."""
        self.region_scheduler.reset()

    def __repr__(self) -> str:
        return "HybridScheduler(learned={}, rule={})".format(
            len(self._learned_names),
            len(self.region_scheduler._program.regions) - len(self._learned_names),
        )

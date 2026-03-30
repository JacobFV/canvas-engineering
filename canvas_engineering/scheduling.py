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

from typing import Dict, Optional, Set

from canvas_engineering.program import CanvasProgram, ClockSpec


class RegionScheduler:
    """Evaluates clock rules to decide which regions fire each step.

    Each region's ClockSpec determines when it fires:
    - periodic: fires when external_t % period == 0
    - on_event: fires when a residual summary exceeds a threshold
    - boundary: fires on a named lifecycle event

    Regions without a ClockSpec always fire (backward compatible).

    Cooldown suppresses re-firing for N steps after a fire.
    max_silence forces firing after N steps of silence regardless.
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

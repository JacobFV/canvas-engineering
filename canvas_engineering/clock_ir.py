"""Clock expression IR: composable AST for firing rules.

ClockExpr nodes describe *when* a region fires. They compose via
And/Or/Not combinators and serialize to/from plain dicts for JSON
persistence.

Leaf nodes:
    Periodic(n, domain)   — fire every N steps in the given time domain
    OnEvent(source, gt, lt, domain) — fire when a residual summary crosses a threshold
    Boundary(name)        — fire on a named lifecycle event

Combinators:
    And(a, b)     — fire when both children fire
    Or(a, b)      — fire when either child fires
    Not(child)    — invert child (fire when child does not fire)

Decorators:
    Cooldown(child, steps)     — suppress re-firing for N steps after fire
    MaxSilence(child, steps)   — force fire after N steps of silence

Sugar functions:
    periodic(n, domain)   — shorthand for Periodic
    on(source, gt, lt)    — shorthand for OnEvent
    boundary(name)        — shorthand for Boundary

Usage:
    from canvas_engineering.clock_ir import periodic, on, boundary, And, Or

    # Fire every 4 steps OR when prediction error > 0.5
    expr = Or(periodic(4), on("err.prediction", gt=0.5))

    # Fire every step but not during cooldown, with max silence safety net
    expr = MaxSilence(Cooldown(periodic(1), steps=3), steps=10)

    # Serialize / deserialize
    d = expr.to_dict()
    loaded = ClockExpr.from_dict(d)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ── Context passed to evaluate() ───────────────────────────────────


@dataclass
class ClockContext:
    """Runtime context for evaluating clock expressions.

    Attributes:
        external_t: Current external timestep.
        summaries: Residual summaries dict {region: {kind: float}}.
        boundary: Optional named boundary event this step.
        last_fired: Timestep the owning region last fired (or -1 if never).
        cooldown_until: Timestep until which cooldown is active (0 if none).
    """
    external_t: int = 0
    summaries: Optional[Dict[str, Dict[str, float]]] = None
    boundary: Optional[str] = None
    last_fired: int = -1
    cooldown_until: int = 0


# ── Base class ──────────────────────────────────────────────────────


class ClockExpr:
    """Base class for clock expression AST nodes."""

    def evaluate(self, ctx: ClockContext) -> bool:
        """Evaluate this expression in the given context."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> ClockExpr:
        """Deserialize from a dict.  Dispatches on the 'type' key."""
        _REGISTRY = {
            "periodic": Periodic,
            "on_event": OnEvent,
            "boundary_expr": BoundaryExpr,
            "and": And,
            "or": Or,
            "not": Not,
            "cooldown": Cooldown,
            "max_silence": MaxSilence,
        }
        kind = d.get("type", "")
        factory = _REGISTRY.get(kind)
        if factory is None:
            raise ValueError(f"Unknown ClockExpr type: {kind!r}")
        return factory._from_dict(d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Leaf nodes ──────────────────────────────────────────────────────


class Periodic(ClockExpr):
    """Fire every *period* steps in the given time domain."""

    def __init__(self, period: int = 1, domain: str = "external"):
        self.period = max(period, 1)
        self.domain = domain

    def evaluate(self, ctx: ClockContext) -> bool:
        return ctx.external_t % self.period == 0

    def to_dict(self) -> dict:
        d: dict = {"type": "periodic", "period": self.period}
        if self.domain != "external":
            d["domain"] = self.domain
        return d

    @classmethod
    def _from_dict(cls, d: dict) -> Periodic:
        return cls(period=d.get("period", 1), domain=d.get("domain", "external"))

    def __repr__(self) -> str:
        return f"Periodic(period={self.period}, domain={self.domain!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Periodic):
            return NotImplemented
        return self.period == other.period and self.domain == other.domain


class OnEvent(ClockExpr):
    """Fire when a residual summary crosses a threshold.

    The *source* string has the format ``"region_name.kind_name"`` and
    is looked up in ``ctx.summaries``.  At least one of *gt* or *lt*
    must be supplied; when both are given the condition is ``gt < val < lt``
    (i.e., value must be within a band).
    """

    def __init__(
        self,
        source: str,
        gt: Optional[float] = None,
        lt: Optional[float] = None,
        domain: str = "external",
    ):
        if gt is None and lt is None:
            raise ValueError("OnEvent requires at least one of gt= or lt=")
        self.source = source
        self.gt = gt
        self.lt = lt
        self.domain = domain

    def evaluate(self, ctx: ClockContext) -> bool:
        if ctx.summaries is None:
            return False
        parts = self.source.rsplit(".", 1)
        if len(parts) != 2:
            return False
        region, kind = parts
        val = ctx.summaries.get(region, {}).get(kind, 0.0)
        if self.gt is not None and val <= self.gt:
            return False
        if self.lt is not None and val >= self.lt:
            return False
        return True

    def to_dict(self) -> dict:
        d: dict = {"type": "on_event", "source": self.source}
        if self.gt is not None:
            d["gt"] = self.gt
        if self.lt is not None:
            d["lt"] = self.lt
        if self.domain != "external":
            d["domain"] = self.domain
        return d

    @classmethod
    def _from_dict(cls, d: dict) -> OnEvent:
        return cls(
            source=d["source"],
            gt=d.get("gt"),
            lt=d.get("lt"),
            domain=d.get("domain", "external"),
        )

    def __repr__(self) -> str:
        parts = [f"source={self.source!r}"]
        if self.gt is not None:
            parts.append(f"gt={self.gt}")
        if self.lt is not None:
            parts.append(f"lt={self.lt}")
        return f"OnEvent({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OnEvent):
            return NotImplemented
        return (self.source == other.source and self.gt == other.gt
                and self.lt == other.lt and self.domain == other.domain)


class BoundaryExpr(ClockExpr):
    """Fire on a named lifecycle boundary event."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, ctx: ClockContext) -> bool:
        return ctx.boundary is not None and ctx.boundary == self.name

    def to_dict(self) -> dict:
        return {"type": "boundary_expr", "name": self.name}

    @classmethod
    def _from_dict(cls, d: dict) -> BoundaryExpr:
        return cls(name=d["name"])

    def __repr__(self) -> str:
        return f"BoundaryExpr({self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundaryExpr):
            return NotImplemented
        return self.name == other.name


# ── Combinators ─────────────────────────────────────────────────────


class And(ClockExpr):
    """Fire when both children fire."""

    def __init__(self, left: ClockExpr, right: ClockExpr):
        self.left = left
        self.right = right

    def evaluate(self, ctx: ClockContext) -> bool:
        return self.left.evaluate(ctx) and self.right.evaluate(ctx)

    def to_dict(self) -> dict:
        return {"type": "and", "left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> And:
        return cls(
            left=ClockExpr.from_dict(d["left"]),
            right=ClockExpr.from_dict(d["right"]),
        )

    def __repr__(self) -> str:
        return f"And({self.left!r}, {self.right!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, And):
            return NotImplemented
        return self.left == other.left and self.right == other.right


class Or(ClockExpr):
    """Fire when either child fires."""

    def __init__(self, left: ClockExpr, right: ClockExpr):
        self.left = left
        self.right = right

    def evaluate(self, ctx: ClockContext) -> bool:
        return self.left.evaluate(ctx) or self.right.evaluate(ctx)

    def to_dict(self) -> dict:
        return {"type": "or", "left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> Or:
        return cls(
            left=ClockExpr.from_dict(d["left"]),
            right=ClockExpr.from_dict(d["right"]),
        )

    def __repr__(self) -> str:
        return f"Or({self.left!r}, {self.right!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Or):
            return NotImplemented
        return self.left == other.left and self.right == other.right


class Not(ClockExpr):
    """Invert: fire when the child does NOT fire."""

    def __init__(self, child: ClockExpr):
        self.child = child

    def evaluate(self, ctx: ClockContext) -> bool:
        return not self.child.evaluate(ctx)

    def to_dict(self) -> dict:
        return {"type": "not", "child": self.child.to_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> Not:
        return cls(child=ClockExpr.from_dict(d["child"]))

    def __repr__(self) -> str:
        return f"Not({self.child!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Not):
            return NotImplemented
        return self.child == other.child


# ── Decorators ──────────────────────────────────────────────────────


class Cooldown(ClockExpr):
    """Suppress re-firing for *steps* steps after the child fires."""

    def __init__(self, child: ClockExpr, steps: int = 1):
        self.child = child
        self.steps = max(steps, 0)

    def evaluate(self, ctx: ClockContext) -> bool:
        # If in cooldown, suppress
        if ctx.external_t < ctx.cooldown_until:
            return False
        return self.child.evaluate(ctx)

    def to_dict(self) -> dict:
        return {"type": "cooldown", "child": self.child.to_dict(), "steps": self.steps}

    @classmethod
    def _from_dict(cls, d: dict) -> Cooldown:
        return cls(child=ClockExpr.from_dict(d["child"]), steps=d.get("steps", 1))

    def __repr__(self) -> str:
        return f"Cooldown({self.child!r}, steps={self.steps})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cooldown):
            return NotImplemented
        return self.child == other.child and self.steps == other.steps


class MaxSilence(ClockExpr):
    """Force fire after *steps* steps of silence regardless of child."""

    def __init__(self, child: ClockExpr, steps: int = 10):
        self.child = child
        self.steps = max(steps, 1)

    def evaluate(self, ctx: ClockContext) -> bool:
        # Force fire if silent too long
        if ctx.last_fired < 0 or (ctx.external_t - ctx.last_fired) >= self.steps:
            return True
        return self.child.evaluate(ctx)

    def to_dict(self) -> dict:
        return {"type": "max_silence", "child": self.child.to_dict(), "steps": self.steps}

    @classmethod
    def _from_dict(cls, d: dict) -> MaxSilence:
        return cls(child=ClockExpr.from_dict(d["child"]), steps=d.get("steps", 10))

    def __repr__(self) -> str:
        return f"MaxSilence({self.child!r}, steps={self.steps})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaxSilence):
            return NotImplemented
        return self.child == other.child and self.steps == other.steps


# ── Sugar ───────────────────────────────────────────────────────────


def periodic(n: int = 1, domain: str = "external") -> ClockExpr:
    """Create a Periodic clock expression."""
    return Periodic(period=n, domain=domain)


def on(source: str, gt: Optional[float] = None, lt: Optional[float] = None,
       domain: str = "external") -> ClockExpr:
    """Create an OnEvent clock expression."""
    return OnEvent(source=source, gt=gt, lt=lt, domain=domain)


def boundary(name: str) -> ClockExpr:
    """Create a BoundaryExpr clock expression."""
    return BoundaryExpr(name=name)

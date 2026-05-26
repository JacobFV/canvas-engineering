"""Typed process semantics for canvas regions and connections.

CanvasProgram layers process semantics on top of CanvasSchema.
Each region gets a RegionProgram describing its family (what kind of
state it holds), carrier (what dynamics govern it), clock (when it
updates), learning recipe, and compile mode.

This is the v2 core addition: upgrading the library from a typed layout
DSL to a typed process compiler.

Usage:
    from canvas_engineering import (
        CanvasProgram, RegionProgram, ConnectionProgram,
        ClockSpec, LearningSpec, compile_program,
    )

    program = CanvasProgram(
        schema=schema,
        regions={
            "vision": RegionProgram(family="observation", carrier="deterministic"),
            "belief": RegionProgram(family="state", tags=("belief", "object")),
            "memory": RegionProgram(family="memory", compile_mode="export"),
            "error":  RegionProgram(family="residual"),
            "action": RegionProgram(family="action"),
        },
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from canvas_engineering.schema import CanvasSchema
from canvas_engineering.clock_ir import ClockExpr


# ── Constants ────────────────────────────────────────────────────────

REGION_FAMILIES: Set[str] = {"observation", "state", "memory", "residual", "action"}

CARRIERS: Set[str] = {"deterministic", "diffusive", "filter", "memory", "residual"}

OPERATORS: Set[str] = {
    "attend", "observe", "predict", "correct", "bind",
    "retrieve", "write", "act", "compress", "integrate",
    "intervene", "emit_residual",
}

WRITE_MODES: Set[str] = {"add", "replace", "gate"}

COMPILE_MODES: Set[str] = {"runtime", "freeze", "constant", "export"}


# ── Specs ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClockSpec:
    """When a region updates.

    Args:
        domain: Time domain. "external" = environment time, "boundary" = lifecycle events.
        mode: Firing rule. "periodic" = every N steps, "on_event" = threshold on
            residual summary, "boundary" = fires on named boundary event.
        period: For periodic mode, fire every N steps.
        event_source: For on_event/boundary mode, the source identifier.
            Format: "region_name.kind_name" for on_event, event name for boundary.
        event_threshold: For on_event mode, fire when summary exceeds this.
        cooldown: After firing, suppress for this many steps.
        max_silence: Force fire after this many steps of silence.
    """
    domain: str = "external"
    mode: str = "periodic"
    period: int = 1
    event_source: Optional[str] = None
    event_threshold: float = 0.0
    cooldown: int = 0
    max_silence: Optional[int] = None
    max_inner_steps: Optional[int] = None
    expr: Optional[ClockExpr] = None


@dataclass(frozen=True)
class LearningSpec:
    """How a region learns during training.

    Args:
        mode: Learning paradigm. "supervised" = task loss, "ssl_prediction" =
            self-supervised next-step/masked prediction, "posterior_match" =
            train student to match teacher posterior, "retrieval" = retrieval
            accuracy + write utility, "calibration" = uncertainty calibration,
            "none" = no learning.
        losses: Named loss functions to apply. Interpreted by the training loop.
        compile_mode: What happens at deploy. "runtime" = keep live, "freeze" =
            no more learning, "distill" = train small student, "constant" =
            materialize as buffer, "export" = save to disk and remove.
    """
    mode: str = "supervised"
    losses: Tuple[str, ...] = ()
    compile_mode: str = "runtime"


# ── Programs ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConstraintSpec:
    """Structural constraints on a region's dynamics.

    Constraints are declarative assertions that the region's learned
    dynamics should respect.  They are checked by ``validate_constraints()``
    and can be used by the compiler for optimization.

    Args:
        equivariance: Symmetry the region must respect.
            "translation" = shift-equivariant, "rotation" = rotation-equivariant,
            "permutation" = permutation-equivariant. None = unconstrained.
        conservation: Conservation law.
            "capacity" = total activation is conserved (competition),
            "energy" = total energy is conserved (Hamiltonian).
            None = unconstrained.
        causal_direction: Causal constraint on outgoing connections.
            "acyclic" = no cycles in the connection graph rooted here,
            "forward_only" = only connects to regions with later temporal
            bounds. None = unconstrained.
        monotonicity: Monotonicity constraint on the region's state.
            "non_decreasing" = state values can only increase over time,
            "non_increasing" = state values can only decrease over time.
            None = unconstrained.
    """
    equivariance: Optional[str] = None
    conservation: Optional[str] = None
    causal_direction: Optional[str] = None
    monotonicity: Optional[str] = None


_VALID_EQUIVARIANCES = {"translation", "rotation", "permutation"}
_VALID_CONSERVATIONS = {"capacity", "energy"}
_VALID_CAUSAL_DIRECTIONS = {"acyclic", "forward_only"}
_VALID_MONOTONICITIES = {"non_decreasing", "non_increasing"}


@dataclass(frozen=True)
class RegionProgram:
    """Process semantics for a canvas region.

    Args:
        family: What kind of state this region holds. One of: observation,
            state, memory, residual, action. Custom strings allowed.
        tags: Semantic sub-tags for nuance within a family. E.g., a state
            region might have tags=("belief", "object") or tags=("goal",).
        carrier: What dynamics govern this region. "deterministic" = standard
            forward updates, "diffusive" = noise/denoise, "filter" = predict/
            correct, "memory" = persistent lookup, "residual" = error traces.
        clock: When this region updates. None = always active.
        learning: Training recipe. None = use family default.
        compile_mode: Deploy behavior. "runtime" = keep live, "freeze" = no
            more learning, "constant" = materialize, "export" = save & remove.
        constraints: Structural constraints on dynamics. None = unconstrained.
        identity: Identity/slot persistence spec. None = no identity tracking.
    """
    family: str = "state"
    tags: Tuple[str, ...] = ()
    carrier: str = "deterministic"
    clock: Optional[ClockSpec] = None
    learning: Optional[LearningSpec] = None
    compile_mode: str = "runtime"
    constraints: Optional[ConstraintSpec] = None
    identity: Optional["IdentitySpec"] = None

    def effective_learning(self) -> "LearningSpec":
        """Resolve the effective learning recipe for this region.

        Returns the explicit ``learning`` field if set, otherwise the
        family default from ``canvas_engineering.learning.FAMILY_DEFAULTS``.
        Unknown families fall back to a generic ``LearningSpec()``.
        """
        if self.learning is not None:
            return self.learning
        # Local import to avoid a circular learning.py → program.py edge.
        from canvas_engineering.learning import default_learning
        return default_learning(self.family)

    def effective_compile_mode(self) -> str:
        """Resolve the effective compile mode (region override > learning > runtime)."""
        if self.compile_mode != "runtime":
            return self.compile_mode
        ls = self.effective_learning()
        return ls.compile_mode


@dataclass(frozen=True)
class ConnectionProgram:
    """Process semantics for a canvas connection.

    Args:
        operator: Semantic intent of this edge. "attend" = generic attention
            (backward compat default), "observe" = write evidence, "predict" =
            propagate forward, "correct" = error-driven update, etc. Each
            operator carries default (fn, write_mode, trigger) semantics —
            see ``OPERATOR_DEFAULTS``.
        trigger: Optional condition for firing. Serializable string expression
            referencing residual summaries. E.g., "vision.err.prediction > 0.25".
            When None, the operator's default trigger applies (if any).
        write_mode: How output accumulates at destination. "add" = additive
            (current behavior), "replace" = overwrite, "gate" = learned gate.
            When left at the default and the operator declares a non-default
            write_mode, the operator's default applies at dispatch time.
        mask_spec: Optional ``MaskSpec`` controlling the sparsity pattern of
            attention between src and dst (full / tile / sparse). When None,
            the dispatcher uses dense src↔dst attention.
    """
    operator: str = "attend"
    trigger: Optional[str] = None
    write_mode: str = "add"
    mask_spec: Optional["MaskSpec"] = None


# ── Per-operator dispatch defaults ───────────────────────────────────
#
# Each operator maps to a default (attention fn, write_mode, trigger).
# At dispatch time, ConnectionProgram values OVERRIDE these:
#   - ``cp.write_mode`` overrides if it's not the bare default "add".
#   - ``cp.trigger`` overrides if not None.
#   - The connection's explicit ``fn=`` (on the Connection itself)
#     always wins over the operator's default fn.

OPERATOR_DEFAULTS: Dict[str, Tuple[Optional[str], str, Optional[str]]] = {
    # operator       → (default_fn,             default_write_mode, default_trigger)
    "attend":         (None,                    "add",     None),
    "observe":        (None,                    "add",     None),
    "predict":        (None,                    "add",     None),
    "correct":        (None,                    "add",     None),
    "bind":           (None,                    "add",     None),
    "retrieve":       ("cross_attention",       "add",     None),
    "write":          ("copy_attention",        "replace", None),
    "act":            ("copy_attention",        "replace", None),
    "compress":       ("pooling_attention",     "add",     None),
    "integrate":      (None,                    "add",     None),
    "intervene":      (None,                    "replace", None),
    "emit_residual":  (None,                    "add",     None),
}


def resolve_operator_defaults(
    cp: Optional[ConnectionProgram],
    base_fn: str,
) -> Tuple[str, str, Optional[str]]:
    """Resolve the effective (fn, write_mode, trigger) for an edge.

    Args:
        cp: The ConnectionProgram for this edge (or None).
        base_fn: The attention fn already resolved from the Connection
            and region defaults (the "explicit" fn — wins over operator).

    Returns:
        Tuple of (effective_fn, effective_write_mode, effective_trigger).
        For unknown operators all fields fall back to (base_fn, "add", None).
    """
    if cp is None:
        return base_fn, "add", None

    default_fn, default_write_mode, default_trigger = OPERATOR_DEFAULTS.get(
        cp.operator, (None, "add", None)
    )

    # Connection's explicit fn (base_fn) always wins; operator default only
    # applies if no explicit fn was set. We can't tell here whether base_fn
    # is "explicit" vs "fallback"; the dispatcher hands us the resolved
    # value. Heuristic: when operator declares a fn AND base_fn is the
    # registry default "cross_attention", prefer the operator default.
    if default_fn is not None and base_fn == "cross_attention":
        fn = default_fn
    else:
        fn = base_fn

    # write_mode override: cp explicit value wins; operator default applies
    # when cp keeps the bare "add" default.
    write_mode = cp.write_mode if cp.write_mode != "add" else default_write_mode

    # trigger override: cp explicit value wins; operator default applies
    # when cp.trigger is None.
    trigger = cp.trigger if cp.trigger is not None else default_trigger

    return fn, write_mode, trigger


# ── Trigger expressions ──────────────────────────────────────────────


_TRIGGER_OPS = ("<=", ">=", "==", "!=", "<", ">")


def evaluate_trigger(
    trigger: Optional[str],
    summaries: Optional[Dict[str, Dict[str, float]]],
) -> bool:
    """Evaluate a ConnectionProgram.trigger string against residual summaries.

    Syntax: ``"<region>.<kind> <op> <number>"`` where ``<op>`` is one of
    ``< > <= >= == !=``. Multiple clauses may be ANDed with ``&&``.
    A ``None`` trigger always evaluates to True. A trigger referencing
    summaries that are missing evaluates to False (cannot fire without
    evidence).
    """
    if trigger is None:
        return True
    s = trigger.strip()
    if not s:
        return True
    if summaries is None:
        return False
    for clause in s.split("&&"):
        clause = clause.strip()
        if not clause:
            continue
        op_used = None
        for op in _TRIGGER_OPS:
            if op in clause:
                op_used = op
                break
        if op_used is None:
            return False
        left, right = clause.split(op_used, 1)
        source = left.strip()
        try:
            threshold = float(right.strip())
        except ValueError:
            return False
        parts = source.rsplit(".", 1)
        if len(parts) != 2:
            return False
        region, kind = parts
        val = summaries.get(region, {}).get(kind)
        if val is None:
            return False
        if op_used == "<" and not (val < threshold): return False
        if op_used == ">" and not (val > threshold): return False
        if op_used == "<=" and not (val <= threshold): return False
        if op_used == ">=" and not (val >= threshold): return False
        if op_used == "==" and not (val == threshold): return False
        if op_used == "!=" and not (val != threshold): return False
    return True


# ── Serialization helpers ────────────────────────────────────────────


def _clock_to_dict(clock: ClockSpec) -> dict:
    """Serialize ClockSpec, omitting default values."""
    d = {}
    if clock.domain != "external":
        d["domain"] = clock.domain
    if clock.mode != "periodic":
        d["mode"] = clock.mode
    if clock.period != 1:
        d["period"] = clock.period
    if clock.event_source is not None:
        d["event_source"] = clock.event_source
    if clock.event_threshold != 0.0:
        d["event_threshold"] = clock.event_threshold
    if clock.cooldown != 0:
        d["cooldown"] = clock.cooldown
    if clock.max_silence is not None:
        d["max_silence"] = clock.max_silence
    if clock.max_inner_steps is not None:
        d["max_inner_steps"] = clock.max_inner_steps
    if clock.expr is not None:
        d["expr"] = clock.expr.to_dict()
    return d


def _clock_from_dict(d: dict) -> ClockSpec:
    """Deserialize ClockSpec with defaults for missing keys."""
    expr_d = d.get("expr")
    expr = ClockExpr.from_dict(expr_d) if expr_d is not None else None
    return ClockSpec(
        domain=d.get("domain", "external"),
        mode=d.get("mode", "periodic"),
        period=d.get("period", 1),
        event_source=d.get("event_source"),
        event_threshold=d.get("event_threshold", 0.0),
        cooldown=d.get("cooldown", 0),
        max_silence=d.get("max_silence"),
        max_inner_steps=d.get("max_inner_steps"),
        expr=expr,
    )


def _learning_to_dict(ls: LearningSpec) -> dict:
    """Serialize LearningSpec, omitting default values."""
    d = {}
    if ls.mode != "supervised":
        d["mode"] = ls.mode
    if ls.losses:
        d["losses"] = list(ls.losses)
    if ls.compile_mode != "runtime":
        d["compile_mode"] = ls.compile_mode
    return d


def _learning_from_dict(d: dict) -> LearningSpec:
    """Deserialize LearningSpec with defaults for missing keys."""
    return LearningSpec(
        mode=d.get("mode", "supervised"),
        losses=tuple(d.get("losses", ())),
        compile_mode=d.get("compile_mode", "runtime"),
    )


def _constraint_to_dict(cs: ConstraintSpec) -> dict:
    """Serialize ConstraintSpec, omitting None values."""
    d = {}
    if cs.equivariance is not None:
        d["equivariance"] = cs.equivariance
    if cs.conservation is not None:
        d["conservation"] = cs.conservation
    if cs.causal_direction is not None:
        d["causal_direction"] = cs.causal_direction
    if cs.monotonicity is not None:
        d["monotonicity"] = cs.monotonicity
    return d


def _constraint_from_dict(d: dict) -> ConstraintSpec:
    """Deserialize ConstraintSpec with defaults for missing keys."""
    return ConstraintSpec(
        equivariance=d.get("equivariance"),
        conservation=d.get("conservation"),
        causal_direction=d.get("causal_direction"),
        monotonicity=d.get("monotonicity"),
    )


def _identity_to_dict(spec: "IdentitySpec") -> dict:
    """Serialize IdentitySpec, omitting default values."""
    d: Dict = {}
    if spec.mode != "persistent":
        d["mode"] = spec.mode
    if spec.slot_capacity != 64:
        d["slot_capacity"] = spec.slot_capacity
    if spec.binding_fn != "cross_attention":
        d["binding_fn"] = spec.binding_fn
    if spec.birth_death:
        d["birth_death"] = spec.birth_death
    return d


def _identity_from_dict(d: dict) -> "IdentitySpec":
    # Import here to avoid circular imports at module level
    from canvas_engineering.identity import IdentitySpec
    return IdentitySpec(
        mode=d.get("mode", "persistent"),
        slot_capacity=d.get("slot_capacity", 64),
        binding_fn=d.get("binding_fn", "cross_attention"),
        birth_death=d.get("birth_death", False),
    )


def _region_program_to_dict(rp: RegionProgram) -> dict:
    """Serialize RegionProgram, omitting default values."""
    d = {}
    if rp.family != "state":
        d["family"] = rp.family
    if rp.tags:
        d["tags"] = list(rp.tags)
    if rp.carrier != "deterministic":
        d["carrier"] = rp.carrier
    if rp.clock is not None:
        d["clock"] = _clock_to_dict(rp.clock)
    if rp.learning is not None:
        d["learning"] = _learning_to_dict(rp.learning)
    if rp.compile_mode != "runtime":
        d["compile_mode"] = rp.compile_mode
    if rp.constraints is not None:
        d["constraints"] = _constraint_to_dict(rp.constraints)
    if rp.identity is not None:
        d["identity"] = _identity_to_dict(rp.identity)
    return d


def _region_program_from_dict(d: dict) -> RegionProgram:
    """Deserialize RegionProgram with defaults for missing keys."""
    clock = _clock_from_dict(d["clock"]) if "clock" in d else None
    learning = _learning_from_dict(d["learning"]) if "learning" in d else None
    constraints = _constraint_from_dict(d["constraints"]) if "constraints" in d else None
    identity = _identity_from_dict(d["identity"]) if "identity" in d else None
    return RegionProgram(
        family=d.get("family", "state"),
        tags=tuple(d.get("tags", ())),
        carrier=d.get("carrier", "deterministic"),
        clock=clock,
        learning=learning,
        compile_mode=d.get("compile_mode", "runtime"),
        constraints=constraints,
        identity=identity,
    )


def _connection_program_to_dict(cp: ConnectionProgram) -> dict:
    """Serialize ConnectionProgram, omitting default values."""
    d = {}
    if cp.operator != "attend":
        d["operator"] = cp.operator
    if cp.trigger is not None:
        d["trigger"] = cp.trigger
    if cp.write_mode != "add":
        d["write_mode"] = cp.write_mode
    if cp.mask_spec is not None:
        ms = cp.mask_spec
        md: dict = {"kind": ms.kind}
        if ms.tile_h != 1:
            md["tile_h"] = ms.tile_h
        if ms.tile_w != 1:
            md["tile_w"] = ms.tile_w
        if ms.tile_shape is not None:
            md["tile_shape"] = list(ms.tile_shape)
        d["mask_spec"] = md
    return d


def _connection_program_from_dict(d: dict) -> ConnectionProgram:
    """Deserialize ConnectionProgram with defaults for missing keys."""
    mask_spec = None
    md = d.get("mask_spec")
    if md is not None:
        from canvas_engineering.masks import MaskSpec
        ts = md.get("tile_shape")
        mask_spec = MaskSpec(
            kind=md.get("kind", "full"),
            tile_h=md.get("tile_h", 1),
            tile_w=md.get("tile_w", 1),
            tile_shape=tuple(ts) if ts is not None else None,
        )
    return ConnectionProgram(
        operator=d.get("operator", "attend"),
        trigger=d.get("trigger"),
        write_mode=d.get("write_mode", "add"),
        mask_spec=mask_spec,
    )


# ── CanvasProgram ────────────────────────────────────────────────────


@dataclass
class CanvasProgram:
    """Typed process semantics layered on top of CanvasSchema.

    The schema declares structure (where things live, who talks to whom).
    The program declares behavior (what each region is, how it learns,
    when it updates, what happens at deploy).

    Usage:
        program = CanvasProgram(
            schema=my_schema,
            regions={"obs": RegionProgram(family="observation")},
        )
        program.to_json("my_program.json")
        loaded = CanvasProgram.from_json("my_program.json")
    """
    schema: CanvasSchema
    regions: Dict[str, RegionProgram] = field(default_factory=dict)
    connections: Dict[Tuple[str, str], ConnectionProgram] = field(default_factory=dict)
    version: str = "2.0.0"

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d = self.schema.to_dict()
        d["program_version"] = self.version

        if self.regions:
            rp_dict = {}
            for name, rp in self.regions.items():
                rp_data = _region_program_to_dict(rp)
                if rp_data:  # omit empty dicts (all defaults)
                    rp_dict[name] = rp_data
            if rp_dict:
                d["region_programs"] = rp_dict

        if self.connections:
            cp_dict = {}
            for (src, dst), cp in self.connections.items():
                cp_data = _connection_program_to_dict(cp)
                if cp_data:
                    cp_dict["{}|{}".format(src, dst)] = cp_data
            if cp_dict:
                d["connection_programs"] = cp_dict

        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CanvasProgram":
        """Deserialize from a dict."""
        schema = CanvasSchema.from_dict(d)
        version = d.get("program_version", "2.0.0")

        regions = {}
        for name, rp_data in d.get("region_programs", {}).items():
            regions[name] = _region_program_from_dict(rp_data)

        connections = {}
        for key_str, cp_data in d.get("connection_programs", {}).items():
            parts = key_str.split("|", 1)
            if len(parts) == 2:
                connections[(parts[0], parts[1])] = _connection_program_from_dict(cp_data)

        return cls(
            schema=schema,
            regions=regions,
            connections=connections,
            version=version,
        )

    def to_json(self, path: str) -> None:
        """Serialize to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CanvasProgram":
        """Deserialize from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        """Human-readable summary of the program."""
        lines = ["CanvasProgram (v{}, {} regions programmed):".format(
            self.version, len(self.regions))]

        family_counts: Dict[str, int] = {}
        for rp in self.regions.values():
            family_counts[rp.family] = family_counts.get(rp.family, 0) + 1
        if family_counts:
            fam_str = ", ".join("{}={}".format(f, n)
                                for f, n in sorted(family_counts.items()))
            lines.append("  families: {}".format(fam_str))

        carrier_counts: Dict[str, int] = {}
        for rp in self.regions.values():
            carrier_counts[rp.carrier] = carrier_counts.get(rp.carrier, 0) + 1
        non_default = {k: v for k, v in carrier_counts.items() if k != "deterministic"}
        if non_default:
            car_str = ", ".join("{}={}".format(c, n)
                                for c, n in sorted(non_default.items()))
            lines.append("  non-default carriers: {}".format(car_str))

        compile_counts: Dict[str, int] = {}
        for rp in self.regions.values():
            if rp.compile_mode != "runtime":
                compile_counts[rp.compile_mode] = compile_counts.get(rp.compile_mode, 0) + 1
        if compile_counts:
            comp_str = ", ".join("{}={}".format(m, n)
                                 for m, n in sorted(compile_counts.items()))
            lines.append("  compile modes: {}".format(comp_str))

        if self.connections:
            op_counts: Dict[str, int] = {}
            for cp in self.connections.values():
                if cp.operator != "attend":
                    op_counts[cp.operator] = op_counts.get(cp.operator, 0) + 1
            if op_counts:
                op_str = ", ".join("{}={}".format(o, n)
                                   for o, n in sorted(op_counts.items()))
                lines.append("  operators: {}".format(op_str))

        return "\n".join(lines)

    def __repr__(self) -> str:
        return "CanvasProgram(regions={}, connections={}, version={!r})".format(
            len(self.regions), len(self.connections), self.version)


# ── Constraint validation ───────────────────────────────────────────


def validate_constraints(program: CanvasProgram) -> List[str]:
    """Check structural constraints declared on regions and return violations.

    Validates:
    - ConstraintSpec field values are from known enums
    - causal_direction="acyclic" — no cycles in the connection sub-graph
      rooted at constrained regions
    - causal_direction="forward_only" — constrained regions only connect
      to regions whose temporal bounds start at the same or later time

    Returns:
        List of human-readable violation strings. Empty list = all good.
    """
    violations: List[str] = []

    for name, rp in program.regions.items():
        if rp.constraints is None:
            continue
        cs = rp.constraints

        # Validate enum values
        if cs.equivariance is not None and cs.equivariance not in _VALID_EQUIVARIANCES:
            violations.append(
                "{}: unknown equivariance {!r} (valid: {})".format(
                    name, cs.equivariance, sorted(_VALID_EQUIVARIANCES)))
        if cs.conservation is not None and cs.conservation not in _VALID_CONSERVATIONS:
            violations.append(
                "{}: unknown conservation {!r} (valid: {})".format(
                    name, cs.conservation, sorted(_VALID_CONSERVATIONS)))
        if cs.causal_direction is not None and cs.causal_direction not in _VALID_CAUSAL_DIRECTIONS:
            violations.append(
                "{}: unknown causal_direction {!r} (valid: {})".format(
                    name, cs.causal_direction, sorted(_VALID_CAUSAL_DIRECTIONS)))
        if cs.monotonicity is not None and cs.monotonicity not in _VALID_MONOTONICITIES:
            violations.append(
                "{}: unknown monotonicity {!r} (valid: {})".format(
                    name, cs.monotonicity, sorted(_VALID_MONOTONICITIES)))

    # Gather topology for causal checks
    topology = program.schema.topology
    if topology is None:
        return violations

    # Build adjacency for cycle detection
    adj: Dict[str, Set[str]] = {}
    for conn in topology.connections:
        adj.setdefault(conn.src, set()).add(conn.dst)

    # Check causal_direction constraints
    layout = program.schema.layout
    for name, rp in program.regions.items():
        if rp.constraints is None or rp.constraints.causal_direction is None:
            continue

        if rp.constraints.causal_direction == "acyclic":
            # DFS from this region — should not reach itself
            visited: Set[str] = set()
            stack = list(adj.get(name, set()))
            while stack:
                node = stack.pop()
                if node == name:
                    violations.append(
                        "{}: causal_direction='acyclic' violated — cycle detected".format(name))
                    break
                if node not in visited:
                    visited.add(node)
                    stack.extend(adj.get(node, set()))

        elif rp.constraints.causal_direction == "forward_only":
            # All outgoing connections must go to regions with t0 >= this region's t0
            try:
                src_spec = layout.region_spec(name)
                src_t0 = src_spec.bounds[0]
            except KeyError:
                continue
            for dst_name in adj.get(name, set()):
                if dst_name == name:
                    continue  # self-connections are fine
                try:
                    dst_spec = layout.region_spec(dst_name)
                    dst_t0 = dst_spec.bounds[0]
                except KeyError:
                    continue
                if dst_t0 < src_t0:
                    violations.append(
                        "{}: causal_direction='forward_only' violated — "
                        "connects to {} (t0={} < {})".format(
                            name, dst_name, dst_t0, src_t0))

    # ── equivariance / conservation / monotonicity static checks ─────
    # These are structural sanity checks based on declarative info.
    # ``_EQUIVARIANCE_COMPAT_ATTN`` lists attention fns that are known to
    # preserve each symmetry; anything else is flagged.
    _EQUIVARIANCE_COMPAT_ATTN = {
        "translation": {
            "cross_attention", "linear_attention", "local_attention",
            "sparse_attention", "random_fixed", "perceiver_attention",
            "cogvideox", "mamba_attention", "rwkv_attention", "hyena_attention",
            "none", "copy",
        },
        "rotation": {
            "cross_attention", "linear_attention", "perceiver_attention",
            "none", "copy",
        },
        "permutation": {
            "cross_attention", "linear_attention", "pooling_attention",
            "perceiver_attention", "none", "copy",
        },
    }

    for name, rp in program.regions.items():
        if rp.constraints is None:
            continue
        cs = rp.constraints

        # equivariance: outgoing edges must use a compatible attention fn.
        if cs.equivariance in _EQUIVARIANCE_COMPAT_ATTN:
            compat = _EQUIVARIANCE_COMPAT_ATTN[cs.equivariance]
            for conn in topology.connections:
                if conn.src != name:
                    continue
                fn = topology.resolve_fn(conn, layout)
                if fn not in compat:
                    violations.append(
                        "{}: equivariance={!r} violated — outgoing edge "
                        "to {} uses {!r} (compat: {})".format(
                            name, cs.equivariance, conn.dst, fn, sorted(compat)))

        # conservation: capacity & energy are incompatible with destructive
        # writes on outgoing edges. write_mode='replace' destroys the
        # invariant; learned 'gate' may also leak if it saturates.
        if cs.conservation in {"capacity", "energy"}:
            for (src, dst), cp in program.connections.items():
                if src != name:
                    continue
                if cp.write_mode == "replace":
                    violations.append(
                        "{}: conservation={!r} violated — outgoing edge to "
                        "{} uses write_mode='replace' (destroys invariant)".format(
                            name, cs.conservation, dst))

        # monotonicity: diffusive/filter carriers add stochastic noise or
        # correction terms that can break monotone state evolution.
        if cs.monotonicity in {"non_decreasing", "non_increasing"}:
            if rp.carrier in {"diffusive", "filter"}:
                violations.append(
                    "{}: monotonicity={!r} incompatible with carrier={!r} "
                    "(stochastic/correction dynamics)".format(
                        name, cs.monotonicity, rp.carrier))

    return violations

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
    """
    family: str = "state"
    tags: Tuple[str, ...] = ()
    carrier: str = "deterministic"
    clock: Optional[ClockSpec] = None
    learning: Optional[LearningSpec] = None
    compile_mode: str = "runtime"


@dataclass(frozen=True)
class ConnectionProgram:
    """Process semantics for a canvas connection.

    Args:
        operator: Semantic intent of this edge. "attend" = generic attention
            (backward compat default), "observe" = write evidence, "predict" =
            propagate forward, "correct" = error-driven update, etc.
        trigger: Optional condition for firing. Serializable string expression
            referencing residual summaries. E.g., "vision.err.prediction > 0.25".
        write_mode: How output accumulates at destination. "add" = additive
            (current behavior), "replace" = overwrite, "gate" = learned gate.
    """
    operator: str = "attend"
    trigger: Optional[str] = None
    write_mode: str = "add"


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
    return d


def _clock_from_dict(d: dict) -> ClockSpec:
    """Deserialize ClockSpec with defaults for missing keys."""
    return ClockSpec(
        domain=d.get("domain", "external"),
        mode=d.get("mode", "periodic"),
        period=d.get("period", 1),
        event_source=d.get("event_source"),
        event_threshold=d.get("event_threshold", 0.0),
        cooldown=d.get("cooldown", 0),
        max_silence=d.get("max_silence"),
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
    return d


def _region_program_from_dict(d: dict) -> RegionProgram:
    """Deserialize RegionProgram with defaults for missing keys."""
    clock = _clock_from_dict(d["clock"]) if "clock" in d else None
    learning = _learning_from_dict(d["learning"]) if "learning" in d else None
    return RegionProgram(
        family=d.get("family", "state"),
        tags=tuple(d.get("tags", ())),
        carrier=d.get("carrier", "deterministic"),
        clock=clock,
        learning=learning,
        compile_mode=d.get("compile_mode", "runtime"),
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
    return d


def _connection_program_from_dict(d: dict) -> ConnectionProgram:
    """Deserialize ConnectionProgram with defaults for missing keys."""
    return ConnectionProgram(
        operator=d.get("operator", "attend"),
        trigger=d.get("trigger"),
        write_mode=d.get("write_mode", "add"),
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

"""Browser agent architecture as canvas types with structured information flow.

Defines a browser control agent using canvas-engineering's typed process
semantics.  The agent observes screen pixels and DOM structure, maintains
internal beliefs about page state and task progress, plans multi-step
actions, and emits browser control commands.

Information flow:
    screen + dom --> page_belief   (observe)
    page_belief + instruction --> task_understanding   (correct)
    task_understanding + history --> plan   (predict)
    plan --> action_type + coordinates + text_input   (act)
    page_belief --> prediction_error   (emit_residual)
    prediction_error --> plan   (trigger re-planning on surprise)

Usage:
    from research.browser.browser_canvas import build_browser_program

    bound, program = build_browser_program(d_model=128)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Ensure canvas_engineering is importable
_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

from canvas_engineering import (
    Field,
    Connection,
    CanvasTopology,
    ConnectivityPolicy,
    compile_program,
    CanvasProgram,
    RegionProgram,
    ConnectionProgram,
    ClockSpec,
    RegionScheduler,
    AttentionDispatcher,
    ResidualAccumulator,
    ResidualSpec,
)
from canvas_engineering.schema import CanvasSchema


# ---- Type hierarchy for the browser agent ----


@dataclass
class ScreenObservation:
    """Visual observation of the browser viewport."""
    pixels: Field = Field(
        8, 8, family="observation", carrier="deterministic",
        semantic_type="browser viewport screenshot 224x224 RGB",
    )


@dataclass
class DOMState:
    """Structured DOM representation."""
    elements: Field = Field(
        4, 4, family="observation", carrier="deterministic",
        semantic_type="DOM element embeddings top-k interactive",
    )
    layout: Field = Field(
        2, 4, family="observation",
        semantic_type="page layout bounding boxes",
    )


@dataclass
class AgentState:
    """Internal agent representations."""
    task_understanding: Field = Field(
        3, 3, family="state", tags=("goal",),
        semantic_type="current task understanding from instruction",
    )
    page_belief: Field = Field(
        3, 3, family="state", tags=("belief",),
        carrier="filter",
        semantic_type="belief about current page state and progress",
    )
    plan: Field = Field(
        2, 3, family="state", tags=("planning",),
        semantic_type="next steps plan",
    )
    history: Field = Field(
        2, 4, family="memory", tags=("working",),
        semantic_type="recent action history and outcomes",
    )


@dataclass
class ActionOutput:
    """Browser actions."""
    action_type: Field = Field(
        1, 6, family="action", loss_weight=3.0,
        semantic_type="action type: click scroll type navigate wait done",
    )
    coordinates: Field = Field(
        1, 2, family="action", loss_weight=2.0,
        semantic_type="click/scroll target x y coordinates",
    )
    text_input: Field = Field(
        1, 8, family="action", loss_weight=2.0,
        semantic_type="text to type into focused element",
    )


@dataclass
class Diagnostics:
    """Error and confidence signals."""
    prediction_error: Field = Field(
        1, 4, family="residual",
        semantic_type="page change prediction error",
    )
    confidence: Field = Field(
        1, 2, family="residual",
        semantic_type="action confidence and uncertainty",
    )


@dataclass
class BrowserAgent:
    """Top-level browser agent type.

    Composes observations, internal state, actions, diagnostics, and a
    natural-language task instruction.
    """
    screen: ScreenObservation = dc_field(default_factory=ScreenObservation)
    dom: DOMState = dc_field(default_factory=DOMState)
    agent: AgentState = dc_field(default_factory=AgentState)
    action: ActionOutput = dc_field(default_factory=ActionOutput)
    diagnostics: Diagnostics = dc_field(default_factory=Diagnostics)
    instruction: Field = Field(
        2, 4, family="observation", is_output=False,
        period=16,
        semantic_type="natural language task instruction",
    )


# ---- Region name constants (matching flattened paths from compile_program) ----

# These are the leaf field paths that compile_program produces.
OBSERVATION_REGIONS = [
    "screen.pixels",
    "dom.elements",
    "dom.layout",
    "instruction",
]

STATE_REGIONS = [
    "agent.task_understanding",
    "agent.page_belief",
    "agent.plan",
]

MEMORY_REGIONS = [
    "agent.history",
]

ACTION_REGIONS = [
    "action.action_type",
    "action.coordinates",
    "action.text_input",
]

RESIDUAL_REGIONS = [
    "diagnostics.prediction_error",
    "diagnostics.confidence",
]


def get_all_leaf_regions() -> List[str]:
    """All leaf region names (excluding coarse-grained intermediates)."""
    return (
        OBSERVATION_REGIONS
        + STATE_REGIONS
        + MEMORY_REGIONS
        + ACTION_REGIONS
        + RESIDUAL_REGIONS
    )


# ---- Custom topology: explicit information flow ----


def build_browser_topology(region_names: List[str]) -> CanvasTopology:
    """Build the browser agent's custom topology.

    Defines the causal information flow:
        observe:  screen + dom -> page_belief
        correct:  page_belief + instruction -> task_understanding
        predict:  task_understanding + history -> plan
        act:      plan -> action_type, coordinates, text_input
        emit:     page_belief -> prediction_error
        replan:   prediction_error -> plan  (surprise triggers re-planning)

    Also includes self-connections for every region (self-attention within
    each region) and a few auxiliary connections (history writes,
    confidence from action).
    """
    connections = []

    # Self-connections for all regions (self-attention)
    for name in region_names:
        connections.append(Connection(src=name, dst=name))

    # Observe: screen + dom -> page_belief
    for obs in ["screen.pixels", "dom.elements", "dom.layout"]:
        if obs in region_names and "agent.page_belief" in region_names:
            connections.append(Connection(
                src="agent.page_belief", dst=obs,
                operator="observe",
            ))

    # Correct: page_belief + instruction -> task_understanding
    for src in ["agent.page_belief", "instruction"]:
        if src in region_names and "agent.task_understanding" in region_names:
            connections.append(Connection(
                src="agent.task_understanding", dst=src,
                operator="correct",
            ))

    # Predict: task_understanding + history -> plan
    for src in ["agent.task_understanding", "agent.history"]:
        if src in region_names and "agent.plan" in region_names:
            connections.append(Connection(
                src="agent.plan", dst=src,
                operator="predict",
            ))

    # Act: plan -> action outputs
    for act in ["action.action_type", "action.coordinates", "action.text_input"]:
        if act in region_names and "agent.plan" in region_names:
            connections.append(Connection(
                src=act, dst="agent.plan",
                operator="act",
            ))

    # Emit residual: page_belief -> prediction_error
    if "agent.page_belief" in region_names and "diagnostics.prediction_error" in region_names:
        connections.append(Connection(
            src="diagnostics.prediction_error", dst="agent.page_belief",
            operator="emit_residual",
        ))

    # Action -> confidence residual
    if "action.action_type" in region_names and "diagnostics.confidence" in region_names:
        connections.append(Connection(
            src="diagnostics.confidence", dst="action.action_type",
            operator="emit_residual",
        ))

    # Re-plan on surprise: prediction_error -> plan
    if "diagnostics.prediction_error" in region_names and "agent.plan" in region_names:
        connections.append(Connection(
            src="agent.plan", dst="diagnostics.prediction_error",
            operator="correct",
        ))

    # Write to history: action + page_belief -> history
    for src in ["action.action_type", "agent.page_belief"]:
        if src in region_names and "agent.history" in region_names:
            connections.append(Connection(
                src="agent.history", dst=src,
                operator="write",
            ))

    # DOM elements also inform DOM layout (intra-observation)
    if "dom.elements" in region_names and "dom.layout" in region_names:
        connections.append(Connection(
            src="dom.elements", dst="dom.layout",
            operator="attend",
        ))
        connections.append(Connection(
            src="dom.layout", dst="dom.elements",
            operator="attend",
        ))

    return CanvasTopology(connections=connections)


# ---- Build the full program ----


def build_browser_program(
    d_model: int = 128,
    T: int = 1,
    use_custom_topology: bool = True,
    plan_clock_threshold: float = 0.25,
) -> Tuple["BoundSchema", CanvasProgram]:
    """Build the browser agent's BoundSchema and CanvasProgram.

    Args:
        d_model: Latent dimensionality per position.
        T: Temporal extent of the canvas.
        use_custom_topology: If True, use the structured information flow
            topology instead of the auto-generated dense topology.
        plan_clock_threshold: Prediction error threshold above which the
            plan region re-fires (event-triggered scheduling).

    Returns:
        (BoundSchema, CanvasProgram) tuple ready for model construction.
    """
    from canvas_engineering.types import BoundSchema

    agent = BrowserAgent()

    # compile_program auto-generates topology from connectivity policy
    # and derives RegionPrograms from Field family/tags/carrier.
    bound, program = compile_program(
        agent,
        T=T,
        d_model=d_model,
        connectivity=ConnectivityPolicy(
            intra="dense",
            array_element="isolated",
            temporal="dense",
        ),
    )

    # Override topology with our custom structured flow
    if use_custom_topology:
        # Get all region names from the compiled layout
        region_names = list(bound.schema.layout.regions.keys())
        topology = build_browser_topology(region_names)
        schema = CanvasSchema(
            layout=bound.schema.layout,
            topology=topology,
            version=bound.schema.version,
            metadata=bound.schema.metadata,
        )
        # Rebuild program with new topology
        program = CanvasProgram(
            schema=schema,
            regions=program.regions,
            connections=program.connections,
        )
        # Patch bound schema
        bound = BoundSchema(schema, dict(bound.schema.layout.regions))

    # Add clock specs for event-triggered planning
    region_programs = dict(program.regions)

    # Plan updates on prediction error threshold (event-triggered)
    if "agent.plan" in region_programs:
        old_rp = region_programs["agent.plan"]
        region_programs["agent.plan"] = RegionProgram(
            family=old_rp.family,
            tags=old_rp.tags,
            carrier=old_rp.carrier,
            clock=ClockSpec(
                mode="on_event",
                event_source="diagnostics.prediction_error.prediction",
                event_threshold=plan_clock_threshold,
                cooldown=2,
                max_silence=8,
            ),
            learning=old_rp.learning,
            compile_mode=old_rp.compile_mode,
        )

    # History updates every 2 steps (slower than perception)
    if "agent.history" in region_programs:
        old_rp = region_programs["agent.history"]
        region_programs["agent.history"] = RegionProgram(
            family=old_rp.family,
            tags=old_rp.tags,
            carrier=old_rp.carrier,
            clock=ClockSpec(mode="periodic", period=2),
            learning=old_rp.learning,
            compile_mode=old_rp.compile_mode,
        )

    # Instruction updates rarely (given once per task)
    if "instruction" in region_programs:
        old_rp = region_programs["instruction"]
        region_programs["instruction"] = RegionProgram(
            family=old_rp.family,
            tags=old_rp.tags,
            carrier=old_rp.carrier,
            clock=ClockSpec(mode="periodic", period=16),
            learning=old_rp.learning,
            compile_mode=old_rp.compile_mode,
        )

    program = CanvasProgram(
        schema=program.schema,
        regions=region_programs,
        connections=program.connections,
    )

    return bound, program


def get_region_names(bound) -> List[str]:
    """Return all leaf region names from a compiled BoundSchema."""
    return list(bound.field_names)

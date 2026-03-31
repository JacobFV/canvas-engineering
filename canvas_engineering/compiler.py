"""Program compiler: lower a CanvasProgram to a deploy-ready execution plan.

The compiler applies compile-mode passes to transform a training-time
CanvasProgram into a deploy-ready CompiledProgram with fewer active
regions, frozen parameters, exported memory banks, and eliminated
dead connections.

Usage:
    from canvas_engineering.compiler import ProgramCompiler

    compiler = ProgramCompiler(program)
    compiled = compiler.compile()
    # compiled.active_regions — regions still alive at deploy
    # compiled.frozen_regions — regions with no gradient
    # compiled.exported_memories — saved to disk, removed from graph
    # compiled.active_connections — connections between active regions only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from canvas_engineering.connectivity import Connection, CanvasTopology
from canvas_engineering.canvas import CanvasLayout, RegionSpec
from canvas_engineering.schema import CanvasSchema
from canvas_engineering.program import CanvasProgram


@dataclass
class CompiledProgram:
    """Deploy-ready execution plan produced by ProgramCompiler.

    Attributes:
        schema: Potentially reduced CanvasSchema (inactive regions removed).
        active_regions: Set of region names still alive at deploy.
        frozen_regions: Set of region names with frozen parameters.
        constant_regions: Set of regions materialized as buffers.
        exported_memories: Dict of region names exported to disk.
        active_connections: Connections between active regions only.
    """
    schema: CanvasSchema
    active_regions: Set[str] = field(default_factory=set)
    frozen_regions: Set[str] = field(default_factory=set)
    constant_regions: Set[str] = field(default_factory=set)
    exported_memories: Set[str] = field(default_factory=set)
    active_connections: List[Connection] = field(default_factory=list)
    all_regions: Set[str] = field(default_factory=set)

    @property
    def n_eliminated(self) -> int:
        """Number of regions eliminated from the training graph."""
        return len(self.all_regions) - len(self.active_regions)

    def summary(self) -> str:
        lines = ["CompiledProgram:"]
        lines.append("  active: {} regions, {} connections".format(
            len(self.active_regions), len(self.active_connections)))
        if self.frozen_regions:
            lines.append("  frozen: {}".format(sorted(self.frozen_regions)))
        if self.constant_regions:
            lines.append("  constant: {}".format(sorted(self.constant_regions)))
        if self.exported_memories:
            lines.append("  exported: {}".format(sorted(self.exported_memories)))
        if self.n_eliminated > 0:
            lines.append("  eliminated: {} regions".format(self.n_eliminated))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "CompiledProgram(active={}, frozen={}, exported={})".format(
            len(self.active_regions), len(self.frozen_regions),
            len(self.exported_memories))


class ProgramCompiler:
    """Lowers a CanvasProgram to a deploy-ready CompiledProgram.

    Compile passes applied in order:
    1. freeze: mark compile_mode="freeze" regions (no gradient)
    2. constant: mark compile_mode="constant" regions (materialize as buffer, remove from active)
    3. export: mark compile_mode="export" regions (save to disk, remove from active)
    4. dead elimination: remove connections involving inactive regions

    Usage:
        compiler = ProgramCompiler(program)
        compiled = compiler.compile()
    """

    def __init__(self, program: CanvasProgram):
        self._program = program
        self._all_regions = set(program.schema.layout.regions.keys())
        self._active = set(self._all_regions)
        self._frozen: Set[str] = set()
        self._constant: Set[str] = set()
        self._exported: Set[str] = set()
        self._connections = list(
            program.schema.topology.connections
        ) if program.schema.topology else []

    def compile(self) -> CompiledProgram:
        """Run all compile passes and return a CompiledProgram."""
        self._pass_freeze()
        self._pass_constant()
        self._pass_export()
        self._pass_eliminate_dead()

        return CompiledProgram(
            schema=self._build_reduced_schema(),
            active_regions=set(self._active),
            frozen_regions=set(self._frozen),
            constant_regions=set(self._constant),
            exported_memories=set(self._exported),
            active_connections=list(self._connections),
            all_regions=set(self._all_regions),
        )

    def _pass_freeze(self) -> None:
        """Mark regions with compile_mode="freeze" — no gradient at deploy."""
        for name, rp in self._program.regions.items():
            if name not in self._active:
                continue
            effective_mode = rp.compile_mode
            if rp.learning and rp.learning.compile_mode != "runtime":
                effective_mode = rp.learning.compile_mode
            if effective_mode == "freeze":
                self._frozen.add(name)

    def _pass_constant(self) -> None:
        """Mark regions with compile_mode="constant" — materialize and remove."""
        for name, rp in self._program.regions.items():
            if name not in self._active:
                continue
            effective_mode = rp.compile_mode
            if rp.learning and rp.learning.compile_mode == "constant":
                effective_mode = "constant"
            if effective_mode == "constant":
                self._constant.add(name)
                self._active.discard(name)

    def _pass_export(self) -> None:
        """Mark regions with compile_mode="export" — save and remove."""
        for name, rp in self._program.regions.items():
            if name not in self._active:
                continue
            effective_mode = rp.compile_mode
            if rp.learning and rp.learning.compile_mode == "export":
                effective_mode = "export"
            if effective_mode == "export":
                self._exported.add(name)
                self._active.discard(name)

    def _pass_eliminate_dead(self) -> None:
        """Remove connections involving inactive regions."""
        self._connections = [
            c for c in self._connections
            if c.src in self._active and c.dst in self._active
        ]

    def _build_reduced_schema(self) -> CanvasSchema:
        """Build a schema with only active regions and connections."""
        layout = self._program.schema.layout
        # Keep the full layout (positions don't change) but filter regions
        # that are fully eliminated (constant + exported)
        eliminated = self._constant | self._exported
        if not eliminated:
            return self._program.schema

        active_regions = {
            name: spec for name, spec in layout.regions.items()
            if name not in eliminated
        }
        reduced_layout = CanvasLayout(
            T=layout.T, H=layout.H, W=layout.W, d_model=layout.d_model,
            regions=active_regions, t_current=layout.t_current,
        )
        topology = CanvasTopology(connections=self._connections) if self._connections else None
        return CanvasSchema(
            layout=reduced_layout,
            topology=topology,
            version=self._program.schema.version,
            metadata=self._program.schema.metadata,
        )

    def __repr__(self) -> str:
        return "ProgramCompiler(regions={})".format(len(self._all_regions))

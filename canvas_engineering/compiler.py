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

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

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
        constant_buffers: For each constant region, the materialized tensor
            stored alongside the compiled plan (populated when compile() is
            given a runtime ``module``).
        exported_paths: For each exported region, the file path the
            region's state was written to.
    """
    schema: CanvasSchema
    active_regions: Set[str] = field(default_factory=set)
    frozen_regions: Set[str] = field(default_factory=set)
    constant_regions: Set[str] = field(default_factory=set)
    exported_memories: Set[str] = field(default_factory=set)
    active_connections: List[Connection] = field(default_factory=list)
    all_regions: Set[str] = field(default_factory=set)
    constant_buffers: Dict[str, torch.Tensor] = field(default_factory=dict)
    exported_paths: Dict[str, str] = field(default_factory=dict)

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

    def compile(
        self,
        module: Optional[nn.Module] = None,
        export_dir: Optional[str] = None,
    ) -> CompiledProgram:
        """Run all compile passes and return a CompiledProgram.

        Args:
            module: Optional runtime ``nn.Module`` whose parameters are
                organized by region (the module must expose a ``regions``
                attribute mapping region names to submodules). When
                supplied:
                  - frozen regions have ``.requires_grad_(False)`` applied
                  - constant regions have their parameters detached and
                    re-registered as buffers (no further training)
                  - exported regions are saved to disk via
                    ``torch.save(submodule.state_dict(), path)`` and then
                    detached (``.requires_grad_(False)``) on the running
                    module so they no longer participate in training.
                When ``None`` the modes are still recorded on the
                returned ``CompiledProgram`` for later application.
            export_dir: Directory used for ``compile_mode='export'``
                regions. Defaults to ``./canvas_exports`` if any region
                is exported and ``module`` is supplied.

        Returns:
            CompiledProgram with active/frozen/constant/exported region
            sets and (if ``module`` was given) ``constant_buffers`` and
            ``exported_paths`` populated.
        """
        self._pass_freeze()
        self._pass_constant()
        self._pass_export()
        self._pass_eliminate_dead()

        compiled = CompiledProgram(
            schema=self._build_reduced_schema(),
            active_regions=set(self._active),
            frozen_regions=set(self._frozen),
            constant_regions=set(self._constant),
            exported_memories=set(self._exported),
            active_connections=list(self._connections),
            all_regions=set(self._all_regions),
        )

        if module is not None:
            self._apply_compile_modes(module, compiled, export_dir)

        return compiled

    def _effective_mode(self, rp) -> str:
        """Resolve effective compile mode: explicit > learning > family default."""
        return rp.effective_compile_mode()

    def _pass_freeze(self) -> None:
        """Mark regions with compile_mode="freeze" — no gradient at deploy."""
        for name, rp in self._program.regions.items():
            if name not in self._active:
                continue
            if self._effective_mode(rp) == "freeze":
                self._frozen.add(name)

    def _pass_constant(self) -> None:
        """Mark regions with compile_mode="constant" — materialize and remove."""
        for name, rp in self._program.regions.items():
            if name not in self._active:
                continue
            if self._effective_mode(rp) == "constant":
                self._constant.add(name)
                self._active.discard(name)

    def _pass_export(self) -> None:
        """Mark regions with compile_mode="export" — save and remove."""
        for name, rp in self._program.regions.items():
            if name not in self._active:
                continue
            if self._effective_mode(rp) == "export":
                self._exported.add(name)
                self._active.discard(name)

    @staticmethod
    def _get_region_submodule(module: nn.Module, name: str) -> Optional[nn.Module]:
        """Resolve the runtime submodule for a region, if available.

        Looks first at ``module.regions[name]`` (the recommended layout),
        then falls back to ``getattr(module, name)`` for simpler modules.
        """
        regions_attr = getattr(module, "regions", None)
        if regions_attr is not None:
            try:
                if isinstance(regions_attr, dict):
                    sub = regions_attr.get(name)
                else:
                    sub = regions_attr[name]
                if sub is not None:
                    return sub
            except (KeyError, IndexError, TypeError):
                pass
        sub = getattr(module, name, None)
        if isinstance(sub, nn.Module):
            return sub
        return None

    def _apply_compile_modes(
        self,
        module: nn.Module,
        compiled: CompiledProgram,
        export_dir: Optional[str],
    ) -> None:
        """Materialize freeze / constant / export on a runtime module.

        - freeze: ``param.requires_grad_(False)`` on each region submodule.
        - constant: detach parameters, capture them as ``constant_buffers``
          on the CompiledProgram, and replace the live parameter with a
          frozen buffer of identical values (re-registered via
          ``register_buffer``). Subsequent forward passes still see the
          values but no gradient flows.
        - export: serialize the region's ``state_dict`` to disk via
          ``torch.save``, record the path on ``exported_paths``, and
          freeze the live submodule.
        """
        if not self._frozen and not self._constant and not self._exported:
            return

        # Freeze pass.
        for name in self._frozen:
            sub = self._get_region_submodule(module, name)
            if sub is None:
                continue
            for p in sub.parameters():
                p.requires_grad_(False)

        # Constant pass: replace parameters with same-valued buffers.
        for name in self._constant:
            sub = self._get_region_submodule(module, name)
            if sub is None:
                continue
            captured: Dict[str, torch.Tensor] = {}
            param_names: List[Tuple[nn.Module, str]] = []
            for owner in sub.modules():
                for pname, p in list(owner.named_parameters(recurse=False)):
                    captured[f"{name}.{pname}"] = p.detach().clone()
                    param_names.append((owner, pname))
            for owner, pname in param_names:
                p = getattr(owner, pname)
                value = p.detach().clone()
                delattr(owner, pname)
                owner.register_buffer(pname, value)
            compiled.constant_buffers.update(captured)

        # Export pass: serialize and freeze.
        if self._exported:
            target_dir = export_dir or os.path.join(os.getcwd(), "canvas_exports")
            os.makedirs(target_dir, exist_ok=True)
            for name in self._exported:
                sub = self._get_region_submodule(module, name)
                if sub is None:
                    continue
                path = os.path.join(target_dir, f"{name}.pt")
                torch.save(sub.state_dict(), path)
                # Also drop a small JSON manifest so external tools can
                # discover the export without loading the .pt file.
                manifest = {
                    "region": name,
                    "family": self._program.regions[name].family
                    if name in self._program.regions else "state",
                    "path": path,
                }
                with open(path + ".json", "w") as fh:
                    json.dump(manifest, fh)
                compiled.exported_paths[name] = path
                for p in sub.parameters():
                    p.requires_grad_(False)

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
            T=layout.T, spatial_shape=layout.spatial_shape,
            d_model=layout.d_model,
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

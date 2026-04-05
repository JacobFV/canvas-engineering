"""Cortex registry: spatial grouping and locality for regions.

A CortexSpec declares a named spatial zone on the canvas with a
preferred tiling and a local attention backend. Regions assigned to
the same cortex share a key-value cache (when ``shared_cache=True``)
and can exploit spatial locality for efficient local attention.

Usage:
    from canvas_engineering.cortex import CortexSpec, CortexRegistry

    reg = CortexRegistry()
    reg.register(CortexSpec(name="visual", bounds=(0, 4, 0, 8, 0, 8), tile=(4, 4)))
    reg.assign("camera", "visual")
    reg.assign("depth", "visual")
    assert reg.same_cortex("camera", "depth")

    spec = reg.cortex_for("camera")
    regions = reg.regions_in("visual")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CortexSpec:
    """Declares a named spatial zone on the canvas.

    Args:
        name: Unique identifier for this cortex.
        bounds: Spatial-temporal extent as flat tuple (t0, t1, s0_lo, s0_hi, ...).
            Length = 2 + 2 * n_spatial_dims.
        tile: Preferred tiling per spatial dimension. E.g. (4, 4) for 2D.
        local_backend: Attention backend for intra-cortex connections.
            Default "local_attention".
        shared_cache: Whether regions in this cortex share a KV cache
            during inference. Default True.
    """
    name: str
    bounds: Tuple[int, ...] = (0, 1, 0, 1, 0, 1)
    tile: Tuple[int, ...] = (4, 4)
    local_backend: str = "local_attention"
    shared_cache: bool = True


class CortexRegistry:
    """Manages cortex assignments for regions.

    Regions can be assigned to at most one cortex. The registry tracks
    which cortex each region belongs to and provides queries for
    co-located region groups.
    """

    def __init__(self) -> None:
        self._cortices: Dict[str, CortexSpec] = {}
        self._region_to_cortex: Dict[str, str] = {}

    def register(self, spec: CortexSpec) -> None:
        """Register a cortex specification.

        Raises ValueError if a cortex with the same name already exists.
        """
        if spec.name in self._cortices:
            raise ValueError("Cortex {!r} already registered".format(spec.name))
        self._cortices[spec.name] = spec

    def unregister(self, name: str) -> None:
        """Remove a cortex and all its region assignments."""
        if name not in self._cortices:
            raise KeyError("Cortex {!r} not registered".format(name))
        del self._cortices[name]
        to_remove = [r for r, c in self._region_to_cortex.items() if c == name]
        for r in to_remove:
            del self._region_to_cortex[r]

    def assign(self, region: str, cortex_name: str) -> None:
        """Assign a region to a cortex.

        Raises KeyError if the cortex is not registered.
        Raises ValueError if the region is already assigned to a different cortex.
        """
        if cortex_name not in self._cortices:
            raise KeyError("Cortex {!r} not registered".format(cortex_name))
        existing = self._region_to_cortex.get(region)
        if existing is not None and existing != cortex_name:
            raise ValueError(
                "Region {!r} already assigned to cortex {!r}, "
                "cannot reassign to {!r}".format(region, existing, cortex_name))
        self._region_to_cortex[region] = cortex_name

    def unassign(self, region: str) -> None:
        """Remove a region's cortex assignment."""
        if region not in self._region_to_cortex:
            raise KeyError("Region {!r} has no cortex assignment".format(region))
        del self._region_to_cortex[region]

    def cortex_for(self, region: str) -> Optional[CortexSpec]:
        """Get the cortex assigned to a region, or None."""
        name = self._region_to_cortex.get(region)
        if name is None:
            return None
        return self._cortices.get(name)

    def regions_in(self, cortex_name: str) -> List[str]:
        """Get all regions assigned to a cortex, sorted alphabetically."""
        return sorted(
            r for r, c in self._region_to_cortex.items() if c == cortex_name
        )

    def same_cortex(self, a: str, b: str) -> bool:
        """Check if two regions are in the same cortex."""
        ca = self._region_to_cortex.get(a)
        cb = self._region_to_cortex.get(b)
        if ca is None or cb is None:
            return False
        return ca == cb

    @property
    def cortex_names(self) -> List[str]:
        """All registered cortex names, sorted."""
        return sorted(self._cortices.keys())

    @property
    def n_cortices(self) -> int:
        return len(self._cortices)

    @property
    def n_assigned(self) -> int:
        """Number of regions with cortex assignments."""
        return len(self._region_to_cortex)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["CortexRegistry({} cortices, {} assignments):".format(
            self.n_cortices, self.n_assigned)]
        for name in self.cortex_names:
            spec = self._cortices[name]
            regions = self.regions_in(name)
            lines.append("  {} (tile={}, backend={}, cache={}): {} regions{}".format(
                name, spec.tile, spec.local_backend, spec.shared_cache,
                len(regions),
                " [{}]".format(", ".join(regions)) if regions else "",
            ))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "CortexRegistry(cortices={}, assigned={})".format(
            self.n_cortices, self.n_assigned)

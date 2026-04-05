"""Attention mask specifications and utilities.

MaskSpec controls the sparsity pattern of attention between regions.
The default "full" mode creates dense attention; "tile" divides the
spatial grid into tiles for block-sparse attention; "sparse" uses a
greedy rectangle covering of a binary mask.

Usage:
    from canvas_engineering.masks import MaskSpec, mask_to_index_pairs, rect_cover

    spec = MaskSpec(kind="tile", tile_h=2, tile_w=2)
    pairs = mask_to_index_pairs(spec, layout, "obs", "state")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product as _product
from typing import List, Optional, Tuple

import torch

from canvas_engineering.canvas import CanvasLayout, _get_bounds, _parse_bounds


@dataclass(frozen=True)
class MaskSpec:
    """Attention mask specification for a connection.

    Args:
        kind: Mask type. "full" = dense attention (default). "tile" =
            tile-based block-sparse attention. "sparse" = greedy rectangle
            covering of an explicit binary mask.
        tile_h: For kind="tile", tile height (2D compat).
        tile_w: For kind="tile", tile width (2D compat).
        tile_shape: For kind="tile", N-D tile shape. Mutually exclusive
            with tile_h/tile_w.
    """
    kind: str = "full"
    tile_h: int = 1
    tile_w: int = 1
    tile_shape: Optional[Tuple[int, ...]] = None

    @property
    def _tile_shape(self) -> Tuple[int, ...]:
        """Resolved tile shape (from tile_shape or tile_h/tile_w)."""
        if self.tile_shape is not None:
            return self.tile_shape
        return (self.tile_h, self.tile_w)


class Rect:
    """Axis-aligned hyperrectangle in N-dimensional space.

    ranges is a tuple of (lo, hi) pairs, one per dimension.
    For (T, H, W): ranges = ((t0, t1), (h0, h1), (w0, w1)).

    Supports backward-compat construction: Rect(t0, t1, h0, h1, w0, w1).
    """
    __slots__ = ('ranges',)

    def __init__(self, *args, ranges: Optional[Tuple[Tuple[int, int], ...]] = None):
        if ranges is not None:
            object.__setattr__(self, 'ranges', ranges)
        elif len(args) == 1 and isinstance(args[0], tuple) and isinstance(args[0][0], tuple):
            # Rect(((t0,t1), (h0,h1), ...))
            object.__setattr__(self, 'ranges', args[0])
        elif len(args) >= 2 and len(args) % 2 == 0:
            # Backward compat: Rect(t0, t1, h0, h1, w0, w1)
            pairs = tuple((args[i], args[i + 1]) for i in range(0, len(args), 2))
            object.__setattr__(self, 'ranges', pairs)
        else:
            raise TypeError(
                "Rect() requires even number of positional args (lo, hi pairs) "
                "or ranges=((lo, hi), ...)"
            )

    def __eq__(self, other):
        if isinstance(other, Rect):
            return self.ranges == other.ranges
        return NotImplemented

    def __hash__(self):
        return hash(self.ranges)

    def __repr__(self):
        return f"Rect(ranges={self.ranges})"

    @property
    def volume(self) -> int:
        """Number of positions covered."""
        return math.prod(hi - lo for lo, hi in self.ranges)

    # Backward-compat properties for 3D (T, H, W) callers
    @property
    def t0(self) -> int:
        return self.ranges[0][0]

    @property
    def t1(self) -> int:
        return self.ranges[0][1]

    @property
    def h0(self) -> int:
        return self.ranges[1][0] if len(self.ranges) > 1 else 0

    @property
    def h1(self) -> int:
        return self.ranges[1][1] if len(self.ranges) > 1 else 0

    @property
    def w0(self) -> int:
        return self.ranges[2][0] if len(self.ranges) > 2 else 0

    @property
    def w1(self) -> int:
        return self.ranges[2][1] if len(self.ranges) > 2 else 0


def _largest_hyperrect_at(mask: torch.Tensor, start: Tuple[int, ...]) -> Rect:
    """Find the largest axis-aligned hyperrectangle of True values.

    Expands greedily from last dimension to first.
    """
    n = mask.ndim
    if not mask[start]:
        return Rect(tuple((s, s) for s in start))

    lo = list(start)
    hi = [s + 1 for s in start]

    # Expand from last dim to first (w → h → t ordering preserved)
    for d in reversed(range(n)):
        while hi[d] < mask.shape[d]:
            # Build slice that extends dim d by 1 while keeping others fixed
            test = tuple(
                slice(lo[i], hi[i]) if i != d else slice(hi[d], hi[d] + 1)
                for i in range(n)
            )
            if mask[test].all():
                hi[d] += 1
            else:
                break

    return Rect(tuple((lo[i], hi[i]) for i in range(n)))


def rect_cover(mask: torch.Tensor, min_area: int = 4) -> List[Rect]:
    """Greedy hyperrectangle covering of an N-D boolean mask.

    Args:
        mask: N-dimensional boolean tensor (e.g. (T, H, W) or (T, X)).
        min_area: Minimum volume for a rectangle.

    Returns:
        List of Rect objects covering all True positions in mask.
    """
    mask = mask.clone().bool()
    n = mask.ndim
    rects: List[Rect] = []

    while mask.any():
        best_rect: Optional[Rect] = None
        best_vol = 0

        positions = mask.nonzero(as_tuple=False)
        if positions.numel() == 0:
            break

        n_to_try = min(len(positions), 256)
        for i in range(n_to_try):
            coords = tuple(positions[i].tolist())
            if not mask[coords]:
                continue
            rect = _largest_hyperrect_at(mask, coords)
            if rect.volume > best_vol:
                best_vol = rect.volume
                best_rect = rect

        if best_rect is None or best_vol == 0:
            for i in range(len(positions)):
                coords = tuple(positions[i].tolist())
                if mask[coords]:
                    rects.append(Rect(tuple((c, c + 1) for c in coords)))
                    mask[coords] = False
            break

        rects.append(best_rect)
        # Mark covered via N-D slice
        cover_slice = tuple(slice(lo, hi) for lo, hi in best_rect.ranges)
        mask[cover_slice] = False

    return rects


# Keep for backward compat (old code used 3-arg signature)
def _largest_rectangle_at(mask: torch.Tensor, start_t: int,
                          start_h: int, start_w: int) -> Rect:
    """Backward-compat wrapper for 3D masks."""
    return _largest_hyperrect_at(mask, (start_t, start_h, start_w))


def _bounds_to_thw(bounds: Tuple[int, ...]) -> Tuple[int, ...]:
    """Extract bounds tuple (identity for backward compat)."""
    return bounds


def mask_to_index_pairs(
    mask_spec: MaskSpec,
    layout: CanvasLayout,
    src: str,
    dst: str,
) -> List[Tuple[List[int], List[int]]]:
    """Convert a MaskSpec to pairs of (src_indices, dst_indices) for dispatch.

    Args:
        mask_spec: The mask specification.
        layout: Canvas layout with region definitions.
        src: Source region name.
        dst: Destination region name.

    Returns:
        List of (src_indices, dst_indices) pairs. Each pair represents
        a group of src positions that should attend to a group of dst
        positions. For "full" mode, returns a single pair with all indices.
    """
    if mask_spec.kind == "full":
        return [(layout.region_indices(src), layout.region_indices(dst))]

    elif mask_spec.kind == "tile":
        src_bounds = _get_bounds(layout.regions[src])
        dst_bounds = _get_bounds(layout.regions[dst])

        n_sp = layout.n_spatial_dims
        (src_t0, src_t1), src_spatial = _parse_bounds(src_bounds, n_sp)
        (dst_t0, dst_t1), dst_spatial = _parse_bounds(dst_bounds, n_sp)

        tile = mask_spec._tile_shape
        # Pad tile shape to match n_spatial_dims
        if len(tile) < n_sp:
            tile = tile + (1,) * (n_sp - len(tile))

        pairs: List[Tuple[List[int], List[int]]] = []

        # Build tile grid ranges over src spatial dims
        tile_ranges = []
        for d in range(n_sp):
            lo, hi = src_spatial[d]
            tile_ranges.append(range(lo, hi, max(tile[d], 1)))

        for tile_origin in _product(*tile_ranges):
            # Compute tile extent in each spatial dim
            tile_lo = list(tile_origin)
            tile_hi = [min(tile_lo[d] + tile[d], src_spatial[d][1]) for d in range(n_sp)]

            # Collect src indices in this tile (all timesteps)
            src_idx: List[int] = []
            spatial_ranges_src = tuple((tile_lo[d], tile_hi[d]) for d in range(n_sp))
            for t in range(src_t0, src_t1):
                for coords in _product(*(range(lo, hi) for lo, hi in spatial_ranges_src)):
                    src_idx.append(layout.flat_index(t, *coords))

            # Collect dst indices in corresponding spatial tile
            dst_tile_lo = [max(tile_lo[d], dst_spatial[d][0]) for d in range(n_sp)]
            dst_tile_hi = [min(tile_hi[d], dst_spatial[d][1]) for d in range(n_sp)]

            has_overlap = all(dst_tile_lo[d] < dst_tile_hi[d] for d in range(n_sp))
            if not has_overlap:
                dst_idx = layout.region_indices(dst)
            else:
                dst_idx_list: List[int] = []
                spatial_ranges_dst = tuple((dst_tile_lo[d], dst_tile_hi[d]) for d in range(n_sp))
                for t in range(dst_t0, dst_t1):
                    for coords in _product(*(range(lo, hi) for lo, hi in spatial_ranges_dst)):
                        dst_idx_list.append(layout.flat_index(t, *coords))
                dst_idx = dst_idx_list if dst_idx_list else layout.region_indices(dst)

            if src_idx and dst_idx:
                pairs.append((src_idx, dst_idx))

        if not pairs:
            pairs = [(layout.region_indices(src), layout.region_indices(dst))]
        return pairs

    elif mask_spec.kind == "sparse":
        return [(layout.region_indices(src), layout.region_indices(dst))]

    else:
        return [(layout.region_indices(src), layout.region_indices(dst))]

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

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from canvas_engineering.canvas import CanvasLayout


@dataclass(frozen=True)
class MaskSpec:
    """Attention mask specification for a connection.

    Args:
        kind: Mask type. "full" = dense attention (default). "tile" =
            tile-based block-sparse attention. "sparse" = greedy rectangle
            covering of an explicit binary mask.
        tile_h: For kind="tile", tile height in spatial rows.
        tile_w: For kind="tile", tile width in spatial columns.
    """
    kind: str = "full"
    tile_h: int = 1
    tile_w: int = 1


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in (T, H, W) space.

    Represents a contiguous block: t in [t0, t1), h in [h0, h1), w in [w0, w1).
    """
    t0: int
    t1: int
    h0: int
    h1: int
    w0: int
    w1: int

    @property
    def volume(self) -> int:
        """Number of positions covered."""
        return (self.t1 - self.t0) * (self.h1 - self.h0) * (self.w1 - self.w0)


def _largest_rectangle_at(mask: torch.Tensor, start_t: int, start_h: int, start_w: int) -> Rect:
    """Find the largest axis-aligned rectangle of True values starting at (t, h, w).

    Expands greedily along w, then h, then t, keeping the rectangle valid.
    """
    T, H, W = mask.shape
    if not mask[start_t, start_h, start_w]:
        return Rect(start_t, start_t, start_h, start_h, start_w, start_w)

    # Find max w extent at (start_t, start_h)
    max_w = start_w
    while max_w + 1 < W and mask[start_t, start_h, max_w + 1]:
        max_w += 1

    # Find max h extent keeping all w positions valid
    max_h = start_h
    while max_h + 1 < H:
        # Check if the entire row [start_w..max_w] at (start_t, max_h+1) is True
        if mask[start_t, max_h + 1, start_w:max_w + 1].all():
            max_h += 1
        else:
            break

    # Find max t extent keeping all (h, w) positions valid
    max_t = start_t
    while max_t + 1 < T:
        if mask[max_t + 1, start_h:max_h + 1, start_w:max_w + 1].all():
            max_t += 1
        else:
            break

    return Rect(start_t, max_t + 1, start_h, max_h + 1, start_w, max_w + 1)


def rect_cover(mask: torch.Tensor, min_area: int = 4) -> List[Rect]:
    """Greedy rectangle covering of a binary 3D mask.

    Iterates through True positions, finds the largest axis-aligned
    rectangle at each, marks it covered, and repeats until all True
    positions are covered.

    Args:
        mask: (T, H, W) boolean tensor.
        min_area: Minimum volume for a rectangle. Smaller rectangles are
            still emitted to ensure coverage, but the algorithm prefers
            larger ones first.

    Returns:
        List of Rect objects covering all True positions in mask.
    """
    mask = mask.clone().bool()
    T, H, W = mask.shape
    rects: List[Rect] = []

    while mask.any():
        # Find the best rectangle (largest volume) across all uncovered starts
        best_rect: Optional[Rect] = None
        best_vol = 0

        # Get all True positions
        positions = mask.nonzero(as_tuple=False)
        if positions.numel() == 0:
            break

        # Try a sample of starting positions (for efficiency, limit to first few)
        # In practice the mask is small enough to try all.
        n_to_try = min(len(positions), 256)
        for i in range(n_to_try):
            t_i, h_i, w_i = positions[i].tolist()
            if not mask[t_i, h_i, w_i]:
                continue
            rect = _largest_rectangle_at(mask, t_i, h_i, w_i)
            if rect.volume > best_vol:
                best_vol = rect.volume
                best_rect = rect

        if best_rect is None or best_vol == 0:
            # Single remaining positions — emit 1x1x1 rects
            for i in range(len(positions)):
                t_i, h_i, w_i = positions[i].tolist()
                if mask[t_i, h_i, w_i]:
                    rects.append(Rect(t_i, t_i + 1, h_i, h_i + 1, w_i, w_i + 1))
                    mask[t_i, h_i, w_i] = False
            break

        rects.append(best_rect)
        # Mark covered
        mask[best_rect.t0:best_rect.t1,
             best_rect.h0:best_rect.h1,
             best_rect.w0:best_rect.w1] = False

    return rects


def _bounds_to_thw(bounds: Tuple[int, int, int, int, int, int]) -> Tuple[int, int, int, int, int, int]:
    """Extract t0, t1, h0, h1, w0, w1 from bounds tuple."""
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
        from canvas_engineering.canvas import _get_bounds
        src_bounds = _get_bounds(layout.regions[src])
        dst_bounds = _get_bounds(layout.regions[dst])

        src_t0, src_t1, src_h0, src_h1, src_w0, src_w1 = src_bounds
        dst_t0, dst_t1, dst_h0, dst_h1, dst_w0, dst_w1 = dst_bounds

        tile_h = max(mask_spec.tile_h, 1)
        tile_w = max(mask_spec.tile_w, 1)

        pairs: List[Tuple[List[int], List[int]]] = []

        # For each tile in the src region, find the corresponding tile in dst
        # Tiles are defined in absolute spatial coordinates
        # We iterate over tiles and collect positions from both regions
        # that fall within each tile
        src_h_range = range(src_h0, src_h1, tile_h)
        src_w_range = range(src_w0, src_w1, tile_w)

        for th in src_h_range:
            for tw in src_w_range:
                th_end = min(th + tile_h, src_h1)
                tw_end = min(tw + tile_w, src_w1)

                # Collect src indices in this tile (all timesteps)
                src_idx: List[int] = []
                for t in range(src_t0, src_t1):
                    for h in range(th, th_end):
                        for w in range(tw, tw_end):
                            src_idx.append(t * (layout.H * layout.W) + h * layout.W + w)

                # Collect dst indices in corresponding spatial tile (all timesteps)
                # Map tile coordinates to dst space
                dst_th = max(th, dst_h0)
                dst_th_end = min(th_end, dst_h1)
                dst_tw = max(tw, dst_w0)
                dst_tw_end = min(tw_end, dst_w1)

                if dst_th >= dst_th_end or dst_tw >= dst_tw_end:
                    # No overlap — use all dst indices as fallback
                    dst_idx = layout.region_indices(dst)
                else:
                    dst_idx_list: List[int] = []
                    for t in range(dst_t0, dst_t1):
                        for h in range(dst_th, dst_th_end):
                            for w in range(dst_tw, dst_tw_end):
                                dst_idx_list.append(
                                    t * (layout.H * layout.W) + h * layout.W + w)
                    dst_idx = dst_idx_list if dst_idx_list else layout.region_indices(dst)

                if src_idx and dst_idx:
                    pairs.append((src_idx, dst_idx))

        if not pairs:
            # Fallback: full attention
            pairs = [(layout.region_indices(src), layout.region_indices(dst))]
        return pairs

    elif mask_spec.kind == "sparse":
        # For sparse, return full — the caller should use rect_cover
        # on an explicit mask tensor to generate sparse index pairs.
        return [(layout.region_indices(src), layout.region_indices(dst))]

    else:
        # Unknown kind — fall back to full
        return [(layout.region_indices(src), layout.region_indices(dst))]

"""Canvas Types: compositional type system for latent space.

Declare structured types whose fields are latent regions. Types compose
(nesting), specialize (inheritance), and replicate (lists). A compiler
flattens any type hierarchy into a CanvasSchema -- a concrete CanvasLayout +
CanvasTopology ready for the diffusion process.

Works with dataclasses, Pydantic models, or any object with Field attributes.

Example:
    from dataclasses import dataclass
    from canvas_engineering.types import Field, compile_schema

    @dataclass
    class Robot:
        camera: Field = Field(12, 12)
        joints: Field = Field(1, 8)
        action: Field = Field(1, 8, loss_weight=2.0)

    bound = compile_schema(Robot(), T=8, H=16, W=16, d_model=256)
    canvas = bound.build_canvas()
    batch = bound.create_batch(4)
    bound["camera"].place(batch, camera_embs)
"""

import math
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from canvas_engineering.canvas import CanvasLayout, RegionSpec, SpatiotemporalCanvas
from canvas_engineering.connectivity import CanvasTopology, Connection
from canvas_engineering.schema import CanvasSchema


# -- Auto-wiring: family pair → default operator ----------------------

DEFAULT_WIRING: Dict[Tuple[str, str], str] = {
    ("observation", "state"):       "observe",
    ("state", "observation"):       "predict",
    ("state", "state"):             "integrate",
    ("state", "memory"):            "write",
    ("memory", "state"):            "retrieve",
    ("state", "action"):            "act",
    ("action", "state"):            "intervene",
    ("state", "residual"):          "emit_residual",
    ("observation", "residual"):    "emit_residual",
    ("observation", "observation"): "attend",
    ("action", "action"):           "attend",
    ("memory", "memory"):           "attend",
    ("residual", "residual"):       "attend",
}


def _apply_operators(
    connections: List[Connection],
    family_map: Dict[str, str],
) -> List[Connection]:
    """Replace default operator with family-derived operator where possible.

    Called only from compile_program(), never from compile_schema().
    """
    result = []
    for c in connections:
        src_family = family_map.get(c.src)
        dst_family = family_map.get(c.dst)
        if src_family and dst_family:
            op = DEFAULT_WIRING.get((src_family, dst_family), "attend")
        else:
            op = c.operator
        if op != c.operator:
            result.append(Connection(
                src=c.src, dst=c.dst, weight=c.weight,
                t_src=c.t_src, t_dst=c.t_dst, fn=c.fn,
                operator=op, write_mode=c.write_mode,
                temporal_fill=c.temporal_fill,
                interpolation_order=c.interpolation_order,
            ))
        else:
            result.append(c)
    return result


# -- Field Declaration ------------------------------------------------

@dataclass(frozen=True)
class Field:
    """A latent region declaration.

    Declares a region of spatial_shape positions on the canvas grid, each
    carrying the canvas's d_model dimensionality. Default (1, 1) = scalar
    (1 position) for 2D spatial layouts.

    For 2D, ``Field(h, w)`` is the standard form (backward compatible).
    For arbitrary dimensions, use ``Field(spatial_shape=(8,))`` for 1D,
    ``Field(spatial_shape=(4, 4, 4))`` for 3D, etc.

    Args:
        h: Height on the canvas grid. Default 1. (2D compat)
        w: Width on the canvas grid. Default 1. (2D compat)
        period: Canvas frames per real-world update (1 = every frame).
        is_output: Whether this field participates in diffusion loss.
        loss_weight: Relative loss weight for positions in this field.
        attn: Default attention function type for outgoing connections.
        semantic_type: Human-readable modality description.
        temporal_extent: Number of timesteps this field spans. None = full T.
        family: Region family for v2 process semantics. None = infer default
            ("state"). One of: observation, state, memory, residual, action.
        tags: Semantic sub-tags within a family. E.g., ("belief", "object").
        carrier: Dynamics carrier. None = infer default ("deterministic").
            One of: deterministic, diffusive, filter, memory, residual.
        spatial_shape: Arbitrary spatial dimensions, e.g. (8,) for 1D,
            (4, 4, 4) for 3D. Mutually exclusive with h/w.
    """
    h: int = 1
    w: int = 1
    period: int = 1
    is_output: bool = True
    loss_weight: float = 1.0
    attn: str = "cross_attention"
    semantic_type: Optional[str] = None
    temporal_extent: Optional[int] = None
    family: Optional[str] = None
    tags: Tuple[str, ...] = ()
    carrier: Optional[str] = None
    spatial_shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.spatial_shape is not None:
            # Explicit spatial_shape provided — h/w must be defaults
            if self.h != 1 or self.w != 1:
                raise ValueError(
                    "Cannot specify both spatial_shape and non-default h/w"
                )
            # Backfill h/w from spatial_shape for compat reads
            if len(self.spatial_shape) >= 1:
                object.__setattr__(self, 'h', self.spatial_shape[0])
            if len(self.spatial_shape) >= 2:
                object.__setattr__(self, 'w', self.spatial_shape[1])
        else:
            # Derive spatial_shape from h/w (the 2D default path)
            object.__setattr__(self, 'spatial_shape', (self.h, self.w))

    @property
    def num_positions(self) -> int:
        """Spatial positions per timestep: prod(spatial_shape)."""
        return math.prod(self.spatial_shape)


# -- Layout Strategy --------------------------------------------------

class LayoutStrategy(Enum):
    """How to arrange fields on the (H, W) grid."""
    PACKED = "packed"
    INTERLEAVED = "interleaved"


# -- Connectivity Policy ----------------------------------------------

@dataclass
class ConnectivityPolicy:
    """Default connectivity rules for compiled schemas.

    Every nested type and array element automatically gets a coarse-grained
    field at its own path. Parent fields connect bidirectionally to the
    coarse-grained field; the coarse-grained field connects bidirectionally
    to child fields.

    Coarse-grained field size is configured on the types themselves:
      - Set ``__coarse__ = Field(h, w)`` on a class to define its
        default coarse-grained field size when it appears as a child anywhere.
      - Set ``metadata={"coarse": Field(h, w)}`` on a dataclass field
        to override the coarse-grained field for that specific parent->child edge.
      - Falls back to Field(1, 1) if neither is set.

    Args:
        intra: How fields within a single type connect.
            "dense" (default), "isolated", "causal_chain", "star"
        array_element: How elements of the same list connect.
            "isolated" (default), "dense", "matched_fields", "ring"
        temporal: Temporal constraint policy for all connections.
            "dense" (default), "causal", "same_frame"
    """
    intra: str = "dense"
    array_element: str = "isolated"
    temporal: str = "dense"


# -- Tree Walking (Internal) ------------------------------------------

@dataclass
class _FieldEntry:
    """A Field found during tree walking."""
    path: str
    field: Field
    local_name: str


@dataclass
class _TypeNode:
    """A node in the walked type hierarchy."""
    path: str
    fields: List[_FieldEntry]
    children: List['_TypeNode']
    arrays: Dict[str, List['_TypeNode']]
    parent: Optional['_TypeNode']
    coarse_field: Optional[Field] = None  # resolved coarse-grained field config for this node


_SKIP_TYPES = (int, float, str, bool, type(None), bytes, torch.Tensor, Field)


def _get_attrs(obj: Any) -> List[str]:
    """Get attribute names, supporting Pydantic, dataclasses, and plain objects."""
    # Pydantic v2
    if hasattr(obj, 'model_fields'):
        return list(obj.model_fields.keys())
    # dataclass
    if hasattr(obj, '__dataclass_fields__'):
        return list(obj.__dataclass_fields__.keys())
    # Plain object
    if hasattr(obj, '__dict__'):
        return [k for k in obj.__dict__ if not k.startswith('_')]
    return []


def _has_canvas_fields(obj: Any) -> bool:
    """Check if an object has any Field attributes (directly or nested)."""
    if isinstance(obj, _SKIP_TYPES):
        return False
    for attr_name in _get_attrs(obj):
        try:
            value = getattr(obj, attr_name)
        except Exception:
            continue
        if isinstance(value, Field):
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Field):
                    return True
                if not isinstance(item, _SKIP_TYPES) and _has_canvas_fields(item):
                    return True
        elif not isinstance(value, _SKIP_TYPES) and _has_canvas_fields(value):
            return True
    return False


def _resolve_coarse_field(child_obj: Any, parent_obj: Any, attr_name: str) -> Field:
    """Determine the coarse-grained Field for a child.

    Priority:
      1. Parent's dataclass field metadata: ``metadata={"coarse": Field(...)}``
      2. Child type's ``__coarse__`` class attribute
      3. Default: ``Field(1, 1)``
    """
    # 1. Parent field metadata override
    if hasattr(parent_obj, '__dataclass_fields__'):
        dc_fields = parent_obj.__dataclass_fields__
        if attr_name in dc_fields:
            meta = dc_fields[attr_name].metadata
            if meta and "coarse" in meta:
                gw = meta["coarse"]
                if isinstance(gw, Field):
                    return gw

    # 2. Child type's __coarse__
    child_type = type(child_obj)
    type_gw = getattr(child_type, '__coarse__', None)
    if isinstance(type_gw, Field):
        return type_gw

    # 3. Default
    return Field(1, 1)


def _walk(obj: Any, path: str = "", local_name: str = "",
          parent: Optional[_TypeNode] = None,
          parent_obj: Any = None) -> _TypeNode:
    """Walk an object tree and build a _TypeNode hierarchy."""
    node = _TypeNode(
        path=path, fields=[], children=[],
        arrays={}, parent=parent,
    )

    for attr_name in _get_attrs(obj):
        try:
            value = getattr(obj, attr_name)
        except Exception:
            continue

        child_path = "{}.{}".format(path, attr_name) if path else attr_name

        if isinstance(value, Field):
            node.fields.append(_FieldEntry(child_path, value, attr_name))

        elif isinstance(value, (list, tuple)):
            elements = []
            for i, item in enumerate(value):
                if not isinstance(item, _SKIP_TYPES) and _has_canvas_fields(item):
                    elem_path = "{}[{}]".format(child_path, i)
                    elem_node = _walk(item, elem_path, attr_name,
                                      parent=node, parent_obj=obj)
                    elem_node.coarse_field = _resolve_coarse_field(item, obj, attr_name)
                    elements.append(elem_node)
            if elements:
                node.arrays[attr_name] = elements

        elif not isinstance(value, _SKIP_TYPES):
            if _has_canvas_fields(value):
                child_node = _walk(value, child_path, attr_name,
                                   parent=node, parent_obj=obj)
                child_node.coarse_field = _resolve_coarse_field(value, obj, attr_name)
                node.children.append(child_node)

    return node


def _flatten_fields(node: _TypeNode) -> List[Tuple[str, Field, str]]:
    """Flatten tree into [(path, Field, local_name), ...]."""
    result = []
    for fe in node.fields:
        result.append((fe.path, fe.field, fe.local_name))
    for child in node.children:
        result.extend(_flatten_fields(child))
    for elements in node.arrays.values():
        for elem in elements:
            result.extend(_flatten_fields(elem))
    return result


# -- Coarse Field Insertion (Internal) --------------------------------

def _block_weighted_mean_period_from_tree(node: _TypeNode) -> int:
    """Get block-count-weighted mean period from all Fields in a subtree.

    Each field's period is weighted by its canvas footprint (h * w).
    Fields with more positions have more influence on the coarse-grained
    summary's update rate, so the summary tracks the dominant timescale
    of the subtree rather than giving every field an equal vote.
    """
    all_fields = _flatten_fields_raw(node)
    if not all_fields:
        return 1
    weighted_sum = 0
    total_weight = 0
    for fe in all_fields:
        block_count = fe.field.h * fe.field.w
        weighted_sum += fe.field.period * block_count
        total_weight += block_count
    if total_weight == 0:
        return 1
    return max(1, round(weighted_sum / total_weight))


def _flatten_fields_raw(node: _TypeNode) -> List[_FieldEntry]:
    """Flatten to raw _FieldEntry list (for internal use)."""
    result = list(node.fields)
    for child in node.children:
        result.extend(_flatten_fields_raw(child))
    for elems in node.arrays.values():
        for elem in elems:
            result.extend(_flatten_fields_raw(elem))
    return result


def _insert_coarse_fields(node: _TypeNode) -> _TypeNode:
    """Insert coarse-grained fields between parent and each child/array-element.

    For each child node, creates a new intermediate node containing a
    coarse-grained field at the child's path. The original child becomes a
    grandchild. Cross-level attention bottlenecks through the coarse-grained field.

    Coarse-grained field params come from the child's ``coarse_field`` (resolved
    during _walk from ``__coarse__`` class attrs and field metadata).

    Before: Parent -> [Child1(fields...), Child2(fields...)]
    After:  Parent -> [CG1(coarse) -> Child1(fields...), CG2(coarse) -> Child2(fields...)]

    Returns the modified node (in-place).
    """
    # Process children
    new_children = []
    for child in node.children:
        _insert_coarse_fields(child)  # recurse first

        local = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
        cg_template = child.coarse_field or Field(1, 1)
        mp = _block_weighted_mean_period_from_tree(child)
        coarse_field = Field(
            cg_template.h, cg_template.w, period=mp,
            semantic_type=cg_template.semantic_type or "coarse: {}".format(child.path),
            is_output=cg_template.is_output,
            loss_weight=cg_template.loss_weight,
            attn=cg_template.attn,
        )
        coarse_entry = _FieldEntry(
            path=child.path, field=coarse_field, local_name=local,
        )

        coarse_node = _TypeNode(
            path=child.path,
            fields=[coarse_entry],
            children=[child],
            arrays={},
            parent=node,
        )
        child.parent = coarse_node
        new_children.append(coarse_node)
    node.children = new_children

    # Process array elements
    new_arrays = {}
    for array_name, elements in node.arrays.items():
        new_elements = []
        for elem in elements:
            _insert_coarse_fields(elem)  # recurse first

            cg_template = elem.coarse_field or Field(1, 1)
            mp = _block_weighted_mean_period_from_tree(elem)
            coarse_field = Field(
                cg_template.h, cg_template.w, period=mp,
                semantic_type=cg_template.semantic_type or "coarse: {}".format(elem.path),
                is_output=cg_template.is_output,
                loss_weight=cg_template.loss_weight,
                attn=cg_template.attn,
            )
            coarse_entry = _FieldEntry(
                path=elem.path, field=coarse_field, local_name=array_name,
            )

            coarse_node = _TypeNode(
                path=elem.path,
                fields=[coarse_entry],
                children=[elem],
                arrays={},
                parent=node,
            )
            elem.parent = coarse_node
            new_elements.append(coarse_node)
        new_arrays[array_name] = new_elements
    node.arrays = new_arrays

    return node


# -- Auto-sizing (Internal) -------------------------------------------

def _auto_canvas_size(
    fields: List[Tuple[str, int, int]],
    pad: float = 1.15,
) -> Tuple[int, int]:
    """Compute minimum (H, W) that can pack all 2D fields.

    Thin wrapper around _auto_spatial_shape for backward compat.
    """
    nd_fields = [(p, (h, w)) for p, h, w in fields]
    shape = _auto_spatial_shape(nd_fields, 2, pad)
    return (shape[0], shape[1])


def _auto_spatial_shape(
    fields: List[Tuple[str, Tuple[int, ...]]],
    n_dims: int,
    pad: float = 1.15,
) -> Tuple[int, ...]:
    """Compute minimum spatial_shape that can pack all fields.

    Args:
        fields: [(path, spatial_shape), ...] from flattened field list.
        n_dims: Number of spatial dimensions (1, 2, 3, ...).
        pad: Padding factor (1.15 = 15% slack).

    Returns:
        Spatial shape tuple of length n_dims.
    """
    if not fields:
        return tuple(4 for _ in range(n_dims))

    if n_dims == 1:
        total = sum(s[0] for _, s in fields)
        return (int(math.ceil(total * pad)),)

    if n_dims == 2:
        # Existing sqrt-area strip-packing heuristic
        max_w = max(s[1] for _, s in fields)
        max_h = max(s[0] for _, s in fields)
        total_area = sum(s[0] * s[1] for _, s in fields)

        side = int(math.ceil(math.sqrt(total_area * pad)))
        W = max(side, max_w)

        # Simulate strip packing to find needed H
        row_h = 0
        row_w = 0
        row_max_h = 0
        for _, s in fields:
            h, w = s[0], s[1]
            if row_w + w > W:
                row_h += row_max_h
                row_w = 0
                row_max_h = 0
            row_w += w
            row_max_h = max(row_max_h, h)
        needed_H = row_h + row_max_h
        H = max(int(math.ceil(needed_H * pad)), max_h)
        return (H, W)

    # 3D+: nth-root of total volume, then validate with greedy packing
    total_vol = sum(math.prod(s) for _, s in fields)
    side = int(math.ceil(total_vol ** (1.0 / n_dims) * pad ** (1.0 / n_dims)))
    # Ensure each dim is at least as large as the biggest field in that dim
    shape = []
    for d in range(n_dims):
        max_d = max(s[d] for _, s in fields)
        shape.append(max(side, max_d))
    return tuple(shape)


# -- Packing (Internal) -----------------------------------------------

def _pack_1d(
    fields: List[Tuple[str, Tuple[int, ...]]],
    spatial_shape: Tuple[int, ...],
) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    """Linear 1D packing: stack fields end-to-end.

    Returns dict of path -> ((x0, x1),).
    """
    size = spatial_shape[0]
    placements = {}
    offset = 0
    for path, shape in fields:
        w = shape[0]
        if offset + w > size:
            raise ValueError(
                "Cannot pack '{}' (size {}): spatial dim {} exceeded at offset {}. "
                "Need at least {}".format(path, w, size, offset, offset + w)
            )
        placements[path] = ((offset, offset + w),)
        offset += w
    return placements


def _pack_2d(
    fields: List[Tuple[str, Tuple[int, ...]]],
    spatial_shape: Tuple[int, ...],
) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    """Row-based 2D strip packing: place fields left-to-right, top-to-bottom.

    Returns dict of path -> ((h0, h1), (w0, w1)).
    """
    H, W = spatial_shape[0], spatial_shape[1]
    placements = {}
    row_h = 0
    row_w = 0
    row_max_h = 0

    for path, shape in fields:
        h, w = shape[0], shape[1]
        if w > W:
            raise ValueError(
                "Field '{}' width {} exceeds canvas W={}".format(path, w, W)
            )
        if row_w + w > W:
            row_h += row_max_h
            row_w = 0
            row_max_h = 0
        if row_h + h > H:
            raise ValueError(
                "Cannot pack '{}' ({}x{}): grid H={} exceeded at row_h={}. "
                "Need at least H={}".format(path, h, w, H, row_h, row_h + h)
            )
        placements[path] = ((row_h, row_h + h), (row_w, row_w + w))
        row_w += w
        row_max_h = max(row_max_h, h)

    return placements


def _pack_nd(
    fields: List[Tuple[str, Tuple[int, ...]]],
    spatial_shape: Tuple[int, ...],
) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    """N-dimensional packing dispatcher.

    - 1D: linear concatenation
    - 2D: row-based strip packing
    - 3D+: stack along dim-0, recursively pack remaining dims
    """
    n = len(spatial_shape)
    if n == 1:
        return _pack_1d(fields, spatial_shape)
    if n == 2:
        return _pack_2d(fields, spatial_shape)

    # 3D+: stack along first dimension, pack (n-1)-D slices
    dim0_size = spatial_shape[0]
    sub_shape = spatial_shape[1:]
    placements = {}
    d0_offset = 0
    for path, shape in fields:
        d0_extent = shape[0]
        if d0_offset + d0_extent > dim0_size:
            raise ValueError(
                "Cannot pack '{}' (dim0 size {}): spatial dim 0 = {} exceeded "
                "at offset {}".format(path, d0_extent, dim0_size, d0_offset)
            )
        # Pack sub-dimensions at origin (each field gets its own slice)
        sub_ranges = tuple((0, shape[i]) for i in range(1, len(shape)))
        # Validate sub-dimensions fit
        for i, (_, hi) in enumerate(sub_ranges):
            if hi > sub_shape[i]:
                raise ValueError(
                    "Cannot pack '{}': dim {} size {} exceeds canvas "
                    "dim {}".format(path, i + 1, hi, sub_shape[i])
                )
        placements[path] = ((d0_offset, d0_offset + d0_extent),) + sub_ranges
        d0_offset += d0_extent
    return placements


def _pack_strip(
    fields: List[Tuple[str, int, int]],
    H: int, W: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Row-based strip packing (backward-compat wrapper).

    Returns dict of path -> (h0, h1, w0, w1).
    """
    nd_fields = [(p, (h, w)) for p, h, w in fields]
    nd_result = _pack_2d(nd_fields, (H, W))
    return {p: (r[0][0], r[0][1], r[1][0], r[1][1]) for p, r in nd_result.items()}


def _pack_interleaved(
    fields: List[Tuple[str, int, int, str]],
    H: int, W: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Interleaved packing: group by local field name, then strip-pack.

    Fields with the same local name (e.g., all "thought" fields from parent
    and children) are placed adjacent on the grid. This exploits spatial
    locality in pretrained models.
    """
    from collections import OrderedDict
    groups = OrderedDict()  # type: Dict[str, List[Tuple[str, int, int]]]
    for path, h, w, local_name in fields:
        groups.setdefault(local_name, []).append((path, h, w))

    ordered = []
    for group in groups.values():
        ordered.extend(group)

    return _pack_strip(ordered, H, W)


# -- Connectivity Generation (Internal) -------------------------------

def _intra_connections(node: _TypeNode, policy: ConnectivityPolicy) -> List[Connection]:
    """Connect fields within a single type instance."""
    conns = []
    fields = node.fields

    if policy.intra == "dense":
        for a in fields:
            for b in fields:
                conns.append(Connection(src=a.path, dst=b.path))

    elif policy.intra == "isolated":
        for f in fields:
            conns.append(Connection(src=f.path, dst=f.path))

    elif policy.intra == "causal_chain":
        for f in fields:
            conns.append(Connection(src=f.path, dst=f.path))
        for i in range(1, len(fields)):
            conns.append(Connection(src=fields[i].path, dst=fields[i - 1].path))

    elif policy.intra == "star":
        if fields:
            hub = fields[0]
            conns.append(Connection(src=hub.path, dst=hub.path))
            for spoke in fields[1:]:
                conns.append(Connection(src=spoke.path, dst=spoke.path))
                conns.append(Connection(src=hub.path, dst=spoke.path))
                conns.append(Connection(src=spoke.path, dst=hub.path))

    return conns


def _parent_child_connections(
    parent: _TypeNode, child: _TypeNode,
) -> List[Connection]:
    """Connect parent fields to child fields (bidirectional).

    With coarse-grained field insertion, this is always called between a
    parent node and a coarse-grained node (or coarse-grained node and its
    child). The coarse-grained field provides the bottleneck -- no policy
    dispatch needed.
    """
    conns = []
    for pf in parent.fields:
        for cf in child.fields:
            conns.append(Connection(src=pf.path, dst=cf.path))
            conns.append(Connection(src=cf.path, dst=pf.path))
    return conns


def _array_element_connections(
    elements: List[_TypeNode], policy: ConnectivityPolicy,
) -> List[Connection]:
    """Connect elements of the same array to each other."""
    conns = []

    if policy.array_element == "isolated":
        return conns

    elif policy.array_element == "dense":
        for i, a in enumerate(elements):
            for j, b in enumerate(elements):
                if i != j:
                    for fa in a.fields:
                        for fb in b.fields:
                            conns.append(Connection(src=fa.path, dst=fb.path))

    elif policy.array_element == "matched_fields":
        for i, a in enumerate(elements):
            for j, b in enumerate(elements):
                if i != j:
                    for fa in a.fields:
                        for fb in b.fields:
                            if fa.local_name == fb.local_name:
                                conns.append(Connection(src=fa.path, dst=fb.path))

    elif policy.array_element == "ring":
        for i in range(len(elements)):
            j = (i + 1) % len(elements)
            if i != j:
                for fa in elements[i].fields:
                    for fb in elements[j].fields:
                        conns.append(Connection(src=fa.path, dst=fb.path))
                        conns.append(Connection(src=fb.path, dst=fa.path))

    return conns


def _generate_connections(
    node: _TypeNode, policy: ConnectivityPolicy,
) -> List[Connection]:
    """Generate all connections from the type hierarchy."""
    conns = []

    # Intra-type for this node
    conns.extend(_intra_connections(node, policy))

    # Recurse into single children
    for child in node.children:
        conns.extend(_generate_connections(child, policy))
        conns.extend(_parent_child_connections(node, child))

    # Recurse into array elements
    for elements in node.arrays.values():
        for elem in elements:
            conns.extend(_generate_connections(elem, policy))
            conns.extend(_parent_child_connections(node, elem))
        conns.extend(_array_element_connections(elements, policy))

    return conns


def _apply_temporal(
    connections: List[Connection], policy: ConnectivityPolicy,
) -> List[Connection]:
    """Apply temporal constraints to all connections."""
    if policy.temporal == "dense":
        return connections

    result = []
    for c in connections:
        if policy.temporal == "same_frame":
            result.append(Connection(
                src=c.src, dst=c.dst, weight=c.weight,
                t_src=0, t_dst=0, fn=c.fn,
                operator=c.operator, write_mode=c.write_mode,
                temporal_fill=c.temporal_fill,
                interpolation_order=c.interpolation_order,
            ))
        elif policy.temporal == "causal":
            if c.src == c.dst:
                # Self-connection: same-frame + prev-frame
                result.append(Connection(
                    src=c.src, dst=c.dst, weight=c.weight,
                    t_src=0, t_dst=0, fn=c.fn,
                    operator=c.operator, write_mode=c.write_mode,
                    temporal_fill=c.temporal_fill,
                    interpolation_order=c.interpolation_order,
                ))
                result.append(Connection(
                    src=c.src, dst=c.dst, weight=c.weight,
                    t_src=0, t_dst=-1, fn=c.fn,
                    operator=c.operator, write_mode=c.write_mode,
                    temporal_fill=c.temporal_fill,
                    interpolation_order=c.interpolation_order,
                ))
            else:
                # Cross-connection: prev-frame only
                result.append(Connection(
                    src=c.src, dst=c.dst, weight=c.weight,
                    t_src=0, t_dst=-1, fn=c.fn,
                    operator=c.operator, write_mode=c.write_mode,
                    temporal_fill=c.temporal_fill,
                    interpolation_order=c.interpolation_order,
                ))
    return result


def _deduplicate(connections: List[Connection]) -> List[Connection]:
    """Remove duplicate connections."""
    seen = set()
    result = []
    for c in connections:
        key = (c.src, c.dst, c.weight, c.t_src, c.t_dst, c.fn,
               c.operator, c.write_mode,
               c.temporal_fill, c.interpolation_order)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


# -- Bound Schema -----------------------------------------------------

class BoundField:
    """A Field bound to a compiled canvas region.

    Provides convenient access to place/extract operations via a canvas.
    """

    def __init__(self, region_name: str, spec: RegionSpec,
                 bound_schema: 'BoundSchema'):
        self.region_name = region_name
        self.spec = spec
        self._schema = bound_schema

    @property
    def num_positions(self) -> int:
        """Total positions in this region."""
        bounds = self.spec.bounds
        result = 1
        for i in range(0, len(bounds), 2):
            result *= bounds[i + 1] - bounds[i]
        return result

    def place(self, batch: torch.Tensor, embeddings: torch.Tensor,
              canvas: Optional[SpatiotemporalCanvas] = None) -> torch.Tensor:
        """Write embeddings into this field's canvas region."""
        c = canvas if canvas is not None else self._schema._canvas
        if c is None:
            raise RuntimeError(
                "No canvas bound. Call build_canvas() first or pass canvas=")
        return c.place(batch, embeddings, self.region_name)

    def extract(self, batch: torch.Tensor,
                canvas: Optional[SpatiotemporalCanvas] = None) -> torch.Tensor:
        """Read embeddings from this field's canvas region."""
        c = canvas if canvas is not None else self._schema._canvas
        if c is None:
            raise RuntimeError(
                "No canvas bound. Call build_canvas() first or pass canvas=")
        return c.extract(batch, self.region_name)

    def indices(self) -> List[int]:
        """Flat indices for this field's region."""
        return self._schema.layout.region_indices(self.region_name)

    def __repr__(self) -> str:
        bounds = self.spec.bounds
        t0, t1 = bounds[0], bounds[1]
        spatial_extents = [
            bounds[i + 1] - bounds[i] for i in range(2, len(bounds), 2)
        ]
        shape_str = "x".join(str(s) for s in spatial_extents)
        return "BoundField('{}', t={}-{}, {})".format(
            self.region_name, t0, t1, shape_str)


class BoundSchema:
    """A compiled type hierarchy bound to a CanvasSchema.

    Provides field access by dotted path and optional canvas binding.

    Example:
        bound = compile_schema(robot, T=8, H=16, W=16, d_model=256)
        canvas = bound.build_canvas()
        batch = bound.create_batch(4)
        bound["camera"].place(batch, camera_embs)
        actions = bound["action"].extract(batch)
    """

    def __init__(self, schema: CanvasSchema,
                 region_specs: Dict[str, RegionSpec]):
        self.schema = schema
        self._canvas = None  # type: Optional[SpatiotemporalCanvas]
        self._fields = {
            path: BoundField(path, spec, self)
            for path, spec in region_specs.items()
        }

    @property
    def layout(self) -> CanvasLayout:
        return self.schema.layout

    @property
    def topology(self) -> Optional[CanvasTopology]:
        return self.schema.topology

    @property
    def fields(self) -> Dict[str, BoundField]:
        return dict(self._fields)

    @property
    def field_names(self) -> List[str]:
        return list(self._fields.keys())

    def __getitem__(self, path: str) -> BoundField:
        return self._fields[path]

    def __contains__(self, path: str) -> bool:
        return path in self._fields

    def build_canvas(self, semantic_conditioner=None) -> SpatiotemporalCanvas:
        """Create and store a SpatiotemporalCanvas for this schema.

        Args:
            semantic_conditioner: Optional SemanticConditioner for semantic
                embedding conditioning. If provided, replaces learned modality
                embeddings with frozen semantic embeddings + learned residuals.
        """
        self._canvas = SpatiotemporalCanvas(
            self.layout, semantic_conditioner=semantic_conditioner
        )
        return self._canvas

    def create_batch(self, batch_size: int) -> torch.Tensor:
        """Create an empty canvas batch. Builds canvas if needed."""
        if self._canvas is None:
            self.build_canvas()
        return self._canvas.create_empty(batch_size)

    def build_semantic_conditioner(
        self,
        embed_fn,
        embed_dim: int = 1536,
        freeze_embeddings: bool = True,
        learn_residuals: bool = True,
    ):
        """Build a SemanticConditioner from this schema's field paths.

        Computes semantic embeddings for all fields using the provided
        embedding function, then creates a SemanticConditioner.

        Args:
            embed_fn: Callable taking list[str] -> list[list[float]].
                Each string is a semantic type description.
            embed_dim: Dimension of the embeddings returned by embed_fn.
            freeze_embeddings: If True, freeze the raw embeddings.
            learn_residuals: If True, add learned residuals.

        Returns:
            SemanticConditioner ready to use with build_canvas().

        Example:
            conditioner = bound.build_semantic_conditioner(my_embed_fn)
            canvas = bound.build_canvas(semantic_conditioner=conditioner)
        """
        from canvas_engineering.semantic import (
            SemanticConditioner,
            compute_semantic_embeddings,
        )

        # Collect semantic types from region specs
        semantic_types = {}
        for path, bf in self._fields.items():
            if bf.spec.semantic_type:
                semantic_types[path] = bf.spec.semantic_type

        embeddings = compute_semantic_embeddings(
            self.field_names, embed_fn, semantic_types=semantic_types,
        )

        return SemanticConditioner(
            d_model=self.layout.d_model,
            embed_dim=embed_dim,
            region_embeddings=embeddings,
            freeze_embeddings=freeze_embeddings,
            learn_residuals=learn_residuals,
        )

    def summary(self) -> str:
        """Human-readable summary of the compiled schema."""
        lines = ["BoundSchema ({} fields, {} positions):".format(
            len(self._fields), self.layout.num_positions)]
        for name, bf in self._fields.items():
            bounds = bf.spec.bounds
            t0, t1 = bounds[0], bounds[1]
            n = bf.num_positions
            # Format spatial bounds as dim_lo-dim_hi pairs
            spatial_parts = []
            for i in range(2, len(bounds), 2):
                spatial_parts.append("{}-{}".format(bounds[i], bounds[i + 1]))
            spatial_str = ", ".join(spatial_parts)
            lines.append("  {} -> (t={}-{}, {}) {}pos".format(
                name, t0, t1, spatial_str, n))
        if self.topology:
            lines.append("  {} connections".format(
                len(self.topology.connections)))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "BoundSchema(fields={}, positions={})".format(
            len(self._fields), self.layout.num_positions)


# -- Compilation ------------------------------------------------------

def compile_schema(
    root: Any,
    T: int = 1,
    spatial_shape: Optional[Tuple[int, ...]] = None,
    d_model: int = 64,
    connectivity: Optional[ConnectivityPolicy] = None,
    layout_strategy: Union[str, LayoutStrategy] = LayoutStrategy.PACKED,
    t_current: int = 0,
    # Backward compat — converted to spatial_shape
    H: Optional[int] = None,
    W: Optional[int] = None,
) -> BoundSchema:
    """Compile an object hierarchy into a BoundSchema.

    Walks the object tree, finds Field attributes, packs them onto a
    (T, *spatial_shape) grid, generates connectivity based on the type
    hierarchy, and returns a BoundSchema wrapping the compiled CanvasSchema.

    Works with dataclasses, Pydantic models, or plain objects.

    When spatial_shape is None (and H/W are also None), the spatial
    dimensions are auto-computed from the declared field dimensions --
    like a C compiler sizing a struct from its members.

    Args:
        root: Object with Field attributes. Lists create array regions.
        T: Temporal extent of the canvas. Default 1.
        spatial_shape: Spatial dimensions, e.g. (16, 16) for 2D or (32,)
            for 1D. None = auto-sized.
        d_model: Latent dimensionality per position. Default 64.
        connectivity: Connectivity policy.
        layout_strategy: "packed" (default) or "interleaved".
        t_current: Timestep boundary for output mask.
        H: (deprecated) Height. Use spatial_shape instead.
        W: (deprecated) Width. Use spatial_shape instead.

    Returns:
        BoundSchema with compiled layout, topology, and bound field access.

    Example:
        @dataclass
        class Robot:
            camera: Field = Field(12, 12)
            joints: Field = Field(1, 8)
            action: Field = Field(1, 8, loss_weight=2.0)

        # Auto-sized canvas:
        bound = compile_schema(Robot(), T=8, d_model=256)

        # Explicit 2D canvas:
        bound = compile_schema(Robot(), T=8, spatial_shape=(16, 16), d_model=256)

        # Backward-compat (H/W):
        bound = compile_schema(Robot(), T=8, H=16, W=16, d_model=256)
    """
    # Resolve spatial_shape from H/W compat
    if H is not None or W is not None:
        if spatial_shape is not None:
            raise ValueError("Cannot specify both spatial_shape and H/W")
        # H/W path: both may be None (auto-sized below)

    if connectivity is None:
        connectivity = ConnectivityPolicy()

    if isinstance(layout_strategy, str):
        layout_strategy = LayoutStrategy(layout_strategy)

    # 1. Walk the object tree
    tree = _walk(root)

    # 1b. Insert coarse-grained fields for all nested types
    _insert_coarse_fields(tree)

    # 2. Flatten to field list
    flat = _flatten_fields(tree)

    if not flat:
        raise ValueError("No Field attributes found in object tree")

    # 2b. Determine spatial dimensions and n_dims
    # Infer n_dims from fields or explicit spatial_shape
    if spatial_shape is not None:
        n_dims = len(spatial_shape)
    elif H is not None or W is not None:
        n_dims = 2
    else:
        # Infer from field spatial_shape — use max dimensionality across fields
        n_dims = max(len(f.spatial_shape) for _, f, _ in flat)

    # 2c. Auto-size spatial_shape if not fully specified
    if spatial_shape is None:
        if H is not None or W is not None:
            # Legacy H/W path with auto-sizing
            if H is None or W is None:
                pack_dims = [(path, f.h, f.w) for path, f, _ in flat]
                auto_H, auto_W = _auto_canvas_size(pack_dims)
                if H is None:
                    H = auto_H
                if W is None:
                    W = auto_W
            spatial_shape = (H, W)
        else:
            # Full auto-size from field spatial_shapes
            pack_fields = [(path, f.spatial_shape) for path, f, _ in flat]
            # Pad field shapes to n_dims if needed (1-pad trailing dims)
            padded = []
            for path, s in pack_fields:
                if len(s) < n_dims:
                    s = s + (1,) * (n_dims - len(s))
                padded.append((path, s))
            spatial_shape = _auto_spatial_shape(padded, n_dims)

    # 3. Pack onto grid
    if n_dims == 2 and layout_strategy == LayoutStrategy.INTERLEAVED:
        # Interleaved only supported for 2D
        pack_input = [(path, f.h, f.w, local) for path, f, local in flat]
        hw_bounds = _pack_interleaved(pack_input, spatial_shape[0], spatial_shape[1])
        # Convert to nd_bounds format
        nd_bounds = {
            p: ((b[0], b[1]), (b[2], b[3])) for p, b in hw_bounds.items()
        }
    else:
        # N-D packing
        pack_fields = [(path, f.spatial_shape) for path, f, _ in flat]
        # Pad field shapes to n_dims
        padded = []
        for path, s in pack_fields:
            if len(s) < n_dims:
                s = s + (1,) * (n_dims - len(s))
            padded.append((path, s))
        nd_bounds = _pack_nd(padded, spatial_shape)

    # 4. Build RegionSpecs with full bounds (t0, t1, s0_lo, s0_hi, ...)
    from canvas_engineering.semantic import auto_semantic_type

    field_map = {path: f for path, f, _ in flat}
    regions = {}  # type: Dict[str, Union[Tuple, RegionSpec]]
    for path, spatial_ranges in nd_bounds.items():
        f = field_map[path]
        t_extent = f.temporal_extent if f.temporal_extent is not None else T
        t_extent = min(t_extent, T)
        sem_type = f.semantic_type if f.semantic_type else auto_semantic_type(path)
        carrier = f.carrier if f.carrier else "deterministic"
        # Flatten spatial ranges into bounds tuple: (t0, t1, s0_lo, s0_hi, ...)
        flat_spatial = []
        for lo, hi in spatial_ranges:
            flat_spatial.extend([lo, hi])
        regions[path] = RegionSpec(
            bounds=(0, t_extent) + tuple(flat_spatial),
            period=f.period,
            is_output=f.is_output,
            loss_weight=f.loss_weight,
            default_attn=f.attn,
            semantic_type=sem_type,
            carrier=carrier,
        )

    # 5. Generate connectivity
    connections = _generate_connections(tree, connectivity)
    connections = _apply_temporal(connections, connectivity)
    connections = _deduplicate(connections)

    # 6. Assemble
    layout = CanvasLayout(
        T=T, spatial_shape=spatial_shape, d_model=d_model,
        regions=regions,
        t_current=t_current,
    )
    topology = CanvasTopology(connections=connections) if connections else None
    schema = CanvasSchema(layout=layout, topology=topology)

    return BoundSchema(schema, regions)


# -- Program Compilation ----------------------------------------------

def compile_program(
    root: Any,
    T: int = 1,
    spatial_shape: Optional[Tuple[int, ...]] = None,
    d_model: int = 64,
    connectivity: Optional[ConnectivityPolicy] = None,
    layout_strategy: Union[str, LayoutStrategy] = LayoutStrategy.PACKED,
    t_current: int = 0,
    # Backward compat
    H: Optional[int] = None,
    W: Optional[int] = None,
) -> Tuple["BoundSchema", "CanvasProgram"]:
    """Compile an object hierarchy into a BoundSchema + CanvasProgram.

    Same as compile_schema() but also generates a CanvasProgram with
    RegionPrograms derived from Field's family/tags/carrier attributes.
    Fields without family/carrier get default RegionPrograms.

    Args:
        root: Object with Field attributes.
        T: Temporal extent.
        spatial_shape: Spatial dimensions. None = auto-sized.
        d_model: Latent dimensionality.
        connectivity: Connectivity policy.
        layout_strategy: Packing strategy.
        t_current: Timestep boundary for output mask.
        H: (deprecated) Use spatial_shape instead.
        W: (deprecated) Use spatial_shape instead.

    Returns:
        (BoundSchema, CanvasProgram) tuple.

    Example:
        @dataclass
        class Robot:
            camera: Field = Field(12, 12, family="observation")
            joints: Field = Field(1, 8, family="observation", carrier="deterministic")
            belief: Field = Field(4, 4, family="state", tags=("belief",))
            action: Field = Field(1, 8, family="action", loss_weight=2.0)

        bound, program = compile_program(Robot(), T=8, d_model=256)
    """
    from canvas_engineering.program import CanvasProgram, RegionProgram

    # 1. Compile schema
    bound = compile_schema(
        root, T=T, spatial_shape=spatial_shape, d_model=d_model,
        connectivity=connectivity,
        layout_strategy=layout_strategy,
        t_current=t_current,
        H=H, W=W,
    )

    # 2. Walk the type tree to extract Field → RegionProgram mapping
    tree = _walk(root)
    _insert_coarse_fields(tree)
    flat = _flatten_fields(tree)
    field_map = {path: f for path, f, _ in flat}

    # 3. Build RegionProgram for each region in the compiled schema
    region_programs = {}
    for path in bound.field_names:
        f = field_map.get(path)
        if f is not None:
            region_programs[path] = RegionProgram(
                family=f.family if f.family else "state",
                tags=f.tags,
                carrier=f.carrier if f.carrier else "deterministic",
            )
        else:
            region_programs[path] = RegionProgram()

    # 4. Apply family-derived operators to connections
    family_map = {}
    for path, f, _ in flat:
        if f.family:
            family_map[path] = f.family
    schema = bound.schema
    if schema.topology and family_map:
        wired_conns = _apply_operators(schema.topology.connections, family_map)
        schema = CanvasSchema(
            layout=schema.layout,
            topology=CanvasTopology(connections=wired_conns),
            version=schema.version,
            metadata=schema.metadata,
        )

    # 5. Build CanvasProgram
    program = CanvasProgram(
        schema=schema,
        regions=region_programs,
    )

    return bound, program

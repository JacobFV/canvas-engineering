"""Canvas Types: compositional type system for latent space.

Declare structured types whose fields are latent regions. Types compose
(nesting), specialize (inheritance), and replicate (lists). A compiler
flattens any type hierarchy into a CanvasSchema — a concrete CanvasLayout +
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

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from canvas_engineering.canvas import CanvasLayout, RegionSpec, SpatiotemporalCanvas
from canvas_engineering.connectivity import CanvasTopology, Connection
from canvas_engineering.schema import CanvasSchema


# ── Field Declaration ────────────────────────────────────────────────

@dataclass(frozen=True)
class Field:
    """A latent region declaration.

    Declares a region of h x w positions on the canvas grid, each carrying
    the canvas's d_model dimensionality. Default (1, 1) = scalar (1 position).

    Args:
        h: Height on the canvas grid. Default 1.
        w: Width on the canvas grid. Default 1.
        period: Canvas frames per real-world update (1 = every frame).
        is_output: Whether this field participates in diffusion loss.
        loss_weight: Relative loss weight for positions in this field.
        attn: Default attention function type for outgoing connections.
        semantic_type: Human-readable modality description.
        temporal_extent: Number of timesteps this field spans. None = full T.
    """
    h: int = 1
    w: int = 1
    period: int = 1
    is_output: bool = True
    loss_weight: float = 1.0
    attn: str = "cross_attention"
    semantic_type: Optional[str] = None
    temporal_extent: Optional[int] = None

    @property
    def num_positions(self) -> int:
        """Spatial positions per timestep (h * w)."""
        return self.h * self.w


# ── Layout Strategy ──────────────────────────────────────────────────

class LayoutStrategy(Enum):
    """How to arrange fields on the (H, W) grid."""
    PACKED = "packed"
    INTERLEAVED = "interleaved"


# ── Connectivity Policy ──────────────────────────────────────────────

@dataclass
class ConnectivityPolicy:
    """Default connectivity rules for compiled schemas.

    Every nested type and array element automatically gets a coarse-grained field
    field at its own path. Parent fields connect bidirectionally to the
    coarse-grained field; the coarse-grained field connects bidirectionally to child fields.

    Coarse-grained field size is configured on the types themselves:
      - Set ``__coarse__ = Field(h, w)`` on a class to define its
        default coarse-grained field size when it appears as a child anywhere.
      - Set ``metadata={"coarse": Field(h, w)}`` on a dataclass field
        to override the coarse-grained field for that specific parent→child edge.
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


# ── Tree Walking (Internal) ──────────────────────────────────────────

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
    """Determine coarse-grained field Field for a child.

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


# ── Gateway Insertion (Internal) ──────────────────────────────────────

def _median_period_from_tree(node: _TypeNode) -> int:
    """Get median period from all Fields in a subtree."""
    periods = []
    for fe in node.fields:
        periods.append(fe.field.period)
    for child in node.children:
        periods.extend(
            fe.field.period for fe in _flatten_fields_raw(child)
        )
    for elems in node.arrays.values():
        for elem in elems:
            periods.extend(
                fe.field.period for fe in _flatten_fields_raw(elem)
            )
    if not periods:
        return 1
    periods.sort()
    return periods[len(periods) // 2]


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

    Before: Parent → [Child1(fields...), Child2(fields...)]
    After:  Parent → [GW1(coarse-grained field) → Child1(fields...), GW2(coarse-grained field) → Child2(fields...)]

    Returns the modified node (in-place).
    """
    # Process children
    new_children = []
    for child in node.children:
        _insert_coarse_fields(child)  # recurse first

        local = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
        cg_template = child.coarse_field or Field(1, 1)
        mp = _median_period_from_tree(child)
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
            mp = _median_period_from_tree(elem)
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


# ── Auto-sizing (Internal) ──────────────────────────────────────────

def _auto_canvas_size(
    fields: List[Tuple[str, int, int]],
    pad: float = 1.15,
) -> Tuple[int, int]:
    """Compute minimum (H, W) that can pack all fields.

    Uses the same strip-packing algorithm as _pack_strip. Computes the
    smallest roughly-square grid that fits everything, with a padding factor.

    Args:
        fields: [(path, h, w), ...] from flattened field list.
        pad: Padding factor (1.15 = 15% slack).

    Returns:
        (H, W) tuple.
    """
    if not fields:
        return (4, 4)

    import math

    max_w = max(w for _, _, w in fields)
    max_h = max(h for _, h, _ in fields)
    total_area = sum(h * w for _, h, w in fields)

    side = int(math.ceil(math.sqrt(total_area * pad)))
    W = max(side, max_w)

    # Simulate strip packing to find needed H
    row_h = 0
    row_w = 0
    row_max_h = 0
    for _, h, w in fields:
        if row_w + w > W:
            row_h += row_max_h
            row_w = 0
            row_max_h = 0
        row_w += w
        row_max_h = max(row_max_h, h)
    needed_H = row_h + row_max_h
    H = max(int(math.ceil(needed_H * pad)), max_h)
    return (H, W)


# ── Packing (Internal) ───────────────────────────────────────────────

def _pack_strip(
    fields: List[Tuple[str, int, int]],
    H: int, W: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Row-based strip packing: place fields left-to-right, top-to-bottom.

    Returns dict of path -> (h0, h1, w0, w1).
    """
    placements = {}
    row_h = 0
    row_w = 0
    row_max_h = 0

    for path, h, w in fields:
        if w > W:
            raise ValueError(
                "Field '{}' width {} exceeds canvas W={}".format(path, w, W)
            )
        if row_w + w > W:
            # Start new row
            row_h += row_max_h
            row_w = 0
            row_max_h = 0
        if row_h + h > H:
            raise ValueError(
                "Cannot pack '{}' ({}x{}): grid H={} exceeded at row_h={}. "
                "Need at least H={}".format(path, h, w, H, row_h, row_h + h)
            )
        placements[path] = (row_h, row_h + h, row_w, row_w + w)
        row_w += w
        row_max_h = max(row_max_h, h)

    return placements


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

    # Flatten groups in order: all "thought" together, then all "goal", etc.
    ordered = []
    for group in groups.values():
        ordered.extend(group)

    return _pack_strip(ordered, H, W)


# ── Connectivity Generation (Internal) ───────────────────────────────

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

    With coarse-grained field insertion, this is always called between a parent node
    and a coarse-grained field node (or coarse-grained field node and its child). The coarse-grained field
    provides the bottleneck — no policy dispatch needed.
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
            ))
        elif policy.temporal == "causal":
            if c.src == c.dst:
                # Self-connection: same-frame + prev-frame
                result.append(Connection(
                    src=c.src, dst=c.dst, weight=c.weight,
                    t_src=0, t_dst=0, fn=c.fn,
                ))
                result.append(Connection(
                    src=c.src, dst=c.dst, weight=c.weight,
                    t_src=0, t_dst=-1, fn=c.fn,
                ))
            else:
                # Cross-connection: prev-frame only
                result.append(Connection(
                    src=c.src, dst=c.dst, weight=c.weight,
                    t_src=0, t_dst=-1, fn=c.fn,
                ))
    return result


def _deduplicate(connections: List[Connection]) -> List[Connection]:
    """Remove duplicate connections."""
    seen = set()
    result = []
    for c in connections:
        key = (c.src, c.dst, c.weight, c.t_src, c.t_dst, c.fn)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


# ── Bound Schema ─────────────────────────────────────────────────────

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
        """Total positions in this region (T_extent * h * w)."""
        t0, t1, h0, h1, w0, w1 = self.spec.bounds
        return (t1 - t0) * (h1 - h0) * (w1 - w0)

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
        t0, t1, h0, h1, w0, w1 = self.spec.bounds
        return "BoundField('{}', t={}-{}, {}x{})".format(
            self.region_name, t0, t1, h1 - h0, w1 - w0)


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
            embed_fn: Callable taking list[str] → list[list[float]].
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
            t0, t1, h0, h1, w0, w1 = bf.spec.bounds
            n = bf.num_positions
            lines.append("  {} -> (t={}-{}, h={}-{}, w={}-{}) {}pos".format(
                name, t0, t1, h0, h1, w0, w1, n))
        if self.topology:
            lines.append("  {} connections".format(
                len(self.topology.connections)))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "BoundSchema(fields={}, positions={})".format(
            len(self._fields), self.layout.num_positions)


# ── Compilation ──────────────────────────────────────────────────────

def compile_schema(
    root: Any,
    T: int = 1,
    H: Optional[int] = None,
    W: Optional[int] = None,
    d_model: int = 64,
    connectivity: Optional[ConnectivityPolicy] = None,
    layout_strategy: Union[str, LayoutStrategy] = LayoutStrategy.PACKED,
    t_current: int = 0,
) -> BoundSchema:
    """Compile an object hierarchy into a BoundSchema.

    Walks the object tree, finds Field attributes, packs them onto a (T, H, W)
    grid, generates connectivity based on the type hierarchy, and returns a
    BoundSchema wrapping the compiled CanvasSchema.

    Works with dataclasses, Pydantic models, or plain objects.

    When H and/or W are None, they are auto-computed from the declared field
    dimensions — like a C compiler sizing a struct from its members.

    Every nested type and array element automatically gets a coarse-grained field
    field at its own path. Coarse-grained field size is configured on the types:
      - ``__coarse__ = Field(h, w)`` on the child class
      - ``metadata={"coarse": Field(h, w)}`` on the parent's field
      - Falls back to Field(1, 1)

    Args:
        root: Object with Field attributes. Lists create array regions.
        T: Temporal extent of the canvas. Default 1.
        H: Height of the canvas grid. None = auto-sized.
        W: Width of the canvas grid. None = auto-sized.
        d_model: Latent dimensionality per position. Default 64.
        connectivity: Connectivity policy. Default: dense intra,
            isolated arrays, dense temporal.
        layout_strategy: "packed" (default) or "interleaved".
        t_current: Timestep boundary for output mask.

    Returns:
        BoundSchema with compiled layout, topology, and bound field access.

    Example:
        from dataclasses import dataclass

        @dataclass
        class Robot:
            camera: Field = Field(12, 12)
            joints: Field = Field(1, 8)
            action: Field = Field(1, 8, loss_weight=2.0)

        # Auto-sized canvas:
        bound = compile_schema(Robot(), T=8, d_model=256)

        # Explicit canvas:
        bound = compile_schema(Robot(), T=8, H=16, W=16, d_model=256)
    """
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

    # 2b. Auto-size canvas if H or W not specified
    if H is None or W is None:
        pack_dims = [(path, f.h, f.w) for path, f, _ in flat]
        auto_H, auto_W = _auto_canvas_size(pack_dims)
        if H is None:
            H = auto_H
        if W is None:
            W = auto_W

    # 3. Pack onto grid
    if layout_strategy == LayoutStrategy.INTERLEAVED:
        pack_input = [(path, f.h, f.w, local) for path, f, local in flat]
        hw_bounds = _pack_interleaved(pack_input, H, W)
    else:
        pack_input = [(path, f.h, f.w) for path, f, _ in flat]
        hw_bounds = _pack_strip(pack_input, H, W)

    # 4. Build RegionSpecs with full bounds (t0, t1, h0, h1, w0, w1)
    from canvas_engineering.semantic import auto_semantic_type

    field_map = {path: f for path, f, _ in flat}
    regions = {}  # type: Dict[str, Union[Tuple, RegionSpec]]
    for path, (h0, h1, w0, w1) in hw_bounds.items():
        f = field_map[path]
        t_extent = f.temporal_extent if f.temporal_extent is not None else T
        t_extent = min(t_extent, T)
        # Auto-generate semantic_type from path if not explicitly set
        sem_type = f.semantic_type if f.semantic_type else auto_semantic_type(path)
        regions[path] = RegionSpec(
            bounds=(0, t_extent, h0, h1, w0, w1),
            period=f.period,
            is_output=f.is_output,
            loss_weight=f.loss_weight,
            default_attn=f.attn,
            semantic_type=sem_type,
        )

    # 5. Generate connectivity
    connections = _generate_connections(tree, connectivity)
    connections = _apply_temporal(connections, connectivity)
    connections = _deduplicate(connections)

    # 6. Assemble
    layout = CanvasLayout(
        T=T, H=H, W=W, d_model=d_model,
        regions=regions,
        t_current=t_current,
    )
    topology = CanvasTopology(connections=connections) if connections else None
    schema = CanvasSchema(layout=layout, topology=topology)

    return BoundSchema(schema, regions)

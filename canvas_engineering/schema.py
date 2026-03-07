"""Canvas Schema: typed, serializable canvas declarations.

A CanvasSchema bundles a CanvasLayout + optional CanvasTopology into a single
object that can be serialized to JSON, compared for compatibility, and used
to estimate transfer distance between modalities.

    schema = CanvasSchema(layout=layout, topology=topology)
    schema.to_json("my_schema.json")
    loaded = CanvasSchema.from_json("my_schema.json")

    pairs = schema.compatible_regions(other_schema, threshold=0.3)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from canvas_engineering.canvas import CanvasLayout, RegionSpec, transfer_distance
from canvas_engineering.connectivity import Connection, CanvasTopology


@dataclass
class CanvasSchema:
    """A complete, serializable declaration of canvas structure.

    Bundles layout (geometry + region specs) with topology (connectivity).
    Designed to be the portable "type signature" for a canvas-based model.
    """
    layout: CanvasLayout
    topology: Optional[CanvasTopology] = None
    version: str = "0.2.0"
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        regions = {}
        for name, r in self.layout.regions.items():
            if isinstance(r, RegionSpec):
                rd = {"bounds": list(r.bounds)}
                if r.period != 1:
                    rd["period"] = r.period
                if not r.is_output:
                    rd["is_output"] = False
                if r.loss_weight != 1.0:
                    rd["loss_weight"] = r.loss_weight
                if r.semantic_type is not None:
                    rd["semantic_type"] = r.semantic_type
                if r.semantic_embedding is not None:
                    rd["semantic_embedding"] = list(r.semantic_embedding)
                if r.embedding_model != "openai/text-embedding-3-small":
                    rd["embedding_model"] = r.embedding_model
                regions[name] = rd
            else:
                regions[name] = {"bounds": list(r)}

        d = {
            "schema_version": self.version,
            "layout": {
                "T": self.layout.T,
                "H": self.layout.H,
                "W": self.layout.W,
                "d_model": self.layout.d_model,
                "t_current": self.layout.t_current,
            },
            "regions": regions,
        }

        if self.topology is not None:
            conns = []
            for c in self.topology.connections:
                cd = {"src": c.src, "dst": c.dst}
                if c.weight != 1.0:
                    cd["weight"] = c.weight
                if c.t_src is not None:
                    cd["t_src"] = c.t_src
                if c.t_dst is not None:
                    cd["t_dst"] = c.t_dst
                conns.append(cd)
            d["topology"] = conns

        if self.metadata:
            d["metadata"] = self.metadata

        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CanvasSchema":
        """Deserialize from a JSON-compatible dict."""
        layout_d = d["layout"]
        regions_d = d["regions"]

        regions: Dict[str, Union[Tuple, RegionSpec]] = {}
        for name, rd in regions_d.items():
            bounds = tuple(rd["bounds"])
            has_spec_fields = any(
                k in rd for k in ("period", "is_output", "loss_weight",
                                  "semantic_type", "semantic_embedding",
                                  "embedding_model")
            )
            if has_spec_fields:
                kwargs = {"bounds": bounds}
                if "period" in rd:
                    kwargs["period"] = rd["period"]
                if "is_output" in rd:
                    kwargs["is_output"] = rd["is_output"]
                if "loss_weight" in rd:
                    kwargs["loss_weight"] = rd["loss_weight"]
                if "semantic_type" in rd:
                    kwargs["semantic_type"] = rd["semantic_type"]
                if "semantic_embedding" in rd:
                    kwargs["semantic_embedding"] = tuple(rd["semantic_embedding"])
                if "embedding_model" in rd:
                    kwargs["embedding_model"] = rd["embedding_model"]
                regions[name] = RegionSpec(**kwargs)
            else:
                regions[name] = bounds

        layout = CanvasLayout(
            T=layout_d["T"],
            H=layout_d["H"],
            W=layout_d["W"],
            d_model=layout_d["d_model"],
            regions=regions,
            t_current=layout_d.get("t_current", 0),
        )

        topology = None
        if "topology" in d:
            conns = []
            for cd in d["topology"]:
                conns.append(Connection(
                    src=cd["src"],
                    dst=cd["dst"],
                    weight=cd.get("weight", 1.0),
                    t_src=cd.get("t_src", None),
                    t_dst=cd.get("t_dst", None),
                ))
            topology = CanvasTopology(connections=conns)

        return cls(
            layout=layout,
            topology=topology,
            version=d.get("schema_version", "0.2.0"),
            metadata=d.get("metadata", {}),
        )

    def to_json(self, path: Union[str, Path]) -> None:
        """Write schema to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "CanvasSchema":
        """Load schema from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def compatible_regions(
        self,
        other: "CanvasSchema",
        threshold: float = 0.3,
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of regions across two schemas with low transfer distance.

        Returns (self_region, other_region, distance) sorted by distance,
        filtered to distance <= threshold. Only compares regions that both
        have semantic_embedding set and share the same embedding_model.
        """
        pairs = []
        for name_a, r_a in self.layout.regions.items():
            spec_a = self.layout.region_spec(name_a)
            if spec_a.semantic_embedding is None:
                continue
            for name_b, r_b in other.layout.regions.items():
                spec_b = other.layout.region_spec(name_b)
                if spec_b.semantic_embedding is None:
                    continue
                if spec_a.embedding_model != spec_b.embedding_model:
                    continue
                dist = transfer_distance(spec_a, spec_b)
                if dist <= threshold:
                    pairs.append((name_a, name_b, dist))
        pairs.sort(key=lambda x: x[2])
        return pairs

"""canvas-engine: A type system for latent dynamics.

Declare structured multimodal latent spaces for video diffusion transformers.
Graft looped attention for weight-sharing regularization.

Quick start:
    from canvas_engineering import CanvasLayout, RegionSpec, CanvasSchema
    from canvas_engineering import graft_looped_blocks, CurriculumScheduler
    from canvas_engineering import Field, compile_schema
"""

from canvas_engineering.canvas import (
    ATTENTION_TYPES, CanvasLayout, RegionSpec, SpatiotemporalCanvas,
    transfer_distance,
)
from canvas_engineering.looped_block import LoopedBlockWrapper
from canvas_engineering.graft import graft_looped_blocks, freeze_full, freeze_half
from canvas_engineering.curriculum import CurriculumScheduler
from canvas_engineering.action_heads import ActionHead
from canvas_engineering.sharpening import SharpeningSchedule
from canvas_engineering.connectivity import Connection, CanvasTopology
from canvas_engineering.schema import CanvasSchema
from canvas_engineering.types import (
    Field, LayoutStrategy, ConnectivityPolicy,
    BoundField, BoundSchema, compile_schema,
)
from canvas_engineering.semantic import (
    SemanticConditioner,
    auto_semantic_type,
    compute_semantic_embeddings,
)

__version__ = "0.1.1"
# TODO: Bump to 0.2.0 before deploying general-unified-world-model 0.0.3
__all__ = [
    "ATTENTION_TYPES",
    "CanvasLayout",
    "RegionSpec",
    "SpatiotemporalCanvas",
    "transfer_distance",
    "CanvasSchema",
    "LoopedBlockWrapper",
    "graft_looped_blocks",
    "freeze_full",
    "freeze_half",
    "CurriculumScheduler",
    "ActionHead",
    "SharpeningSchedule",
    "Connection",
    "CanvasTopology",
    "Field",
    "LayoutStrategy",
    "ConnectivityPolicy",
    "BoundField",
    "BoundSchema",
    "compile_schema",
    "SemanticConditioner",
    "auto_semantic_type",
    "compute_semantic_embeddings",
]

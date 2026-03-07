"""canvas-engine: A type system for latent dynamics.

Declare structured multimodal latent spaces for video diffusion transformers.
Graft looped attention for weight-sharing regularization.

Quick start:
    from canvas_engine import CanvasLayout, RegionSpec, CanvasSchema
    from canvas_engine import graft_looped_blocks, CurriculumScheduler
"""

from canvas_engine.canvas import (
    CanvasLayout, RegionSpec, SpatiotemporalCanvas, transfer_distance,
)
from canvas_engine.looped_block import LoopedBlockWrapper
from canvas_engine.graft import graft_looped_blocks, freeze_full, freeze_half
from canvas_engine.curriculum import CurriculumScheduler
from canvas_engine.action_heads import ActionHead
from canvas_engine.sharpening import SharpeningSchedule
from canvas_engine.connectivity import Connection, CanvasTopology
from canvas_engine.schema import CanvasSchema

__version__ = "0.2.0"
__all__ = [
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
]

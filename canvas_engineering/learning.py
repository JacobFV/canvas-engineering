"""Per-family learning defaults.

Each of the 5 region families has a default training recipe: what kind
of learning to apply, which losses to use, and what compile mode to
target at deploy.

Users override these defaults by setting LearningSpec on their
RegionProgram explicitly. The defaults serve as sane starting points
that encode the structural prior for each family.

Usage:
    from canvas_engineering.learning import default_learning

    ls = default_learning("observation")
    # LearningSpec(mode="ssl_prediction", losses=("next_step", "masked_prediction"), compile_mode="freeze")
"""

from __future__ import annotations

from typing import Dict

from canvas_engineering.program import LearningSpec


FAMILY_DEFAULTS: Dict[str, LearningSpec] = {
    "observation": LearningSpec(
        mode="ssl_prediction",
        losses=("next_step", "masked_prediction"),
        compile_mode="freeze",
    ),
    "state": LearningSpec(
        mode="posterior_match",
        losses=("predictive_consistency", "calibration"),
        compile_mode="runtime",
    ),
    "memory": LearningSpec(
        mode="retrieval",
        losses=("retrieval_accuracy", "write_utility"),
        compile_mode="export",
    ),
    "residual": LearningSpec(
        mode="calibration",
        losses=("calibration", "sparsity"),
        compile_mode="freeze",
    ),
    "action": LearningSpec(
        mode="supervised",
        losses=("task",),
        compile_mode="freeze",
    ),
}


def default_learning(family: str) -> LearningSpec:
    """Get the default LearningSpec for a region family.

    Returns the family's default if known, otherwise a generic
    LearningSpec with all defaults (supervised, no specific losses,
    runtime compile mode).

    Args:
        family: Region family name.

    Returns:
        LearningSpec with family-appropriate defaults.
    """
    return FAMILY_DEFAULTS.get(family, LearningSpec())

"""Curriculum scheduler for gradually increasing loop count during training."""

from typing import List


class CurriculumScheduler:
    """Ramp loop count from 1 to max_loops over training.

    Example:
        scheduler = CurriculumScheduler(max_loops=3, total_steps=5000)
        for step in range(5000):
            n = scheduler.step(looped_blocks, step)
            # n = 1 for steps 0-1666, 2 for 1667-3333, 3 for 3334-5000
    """

    def __init__(self, max_loops: int = 3, total_steps: int = 5000, warmup_fraction: float = 0.0):
        self.max_loops = max_loops
        self.total_steps = total_steps
        self.warmup_fraction = warmup_fraction

    def get_loops(self, step: int) -> int:
        """Compute number of loops for a given training step."""
        if self.max_loops == 1:
            return 1
        # Divide training into max_loops equal phases
        phase_len = self.total_steps / self.max_loops
        current_phase = min(int(step / phase_len), self.max_loops - 1)
        return current_phase + 1

    def step(self, looped_blocks: List, step: int) -> int:
        """Update all looped blocks and return the current loop count."""
        n = self.get_loops(step)
        for block in looped_blocks:
            if hasattr(block, "set_loops"):
                block.set_loops(n)
        return n

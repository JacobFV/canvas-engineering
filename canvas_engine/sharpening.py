"""Progressive Sharpening: loop-indexed inverse temperature schedule.

Early loops use soft attention (beta ~1) for broad gradient flow.
Later loops use sharp attention (beta >> 1) for precise, near-discrete patterns.
This bridges the soft-to-sharp discontinuity that prevents SGD from learning
discrete-like computations.
"""

import math


class SharpeningSchedule:
    """Compute inverse temperature beta as a function of loop index.

    Example:
        schedule = SharpeningSchedule(max_loops=4, beta_min=1.0, beta_max=4.0)
        beta = schedule(loop_idx=2)  # -> 3.0 (linear interpolation)
    """

    def __init__(
        self,
        max_loops: int = 4,
        beta_min: float = 1.0,
        beta_max: float = 4.0,
        schedule: str = "linear",
    ):
        self.max_loops = max_loops
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.schedule = schedule

    def __call__(self, loop_idx: int) -> float:
        if self.max_loops <= 1:
            return self.beta_min
        t = loop_idx / (self.max_loops - 1)  # 0 -> 1
        if self.schedule == "linear":
            return self.beta_min + t * (self.beta_max - self.beta_min)
        elif self.schedule == "exponential":
            return self.beta_min * (self.beta_max / self.beta_min) ** t
        elif self.schedule == "cosine":
            return self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (1 - math.cos(math.pi * t))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def __repr__(self) -> str:
        return f"SharpeningSchedule({self.schedule}, beta=[{self.beta_min}, {self.beta_max}], loops={self.max_loops})"

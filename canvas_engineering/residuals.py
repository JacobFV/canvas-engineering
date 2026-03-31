"""Residual signal tracking for scheduling and diagnostics.

ResidualAccumulator maintains running scalar EMA summaries of error
signals emitted by residual-carrier regions. These summaries drive
event-triggered scheduling (phase 4) and provide diagnostics for
monitoring prediction error, uncertainty, and novelty across regions.

Usage:
    from canvas_engineering.residuals import ResidualSpec, ResidualAccumulator

    spec = ResidualSpec(kinds=("prediction", "novelty"), decay=0.95)
    accumulator = ResidualAccumulator(["vision.err", "belief.err"], spec)

    # During forward pass:
    accumulator.update("vision.err", error_tensor)

    # Read summaries:
    summaries = accumulator.summaries()
    # {"vision.err": {"prediction": 0.12, "novelty": 0.03}, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ResidualSpec:
    """Declares what error signals a region emits and how to summarize them.

    Args:
        kinds: Named error signal types to track. Each gets its own
            running scalar summary.
        reduce: How to reduce an error tensor to scalars. "max_mean" =
            [max, mean] sliced to n_kinds. "mean" = mean only. "max" =
            max only. "l2" = L2 norm.
        decay: EMA decay factor. summary = decay * old + (1 - decay) * new.
            Higher decay = smoother, slower response. 0.95 is a good default.
    """
    kinds: Tuple[str, ...] = ("prediction",)
    reduce: str = "max_mean"
    decay: float = 0.95


class ResidualAccumulator(nn.Module):
    """Tracks running scalar summaries of residual signals per region.

    Not a learned module — uses nn.Module only for device management
    and state_dict integration. No trainable parameters.

    The accumulator stores a (n_regions, n_kinds) buffer of running EMA
    scalars. Each call to update() reduces an error tensor to scalars
    and blends them into the running summary.

    Usage:
        acc = ResidualAccumulator(["err_a", "err_b"], ResidualSpec())
        acc.update("err_a", error_tensor)  # (B, N, D) or (B, N) or scalar
        print(acc.summaries())  # {"err_a": {"prediction": 0.12}, ...}
    """

    def __init__(
        self,
        region_names: List[str],
        spec: Optional[ResidualSpec] = None,
    ):
        super().__init__()
        self.spec = spec or ResidualSpec()
        self._region_names = sorted(region_names)
        self._name_to_idx = {n: i for i, n in enumerate(self._region_names)}

        n = len(self._region_names)
        n_kinds = len(self.spec.kinds)
        self.register_buffer("_summaries", torch.zeros(n, n_kinds))
        self.register_buffer("_step_counts", torch.zeros(n, dtype=torch.long))

    def update(self, region: str, error: torch.Tensor) -> None:
        """Update running summary for a region.

        Args:
            region: Region name (must be in the region_names list).
            error: Error tensor of any shape. Reduced to scalar(s) via
                the spec's reduce mode, then blended into the EMA.
        """
        idx = self._name_to_idx[region]
        reduced = self._reduce(error)
        decay = self.spec.decay
        self._summaries[idx] = decay * self._summaries[idx] + (1 - decay) * reduced
        self._step_counts[idx] += 1

    def _reduce(self, error: torch.Tensor) -> torch.Tensor:
        """Reduce an error tensor to a 1D tensor of length n_kinds."""
        n_kinds = len(self.spec.kinds)
        flat = error.detach().float()

        if self.spec.reduce == "mean":
            val = flat.mean()
            return val.unsqueeze(0).expand(n_kinds)
        elif self.spec.reduce == "max":
            val = flat.max()
            return val.unsqueeze(0).expand(n_kinds)
        elif self.spec.reduce == "l2":
            val = flat.norm()
            return val.unsqueeze(0).expand(n_kinds)
        else:  # max_mean (default)
            max_val = flat.max()
            mean_val = flat.mean()
            if n_kinds == 1:
                return mean_val.unsqueeze(0)
            elif n_kinds == 2:
                return torch.stack([max_val, mean_val])
            else:
                # Repeat max, mean pattern
                vals = torch.stack([max_val, mean_val])
                return vals.repeat((n_kinds + 1) // 2)[:n_kinds]

    def summaries(self) -> Dict[str, Dict[str, float]]:
        """Get current summaries as {region: {kind: value}}.

        Returns a plain dict of floats (detached from compute graph).
        """
        result = {}
        for name in self._region_names:
            idx = self._name_to_idx[name]
            result[name] = {
                kind: self._summaries[idx, k].item()
                for k, kind in enumerate(self.spec.kinds)
            }
        return result

    def summary_tensor(self) -> torch.Tensor:
        """Get raw (n_regions, n_kinds) summary tensor."""
        return self._summaries

    def reset(self) -> None:
        """Reset all summaries to zero."""
        self._summaries.zero_()
        self._step_counts.zero_()

    @property
    def region_names(self) -> List[str]:
        return list(self._region_names)

    def __repr__(self) -> str:
        return "ResidualAccumulator(regions={}, kinds={}, decay={})".format(
            len(self._region_names), self.spec.kinds, self.spec.decay)

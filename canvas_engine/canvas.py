"""Spatiotemporal Canvas: the unified representation space.

The canvas is a 3D grid (T, H, W) of d_model-dimensional vectors.
Each modality occupies designated regions. The diffusion process operates
on "output" regions; "input" regions serve as conditioning context.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class RegionSpec:
    """Declarative specification for a canvas region.

    Args:
        bounds: (t0, t1, h0, h1, w0, w1) spatial-temporal extent.
        period: Canvas frames per real-world update (1 = every frame).
            A region with period=4 spanning t=0..3 maps to real frames 0,4,8,12.
        is_output: Whether this region participates in diffusion loss.
        loss_weight: Relative loss weight for positions in this region.
        semantic_type: Human-readable modality description, e.g.
            "RGB video 224x224 30fps from front-facing monocular camera".
            This is the source of truth — the embedding is derived from it.
        semantic_embedding: Frozen vector from embedding_model applied to
            semantic_type. Used to compute transfer distance between modalities.
            Must be re-derived if semantic_type or embedding_model changes.
        embedding_model: Identifier of the model that produced semantic_embedding.
            Should stay constant within a project/ecosystem. Declared explicitly
            so different communities can use different models and so the embedding
            can always be re-derived.
    """
    bounds: Tuple[int, int, int, int, int, int]
    period: int = 1
    is_output: bool = True
    loss_weight: float = 1.0
    semantic_type: Optional[str] = None
    semantic_embedding: Optional[Tuple[float, ...]] = None
    embedding_model: str = "openai/text-embedding-3-small"


def transfer_distance(a: RegionSpec, b: RegionSpec) -> float:
    """Cosine distance between two regions' semantic type embeddings.

    Returns a value in [0, 2]: 0 = identical modality, 1 = orthogonal, 2 = opposite.
    Lower distance → cheaper to bridge (fewer adapter layers, less data).

    Both specs must have semantic_embedding set and use the same embedding_model.
    """
    if a.semantic_embedding is None or b.semantic_embedding is None:
        raise ValueError("Both RegionSpecs must have semantic_embedding set")
    if a.embedding_model != b.embedding_model:
        raise ValueError(
            f"Embedding model mismatch: {a.embedding_model!r} vs {b.embedding_model!r}. "
            "Transfer distance is only meaningful within the same embedding space."
        )
    va = torch.tensor(a.semantic_embedding, dtype=torch.float32)
    vb = torch.tensor(b.semantic_embedding, dtype=torch.float32)
    if va.shape != vb.shape:
        raise ValueError(
            f"Embedding dimension mismatch: {va.shape[0]} vs {vb.shape[0]}"
        )
    cos_sim = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
    return 1.0 - cos_sim


def _get_bounds(region: Union[Tuple, RegionSpec]) -> Tuple[int, int, int, int, int, int]:
    """Extract bounds from a raw tuple or RegionSpec."""
    if isinstance(region, RegionSpec):
        return region.bounds
    return region


@dataclass
class CanvasLayout:
    """Declarative canvas geometry and modality region assignments.

    Example:
        layout = CanvasLayout(
            T=5, H=8, W=8, d_model=256,
            regions={
                "visual": (0, 5, 0, 6, 0, 6),   # 5 frames of 6x6 patches
                "action": (0, 5, 6, 7, 0, 1),    # per-frame actions
                "reward": (2, 3, 7, 8, 0, 1),    # single reward slot
            },
            t_current=2,  # t >= 2 is "future" (diffusion output)
        )
    """
    T: int
    H: int
    W: int
    d_model: int
    regions: Dict[str, Union[Tuple[int, int, int, int, int, int], RegionSpec]] = field(default_factory=dict)
    t_current: int = 0

    @property
    def num_positions(self) -> int:
        return self.T * self.H * self.W

    def region_spec(self, name: str) -> RegionSpec:
        """Return the RegionSpec for a named region, wrapping raw tuples with defaults."""
        r = self.regions[name]
        if isinstance(r, RegionSpec):
            return r
        return RegionSpec(bounds=r)

    def region_size(self, name: str) -> Tuple[int, int, int]:
        t0, t1, h0, h1, w0, w1 = _get_bounds(self.regions[name])
        return (t1 - t0, h1 - h0, w1 - w0)

    def region_numel(self, name: str) -> int:
        t, h, w = self.region_size(name)
        return t * h * w

    def region_indices(self, name: str) -> List[int]:
        """Flat indices for a named region."""
        t0, t1, h0, h1, w0, w1 = _get_bounds(self.regions[name])
        indices = []
        for t in range(t0, t1):
            for h in range(h0, h1):
                for w in range(w0, w1):
                    indices.append(t * (self.H * self.W) + h * self.W + w)
        return indices

    def region_indices_at_t(self, name: str, t_abs: int) -> List[int]:
        """Flat indices for a named region at a specific absolute timestep.

        Returns empty list if t_abs is outside the region's temporal extent.
        """
        t0, t1, h0, h1, w0, w1 = _get_bounds(self.regions[name])
        if t_abs < t0 or t_abs >= t1:
            return []
        indices = []
        for h in range(h0, h1):
            for w in range(w0, w1):
                indices.append(t_abs * (self.H * self.W) + h * self.W + w)
        return indices

    def region_timesteps(self, name: str) -> List[int]:
        """Absolute timesteps covered by a named region."""
        bounds = _get_bounds(self.regions[name])
        t0, t1 = bounds[0], bounds[1]
        return list(range(t0, t1))

    def output_mask(self) -> List[int]:
        """Flat indices that are diffusion outputs (t >= t_current)."""
        indices = []
        for t in range(self.t_current, self.T):
            for h in range(self.H):
                for w in range(self.W):
                    indices.append(t * (self.H * self.W) + h * self.W + w)
        return indices

    def loss_weight_mask(self, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """Per-position loss weights as a (N,) tensor.

        Positions in is_output=True regions get their loss_weight;
        is_output=False or uncovered positions get 0.
        Overlapping regions accumulate weights additively.
        """
        weights = torch.zeros(self.num_positions, device=device)
        for name in self.regions:
            spec = self.region_spec(name)
            if not spec.is_output:
                continue
            indices = self.region_indices(name)
            idx = torch.tensor(indices, device=device, dtype=torch.long)
            weights[idx] += spec.loss_weight
        return weights

    def real_frame(self, name: str, canvas_t: int) -> int:
        """Map a canvas timestep index (relative to region start) to a real-world frame.

        real_frame = canvas_t * period
        """
        spec = self.region_spec(name)
        return canvas_t * spec.period

    def canvas_frame(self, name: str, real_t: int) -> Optional[int]:
        """Map a real-world frame to a canvas timestep index (relative to region start).

        Returns None if real_t is not aligned to this region's period.
        """
        spec = self.region_spec(name)
        if spec.period == 0:
            return None
        q, r = divmod(real_t, spec.period)
        if r != 0:
            return None
        t_extent = spec.bounds[1] - spec.bounds[0]
        if q < 0 or q >= t_extent:
            return None
        return q


class SinusoidalPositionalEncoding3D(nn.Module):
    """3D sinusoidal positional encoding for (T, H, W) grid."""

    def __init__(self, d_model: int, max_T: int = 32, max_H: int = 64, max_W: int = 64):
        super().__init__()
        d_t = d_model // 3
        d_h = d_model // 3
        d_w = d_model - d_t - d_h

        pe_t = self._make_1d(max_T, d_t)
        pe_h = self._make_1d(max_H, d_h)
        pe_w = self._make_1d(max_W, d_w)

        pe = torch.zeros(max_T, max_H, max_W, d_model)
        pe[:, :, :, :d_t] = pe_t[:, None, None, :]
        pe[:, :, :, d_t:d_t + d_h] = pe_h[None, :, None, :]
        pe[:, :, :, d_t + d_h:] = pe_w[None, None, :, :]
        self.register_buffer("pe", pe)

    @staticmethod
    def _make_1d(max_len: int, d: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        if d > 1:
            pe[:, 1::2] = torch.cos(pos * div[: d // 2])
        return pe

    def forward(self, T: int, H: int, W: int) -> torch.Tensor:
        return self.pe[:T, :H, :W, :]


class SpatiotemporalCanvas(nn.Module):
    """Manages the unified canvas tensor with positional + modality embeddings.

    Example:
        canvas_mod = SpatiotemporalCanvas(layout)
        canvas = canvas_mod.create_empty(batch_size=4)  # (4, T*H*W, d_model)
        canvas = canvas_mod.place(canvas, visual_embs, "visual")
        action_embs = canvas_mod.extract(canvas, "action")
    """

    def __init__(self, layout: CanvasLayout):
        super().__init__()
        self.layout = layout
        self.pos_enc = SinusoidalPositionalEncoding3D(
            layout.d_model, max_T=layout.T, max_H=layout.H, max_W=layout.W
        )
        self.empty_token = nn.Parameter(torch.randn(layout.d_model) * 0.02)
        self.modality_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(layout.d_model) * 0.02) for name in layout.regions}
        )

    def create_empty(self, batch_size: int) -> torch.Tensor:
        """(B, N, d_model) canvas filled with empty tokens + positional encoding."""
        L = self.layout
        canvas = self.empty_token.unsqueeze(0).unsqueeze(0).expand(batch_size, L.num_positions, L.d_model).clone()
        pe = self.pos_enc(L.T, L.H, L.W).reshape(1, L.num_positions, L.d_model)
        return canvas + pe

    def place(self, canvas: torch.Tensor, embeddings: torch.Tensor, region_name: str) -> torch.Tensor:
        """Write embeddings into a named region, adding modality embedding."""
        indices = self.layout.region_indices(region_name)
        n = len(indices)
        if embeddings.shape[1] > n:
            embeddings = embeddings[:, :n]
        elif embeddings.shape[1] < n:
            pad = torch.zeros(embeddings.shape[0], n - embeddings.shape[1], self.layout.d_model, device=embeddings.device)
            embeddings = torch.cat([embeddings, pad], dim=1)
        idx = torch.tensor(indices, device=canvas.device, dtype=torch.long)
        mod_emb = self.modality_embeddings[region_name] if region_name in self.modality_embeddings else 0
        canvas = canvas.clone()
        canvas[:, idx] = embeddings + mod_emb
        return canvas

    def extract(self, canvas: torch.Tensor, region_name: str) -> torch.Tensor:
        """Read embeddings from a named region."""
        indices = self.layout.region_indices(region_name)
        idx = torch.tensor(indices, device=canvas.device, dtype=torch.long)
        return canvas[:, idx]

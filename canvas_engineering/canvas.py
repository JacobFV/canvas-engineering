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
        default_attn: Default attention function type for outgoing connections
            from this region. Connections can override this per-edge. See
            ATTENTION_TYPES for the full registry of supported types.
        carrier: Dynamics carrier for this region. "deterministic" = standard
            forward latent updates (default). "diffusive" = noise/denoise.
            "filter" = predict/correct. "memory" = persistent lookup.
            "residual" = error traces.
    """
    bounds: Tuple[int, int, int, int, int, int]
    period: int = 1
    is_output: bool = True
    loss_weight: float = 1.0
    semantic_type: Optional[str] = None
    semantic_embedding: Optional[Tuple[float, ...]] = None
    embedding_model: str = "openai/text-embedding-3-small"
    default_attn: str = "cross_attention"
    carrier: str = "deterministic"


# Registry of declared attention function types. The schema declares intent;
# the executor decides how to run it. Not all backends support all types —
# a frozen CogVideoX backbone can only honor weight modulation, while a
# custom dispatcher can route each connection to a different implementation.
ATTENTION_TYPES = {
    # --- Dot-product family ---
    "cross_attention": "Standard scaled dot-product QKV attention (softmax). "
        "The default. O(N*M) where N=|src|, M=|dst|.",
    "linear_attention": "Dot-product without softmax normalization. O(N+M) via "
        "kernel trick (elu+1 or ReLU features). Good for low-dimensional or "
        "high-frequency regions where full attention is overkill.",
    "cosine_attention": "Cosine similarity attention — normalized dot-product "
        "without learned temperature. Stable gradients, no scaling by sqrt(d).",
    "sigmoid_attention": "Sigmoid instead of softmax over attention logits. "
        "Each position independently gates each key (no competition). "
        "Good for multi-label / non-exclusive attention patterns.",

    # --- Gating family ---
    "gated": "Gated cross-attention (Flamingo-style). A learned sigmoid gate "
        "on the cross-attention output controls whether to incorporate context. "
        "Good for optional conditioning (goals, instructions, memory).",

    # --- Compression family ---
    "perceiver": "Cross-attend through a learned latent bottleneck. Compresses "
        "dst into a small set of latent vectors, then src attends to those. "
        "O(N*K) where K << M. Good for very large dst regions.",
    "pooling": "Mean-pool dst into a single vector, broadcast to all src "
        "positions. Cheapest possible information transfer. O(M+N). "
        "Good for scalar or low-dimensional conditioning signals.",
    "copy": "Direct tensor transfer — no learned parameters, no attention. "
        "Requires src and dst to have compatible shapes. For broadcast "
        "regions or direct latent sharing between agents.",

    # --- State-space / recurrence family ---
    "mamba": "Selective state-space model (S6). Input-dependent gating over "
        "a compressed state. O(N) sequential, hardware-efficient. "
        "Good for long temporal sequences.",
    "rwkv": "Linear attention with learned exponential decay. O(N) via "
        "recurrent formulation. Time-mixing with position-dependent "
        "forgetting. Good for causal temporal connections.",
    "hyena": "Long convolution operator with data-dependent gating. "
        "O(N log N) via FFT. Sub-quadratic alternative to attention "
        "for very long sequences.",

    # --- Sparse / structured family ---
    "sparse_attention": "Top-k attention — only the k highest logits "
        "survive softmax. Sparse gradient flow. Good for regions that "
        "should selectively bind to specific positions.",
    "local_attention": "Windowed attention — each position only attends "
        "within a local spatial/temporal window. O(N*W) where W is "
        "window size. Good for spatially local interactions.",

    # --- Meta / experimental ---
    "none": "Connection exists in schema but is disabled. Useful for "
        "ablation studies — the edge is declared but produces no "
        "information flow.",
    "random_fixed": "Random sparse attention pattern, frozen at init. "
        "Each position attends to a random fixed subset of dst. "
        "Baseline for measuring whether learned patterns matter.",
    "mixture": "Mixture-of-experts style routing. Each src position is "
        "routed to a subset of dst positions by a learned router. "
        "Sparse but adaptive. Good for multi-modal hubs.",

    # --- Backbone-native ---
    "cogvideox": "CogVideoX-native attention. Within the canvas dispatcher "
        "this is standard cross-attention; when a CogVideoX backbone is "
        "grafted via graft_looped_blocks(), the backbone's native 3D-RoPE "
        "multi-head attention supersedes this entirely. Use as default_attn "
        "on any region trained inside a CogVideoX transformer.",
}


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


class PeriodEmbedding(nn.Module):
    """Learned embedding indexed by log-bucketed temporal period.

    Summed into position representations so the model knows each position's
    native update rate. Combined with temporal positional encoding, this
    lets the model infer staleness when reading held values from slower
    regions via temporal fill.

    Period values are mapped to buckets via log scaling:
        period=1  → bucket 0    (tick-rate)
        period=4  → bucket 2    (hourly)
        period=16 → bucket 4    (daily)
        period=576 → bucket 10  (quarterly)
        period=4608 → bucket 13 (decadal)

    Args:
        d_model: Embedding dimension (must match canvas d_model).
        n_buckets: Number of discrete period buckets.
        max_period: Maximum expected period (for log scaling).
    """

    def __init__(self, d_model: int, n_buckets: int = 16, max_period: int = 10000):
        super().__init__()
        self.n_buckets = n_buckets
        self.max_period = max_period
        self.embedding = nn.Embedding(n_buckets, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def bucket(self, period: int) -> int:
        """Map a period to a bucket index via log scaling."""
        if period <= 1:
            return 0
        log_ratio = math.log(period) / math.log(max(self.max_period, 2))
        b = int(log_ratio * (self.n_buckets - 1)) + 1
        return min(b, self.n_buckets - 1)

    def forward(self, period: int) -> torch.Tensor:
        """Get the (d_model,) embedding vector for a given period."""
        idx = self.bucket(period)
        return self.embedding.weight[idx]


class FamilyCarrierEmbedding(nn.Module):
    """Learned embeddings for region family and carrier, summed into positions.

    Each region on the canvas has a family (observation, state, memory,
    residual, action) and a carrier (deterministic, diffusive, filter,
    memory, residual).  This module produces a (d_model,) vector for
    any family+carrier pair by summing two learned embeddings.

    Unknown families/carriers map to a dedicated "unknown" slot (the last
    row of each embedding table), so the module is safe against custom
    string values.

    Args:
        d_model: Embedding dimension.
    """

    FAMILIES = ["observation", "state", "memory", "residual", "action"]
    CARRIERS = ["deterministic", "diffusive", "filter", "memory", "residual"]

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.family_embedding = nn.Embedding(len(self.FAMILIES) + 1, d_model)  # +1 for unknown
        self.carrier_embedding = nn.Embedding(len(self.CARRIERS) + 1, d_model)
        nn.init.normal_(self.family_embedding.weight, std=0.02)
        nn.init.normal_(self.carrier_embedding.weight, std=0.02)
        self._family_to_idx = {f: i for i, f in enumerate(self.FAMILIES)}
        self._carrier_to_idx = {c: i for i, c in enumerate(self.CARRIERS)}

    def family_idx(self, family: str) -> int:
        """Map a family string to an embedding index."""
        return self._family_to_idx.get(family, len(self.FAMILIES))

    def carrier_idx(self, carrier: str) -> int:
        """Map a carrier string to an embedding index."""
        return self._carrier_to_idx.get(carrier, len(self.CARRIERS))

    def forward(self, family: str, carrier: str) -> torch.Tensor:
        """Get (d_model,) embedding for a family+carrier pair."""
        f_emb = self.family_embedding.weight[self.family_idx(family)]
        c_emb = self.carrier_embedding.weight[self.carrier_idx(carrier)]
        return f_emb + c_emb

    def __repr__(self) -> str:
        return "FamilyCarrierEmbedding(d_model={}, families={}, carriers={})".format(
            self.d_model, len(self.FAMILIES), len(self.CARRIERS))


class SpatiotemporalCanvas(nn.Module):
    """Manages the unified canvas tensor with positional + modality + period embeddings.

    Each position's representation is the sum of:
    1. Content embedding (empty token or placed data)
    2. 3D sinusoidal positional encoding (T, H, W)
    3. Region identity (learned modality embedding or semantic conditioning)
    4. Period embedding (learned, indexed by log-bucketed update period)

    The period embedding tells the model each position's native update rate,
    enabling it to infer staleness when reading held values from slower
    regions via temporal fill connections.

    Supports two modes of per-region identity:
    1. **Learned modality embeddings** (default): one learned vector per region.
    2. **Semantic conditioning**: frozen embeddings from a text model, projected
       to d_model with optional learned residuals. Pass a SemanticConditioner
       to ``__init__`` to enable. Replaces learned modality embeddings.

    Example:
        canvas_mod = SpatiotemporalCanvas(layout)
        canvas = canvas_mod.create_empty(batch_size=4)  # (4, T*H*W, d_model)
        canvas = canvas_mod.place(canvas, visual_embs, "visual")
        action_embs = canvas_mod.extract(canvas, "action")

        # With semantic conditioning:
        from canvas_engineering.semantic import SemanticConditioner
        cond = SemanticConditioner(d_model=256, embed_dim=1536, region_embeddings={...})
        canvas_mod = SpatiotemporalCanvas(layout, semantic_conditioner=cond)
    """

    @staticmethod
    def _sanitize_key(name: str) -> str:
        """Sanitize region name for use as a PyTorch ParameterDict key."""
        return name.replace(".", "__").replace("[", "_").replace("]", "_")

    def __init__(self, layout: CanvasLayout, semantic_conditioner=None, program=None):
        super().__init__()
        self.layout = layout
        self.semantic_conditioner = semantic_conditioner
        self._program = program
        self.pos_enc = SinusoidalPositionalEncoding3D(
            layout.d_model, max_T=layout.T, max_H=layout.H, max_W=layout.W
        )
        self.period_embedding = PeriodEmbedding(layout.d_model)
        self.empty_token = nn.Parameter(torch.randn(layout.d_model) * 0.02)
        # Map region_name -> sanitized key for ParameterDict compatibility
        self._key_map = {name: self._sanitize_key(name) for name in layout.regions}

        if semantic_conditioner is None:
            # Use learned modality embeddings
            self.modality_embeddings = nn.ParameterDict(
                {self._key_map[name]: nn.Parameter(torch.randn(layout.d_model) * 0.02)
                 for name in layout.regions}
            )
        else:
            # Semantic conditioner replaces learned modality embeddings
            self.modality_embeddings = None

        # Optional family+carrier embedding when a CanvasProgram is provided
        if program is not None:
            self.family_carrier_embedding = FamilyCarrierEmbedding(layout.d_model)
        else:
            self.family_carrier_embedding = None

    def create_empty(self, batch_size: int) -> torch.Tensor:
        """(B, N, d_model) canvas filled with empty tokens + positional + period encoding.

        Each position gets: empty_token + 3D_PE + period_embedding.
        If a semantic conditioner is present, also adds semantic conditioning.
        If a program was provided at init, also adds family+carrier embeddings.
        """
        L = self.layout
        canvas = self.empty_token.unsqueeze(0).unsqueeze(0).expand(batch_size, L.num_positions, L.d_model).clone()
        pe = self.pos_enc(L.T, L.H, L.W).reshape(1, L.num_positions, L.d_model)
        canvas = canvas + pe

        # Sum period embedding for each region's positions
        for name in L.regions:
            spec = L.region_spec(name)
            period_emb = self.period_embedding(spec.period)
            indices = L.region_indices(name)
            if indices:
                idx = torch.tensor(indices, device=canvas.device, dtype=torch.long)
                canvas[:, idx] = canvas[:, idx] + period_emb.to(canvas.device)

        # Sum family+carrier embedding when a program is available
        if self.family_carrier_embedding is not None and self._program is not None:
            for name in L.regions:
                rp = self._program.regions.get(name)
                if rp is not None:
                    fc_emb = self.family_carrier_embedding(rp.family, rp.carrier)
                    indices = L.region_indices(name)
                    if indices:
                        idx = torch.tensor(indices, device=canvas.device, dtype=torch.long)
                        canvas[:, idx] = canvas[:, idx] + fc_emb.to(canvas.device)

        if self.semantic_conditioner is not None:
            canvas = self.semantic_conditioner.condition_canvas(canvas, L)

        return canvas

    def place(self, canvas: torch.Tensor, embeddings: torch.Tensor, region_name: str) -> torch.Tensor:
        """Write embeddings into a named region, adding modality + period + family/carrier embeddings."""
        indices = self.layout.region_indices(region_name)
        n = len(indices)
        if embeddings.shape[1] > n:
            embeddings = embeddings[:, :n]
        elif embeddings.shape[1] < n:
            pad = torch.zeros(embeddings.shape[0], n - embeddings.shape[1], self.layout.d_model, device=embeddings.device)
            embeddings = torch.cat([embeddings, pad], dim=1)
        idx = torch.tensor(indices, device=canvas.device, dtype=torch.long)

        # Add region identity: semantic conditioning or learned modality embedding
        if self.semantic_conditioner is not None:
            if region_name in self.semantic_conditioner._name_to_idx:
                region_emb = self.semantic_conditioner.get_conditioning(region_name).to(canvas.device)
            else:
                region_emb = 0
        elif self.modality_embeddings is not None:
            key = self._key_map.get(region_name, self._sanitize_key(region_name))
            region_emb = self.modality_embeddings[key] if key in self.modality_embeddings else 0
        else:
            region_emb = 0

        # Add period embedding so the model knows this region's update rate
        spec = self.layout.region_spec(region_name)
        period_emb = self.period_embedding(spec.period).to(canvas.device)

        # Add family+carrier embedding when a program is available
        fc_emb = 0
        if self.family_carrier_embedding is not None and self._program is not None:
            rp = self._program.regions.get(region_name)
            if rp is not None:
                fc_emb = self.family_carrier_embedding(rp.family, rp.carrier).to(canvas.device)

        canvas = canvas.clone()
        canvas[:, idx] = embeddings + region_emb + period_emb + fc_emb
        return canvas

    def extract(self, canvas: torch.Tensor, region_name: str) -> torch.Tensor:
        """Read embeddings from a named region."""
        indices = self.layout.region_indices(region_name)
        idx = torch.tensor(indices, device=canvas.device, dtype=torch.long)
        return canvas[:, idx]

"""Semantic conditioning for canvas dynamics.

Positions on the canvas carry frozen semantic embeddings that tell the model
*what* each region represents. This enables a single set of transformer weights
to learn different dynamics for different field types — conditioned on semantic
identity rather than hardcoded heads.

Design:
    - Frozen embeddings from a text model preserve transfer distance structure
    - Learned residuals (zero-init, like looped attention) adapt to dynamics
      that text similarity doesn't capture
    - Shared linear projector (embed_dim → d_model) enables generalization
      to unseen fields at inference time
    - Compatible with SpatiotemporalCanvas: replaces learned modality embeddings
      when a conditioner is provided

# TODO: Meta-learn semantic embeddings via prediction/infilling/reverse-diffusion
# across diverse projections of the same world model. The text embeddings are a
# good initialization, but the optimal embedding space for dynamics conditioning
# may differ from the text embedding space. A meta-learning loop over many
# (projection, dataset) pairs could learn a better mapping.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from canvas_engineering.canvas import CanvasLayout


def auto_semantic_type(path: str) -> str:
    """Generate a human-readable semantic type from a dotted field path.

    Converts structured paths into readable descriptions.

    Examples:
        "country_us.macro.output.gdp_nowcast"
        → "country us / macro / output / gdp nowcast"

        "financial.yield_curves.ten_year"
        → "financial / yield curves / ten year"

        "firm_AAPL.financials.revenue"
        → "firm AAPL / financials / revenue"
    """
    # Split on dots to get hierarchy levels
    parts = path.split(".")
    # Replace underscores with spaces in each part
    readable_parts = []
    for part in parts:
        # Handle array indices like [0]
        part = re.sub(r"\[(\d+)\]", r" \1", part)
        # Replace underscores with spaces, but preserve case of acronyms
        readable = part.replace("_", " ")
        readable_parts.append(readable)
    return " / ".join(readable_parts)


class SemanticConditioner(nn.Module):
    """Conditions canvas positions on their semantic type embeddings.

    Each region on the canvas has a semantic type (e.g., "GDP quarterly nowcast")
    described by a frozen embedding vector from a text embedding model. The
    conditioner projects these to d_model and adds optional learned residuals.

    Args:
        d_model: Canvas latent dimension.
        embed_dim: Dimension of the input semantic embeddings (e.g. 1536
            for OpenAI text-embedding-3-small).
        region_embeddings: Dict mapping region names to embedding vectors.
        freeze_embeddings: If True (default), raw semantic embeddings are
            registered as buffers (non-trainable). Set False to fine-tune them.
        learn_residuals: If True (default), add zero-initialized learned
            residuals on top of the projected embeddings. These adapt the
            conditioning to capture dynamics that text similarity doesn't.
        projector_bias: Whether the projection layer has bias. Default False
            for cleaner zero-init behavior.

    Example:
        # From precomputed embeddings
        conditioner = SemanticConditioner(
            d_model=64,
            embed_dim=1536,
            region_embeddings={
                "financial.yield_curves.ten_year": (0.12, -0.05, ...),
                "country_us.macro.output.gdp_nowcast": (0.31, 0.08, ...),
            },
        )

        # Condition a canvas
        canvas = conditioner.condition_canvas(canvas, layout)

        # Or get a single region's conditioning vector
        cond = conditioner.get_conditioning("financial.yield_curves.ten_year")
    """

    def __init__(
        self,
        d_model: int,
        embed_dim: int,
        region_embeddings: Dict[str, Union[Tuple[float, ...], List[float], torch.Tensor]],
        freeze_embeddings: bool = True,
        learn_residuals: bool = True,
        projector_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim

        # Shared projector: embed_dim → d_model
        self.projector = nn.Linear(embed_dim, d_model, bias=projector_bias)

        # Build ordered region list and name mapping
        self._region_names = sorted(region_embeddings.keys())
        self._name_to_idx = {n: i for i, n in enumerate(self._region_names)}

        # Stack all embeddings into one tensor
        n_regions = len(self._region_names)
        emb_matrix = torch.zeros(n_regions, embed_dim)
        for i, name in enumerate(self._region_names):
            raw = region_embeddings[name]
            if isinstance(raw, torch.Tensor):
                emb_matrix[i] = raw.float()
            else:
                emb_matrix[i] = torch.tensor(raw, dtype=torch.float32)

        if freeze_embeddings:
            self.register_buffer("_embeddings", emb_matrix)
        else:
            self._embeddings = nn.Parameter(emb_matrix)

        # Learned residuals (zero-init for safe grafting)
        if learn_residuals:
            self.residuals = nn.Parameter(torch.zeros(n_regions, d_model))
        else:
            self.register_buffer("residuals", None)

    @property
    def n_regions(self) -> int:
        return len(self._region_names)

    @property
    def region_names(self) -> List[str]:
        return list(self._region_names)

    def get_conditioning(self, region_name: str) -> torch.Tensor:
        """Get the d_model conditioning vector for a single region.

        Returns:
            (d_model,) tensor.
        """
        idx = self._name_to_idx[region_name]
        projected = self.projector(self._embeddings[idx])
        if self.residuals is not None:
            projected = projected + self.residuals[idx]
        return projected

    def get_all_conditioning(self) -> torch.Tensor:
        """Get conditioning vectors for all regions at once.

        Returns:
            (n_regions, d_model) tensor, ordered by sorted region name.
        """
        projected = self.projector(self._embeddings)  # (n_regions, d_model)
        if self.residuals is not None:
            projected = projected + self.residuals
        return projected

    def condition_canvas(
        self,
        canvas: torch.Tensor,
        layout: CanvasLayout,
    ) -> torch.Tensor:
        """Add semantic conditioning to all canvas positions.

        For each region in the layout that has a semantic embedding, adds
        the projected + residual conditioning vector to all positions in
        that region. Regions not in this conditioner are left unchanged.

        Args:
            canvas: (B, N, d_model) canvas tensor.
            layout: CanvasLayout describing the grid geometry.

        Returns:
            (B, N, d_model) conditioned canvas (new tensor, does not modify input).
        """
        canvas = canvas.clone()
        device = canvas.device

        # Batch-compute all conditioning vectors
        all_cond = self.get_all_conditioning()  # (n_regions, d_model)

        for name in layout.regions:
            if name not in self._name_to_idx:
                continue
            idx = self._name_to_idx[name]
            cond = all_cond[idx].to(device)  # (d_model,)
            pos_indices = layout.region_indices(name)
            pos_idx = torch.tensor(pos_indices, device=device, dtype=torch.long)
            canvas[:, pos_idx] = canvas[:, pos_idx] + cond

        return canvas

    def __repr__(self) -> str:
        frozen = not isinstance(self._embeddings, nn.Parameter)
        residual = self.residuals is not None
        return (
            f"SemanticConditioner(d_model={self.d_model}, embed_dim={self.embed_dim}, "
            f"n_regions={self.n_regions}, frozen={frozen}, residuals={residual})"
        )


def compute_semantic_embeddings(
    field_paths: List[str],
    embed_fn: Callable[[List[str]], List[List[float]]],
    semantic_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[float, ...]]:
    """Compute semantic embeddings for a list of field paths.

    Args:
        field_paths: List of dotted field paths (e.g. from BoundSchema.field_names).
        embed_fn: Callable that takes a list of text strings and returns
            a list of embedding vectors. Signature: (texts: List[str]) -> List[List[float]].
            The embedding model is up to the caller (OpenAI, sentence-transformers, etc.).
        semantic_types: Optional dict mapping field paths to custom semantic
            type descriptions. Paths not in this dict get auto-generated
            descriptions from their path structure.

    Returns:
        Dict mapping field paths to embedding tuples, ready to pass to
        SemanticConditioner.

    Example:
        import openai
        client = openai.OpenAI()

        def embed_fn(texts):
            resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [e.embedding for e in resp.data]

        embeddings = compute_semantic_embeddings(
            bound_schema.field_names,
            embed_fn,
        )

        conditioner = SemanticConditioner(
            d_model=64, embed_dim=1536,
            region_embeddings=embeddings,
        )
    """
    semantic_types = semantic_types or {}

    # Build text descriptions for each field
    texts = []
    for path in field_paths:
        if path in semantic_types:
            texts.append(semantic_types[path])
        else:
            texts.append(auto_semantic_type(path))

    # Call the embedding function
    raw_embeddings = embed_fn(texts)

    # Build result dict
    result = {}
    for path, emb in zip(field_paths, raw_embeddings):
        result[path] = tuple(emb)

    return result

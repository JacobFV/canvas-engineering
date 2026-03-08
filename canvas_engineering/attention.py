"""Attention function implementations for per-connection dispatch.

Each attention function is an nn.Module that takes:
  - queries: (B, N_src, d_model) from src region positions
  - keys:    (B, N_dst, d_model) from dst region positions
  - values:  (B, N_dst, d_model) from dst region positions

And returns:
  - output:  (B, N_src, d_model) updated src representations

The ATTENTION_REGISTRY maps function names (from ATTENTION_TYPES in canvas.py)
to factory callables: factory(d_model, n_heads, **kwargs) -> nn.Module.

Usage:
    from canvas_engineering.attention import create_attention
    attn = create_attention("cross_attention", d_model=256, n_heads=4)
    out = attn(queries, keys, values)
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Standard scaled dot-product QKV attention with softmax.

    O(N_src * N_dst). The default attention type.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = queries.shape
        M = keys.shape[1]
        H = self.n_heads

        q = self.q_proj(queries).view(B, N, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).view(B, M, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(B, M, H, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class LinearAttention(nn.Module):
    """Linear attention via elu+1 kernel trick. O(N_src + N_dst).

    No softmax — uses feature map phi(x) = elu(x) + 1 to decompose
    attention into linear operations. Much cheaper for large regions.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    @staticmethod
    def _elu_feature(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = queries.shape
        M = keys.shape[1]
        H = self.n_heads

        q = self._elu_feature(
            self.q_proj(queries).view(B, N, H, self.head_dim).transpose(1, 2)
        )
        k = self._elu_feature(
            self.k_proj(keys).view(B, M, H, self.head_dim).transpose(1, 2)
        )
        v = self.v_proj(values).view(B, M, H, self.head_dim).transpose(1, 2)

        # Linear attention: O(N + M) via associativity
        # out = (Q @ (K^T @ V)) / (Q @ sum(K))
        kv = k.transpose(-2, -1) @ v  # (B, H, d, d)
        z = k.sum(dim=-2, keepdim=True)  # (B, H, 1, d)

        out = (q @ kv) / (q @ z.transpose(-2, -1) + 1e-6)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class CosineAttention(nn.Module):
    """Cosine similarity attention — normalized dot-product without temperature."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = queries.shape
        M = keys.shape[1]
        H = self.n_heads

        q = self.q_proj(queries).view(B, N, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).view(B, M, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(B, M, H, self.head_dim).transpose(1, 2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class SigmoidAttention(nn.Module):
    """Sigmoid attention — each position independently gates each key.

    Non-exclusive: multiple keys can be attended to simultaneously.
    Good for multi-label patterns where attention isn't zero-sum.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = queries.shape
        M = keys.shape[1]
        H = self.n_heads

        q = self.q_proj(queries).view(B, N, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).view(B, M, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(B, M, H, self.head_dim).transpose(1, 2)

        attn = torch.sigmoid((q @ k.transpose(-2, -1)) * self.scale)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class GatedAttention(nn.Module):
    """Flamingo-style gated cross-attention.

    A learned sigmoid gate controls whether to incorporate cross-attended
    context. The gate starts at zero (pure pass-through), so adding a gated
    connection never hurts the initial model.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.gate = nn.Parameter(torch.zeros(1))  # zero-init = pass-through

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.cross_attn(queries, keys, values)
        return torch.sigmoid(self.gate) * attended


class PoolingAttention(nn.Module):
    """Mean-pool dst positions into a single vector, project, broadcast to src.

    Cheapest possible information transfer: O(M + N).
    Good for scalar conditioning signals (reward, time, regime indicator).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        pooled = values.mean(dim=1, keepdim=True)  # (B, 1, d_model)
        projected = self.proj(pooled)  # (B, 1, d_model)
        return projected.expand_as(queries)


class CopyAttention(nn.Module):
    """Direct tensor transfer — no learned parameters.

    Requires src and dst to have compatible shapes (or truncates/pads).
    For broadcast regions or direct latent sharing between agents.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # No parameters

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = queries.shape
        M = values.shape[1]
        if M >= N:
            return values[:, :N, :]
        # Pad if dst is smaller
        pad = torch.zeros(B, N - M, D, device=values.device, dtype=values.dtype)
        return torch.cat([values, pad], dim=1)


class NoneAttention(nn.Module):
    """Disabled connection — returns zeros. For ablation studies."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(queries)


class PerceiverAttention(nn.Module):
    """Cross-attend through a learned latent bottleneck.

    Compresses dst into K latent vectors, then src attends to those.
    O(N*K + M*K) where K << M. Good for very large dst regions.
    """

    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.0, n_latents: int = 8
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.compress = CrossAttention(d_model, n_heads, dropout)
        self.readout = CrossAttention(d_model, n_heads, dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B = queries.shape[0]
        latents = self.latents.expand(B, -1, -1)
        # Compress: latents attend to dst
        compressed = self.compress(latents, keys, values)
        # Readout: src attends to compressed latents
        return self.readout(queries, compressed, compressed)


class SparseAttention(nn.Module):
    """Top-k attention — only the k highest logits survive softmax.

    Sparse gradient flow. Good for selective binding.
    """

    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.0, top_k: int = 8
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = queries.shape
        M = keys.shape[1]
        H = self.n_heads
        k = min(self.top_k, M)

        q = self.q_proj(queries).view(B, N, H, self.head_dim).transpose(1, 2)
        kk = self.k_proj(keys).view(B, M, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(B, M, H, self.head_dim).transpose(1, 2)

        logits = (q @ kk.transpose(-2, -1)) * self.scale
        # Zero out everything except top-k
        topk_vals, _ = logits.topk(k, dim=-1)
        threshold = topk_vals[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        attn = F.softmax(logits, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


# ── Registry ──────────────────────────────────────────────────────────

ATTENTION_REGISTRY: Dict[str, type] = {
    "cross_attention": CrossAttention,
    "self_attention": CrossAttention,  # same implementation, different name
    "linear_attention": LinearAttention,
    "cosine_attention": CosineAttention,
    "sigmoid_attention": SigmoidAttention,
    "gated": GatedAttention,
    "perceiver": PerceiverAttention,
    "pooling": PoolingAttention,
    "copy": CopyAttention,
    "sparse_attention": SparseAttention,
    "none": NoneAttention,
}

# Types declared in ATTENTION_TYPES but not yet implemented as nn.Modules.
# These fall back to CrossAttention with a warning on first use.
_FALLBACK_TYPES = {
    "mamba", "rwkv", "hyena", "local_attention",
    "random_fixed", "mixture",
}
_warned_fallbacks: set = set()


def create_attention(
    fn_name: str,
    d_model: int,
    n_heads: int = 4,
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    """Create an attention module by function name.

    Args:
        fn_name: Attention function name (from ATTENTION_TYPES).
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        **kwargs: Extra kwargs passed to the attention constructor.

    Returns:
        nn.Module with forward(queries, keys, values) -> output.
    """
    if fn_name in ATTENTION_REGISTRY:
        return ATTENTION_REGISTRY[fn_name](d_model, n_heads, dropout, **kwargs)

    if fn_name in _FALLBACK_TYPES:
        if fn_name not in _warned_fallbacks:
            warnings.warn(
                f"Attention type '{fn_name}' is declared but not yet implemented. "
                f"Falling back to CrossAttention. To implement, add to ATTENTION_REGISTRY.",
                stacklevel=2,
            )
            _warned_fallbacks.add(fn_name)
        return CrossAttention(d_model, n_heads, dropout)

    raise ValueError(
        f"Unknown attention type: {fn_name!r}. "
        f"Available: {sorted(ATTENTION_REGISTRY.keys())}"
    )


def register_attention(name: str, cls: type) -> None:
    """Register a custom attention type at runtime.

    Args:
        name: Attention function name.
        cls: nn.Module class with forward(queries, keys, values) -> output.
    """
    ATTENTION_REGISTRY[name] = cls
    _FALLBACK_TYPES.discard(name)

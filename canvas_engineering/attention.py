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
        topk_vals, _ = logits.topk(k, dim=-1)
        threshold = topk_vals[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        attn = F.softmax(logits, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class LocalAttention(nn.Module):
    """Windowed attention — each query attends only within a local window.

    O(N * W) where W is window size. Positions outside the window are masked
    to -inf before softmax. For spatially or temporally local interactions
    where global attention is wasteful.
    """

    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.0, window_size: int = 8
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
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
        W = self.window_size

        q = self.q_proj(queries).view(B, N, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).view(B, M, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(B, M, H, self.head_dim).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, M)

        # Build local window mask: query i attends to keys [i-W//2, i+W//2]
        qi = torch.arange(N, device=logits.device).unsqueeze(1)  # (N, 1)
        ki = torch.arange(M, device=logits.device).unsqueeze(0)  # (1, M)
        # Scale key indices to query index space for cross-region compatibility
        if M != N:
            ki_scaled = (ki.float() * N / max(M, 1)).long()
        else:
            ki_scaled = ki
        dist = (qi - ki_scaled).abs()
        local_mask = dist > (W // 2)
        logits = logits.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class RandomFixedAttention(nn.Module):
    """Random sparse attention pattern, frozen at initialization.

    Each query position attends to a fixed random subset of key positions.
    The pattern is set at init and never changes (not learned). Useful as
    a baseline to measure whether learned attention patterns matter.
    """

    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.0, n_random: int = 8
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.n_random = n_random
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_shape: Optional[tuple] = None

    def _get_mask(self, N: int, M: int, device: torch.device) -> torch.Tensor:
        shape = (N, M)
        if self._cached_mask is not None and self._cached_shape == shape:
            return self._cached_mask.to(device)

        k = min(self.n_random, M)
        mask = torch.full((N, M), float("-inf"), device=device)
        for i in range(N):
            indices = torch.randperm(M, device=device)[:k]
            mask[i, indices] = 0.0

        self._cached_mask = mask.cpu()
        self._cached_shape = shape
        return mask.to(device)

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

        logits = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._get_mask(N, M, logits.device)
        logits = logits + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class MixtureAttention(nn.Module):
    """Mixture-of-experts style routing. Each query is routed to top-k
    experts (subsets of keys) by a learned router.

    The router produces a (N, n_experts) score matrix. Each expert
    attends to a disjoint slice of keys. Outputs are mixed by router
    weights. Sparse but adaptive.
    """

    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.0,
        n_experts: int = 4, top_k_experts: int = 2,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k_experts = min(top_k_experts, n_experts)
        self.experts = nn.ModuleList([
            CrossAttention(d_model, n_heads, dropout)
            for _ in range(n_experts)
        ])
        self.router = nn.Linear(d_model, n_experts)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = queries.shape
        M = keys.shape[1]

        # Route: (B, N, n_experts)
        router_logits = self.router(queries)
        topk_vals, topk_idx = router_logits.topk(self.top_k_experts, dim=-1)
        router_weights = F.softmax(topk_vals, dim=-1)  # (B, N, top_k)

        # Split keys/values across experts
        chunk_size = max(1, M // self.n_experts)
        expert_outputs = []
        for e in range(self.n_experts):
            start = e * chunk_size
            end = min(start + chunk_size, M) if e < self.n_experts - 1 else M
            k_e = keys[:, start:end]
            v_e = values[:, start:end]
            expert_outputs.append(self.experts[e](queries, k_e, v_e))

        expert_stack = torch.stack(expert_outputs, dim=2)  # (B, N, n_experts, D)

        # Mix by router weights (only top-k experts)
        output = torch.zeros(B, N, D, device=queries.device, dtype=queries.dtype)
        for k_idx in range(self.top_k_experts):
            expert_idx = topk_idx[..., k_idx]  # (B, N)
            weight = router_weights[..., k_idx].unsqueeze(-1)  # (B, N, 1)
            gathered = expert_stack.gather(
                2, expert_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, D)
            ).squeeze(2)
            output = output + weight * gathered

        return output


class MambaAttention(nn.Module):
    """Selective state-space model (S6-style). Input-dependent gating over
    a compressed state. O(N) sequential, hardware-efficient.

    Implements the core Mamba mechanism: a diagonal state-space model with
    input-dependent B, C, and delta parameters. The state compresses the
    key/value sequence into a fixed-size vector that is selectively updated.
    """

    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.0,
        state_size: int = 16, dt_rank: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        dt_rank = dt_rank or max(1, d_model // 16)

        # Input projections (expand to inner dim for processing)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        # SSM parameters (input-dependent)
        self.dt_proj = nn.Linear(dt_rank, d_model)
        self.x_proj = nn.Linear(d_model, dt_rank + state_size * 2)
        # Fixed A parameter (diagonal, log-space for stability)
        A = torch.arange(1, state_size + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.register_buffer("A_log", torch.log(A))
        # Output D residual
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = queries.shape
        M = keys.shape[1]

        # Process key/value sequence through SSM to build context
        x = values  # Use values as input sequence to SSM
        xz = self.in_proj(x)  # (B, M, 2D)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Input-dependent SSM parameters
        x_dbl = self.x_proj(x_ssm)  # (B, M, dt_rank + 2*state_size)
        dt_rank = x_dbl.shape[-1] - 2 * self.state_size
        dt, B_param, C_param = x_dbl.split([dt_rank, self.state_size, self.state_size], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, M, D)

        A = -torch.exp(self.A_log)  # (D, state_size)

        # Discretize and scan (simplified parallel scan via cumulative operations)
        # For each position in the sequence, update state h = exp(A*dt)*h + B*x
        # and compute output y = C*h + D*x
        h = torch.zeros(B, D, self.state_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(M):
            dt_t = dt[:, t].unsqueeze(-1)  # (B, D, 1)
            B_t = B_param[:, t].unsqueeze(1)  # (B, 1, state_size)
            C_t = C_param[:, t].unsqueeze(1)  # (B, 1, state_size)
            x_t = x_ssm[:, t].unsqueeze(-1)  # (B, D, 1)

            dA = torch.exp(A.unsqueeze(0) * dt_t)  # (B, D, state_size)
            dB = dt_t * B_t  # (B, D, state_size)

            h = dA * h + dB * x_t
            y_t = (h * C_t).sum(dim=-1) + self.D * x_ssm[:, t]  # (B, D)
            outputs.append(y_t)

        ssm_out = torch.stack(outputs, dim=1)  # (B, M, D)
        ssm_out = ssm_out * F.silu(z)  # Gated output

        # Use SSM output as context for queries via simple cross-attention-like readout
        # Pool SSM context and project to query space
        context = ssm_out.mean(dim=1, keepdim=True)  # (B, 1, D)
        out = self.out_proj(self.norm(context)).expand(B, N, -1)
        return out


class RWKVAttention(nn.Module):
    """Linear attention with learned exponential decay. O(N) via
    recurrent formulation. Time-mixing with position-dependent
    forgetting — like a transformer that naturally decays old context.

    Implements the RWKV-style WKV (weighted key-value) mechanism:
    each position has a learned decay rate that controls how quickly
    it forgets past keys.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Receptance, key, value projections
        self.r_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # Learned decay (per-head, log-space)
        self.decay = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.01 - 5.0)
        # Bonus for current position
        self.bonus = nn.Parameter(torch.zeros(n_heads, self.head_dim))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = queries.shape
        M = keys.shape[1]
        H = self.n_heads
        D = self.head_dim

        r = self.r_proj(queries).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(keys).view(B, M, H, D).transpose(1, 2)    # (B, H, M, D)
        v = self.v_proj(values).view(B, M, H, D).transpose(1, 2)  # (B, H, M, D)

        w = -torch.exp(self.decay).unsqueeze(0)  # (1, H, D) negative decay rates

        # WKV computation: weighted key-value with exponential decay
        # For each query position q, compute:
        #   wkv_q = sum_t exp(w * (q-t)) * exp(k_t) * v_t / sum_t exp(w * (q-t)) * exp(k_t)
        # We compute this via running sums for efficiency
        outputs = []
        num = torch.zeros(B, H, D, device=queries.device, dtype=queries.dtype)
        den = torch.zeros(B, H, D, device=queries.device, dtype=queries.dtype)

        for t in range(M):
            k_t = k[:, :, t]  # (B, H, D)
            v_t = v[:, :, t]  # (B, H, D)
            ek = torch.exp(k_t.clamp(-10, 10))  # (B, H, D)

            # Current token gets bonus weight
            if t < N:
                bonus_weight = torch.exp(self.bonus.unsqueeze(0) + k_t.clamp(-10, 10))
                wkv_t = (num + bonus_weight * v_t) / (den + bonus_weight + 1e-6)
                outputs.append(wkv_t)

            # Decay and accumulate
            decay_factor = torch.exp(w)  # (1, H, D)
            num = num * decay_factor + ek * v_t
            den = den * decay_factor + ek

        # Pad if M < N
        while len(outputs) < N:
            outputs.append(outputs[-1] if outputs else torch.zeros(B, H, D, device=queries.device))

        wkv = torch.stack(outputs[:N], dim=2)  # (B, H, N, D)

        # Apply receptance gating
        out = torch.sigmoid(r) * wkv
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class HyenaAttention(nn.Module):
    """Long convolution with data-dependent gating. O(N log N) via FFT.

    Implements a simplified Hyena operator: a stack of data-controlled
    long convolutions that provide sub-quadratic sequence mixing.
    Each "order" applies: output = conv(gate(input) * v, filter).
    """

    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.0,
        order: int = 2, max_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.max_len = max_len

        # Input projections: produce (order+1) streams from input
        self.in_proj = nn.Linear(d_model, d_model * (order + 1))
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable convolution filters (one per order, in frequency domain)
        # Parameterized as exponentially decaying sinusoids for stability
        self.filter_params = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, max_len // 2 + 1, 2) * 0.02)
            for _ in range(order)
        ])

        # Data-dependent gates
        self.gates = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(order)
        ])

    def _fft_conv(self, x: torch.Tensor, filter_idx: int) -> torch.Tensor:
        """Apply circular convolution via FFT."""
        B, N, D = x.shape
        # Pad to power of 2 for efficient FFT
        fft_len = max(1, 2 ** math.ceil(math.log2(max(N, 2))))

        x_f = torch.fft.rfft(x, n=fft_len, dim=1)  # (B, fft_len//2+1, D)

        # Build filter in frequency domain
        fp = self.filter_params[filter_idx]  # (D, max_len//2+1, 2)
        freq_len = x_f.shape[1]
        fp_slice = fp[:, :freq_len]  # (D, freq_len, 2)
        h_f = torch.view_as_complex(fp_slice.contiguous())  # (D, freq_len)
        h_f = h_f.unsqueeze(0).transpose(1, 2)  # (1, freq_len, D)

        y_f = x_f * h_f
        y = torch.fft.irfft(y_f, n=fft_len, dim=1)[:, :N]  # (B, N, D)
        return y

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = queries.shape
        M = keys.shape[1]

        # Use keys/values as the sequence to convolve over
        # Pool to query length if needed
        if M != N:
            x = F.adaptive_avg_pool1d(
                values.transpose(1, 2), N
            ).transpose(1, 2)
        else:
            x = values

        # Project to (order+1) streams
        projections = self.in_proj(x).view(B, N, self.order + 1, D)
        v = projections[:, :, 0]  # value stream
        xs = [projections[:, :, i + 1] for i in range(self.order)]

        # Apply Hyena recurrence: v = conv(gate(x_i) * v, h_i) for each order
        for i in range(self.order):
            gate = torch.sigmoid(self.gates[i](xs[i]))
            v = v * gate
            v = self._fft_conv(v, i)

        return self.out_proj(v)


# ── Registry ──────────────────────────────────────────────────────────

ATTENTION_REGISTRY: Dict[str, type] = {
    "cross_attention": CrossAttention,
    "self_attention": CrossAttention,
    "linear_attention": LinearAttention,
    "cosine_attention": CosineAttention,
    "sigmoid_attention": SigmoidAttention,
    "gated": GatedAttention,
    "perceiver": PerceiverAttention,
    "pooling": PoolingAttention,
    "copy": CopyAttention,
    "sparse_attention": SparseAttention,
    "local_attention": LocalAttention,
    "random_fixed": RandomFixedAttention,
    "mixture": MixtureAttention,
    "mamba": MambaAttention,
    "rwkv": RWKVAttention,
    "hyena": HyenaAttention,
    "none": NoneAttention,
}


def create_attention(
    fn_name: str,
    d_model: int,
    n_heads: int = 4,
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    """Create an attention module by function name.

    All 17 types declared in ATTENTION_TYPES are fully implemented.

    Args:
        fn_name: Attention function name (from ATTENTION_TYPES / ATTENTION_REGISTRY).
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        **kwargs: Extra kwargs passed to the attention constructor
            (e.g. top_k for sparse_attention, window_size for local_attention).

    Returns:
        nn.Module with forward(queries, keys, values) -> output.

    Raises:
        ValueError: If fn_name is not in ATTENTION_REGISTRY.
    """
    if fn_name in ATTENTION_REGISTRY:
        return ATTENTION_REGISTRY[fn_name](d_model, n_heads, dropout, **kwargs)

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

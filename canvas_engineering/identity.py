"""Identity and slot persistence for canvas regions.

IdentitySpec declares how a region manages persistent object identity
over time. SlotBindingModule implements cross-attention based binding
of observations to persistent identity slots.

Modes:
    "persistent" — Slots persist across timesteps. Observations bind to
        existing slots via cross-attention. Good for tracking known
        objects with stable identities.
    "ephemeral" — Slots are re-created each timestep from observations.
        No temporal persistence. Good for detection without tracking.
    "tracklet" — Slots persist but can be born and killed via learned
        birth/death gates. Good for tracking with variable object count.

Usage:
    from canvas_engineering.identity import IdentitySpec, SlotBindingModule

    spec = IdentitySpec(mode="persistent", slot_capacity=32)
    binder = SlotBindingModule(d_model=256, n_slots=32, n_heads=4)

    observations = torch.randn(2, 10, 256)  # (B, N_obs, d_model)
    slots = binder(observations)  # (B, 32, 256)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class IdentitySpec:
    """Declares how a region manages persistent object identity.

    Args:
        mode: Identity management mode.
            "persistent" = slots persist across timesteps (default).
            "ephemeral" = slots re-created each step.
            "tracklet" = persistent with birth/death gating.
        slot_capacity: Maximum number of identity slots.
        binding_fn: How observations bind to slots.
            "cross_attention" (default) = standard cross-attention binding.
        birth_death: If True and mode="tracklet", enable learned
            birth/death gates for dynamic slot management.
    """
    mode: str = "persistent"
    slot_capacity: int = 64
    binding_fn: str = "cross_attention"
    birth_death: bool = False


class _SlotCrossAttention(nn.Module):
    """Cross-attention from slots (queries) to observations (keys/values)."""

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
        """(B, N_q, D), (B, N_kv, D), (B, N_kv, D) -> (B, N_q, D)."""
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


class SlotBindingModule(nn.Module):
    """Binds observations to persistent identity slots via cross-attention.

    Maintains a set of learned slot vectors. Each forward call updates
    slots by cross-attending from slots (queries) to observations
    (keys/values), producing updated slot representations.

    For "tracklet" mode with birth_death=True, also computes a scalar
    gate per slot indicating whether the slot is alive.

    Args:
        d_model: Model dimension.
        n_slots: Number of identity slots.
        n_heads: Number of attention heads for cross-attention.
        birth_death: Whether to enable birth/death gating.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int,
        n_heads: int = 4,
        birth_death: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.binding_attn = _SlotCrossAttention(d_model, n_heads)
        self.layer_norm = nn.LayerNorm(d_model)
        self._birth_death = birth_death

        if birth_death:
            self.life_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.life_gate = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Bind observations to slots via cross-attention.

        Args:
            observations: (B, N_obs, d_model) observation embeddings.

        Returns:
            (B, n_slots, d_model) updated slot representations.
            If birth_death is enabled, dead slots are zeroed out.
        """
        B = observations.shape[0]
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)

        # Cross-attend: slots query observations
        updated = self.binding_attn(slots, observations, observations)
        # Residual + layer norm
        updated = self.layer_norm(updated + slots)

        # Apply birth/death gating if enabled
        if self.life_gate is not None:
            gate = self.life_gate(updated)  # (B, n_slots, 1)
            updated = updated * gate

        return updated

    def alive_mask(self, observations: torch.Tensor) -> torch.Tensor:
        """Get the alive mask for slots after binding.

        Only meaningful when birth_death=True.

        Args:
            observations: (B, N_obs, d_model) observation embeddings.

        Returns:
            (B, n_slots) boolean tensor. True = slot is alive.
        """
        if self.life_gate is None:
            B = observations.shape[0]
            return torch.ones(B, self.n_slots, dtype=torch.bool,
                              device=observations.device)

        B = observations.shape[0]
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)
        updated = self.binding_attn(slots, observations, observations)
        updated = self.layer_norm(updated + slots)
        gate = self.life_gate(updated).squeeze(-1)  # (B, n_slots)
        return gate > 0.5

    def __repr__(self) -> str:
        return "SlotBindingModule(d_model={}, n_slots={}, birth_death={})".format(
            self.d_model, self.n_slots, self._birth_death)

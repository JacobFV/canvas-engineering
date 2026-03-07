"""Action decoder heads for extracting robot actions from diffusion model outputs."""

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Simple MLP action decoder.

    Reads from predicted noise (latent_channels dim) and outputs action_dim.
    Default architecture: Linear -> GELU -> Linear.

    Args:
        latent_channels: Input dimension (e.g., 16 for CogVideoX latent space).
        action_dim: Output action dimension (e.g., 7 for 6DOF + gripper).
        hidden_dim: Hidden layer width. Default: latent_channels // 2.
    """

    def __init__(self, latent_channels: int = 16, action_dim: int = 7, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(latent_channels // 2, 8)
        self.net = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, noise_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise_pred: (B, T, C, H, W) or (B, C) predicted noise from diffusion model.
        Returns:
            actions: (B, action_dim) predicted actions.
        """
        if noise_pred.dim() == 5:
            # Average over spatial and temporal dims, keep channels
            x = noise_pred.mean(dim=(1, 3, 4))  # (B, C)
        elif noise_pred.dim() == 3:
            x = noise_pred.mean(dim=1)  # (B, C)
        else:
            x = noise_pred
        return self.net(x)

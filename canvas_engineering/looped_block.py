"""Generic Looped Block Wrapper.

Wraps ANY transformer block to iterate it L times per forward pass,
adding learned loop embeddings and sigmoid gates at each iteration.
Zero-init embeddings ensure safe grafting onto pretrained models.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple


class LoopedBlockWrapper(nn.Module):
    """Wrap a transformer block for looped execution.

    The original block is called L times. At each iteration:
        h = original(h + loop_emb[l], ...) * gate[l] + h * (1 - gate[l])

    Args:
        original: The transformer block to wrap (kept frozen or trainable).
        block_idx: Index of this block in the model (for logging).
        max_loops: Maximum number of iterations.
        embed_dim: Dimension for loop embeddings (usually d_model or inner_dim).
        gate_init_bias: Initial bias for sigmoid gate (0.0 = 50% blending).
        use_gradient_checkpointing: Recompute activations on backward for memory savings.
    """

    def __init__(
        self,
        original: nn.Module,
        block_idx: int = 0,
        max_loops: int = 4,
        embed_dim: int = 256,
        gate_init_bias: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.original = original
        self.block_idx = block_idx
        self.max_loops = max_loops
        self.embed_dim = embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.current_loops = 1  # Start with 1 loop (identity behavior)

        # Per-iteration learned embeddings (zero-init = no perturbation at start)
        self.loop_emb = nn.Parameter(torch.zeros(max_loops, embed_dim))
        # Per-iteration gates (init at gate_init_bias => sigmoid(0) = 0.5)
        self.loop_gate = nn.Parameter(torch.full((max_loops, 1), gate_init_bias))

    def set_loops(self, n: int):
        """Set the current number of loop iterations (for curriculum)."""
        self.current_loops = min(n, self.max_loops)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Run the original block `current_loops` times with learned iteration signals.

        The first positional argument is assumed to be the hidden states tensor.
        All other args/kwargs are passed through to the original block unchanged.
        """
        h = hidden_states
        for l in range(self.current_loops):
            # Add loop embedding to hidden states
            h_input = h + self.loop_emb[l].unsqueeze(0).unsqueeze(0)

            # Run original block (with optional gradient checkpointing)
            if self.use_gradient_checkpointing and self.current_loops > 1 and torch.is_grad_enabled():
                import torch.utils.checkpoint as ckpt
                _args = args
                _kwargs = kwargs

                def _run(h_in):
                    return self.original(h_in, *_args, **_kwargs)

                h_out = ckpt.checkpoint(_run, h_input, use_reentrant=False)
            else:
                h_out = self.original(h_input, *args, **kwargs)

            # Handle blocks that return tuples (hidden_states, extra_outputs...)
            if isinstance(h_out, tuple):
                h_out = h_out[0]

            # Gated residual mixing
            gate = torch.sigmoid(self.loop_gate[l])
            h = gate * h_out + (1 - gate) * h

        return h

    def trainable_params(self) -> int:
        """Count only the loop-specific trainable parameters."""
        return sum(p.numel() for p in [self.loop_emb, self.loop_gate])

    def extra_repr(self) -> str:
        return (
            f"block_idx={self.block_idx}, max_loops={self.max_loops}, "
            f"current_loops={self.current_loops}, embed_dim={self.embed_dim}, "
            f"trainable_params={self.trainable_params()}"
        )

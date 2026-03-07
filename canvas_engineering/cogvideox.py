"""CogVideoX-specific looped block wrapper.

Handles the dual hidden state (hidden_states + encoder_hidden_states) signature
of CogVideoXBlock, plus the image_rotary_emb and temb conditioning.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class LoopedCogVideoXBlock(nn.Module):
    """Wrap a frozen CogVideoXBlock for looped execution.

    CogVideoX blocks have a specific forward signature:
        forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb=..., **kwargs)
        -> (hidden_states, encoder_hidden_states)

    This wrapper adds per-iteration loop embeddings to both hidden states streams.
    """

    def __init__(self, original_block: nn.Module, block_idx: int, max_loops: int, inner_dim: int):
        super().__init__()
        self.original = original_block
        self.block_idx = block_idx
        self.max_loops = max_loops
        self.inner_dim = inner_dim
        self.current_loops = 1

        # Per-iteration embeddings for the hidden stream
        self.loop_emb = nn.Parameter(torch.zeros(max_loops, inner_dim))
        # Per-iteration embeddings for the encoder stream
        self.loop_emb_enc = nn.Parameter(torch.zeros(max_loops, inner_dim))
        # Per-iteration gating
        self.loop_gate = nn.Parameter(torch.zeros(max_loops, 1))

    def set_loops(self, n: int):
        self.current_loops = min(n, self.max_loops)

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, **kwargs):
        h_in = hidden_states
        e_in = encoder_hidden_states

        for l in range(self.current_loops):
            h_input = h_in + self.loop_emb[l].unsqueeze(0).unsqueeze(0)
            e_input = e_in + self.loop_emb_enc[l].unsqueeze(0).unsqueeze(0)

            # Gradient checkpointing for multi-loop memory savings
            if self.current_loops > 1 and torch.is_grad_enabled():
                import torch.utils.checkpoint as ckpt
                _temb = temb
                _ire = image_rotary_emb
                _kwargs = kwargs

                def _run(h, e):
                    return self.original(h, e, _temb, image_rotary_emb=_ire, **_kwargs)

                h_out, e_out = ckpt.checkpoint(_run, h_input, e_input, use_reentrant=False)
            else:
                h_out, e_out = self.original(
                    h_input, e_input, temb, image_rotary_emb=image_rotary_emb, **kwargs
                )

            gate = torch.sigmoid(self.loop_gate[l])
            h_in = gate * h_out + (1 - gate) * h_in
            e_in = gate * e_out + (1 - gate) * e_in

        return h_in, e_in

    def trainable_params(self) -> int:
        return sum(p.numel() for p in [self.loop_emb, self.loop_emb_enc, self.loop_gate])


def detect_inner_dim(transformer) -> int:
    """Auto-detect inner_dim from a CogVideoX transformer."""
    blocks = transformer.transformer_blocks
    if len(blocks) == 0:
        raise ValueError("No transformer blocks found")
    block = blocks[0]
    # Try common attribute names
    for attr in ["norm1.normalized_shape", "norm1.weight"]:
        obj = block
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__"):
                return obj[0] if isinstance(obj, (list, tuple)) else obj.shape[0]
        except (AttributeError, IndexError):
            continue
    # Fallback: check attention layer
    if hasattr(block, "attn1") and hasattr(block.attn1, "to_q"):
        return block.attn1.to_q.in_features
    logger.warning("Could not detect inner_dim, defaulting to 1920")
    return 1920

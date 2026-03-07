"""Graft looped attention onto existing diffusion models.

The core operation: replace each transformer block with a LoopedBlockWrapper
that iterates the frozen original block multiple times with learned embeddings.
"""

import logging
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

from canvas_engineering.looped_block import LoopedBlockWrapper
from canvas_engineering.action_heads import ActionHead

logger = logging.getLogger(__name__)


def graft_looped_blocks(
    transformer: nn.Module,
    max_loops: int = 3,
    block_attr: str = "transformer_blocks",
    wrapper_class: Optional[Type] = None,
    inner_dim: Optional[int] = None,
    action_dim: int = 7,
    latent_channels: int = 16,
    freeze: str = "full",
) -> Tuple[List[nn.Module], ActionHead]:
    """Graft looped attention onto any transformer model.

    Replaces each block in transformer.<block_attr> with a looped wrapper.
    Returns the list of looped blocks (for optimizer) and an ActionHead.

    Args:
        transformer: The pretrained transformer model.
        max_loops: Maximum loop iterations per block.
        block_attr: Attribute name for the block list (e.g., "transformer_blocks").
        wrapper_class: Custom wrapper class. If None, auto-detects CogVideoX vs generic.
        inner_dim: Embedding dimension for loop params. Auto-detected if None.
        action_dim: Output action dimension for the ActionHead.
        latent_channels: Input channels for ActionHead (e.g., 16 for CogVideoX latents).
        freeze: Freeze strategy: "full", "half", or "none".

    Returns:
        (looped_blocks, action_head): List of wrapped blocks and the action decoder.
    """
    blocks = getattr(transformer, block_attr)
    n_blocks = len(blocks)

    # Auto-detect wrapper class
    if wrapper_class is None:
        block_type = type(blocks[0]).__name__
        if "CogVideoX" in block_type:
            from canvas_engineering.cogvideox import LoopedCogVideoXBlock, detect_inner_dim
            wrapper_class = LoopedCogVideoXBlock
            if inner_dim is None:
                inner_dim = detect_inner_dim(transformer)
        else:
            wrapper_class = LoopedBlockWrapper
            if inner_dim is None:
                # Try to detect from first block
                for name, param in blocks[0].named_parameters():
                    if "weight" in name:
                        inner_dim = param.shape[0]
                        break
                if inner_dim is None:
                    inner_dim = 256

    logger.info(f"Grafting {n_blocks} blocks with {wrapper_class.__name__}, inner_dim={inner_dim}")

    # Replace blocks
    looped_blocks = []
    for i in range(n_blocks):
        original = blocks[i]
        looped = wrapper_class(original, block_idx=i, max_loops=max_loops, inner_dim=inner_dim)
        blocks[i] = looped
        looped_blocks.append(looped)
        logger.info(f"  Block {i}: {type(original).__name__} -> {wrapper_class.__name__}")

    logger.info(f"Grafting complete: {n_blocks} blocks replaced")

    # Apply freeze strategy
    if freeze == "full":
        frozen = freeze_full(transformer)
    elif freeze == "half":
        frozen = freeze_half(transformer)
    else:
        frozen = 0
    logger.info(f"Freeze strategy '{freeze}': {frozen:,} params frozen")

    # Count trainable
    total_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    loop_trainable = sum(b.trainable_params() for b in looped_blocks)
    logger.info(f"Total trainable: {total_trainable:,} (loop params: {loop_trainable:,})")

    # Create action head
    action_head = ActionHead(latent_channels, action_dim)
    logger.info(f"ActionHead: {latent_channels} -> {action_dim} ({sum(p.numel() for p in action_head.parameters())} params)")

    return looped_blocks, action_head


def freeze_full(transformer: nn.Module) -> int:
    """Freeze all non-loop parameters (patch_embed, time_embed, norm_out, proj_out)."""
    frozen = 0
    loop_param_names = {"loop_emb", "loop_emb_enc", "loop_gate"}
    for name, param in transformer.named_parameters():
        # Keep loop params trainable
        if any(lp in name for lp in loop_param_names):
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def freeze_half(transformer: nn.Module) -> int:
    """Freeze only patch_embed (largest input module). Leave time_embed, norms free."""
    frozen = 0
    for name, param in transformer.named_parameters():
        if "patch_embed" in name or "pos_embedding" in name:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
    return frozen

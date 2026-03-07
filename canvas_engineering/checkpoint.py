"""Save and load only loop parameters (not the full model)."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_loop_checkpoint(
    looped_blocks: List[nn.Module],
    action_head: nn.Module,
    path: str,
    metadata: Optional[Dict] = None,
):
    """Save only the trainable loop parameters and action head.

    This is ~0.1% of the full model size since the backbone is frozen.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "loop_blocks": {
            f"block_{i}": {
                name: param.data.cpu()
                for name, param in block.named_parameters()
                if param.requires_grad
            }
            for i, block in enumerate(looped_blocks)
        },
        "action_head": action_head.state_dict(),
    }
    torch.save(state, path)

    if metadata:
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    total_params = sum(p.numel() for d in state["loop_blocks"].values() for p in d.values())
    total_params += sum(p.numel() for p in state["action_head"].values())
    logger.info(f"Saved {total_params:,} params to {path} ({path.stat().st_size / 1024:.1f} KB)")


def load_loop_checkpoint(
    looped_blocks: List[nn.Module],
    action_head: nn.Module,
    path: str,
):
    """Load loop parameters and action head from a checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=True)

    for i, block in enumerate(looped_blocks):
        key = f"block_{i}"
        if key in state["loop_blocks"]:
            block_state = state["loop_blocks"][key]
            for name, param in block.named_parameters():
                if name in block_state:
                    param.data.copy_(block_state[name])

    action_head.load_state_dict(state["action_head"])
    logger.info(f"Loaded checkpoint from {path}")

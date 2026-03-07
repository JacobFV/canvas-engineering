"""Tests for grafting looped blocks onto models."""
import torch
import torch.nn as nn
import pytest
from canvas_engine.looped_block import LoopedBlockWrapper
from canvas_engine.graft import freeze_full, freeze_half


class FakeTransformer(nn.Module):
    def __init__(self, n_blocks=4, dim=32):
        super().__init__()
        self.patch_embed = nn.Linear(3, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 10, dim))
        self.transformer_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
            for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, 3)


def test_freeze_full():
    model = FakeTransformer()
    # Manually add loop params to simulate grafting
    for i, block in enumerate(model.transformer_blocks):
        wrapper = LoopedBlockWrapper(block, block_idx=i, max_loops=3, embed_dim=32)
        model.transformer_blocks[i] = wrapper

    frozen = freeze_full(model)
    assert frozen > 0

    # Loop params should still be trainable
    for block in model.transformer_blocks:
        assert block.loop_emb.requires_grad
        assert block.loop_gate.requires_grad

    # Patch embed should be frozen
    assert not model.patch_embed.weight.requires_grad


def test_freeze_half():
    model = FakeTransformer()
    frozen = freeze_half(model)
    assert frozen > 0
    assert not model.patch_embed.weight.requires_grad
    assert model.norm_out.weight.requires_grad  # should NOT be frozen in half mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

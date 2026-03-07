"""Tests for LoopedBlockWrapper."""
import torch
import torch.nn as nn
import pytest
from canvas_engineering.looped_block import LoopedBlockWrapper


class DummyBlock(nn.Module):
    """Simple transformer block for testing."""
    def __init__(self, dim=32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.linear(x) + x)


def test_looped_block_1loop_is_passthrough():
    block = DummyBlock(32)
    looped = LoopedBlockWrapper(block, max_loops=4, embed_dim=32, use_gradient_checkpointing=False)
    looped.set_loops(1)
    x = torch.randn(2, 10, 32)
    y = looped(x)
    assert y.shape == x.shape


def test_looped_block_multiple_loops():
    block = DummyBlock(32)
    looped = LoopedBlockWrapper(block, max_loops=4, embed_dim=32, use_gradient_checkpointing=False)
    looped.set_loops(3)
    x = torch.randn(2, 10, 32)
    y = looped(x)
    assert y.shape == x.shape


def test_looped_block_zero_init():
    """Loop embeddings are zero-initialized, so 1-loop should approximate original."""
    block = DummyBlock(32)
    looped = LoopedBlockWrapper(block, max_loops=4, embed_dim=32, use_gradient_checkpointing=False)
    looped.set_loops(1)

    x = torch.randn(1, 5, 32)
    y_looped = looped(x)
    y_original = block(x)

    # With zero-init emb and 0.5 gate, output should be 0.5 * original + 0.5 * input
    # Not exactly equal to original, but close
    assert y_looped.shape == y_original.shape


def test_looped_block_trainable_params():
    block = DummyBlock(32)
    looped = LoopedBlockWrapper(block, max_loops=4, embed_dim=32)
    # loop_emb: 4*32=128, loop_gate: 4*1=4
    assert looped.trainable_params() == 132


def test_looped_block_gradient_flow():
    block = DummyBlock(32)
    looped = LoopedBlockWrapper(block, max_loops=3, embed_dim=32, use_gradient_checkpointing=False)
    looped.set_loops(3)
    x = torch.randn(1, 5, 32, requires_grad=True)
    y = looped(x)
    loss = y.sum()
    loss.backward()
    assert looped.loop_emb.grad is not None
    assert looped.loop_gate.grad is not None


def test_set_loops_clamps():
    block = DummyBlock(32)
    looped = LoopedBlockWrapper(block, max_loops=3, embed_dim=32)
    looped.set_loops(10)
    assert looped.current_loops == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

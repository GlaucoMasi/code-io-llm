import torch
from src.model import TransformerBlock

class Config:
    n_embd = 32
    n_head = 4
    block_size = 10

def test_shapes_and_forward_pass():
    config = Config()
    block = TransformerBlock(config)

    x = torch.randn(1, 10, config.n_embd)
    output = block(x)

    assert output.shape == x.shape, f"TransformerBlock Error: Input: {x.shape}, Output: {output.shape}"
    assert not torch.isnan(output).any(), f"TransformerBlock Error: Output: {output}"
    print("Test Shapes and Forward Pass: OK!")

def test_non_linearity():
    config = Config()
    block = TransformerBlock(config)

    for p in block.parameters(): p.data.zero_()
    
    x = torch.randn(1, 10, config.n_embd)
    output = block(x)

    assert torch.allclose(block(x), x), f"FeedForward Error: Residual Connection Failed. Input: {x}, Output: {output}"
    print("Test Residual Connection: OK!")
import torch
from src.model import CausalSelfAttention

class Config:
    n_embd = 32
    n_head = 4
    block_size = 10

def test_shapes():
    config = Config()
    attn = CausalSelfAttention(config)

    x = torch.randn(2, 10, config.n_embd)
    output = attn(x)

    assert output.shape == x.shape, f"CausalSelfAttention Error: Input: {x.shape}, Output: {output.shape}"
    print("Test Shapes: OK!")

def test_forward_pass():
    config = Config()
    attn = CausalSelfAttention(config)

    x = torch.randn(2, 10, config.n_embd)
    output = attn(x)

    assert not torch.isnan(output).any(), f"CausalSelfAttention Error: Output: {output}"
    print("Test Forward Pass: OK!")
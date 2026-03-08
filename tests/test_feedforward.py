import torch
from src.model import FeedForward

class Config:
    n_embd = 32

def test_shapes():
    config = Config()
    ffn = FeedForward(config)

    x = torch.randn(2, 10, config.n_embd)
    output = ffn(x)

    assert output.shape == x.shape, f"FeedForward Error: Input: {x.shape}, Output: {output.shape}"
    print("Test Shapes: OK!")

def test_non_linearity():
    config = Config()
    ffn = FeedForward(config)

    x1 = torch.randn(1, 5, config.n_embd)
    x2 = torch.randn(1, 5, config.n_embd)

    output1 = ffn(x1 + x2)
    output2 = ffn(x1) + ffn(x2)

    assert not torch.allclose(output1, output2, atol=1e-5), f"FeedForward Error: Model is linear. Output1: {output1}, Output2: {output2}"
    print("Test Non-Linearity: OK!")
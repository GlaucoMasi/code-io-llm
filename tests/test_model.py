import torch
import torch.optim as optim
import torch.nn.functional as F
from src.model import CodeIOLLM

class Config:
    vocab_size = 100
    n_embd = 32
    n_head = 4
    n_layer = 2
    block_size = 10

def test_shapes():
    config = Config()
    model = CodeIOLLM(config)

    B, T = 2, 5
    idx = torch.randint(0, config.vocab_size, (B, T))

    logits = model(idx)

    assert logits.shape == (B, T, config.vocab_size), f"CodeIOLLM Error: Expected: {(B, T, config.vocab_size)}, Output: {logits.shape}"
    print("Test Shapes: OK!")

def test_determinism():
    config = Config()
    model = CodeIOLLM(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (1, 5))

    output1 = model(idx)
    output2 = model(idx)

    assert torch.allclose(output1, output2), f"CodeIOLLM Error: Output1: {output1}, Output2: {output2}"
    print("Test Determinism: OK!")

def test_gradients():
    config = Config()
    model = CodeIOLLM(config)

    idx = torch.randint(0, config.vocab_size, (1, 5))
    logits = model(idx)

    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"CodeIOLLM Error: Parameter: {name}"
    print("Test Gradients: OK!")

def test_causal_mask():
    config = Config()
    model = CodeIOLLM(config)

    idx1 = torch.tensor([[1, 2, 3, 4, 5]])
    idx2 = torch.tensor([[1, 2, 3, 4, 99]])

    output1 = model(idx1)
    output2 = model(idx2)

    assert torch.allclose(output1[:, :2, :], output2[:, :2, :]), f"CodeIOLLM Error: Output1: {output1[:, :2, :]}, Output2: {output2[:, :2, :]}"
    print("Test Causal Mask: OK!")

def test_minimal_learning():
    config = Config()
    model = CodeIOLLM(config)

    x = torch.randint(0, config.vocab_size, (1, 5))
    target = torch.randint(0, config.vocab_size, (1, 5))

    initial_logits = model(x)
    initial_loss = F.cross_entropy(initial_logits.view(-1, config.vocab_size), target.view(-1))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(20):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()

    final_logits = model(x)
    final_loss = F.cross_entropy(final_logits.view(-1, config.vocab_size), target.view(-1))

    assert final_loss < initial_loss, f"CodeIOLLM Error: Initial Loss: {initial_loss.item():.4f}, Final Loss: {final_loss.item():.4f}"
    print("Test Minimal Learning: OK!")

def test_overfitting_to_zero():
    config = Config()
    model = CodeIOLLM(config)

    # Memorizzazione
    x = torch.randint(0, config.vocab_size, (1, 5))
    target = x.clone()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(100):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.05, f"CodeIOLLM Error: Final Loss: {loss.item():.4f}"
    print("Test Overfitting to Zero: OK!")
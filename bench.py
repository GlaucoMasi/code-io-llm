"""
bench.py — Training benchmark for CodeIOLLM.

Runs a fixed number of warm-up steps followed by timed benchmark steps,
then prints a summary of throughput, latency, loss progression, and memory.

Usage:
    python3 bench.py [--steps N] [--warmup N] [--config small]
"""

import argparse
import time
import yaml
import torch
import torch.nn as nn

from src.tokenizer.BasicTokenizer import BasicTokenizer
from src.model import CodeIOLLM
from src.dataset import create_dataloaders


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(name: str) -> dict:
    with open(f"configs/{name}.yaml", "r") as f:
        return yaml.safe_load(f)


def load_docs(path: str) -> list[str]:
    # Non più necessario
    pass


def peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float("nan")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(config_name: str = "small", bench_steps: int = 100, warmup_steps: int = 5):
    config = load_config(config_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*56}")
    print(f"  CodeIOLLM Training Benchmark")
    print(f"{'='*56}")
    print(f"  Config   : {config_name}")
    print(f"  Device   : {device}")
    print(f"  Warmup   : {warmup_steps} steps")
    print(f"  Bench    : {bench_steps} steps")
    print(f"  n_embd   : {config['n_embd']}  |  n_layer: {config['n_layer']}  |  n_head: {config['n_head']}")
    print(f"  vocab    : {config['vocab_size']}  |  block_size: {config['block_size']}")
    print(f"{'='*56}\n")

    model = CodeIOLLM(config).to(device)
    model.train()

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    tokenizer = BasicTokenizer()
    tokenizer.load("src/tokenizer/test.model")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step_times:  list[float] = []
    step_tokens: list[int]   = []
    losses:      list[float] = []
    doc_idx = 0

    train_dl, _ = create_dataloaders("data/code_contests_cpp.txt", tokenizer, config["block_size"], batch_size=1)
    train_iter = iter(train_dl)

    total_steps = warmup_steps + bench_steps
    for step in range(total_steps):
        try:
            tokens, shift_labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            tokens, shift_labels = next(train_iter)

        tokens = tokens.to(device)
        shift_labels = shift_labels.to(device)

        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits       = model(tokens)
        
        # In this dataset abstraction, x and y are already shifted
        logits = logits.view(-1, logits.size(-1))
        y_flat = shift_labels.view(-1)

        if (y_flat != -100).sum() == 0:
            # Skip this batch if all tokens are masked out (e.g. falls entirely inside a long prompt)
            continue
            
        loss = loss_fn(logits, y_flat)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0

        if step < warmup_steps:
            label = "warmup"
        else:
            step_times.append(elapsed)
            step_tokens.append(tokens.numel())
            losses.append(loss.item())
            label = "bench "

        print(f"  [{label}] step {step+1:4d}/{total_steps} | "
              f"loss {loss.item():.4f} | {elapsed*1000:.1f} ms | {tokens.numel()} tok")

    # ── Summary ───────────────────────────────────────────────────────────────
    if not step_times:
        print("\n========================================================")
        print("  WARNING: No benchmark steps completed. All batches were skipped entirely.")
        print("========================================================\n")
        return

    avg_ms        = (sum(step_times) / len(step_times)) * 1000
    min_ms        = min(step_times) * 1000
    max_ms        = max(step_times) * 1000
    total_tokens  = sum(step_tokens)
    total_time    = sum(step_times)
    tokens_per_s  = total_tokens / total_time
    steps_per_s   = len(step_times) / total_time
    first_loss    = losses[0] if losses else float('nan')
    last_loss     = losses[-1] if losses else float('nan')
    mem_mb        = peak_memory_mb()

    print(f"\n{'='*56}")
    print(f"  BENCHMARK SUMMARY ({bench_steps} timed steps)")
    print(f"{'='*56}")
    print(f"  Steps/sec        : {steps_per_s:.2f}")
    print(f"  Tokens/sec       : {tokens_per_s:.0f}")
    print(f"  Avg latency      : {avg_ms:.1f} ms/step")
    print(f"  Min / Max latency: {min_ms:.1f} ms / {max_ms:.1f} ms")
    print(f"  Loss (first→last): {first_loss:.4f} → {last_loss:.4f}  "
          f"({'↓' if last_loss < first_loss else '↑'} {abs(last_loss - first_loss):.4f})")
    print(f"  Peak memory      : {mem_mb:.1f} MB")
    print(f"  Total parameters : {total_params:,}")
    print(f"{'='*56}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CodeIOLLM training")
    parser.add_argument("--steps",   type=int, default=100, help="Number of timed steps (default: 100)")
    parser.add_argument("--warmup",  type=int, default=5,   help="Warmup steps before timing (default: 5)")
    parser.add_argument("--config",  type=str, default="small", help="Config name in configs/ (default: small)")
    args = parser.parse_args()

    run_benchmark(config_name=args.config, bench_steps=args.steps, warmup_steps=args.warmup)

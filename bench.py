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


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(name: str) -> dict:
    with open(f"configs/{name}.yaml", "r") as f:
        return yaml.safe_load(f)


def load_docs(path: str) -> list[str]:
    with open(path, "r") as f:
        docs = f.read().splitlines()
    return [d for d in docs if d.strip()]


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
    docs   = load_docs("data/code_contests_cpp.txt")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

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

    total_steps = warmup_steps + bench_steps
    for step in range(total_steps):
        # Grab a non-empty doc
        while True:
            doc = docs[doc_idx % len(docs)]
            doc_idx += 1
            ids = tokenizer.encode(doc)[:config["block_size"]]
            if len(ids) >= 2:
                break

        tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits       = model(tokens)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        loss         = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
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
            step_tokens.append(len(ids))
            losses.append(loss.item())
            label = "bench "

        print(f"  [{label}] step {step+1:4d}/{total_steps} | "
              f"loss {loss.item():.4f} | {elapsed*1000:.1f} ms | {len(ids)} tok")

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_ms        = (sum(step_times) / len(step_times)) * 1000
    min_ms        = min(step_times) * 1000
    max_ms        = max(step_times) * 1000
    total_tokens  = sum(step_tokens)
    total_time    = sum(step_times)
    tokens_per_s  = total_tokens / total_time
    steps_per_s   = len(step_times) / total_time
    first_loss    = losses[0]
    last_loss     = losses[-1]
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

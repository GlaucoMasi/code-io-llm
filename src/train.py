"""
train.py — Training loop for CodeIOLLM.

Trains the CodeIOLLM model on the provided dataset, with a linear warmup + cosine decay learning rate schedule.
Checkpoints are saved periodically, and the best checkpoint (lowest validation loss) is also saved separately.

Usage:
    PYTHONPATH=. python3 src/train.py --config small
"""

import argparse
import json
import math
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.tokenizer.BasicTokenizer import BasicTokenizer
from src.model import CodeIOLLM
from src.dataset import create_dataloaders


# ── Config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="small", help="Nome del config in configs/ (es. small)")
args = parser.parse_args()

config_path = f"configs/{args.config}.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# ── LR schedule: linear warmup + cosine decay ─────────────────────────────────

def get_lr(step: int) -> float:
    warmup = config.get("warmup_steps", 100)
    max_lr = float(config["learning_rate"])
    min_lr = float(config.get("min_lr", max_lr * 0.1))
    n      = config["num_steps"]

    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= n:
        return min_lr
    decay = (step - warmup) / (n - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (max_lr - min_lr)


# ── Evaluation step ─────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, val_loader, eval_iters=50):
    model.eval()
    losses = []
    device = next(model.parameters()).device
    # Take a few batches from val_loader
    val_iter = iter(val_loader)
    while len(losses) < eval_iters:
        try:
            x, y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x, y = next(val_iter)
        x, y = x.to(device), y.to(device)
        
        y_flat = y.view(-1)
        if (y_flat != -100).sum() == 0:
            continue
            
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_flat)
        losses.append(loss.item())
        
    model.train()
    return sum(losses) / len(losses) if losses else float('nan')


# ── Run directory for Summaries ───────────────────────────────────────────────

def make_summary_dir(config: dict) -> str:
    """Genera un nome di cartella descrittivo per i log/summary in 'runs/'."""
    n_steps = config['num_steps']
    steps_str = f"{n_steps//1000}ksteps" if n_steps >= 1000 and n_steps % 1000 == 0 else f"{n_steps}steps"

    parts = [
        f"{config['n_layer']}L",
        f"{config['n_embd']}d",
        f"{config['n_head']}H",
        f"bs{config.get('batch_size', 16)}",
        f"lr{config['learning_rate']}",
        steps_str,
        f"vocab{config['vocab_size']}",
        f"block{config['block_size']}",
    ]
    name = "_".join(parts)
    run_dir = os.path.join("runs", name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model: CodeIOLLM, train_loader, val_loader):
    device    = next(model.parameters()).device
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=0.1, betas=(0.9, 0.95))

    num_steps  = config["num_steps"]
    log_every  = config.get("log_every",  100)
    save_every = config.get("save_every", 1000)
    total_params = sum(p.numel() for p in model.parameters())

    os.makedirs("checkpoints", exist_ok=True)
    summary_dir = make_summary_dir(config)
    run_name = os.path.basename(summary_dir)
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training CodeIOLLM for {num_steps:,} steps")
    print(f"  Params: {total_params:,}")
    print(f"  Checkpoints saving to: {ckpt_dir}")
    print(f"  Summaries saving to: {summary_dir}")
    print(f"{'='*60}\n")

    running_loss = 0.0
    best_val_loss = float('inf')
    best_val_step = 0
    loss_history = []  # lista di (step, train_loss, val_loss)
    model.train()

    train_iter = iter(train_loader)
    t_start = time.time()

    for step in range(num_steps):
        # LR schedule
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)                                       # (B, T, vocab)
        y_flat = y.view(-1)
        
        if (y_flat != -100).sum() == 0:
            continue
            
        loss = loss_fn(logits.view(-1, logits.size(-1)), y_flat)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % log_every == 0 or step == 0:
            avg_loss = running_loss / (log_every if step > 0 else 1)
            running_loss = 0.0
            
            val_loss = estimate_loss(model, val_loader)
            loss_history.append({"step": step + 1, "train_loss": round(avg_loss, 4), "val_loss": round(val_loss, 4)})
            print(f"  step {step+1:5d}/{num_steps} | train loss {avg_loss:.4f} | val loss {val_loss:.4f} | lr {lr:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = step + 1
                best_ckpt_path = os.path.join(ckpt_dir, "model_best.pt")
                torch.save({"step": step + 1, "model": model.state_dict(),
                            "config": config, "val_loss": val_loss}, best_ckpt_path)
                print("  ✓ Best checkpoint updated")

        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_step{step+1}.pt")
            torch.save({"step": step + 1, "model": model.state_dict(),
                        "config": config}, ckpt_path)
            print(f"  ✓ Checkpoint saved → {ckpt_path}")

    # Final checkpoint
    final_ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save({"step": num_steps, "model": model.state_dict(),
                "config": config}, final_ckpt_path)
    print(f"\n  ✓ Final checkpoint saved → {final_ckpt_path}")

    elapsed = time.time() - t_start

    # ── Training summary ──────────────────────────────────────────────────────
    summary = {
        "started_at": datetime.now().isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_params": total_params,
        "training_time_seconds": round(elapsed, 1),
        "training_time_human": f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s",
        "final_train_loss": loss_history[-1]["train_loss"] if loss_history else None,
        "final_val_loss": loss_history[-1]["val_loss"] if loss_history else None,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_step": best_val_step,
        "config": config,
        "loss_history": loss_history,
    }
    summary_path = os.path.join(summary_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Training summary saved → {summary_path}")

    return summary_dir


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    tokenizer = BasicTokenizer()
    tokenizer.load("src/tokenizer/test.model")

    model = CodeIOLLM(config).to(device)
    
    # Enable torch.compile for massive throughput speedups where available
    if hasattr(torch, "compile"):
        print("Compiling model for speed (this takes a moment)...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed, continuing with uncompiled model: {e}")

    # Creazione iterators
    train_dl, val_dl = create_dataloaders("data/code_contests_cpp.txt", tokenizer, config["block_size"], batch_size=config.get("batch_size", 16))

    summary_dir = train(model, train_dl, val_dl)

    # ── Inference demo ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  INFERENCE DEMO (after training)")
    print(f"{'='*60}")

    prompts = [
        "#include",
        "int main() {",
        "for (int i = 0;",
    ]

    model.eval()
    
    demo_output = []
    
    for i, prompt in enumerate(prompts):
        if i > 0:
            print()
            demo_output.append("")
            
        msg1 = f"Prompt: {repr(prompt)}\n{'-' * 40}"
        print(msg1)
        demo_output.append(msg1)
        
        out = getattr(model, "_orig_mod", model).generate(
            tokenizer, prompt,
            block_size=config["block_size"],
            max_tokens=150,
            temperature=0.8,
            top_k=40,
            device=device,
        )
        print(out)
        demo_output.append(out)

    demo_path = os.path.join(summary_dir, "inference_demo.txt")
    with open(demo_path, "w") as f:
        f.write("\n".join(demo_output))
    print(f"\n  ✓ Inference demo saved → {demo_path}")
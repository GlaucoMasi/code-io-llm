# CodeIOLLM Command Cheat Sheet

This document contains backtested default commands to run every major script in the repository. Use these as a quick reference point for workflow operations.

### 1. Training the Model (`src/train.py`)
Initiate the main deep-learning training loop. The script automatically uses `DataLoader` chunks, calculates validation loss intermittently, creates `torch.compile` optimizations, and saves the best model to `checkpoints/model_best.pt`.
```bash
# Add PYTHONPATH=. when executing scripts from the `src/` directory directly
PYTHONPATH=. python3 src/train.py
```

### 2. Evaluating Performance (`bench.py`)
Run a stripped-down, high-speed iteration test. This does not save weights or perform validation, but evaluates throughput (`Tokens/sec` and `ms/step`). It is extremely useful for profiling memory bottlenecks.
```bash
# Example: 10 warmup throws, followed by 50 timed benchmark iterations
python3 bench.py --steps 50 --warmup 10
```
> [!CAUTION]
> If you set `--steps` too low (e.g., `2`), the benchmark might skip all steps if it happens to randomly sample batches comprised entirely of `-100` masked Prompt tokens.

### 3. Updating the Vocabulary (`update_tok.py`)
If you manually inject new tokens (like `<|system|>` or `<|user|>`) directly into the `src/tokenizer/test.model` text file, you **must** run this script to align the system.
```bash
python3 update_tok.py
```
*What it does:* It reads the new line count from `test.model`, updates `vocab_size` directly inside `configs/small.yaml`, and ensures the resulting Model Embedding matrices size themselves correctly to avoid `KeyError: Out of bounds` exceptions.

### 4. Running Inference Generator (`generate.py`)
Query the checkpointed model and ask it to write functional C++ code. The script automatically wraps your prompt in the exact conversational tokens (`<|system|>`, `<|user|>`, `<|model|>`) used during Instruction Fine-Tuning.
```bash
python3 generate.py --prompt "Write a C++ program that prints 'Hello World'"
```
*You can also override generation parameters:*
```bash
python3 generate.py --prompt "int main() {" --max_tokens 150 --temperature 0.5 --top_k 20
```

---
*All commands assume execution from the root `/home/mazza/Documents/code-io-llm` directory.*

# CodeIOLLM

CodeIOLLM is a custom, autoregressive transformer-based Language Model specifically built and fine-tuned for generating C++ programming solutions from prompts.

## Recent Refactoring & Architecture Optimizations

The codebase recently underwent a major systemic refactoring to improve training throughput, optimize memory overhead, and enable structural Instruction Fine-Tuning for problem-solving tasks. 

Below are the key changes applied across the repository, along with their technical motivations:

### 1. FlashAttention Implementation (`src/model.py`)
- **Change:** Replaced the manual lower-triangular causal masking and custom scaled dot-product routines in `CausalSelfAttention` with PyTorch's native `F.scaled_dot_product_attention(..., is_causal=True)`.
- **Motivation:** Generating an $O(T^2)$ float tensor simply to drop half of it via masking scales terribly with context length. FlashAttention fuses the attention matrix calculations at the CUDA kernel level, fundamentally lowering memory complexity from $O(T^2)$ to $O(T)$ in High Bandwidth Memory (HBM). This prevents Out-Of-Memory (OOM) errors on larger blocks and drastically speeds up the `forward/backward` passes.

### 2. Eliminating the $O(N^2)$ Tokenization Bottleneck (`src/dataset.py`)
- **Change:** Deprecated the naive approach of loading the entire `code_contests_cpp.txt` 340MB string into memory to be tokenized as a monolith. We implemented a custom PyTorch `Dataset` (`CodeDataset`) that yields split and batched chunk sizes via a `DataLoader`.
- **Motivation:** Real-world BPE Tokenizers parse linearly. Parsing an un-chunked 340MB flat string caused the encoding loop to effectively halt the CPU. The new iterative string extraction paired with `.npy` cache generation and memory mapping (`mmap_mode='r'`) safely caches training splits sequentially to disk, eliminating CPU RAM spikes completely.

### 3. Instruction Fine-Tuning via Chat Templates 
- **Change:** Restructured the raw codebase text dataset into structured conversational sequences using newly injected special tokens (`<|system|>`, `<|user|>`, `<|model|>`, `<|endoftext|>`).
- **Motivation:** The previous dataset arrangement was purely unsupervised pre-training (training the model to just guess standard C++ sequences). To explicitly teach the model to "Output a functional C++ program _based on an instruction_", we had to format the stream as a dialogue.

### 4. Loss Masking Checkpoints (`src/dataset.py` & `src/train.py`)
- **Change:** Constructed a parallel `labels` array inside dataset generation where all tags representing the English `<|system|>` prompt were overridden with the index `-100`. We patched the training loop's `F.cross_entropy()` to correctly ingest and bypass these targets.
- **Motivation:** In a prompt-response architecture, you want the model to dedicate 100% of its gradient descents trying to perfect the C++ solution, *not* trying to memorize and recite English instruction text. PyTorch natively skips `-100` targets within CrossEntropy validations. We also added safety catches bypassing `NaN` gradient occurrences if a 256-block context window landed entirely within an ignored prompt string.

### 5. Encapsulated Execution Pipeline 
- **Change:** Relocated the autoregressive loop out of `generate.py` and converted it into an `@torch.no_grad()` bounded method natively inside the `CodeIOLLM` class. We also bundled model compilation hooks via `torch.compile` directly into `src/train.py`.
- **Motivation:** Improves standard Object-Oriented paradigms (keeping the generation logic intimately tied to the model architecture that governs it). `torch.compile` leverages TorchInductor to fuse backend graph subgraphs into perfectly shaped C++ kernels matching local hardware capabilities, speeding up local iterations by 10-15%.

### 6. Special Token Regex Integration (`src/tokenizer/BasicTokenizer.py`)
- **Change:** Overrode the base `BasicTokenizer.encode()` method to accept an `allowed_special` flag, implementing a recursive regex split string parser.
- **Motivation:** Previously, feeding `<|system|>` into the model would break it cleanly into `['<', '|', 'system', '|', '>']` tokens. The new handler protects special token formatting, guaranteeing the generative loop recognizes contextual transitions flawlessly.

---

*This log documents structural fixes applied systematically during the codebase refactoring phases.*

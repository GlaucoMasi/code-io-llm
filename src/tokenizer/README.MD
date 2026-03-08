Run metrics
```python
cd src/tokenizer
python3 -m tests.metrics
```
Will output something like
```
Training time: 3.8065s
Merges dictionary len: 244
Vocab count: 500
```

Run tokenizer training
```python
python3 src/tokenizer/train/train.py
```
This will create:
- `test.model` file which is what LLM will use as token vocabulary
- `test.vocab` for human readability purposes, it shows first the list of (character, token-id) and the the pairs [string][string] -> token_id that were merged during tokenizer training. For example `[in][t] -> [int] 306` means that token `in` and `t` to form `int` with `token_id` 306


## Extracting C++ Solutions from CodeContests

`code_contests_cpp.txt` contains 100 correct C++ solutions extracted from the
[DeepMind CodeContests](https://github.com/google-deepmind/code_contests) dataset
(sourced via the HuggingFace mirror `deepmind/code_contests`).

### Source

| Dataset | Split | HuggingFace ID |
|---------|-------|----------------|
| CodeContests | train | `deepmind/code_contests` |

Fields used per problem:
- `solutions.language` — filtered for `CPP` (value `2`)
- `solutions.solution` — the raw C++ source code
- `name`, `source` — written as a comment header before each snippet

### Steps

1. **Install `datasets`** — if not present, the script auto-installs it via pip.
2. **Stream the train split** — uses HuggingFace streaming mode (no full 3 GiB download).
3. **Filter C++ solutions** — iterates problems and collects entries where `language == 2` (CPP).
4. **Cap at 100** — stops once 100 unique, non-empty snippets are gathered.
5. **Write `code_contests_cpp.txt`** — each solution is preceded by:
   ```
   // --- Solution NNN ---
   // Problem : <problem name>
   // Source  : <CODEFORCES | ATCODER | …>
   ```
   and followed by a `=`×72 separator line.

### Run

```bash
python3 train/merge_cc_solutions.py
```


TODO
Add tokenizer save() tests
Add regex like training
Improve N^2 complexity of training to linear time


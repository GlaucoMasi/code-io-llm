"""Extract 100 C++ correct solutions from deepmind/code_contests (HuggingFace).

Usage:
    python3 merge_cc_solutions.py

Requires:
    pip install datasets

Output:
    code_contests_cpp.txt  — 100 C++ solutions merged into a single text file,
    each preceded by a comment header and separated by a divider line.
"""

import subprocess
import sys

# ---------------------------------------------------------------------------
# Auto-install datasets if missing
# ---------------------------------------------------------------------------
try:
    import datasets  # noqa: F401
except ImportError:
    print("Installing 'datasets' library…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    import datasets  # noqa: F401

from datasets import load_dataset

BOS = 501

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_FILE = "code_contests_cpp.txt"
MAX_SOLUTIONS = 100
CPP_LANGUAGE = 2        # Language enum value for C++ in contest_problem.proto
SEPARATOR = "\n" + "=" * 72 + "\n"

# Source enum → human-readable label (from contest_problem.proto)
SOURCE_NAMES = {
    0: "UNKNOWN",
    1: "CODECHEF",
    2: "CODEFORCES",
    3: "HACKEREARTH",
    4: "CODEJAM",
    6: "ATCODER",
    7: "AIZU",
}

# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading deepmind/code_contests (train split, streaming mode)…")
    ds = load_dataset(
        "deepmind/code_contests",
        split="train",
        streaming=True,
    )

    collected: list[str] = []

    for problem in ds:
        if len(collected) >= MAX_SOLUTIONS:
            break

        name   = problem.get("name", "?")
        description = problem.get("description", "No description found")

        source = SOURCE_NAMES.get(problem.get("source", 0), "UNKNOWN")
        solutions = problem.get("solutions", {})

        # solutions is a dict-of-lists: {"language": [...], "solution": [...]}
        languages = solutions.get("language", [])
        codes     = solutions.get("solution", [])

        # Pick the FIRST valid C++ solution for this problem, then move on.
        for lang, code in zip(languages, codes):
            if lang == CPP_LANGUAGE and code and code.strip():
                idx = len(collected) + 1
                header = (
                    f"{description}\n"
                )
                collected.append(header + code.strip())
                print(f"  [{idx:3d}/{MAX_SOLUTIONS}] {source} — {name[:60]}")
                break  # one solution per problem

    print(f"\nWriting {len(collected)} solutions to {OUTPUT_FILE} …")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(SEPARATOR.join(collected))
        f.write("\n" + "=" * 72 + "\n")  # trailing divider

    print(f"Done: wrote {len(collected)} C++ solutions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

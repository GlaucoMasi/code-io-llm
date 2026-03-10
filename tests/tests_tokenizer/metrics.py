import time
from pathlib import Path

from src.tokenizer import BasicTokenizer

TESTS_DIR = Path(__file__).parent


def compression_basic(file_path: Path) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        contents = f.read()

    tokenizer = BasicTokenizer()
    start_time = time.time()
    tokenizer.train(contents, vocab_size=500)
    print(f"Training time: {(time.time() - start_time):.4f}s")
    print(f"Merges dictionary len: {len(tokenizer.merges)}")
    print(f"Vocab count: {len(tokenizer.vocab)}")

    ids = tokenizer.encode(contents)
    assert(len(ids) != 0)
    compression_ratio = len(contents) / len(ids)
    print(f"Compression ratio: {compression_ratio:.4}")


if __name__ == "__main__":
    compression_basic(TESTS_DIR / "taylorswift.txt")

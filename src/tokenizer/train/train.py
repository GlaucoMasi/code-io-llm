from src.tokenizer import BasicTokenizer
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent

def train(input_file, output_file_prefix, vocab_size):
    with open(input_file, "r", encoding="utf-8") as f:
        contents = f.read()

    tokenizer = BasicTokenizer()
    start_time = time.time()
    tokenizer.train(contents, vocab_size=vocab_size)
    print(f"Training time: {(time.time() - start_time):.4f}s")
    print(f"Merges dictionary len: {len(tokenizer.merges)}")
    print(f"Vocab count: {len(tokenizer.vocab)}")
    tokenizer.save(output_file_prefix)
    print("Toeen model saved")


if __name__ == "__main__":
    train(input_file=TESTS_DIR / "code_contests_cpp.txt",
          output_file_prefix="test",
          vocab_size=500)
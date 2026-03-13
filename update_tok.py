import yaml
from src.tokenizer.BasicTokenizer import BasicTokenizer

def main():
    t = BasicTokenizer()
    t.load("src/tokenizer/test.model")
    
    # Base vocabulary is 256 bytes + the trained merges
    base_vocab_size = 256 + len(t.merges)
    print(f"Base (bytes + merges) size: {base_vocab_size}")
    
    special_tokens = ["<|system|>", "<|user|>", "<|model|>", "<|endoftext|>"]
    for i, st in enumerate(special_tokens):
        # Always assign deterministically to avoid shifting IDs on re-runs
        t.special_tokens[st] = base_vocab_size + i
        
    t.vocab = t._build_vocab()
    
    # Calculate exactly what nn.Embedding needs (max index + 1)
    new_vocab_size = max(t.vocab.keys()) + 1
    print(f"New Configured Vocab Size (max_index + 1): {new_vocab_size}")
    
    t.save("src/tokenizer/test")
    
    # Update config
    with open("configs/small.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    config["vocab_size"] = new_vocab_size
    
    with open("configs/small.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print("Updated small.yaml with new vocab_size")

if __name__ == "__main__":
    main()

import yaml
from src.tokenizer.BasicTokenizer import BasicTokenizer

def main():
    t = BasicTokenizer()
    t.load("src/tokenizer/test.model")
    
    current_vocab_size = len(t.vocab)
    print(f"Current vocab size: {current_vocab_size}")
    
    special_tokens = ["<|system|>", "<|user|>", "<|model|>", "<|endoftext|>"]
    for i, st in enumerate(special_tokens):
        t.special_tokens[st] = current_vocab_size + i
        
    t.vocab = t._build_vocab()
    
    new_vocab_size = len(t.vocab)
    print(f"New vocab size: {new_vocab_size}")
    
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

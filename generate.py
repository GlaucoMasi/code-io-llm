"""
generate.py — Autoregressive inference for CodeIOLLM.

Loads a saved checkpoint and generates C++ code from a prompt.

Usage:
    python3 generate.py --prompt "#include" --max_tokens 200 --temperature 0.8 --top_k 40
"""

import argparse
import yaml
import torch
import torch.nn.functional as F

from src.tokenizer.BasicTokenizer import BasicTokenizer
from src.model import CodeIOLLM


def load_model(checkpoint_path: str, config: dict, device: str) -> CodeIOLLM:
    model = CodeIOLLM(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    
    # Strip "_orig_mod." prefix added by torch.compile
    state_dict = state["model"]
    uncompiled_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        uncompiled_state_dict[new_k] = v
        
    model.load_state_dict(uncompiled_state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate C++ code with CodeIOLLM")
    parser.add_argument("--prompt",      type=str,   default="Write a C++ program to reverse an array.")
    parser.add_argument("--checkpoint",  type=str,   default="checkpoints/model_best.pt")
    parser.add_argument("--config",      type=str,   default="small")
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=40)
    args = parser.parse_args()

    with open(f"configs/{args.config}.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BasicTokenizer()
    tokenizer.load("src/tokenizer/test.model")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params) on {device}\n")

    sys_text = "<|system|>\nYou are an expert C++ competitive programmer. Write functionally correct C++ code to solve the following problem.\n<|user|>\n"
    formatted_prompt = f"{sys_text}{args.prompt}\n<|model|>\n"

    print("=" * 60)
    print("PROMPT:")
    print(formatted_prompt)
    print("-" * 60)
    print("GENERATED:")
    output = model.generate(
        tokenizer, formatted_prompt,
        block_size=config["block_size"],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()

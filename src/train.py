
import torch
import torch.nn as nn

import yaml

from src.tokenizer.BasicTokenizer import BasicTokenizer
from src.model import CodeIOLLM

def train(model, configs):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])

    with open('data/code_contests_cpp.txt', 'r') as f:
        docs = f.read().splitlines()

    # Repeat in sequence
    num_steps = configs['num_steps']
    tokenizer = BasicTokenizer()
    tokenizer.load('src/tokenizer/test.model')

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        # TODO Mettere BOS e EOS
        # Take single document, tokenize it, surround it with BOS special token on both sides
        doc = docs[step % len(docs)]

        ids = tokenizer.encode(doc)
        # Truncate to block_size to avoid position embedding out-of-bounds
        ids = ids[:configs['block_size']]
        # Need at least 2 tokens to form (input, target) pairs
        if len(ids) < 2:
            continue
        tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        logits = model(tokens)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()

        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Backward the loss, calculating the gradients with respect to all model parameters.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")


with open('configs/small.yaml', 'r') as file:
    config_small = yaml.safe_load(file)
    
model = CodeIOLLM(config_small)
train(model, config_small)
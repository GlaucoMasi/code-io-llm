import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CodeDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, block_size: int, split: str = "train", test_size: float = 0.1, cache_dir: str = "data/cache"):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(data_path)
        cache_path_ids = os.path.join(cache_dir, f"{filename}_{split}_ids.npy")
        cache_path_labels = os.path.join(cache_dir, f"{filename}_{split}_labels.npy")

        if os.path.exists(cache_path_ids) and os.path.exists(cache_path_labels):
            print(f"Loading {split} split from {cache_path_ids}...")
            self.data_ids = np.load(cache_path_ids, mmap_mode='r')
            self.data_labels = np.load(cache_path_labels, mmap_mode='r')
        else:
            print(f"Tokenizing {data_path} for {split} split with Chat Templates and Loss Masking...")
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()

            docs = text.split("========================================================================")
            docs = [d.strip() for d in docs if d.strip()]

            split_idx = int(len(docs) * (1 - test_size))
            if split == "train":
                docs = docs[:split_idx]
            else:
                docs = docs[split_idx:]

            sys_id = tokenizer.special_tokens.get("<|system|>", 500)
            user_id = tokenizer.special_tokens.get("<|user|>", 501)
            model_id = tokenizer.special_tokens.get("<|model|>", 502)
            end_id = tokenizer.special_tokens.get("<|endoftext|>", 503)

            sys_text = "\nYou are an expert C++ competitive programmer. Write functionally correct C++ code to solve the following problem.\n"
            sys_text_ids = tokenizer.encode(sys_text)

            all_ids = []
            all_labels = []

            for d in docs:
                parts = d.split("#include", 1)
                if len(parts) == 2:
                    prompt_text = parts[0].strip()
                    solution_text = "#include" + parts[1]
                else:
                    prompt_text = d
                    solution_text = ""
                
                prompt_ids = tokenizer.encode("\n" + prompt_text + "\n")
                solution_ids = tokenizer.encode("\n" + solution_text + "\n") if solution_text else []
                
                # Assemble the instruction tuning template
                block_ids = [sys_id] + sys_text_ids + [user_id] + prompt_ids + [model_id]
                
                # Mask out the prompt from the loss (use -100 so CrossEntropy skips them)
                block_labels = [-100] * len(block_ids)
                
                if solution_ids:
                    block_ids.extend(solution_ids + [end_id])
                    block_labels.extend(solution_ids + [end_id])
                
                all_ids.extend(block_ids)
                all_labels.extend(block_labels)

            all_ids = np.array(all_ids, dtype=np.uint16)
            all_labels = np.array(all_labels, dtype=np.int32)
            
            np.save(cache_path_ids, all_ids)
            np.save(cache_path_labels, all_labels)
            
            self.data_ids = np.load(cache_path_ids, mmap_mode='r')
            self.data_labels = np.load(cache_path_labels, mmap_mode='r')

        self.n_tokens = len(self.data_ids)
        print(f"  Loaded {split} split: {self.n_tokens:,} tokens")

    def __len__(self):
        return self.n_tokens - self.block_size

    def __getitem__(self, i):
        chunk_ids = self.data_ids[i: i + self.block_size + 1].astype(np.int64)
        chunk_labels = self.data_labels[i: i + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk_ids[:-1])
        y = torch.from_numpy(chunk_labels[1:])
        return x, y

def create_dataloaders(data_path, tokenizer, block_size, batch_size=4, num_workers=2):
    train_ds = CodeDataset(data_path, tokenizer, block_size, split="train")
    val_ds = CodeDataset(data_path, tokenizer, block_size, split="val")

    # L'uso di shuffle in train farà sì che si sceglieranno offset causali del tutto randomizzati in memoria!
    # Questo è corretto per il fine-tuning o pretraining AR (Autoregressive).
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl

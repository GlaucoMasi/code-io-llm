import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Garantisce che gli output per una determinata posizione in una sequenza siano basati solo sugli output noti nelle posizioni precedenti e non sulle posizioni future
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Configs
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]

        # Proiezione Q, K, V + Proiezioni di output
        self.c_attn = nn.Linear(config["n_embd"], 3 * self.n_embd)
        self.c_proj = nn.Linear(config["n_embd"], self.n_embd)

        # La maschera manuale non serve più grazie a F.scaled_dot_product_attention()
        # mask = torch.tril(torch.ones(config["block_size"], config["block_size"]))
        # self.register_buffer("mask", mask.view(1, 1, config["block_size"], config["block_size"]))

    def forward(self, x):
        # B: batch_size, T: lunghezza sequenza, C: n_embd
        B, T, C = x.size()

        # Calcolo di Q, K e V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # C splittato in n_head teste. Ogni testa analizza una parte diversa degli embedding
        head_dim = C // self.n_head
        # Con la trasposizione si CausalSelfAttentionpassa da (B, T, n_head, head_dim) a (B, n_head, T, head_dim) per il prodotto
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # FlashAttention equivalente: calcola attenzioni in modo efficiente ed evita di instanziare O(N^2) memoria
        # is_causal=True applica automaticamente la maschera triangolare inferiore
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Con la trasposizione si passa da (B, n_head, T, head_dim) a (B, T, C). Anche detto Ricongiungimento
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)

# Una volta create le relazioni tra i token, ora si elaborano le informazioni 
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Espansione di 4 volte la dimensione degli embedding per aumentare la capacità del modello
        self.c_esp = nn.Linear(config["n_embd"], 4 * config["n_embd"])
        
        # Funzione di attivazione standard
        self.gelu = nn.GELU()

        # Contrazione alla dimensione originale degli embedding
        self.c_contr = nn.Linear(4 * config["n_embd"], config["n_embd"])

    def forward(self, x):
        x = self.c_esp(x)
        x = self.gelu(x)
        x = self.c_contr(x)
        return x
        
# Un blocco Transformer combina attenzione, feedforward, normalizzazione e residual connection
# La normalizzazione prende i dati e li normalizza in modo che abbiano media 0 e variaza 1
# La residual connection aggiunge l'input originale all'output del blocco, invece di trasformarlo completamente
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        
        self.ln2 = nn.LayerNorm(config["n_embd"])
        self.ffn = FeedForward(config)

    def forward(self, x):
        # Norm applicata prima dell'attenzione per rendere il training più stabile
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class CodeIOLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Embedding del vocabolario, da token a vettori
        self.token_embedding = nn.Embedding(config["vocab_size"], config["n_embd"])

        # Embedding della posizione, da posizione a vettori
        self.position_embedding = nn.Embedding(config["block_size"], config["n_embd"])

        # Stack di N TransformerBlocks
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layer"])])

        # Normalizzazione finale e Testa di output
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"])

    def forward(self, idx):
        B, T = idx.shape

        # Indici di posizione, 0-based. Mantenendo lo stesso device dell'input
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Somma degli embedding: (Token + Posizione)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        # Forward pass attraverso i blocchi
        x = self.blocks(x)

        # Normalizzazione finale e Logits per la decisione finale
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        prompt: str,
        block_size: int,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        device: str = "cpu",
    ) -> str:
        """
        Genera codice ricorsivamente partendo dal prompt.
        """
        self.eval()
        # Allow special tokens so format labels like <|system|> are recognized
        ids = tokenizer.encode(prompt, allowed_special="all")
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        for _ in range(max_tokens):
            # Crop to block_size context window
            x_cond = x[:, -block_size:]
            logits = self(x_cond)           # (1, T, vocab) the model outputs logits for every position in the prompt, but we only care about the last one
            logits = logits[:, -1, :]       # (1, vocab), here we take only the last one
            
            # Mask out tokens that are not in the vocabulary
            vocab_size = len(tokenizer.vocab)
            if logits.size(-1) > vocab_size:
                logits[:, vocab_size:] = float("-inf")
            elif logits.size(-1) < vocab_size:
                raise ValueError(f"Model vocab size ({logits.size(-1)}) is smaller than tokenizer vocab size ({vocab_size})")

            if temperature > 0.0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1), next token is chosen among this probability distribution (beam search)  instead of taking only the most probable one
            x = torch.cat([x, next_token], dim=1)
            
            # Optional: early stopping check if you introduce a particular token, not needed here
            
        self.train()
        return tokenizer.decode(x[0].tolist())
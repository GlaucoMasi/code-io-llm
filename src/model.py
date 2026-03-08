import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Garantisce che gli output per una determinata posizione in una sequenza siano basati solo sugli output noti nelle posizioni precedenti e non sulle posizioni future
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Configs
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Proiezione Q, K, V + Proiezioni di output
        self.c_attn = nn.Linear(config.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(config.n_embd, self.n_embd)

        # Maschera causale, triangolare inferiore, in modo che il softmax escluda interazioni con token futuri 
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        # register_buffer indica che questo non è un parametro da imparare, è una costante
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

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

        # Attention score. Quanto la query è compatibile con la key 
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

        # Causal masking
        attention = attention.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # Probabilità normalizzate affinchè sommino 1
        attention = F.softmax(attention, dim=3)

        # Contesto pesato considerando la Value 
        y = attention @ v

        # Con la trasposizione si passa da (B, n_head, T, head_dim) a (B, T, C). Anche detto Ricongiungimento
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)

# Una volta create le relazioni tra i token, ora si elaborano le informazioni 
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Espansione di 4 volte la dimensione degli embedding per aumentare la capacità del modello
        self.c_esp = nn.Linear(config.n_embd, 4 * config.n_embd)
        
        # Funzione di attivazione standard
        self.gelu = nn.GELU()

        # Contrazione alla dimensione originale degli embedding
        self.c_contr = nn.Linear(4 * config.n_embd, config.n_embd)

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
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # Norm applicata prima dell'attenzione per rendere il training più stabile
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class CodeIOLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, idx, targets=None):
        return idx
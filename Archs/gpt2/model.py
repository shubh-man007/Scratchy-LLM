import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt

@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 256
    emb_dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    drop_rate: float = 0.1
    qkv_bias: bool = False



class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.token_ids = []
        self.target_ids = []
        
        encoded_text = tokenizer.encode(text, allowed_special = {"<|EOS|>"})
        assert len(encoded_text) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"
        
        for idx in range(0, len(encoded_text) - max_length, stride):
            self.token_ids.append(torch.tensor(encoded_text[idx:idx+max_length]))
            self.target_ids.append(torch.tensor(encoded_text[idx+1:idx+max_length+1]))
        
    def __len__(self):
        return len(self.token_ids)
    
    def __getitem__(self, idx):
        return self.token_ids[idx], self.target_ids[idx]
    
    
     
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_head = d_out // num_heads # d_out = num_heads * d_head
        self.d_in = d_in
        self.W_q = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1))
        
    def forward(self, batch):
        b, num_tokens, d = batch.size()
        
        # (b, num_tokens, d) ---> (b, num_tokens, d_out) = (b, num_tokens, num_heads * d_head)
        q = self.W_q(batch)
        k = self.W_k(batch)
        v = self.W_v(batch)
        
        # (b, num_tokens, d_out) ---> (b, num_tokens, num_heads, d_head):
        q = q.view(b, num_tokens, self.num_heads, self.d_head)
        k = k.view(b, num_tokens, self.num_heads, self.d_head)
        v = v.view(b, num_tokens, self.num_heads, self.d_head)
        
        # (b, num_tokens, num_heads, d_head) ---> (b, num_heads, num_tokens, d_head):
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
        
        # (b, num_heads, num_tokens, d_head) @ (b, num_heads, d_head, num_tokens) ---> (b, num_heads, num_tokens, num_tokens):
        attn_scores = q @ torch.transpose(k, -2, -1)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
        
        # attention weights
        # (b, num_heads, num_tokens, num_tokens):
        attn_weights = torch.softmax(attn_scores / self.d_head ** 0.5, dim = -1)
        
        # dropout
        # (b, num_heads, num_tokens, num_tokens):
        attn_weights = self.dropout(attn_weights)
        
        # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, d_head) ---> (b, num_heads, num_tokens, d_head):
        z = attn_weights @ v
        
        # (b, num_heads, num_tokens, d_head) ---> (b, num_tokens, num_heads, d_head):
        z = torch.transpose(z, 1, 2)
        
        # (b, num_tokens, num_heads, d_head) ---> (b, num_tokens, d_out) = (b, num_tokens, num_heads * d_head):
        z = z.contiguous().view(b, num_tokens, self.num_heads * self.d_head)
        
        # Outward projection (optional)
        z = self.out_proj(z)
        
        return z
    
    
    
class LayerNorm(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.d_out = d_out
        self.epsilon = 1e-5
        self.gamma = nn.Parameter(torch.ones(d_out), requires_grad = True)
        self.beta = nn.Parameter(torch.zeros(d_out), requires_grad = True)
        
    def forward(self, batch):
        mean = batch.mean(dim = -1, keepdim = True)
        var = batch.var(dim = -1, keepdim = True)
        batch_ = (batch - mean) / ((var**0.5) + self.epsilon)
        batch_ = self.gamma * batch_ + self.beta
        return batch_
    
    
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x_ = torch.tensor(0.5)*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x+torch.tensor(0.044715)*(x**3))))
        return x_
    
    

# MLP Layer:   
class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
    
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.mha = MultiHeadAttention(
            cfg.n_heads,
            cfg.emb_dim,
            cfg.emb_dim,
            cfg.context_length,
            cfg.drop_rate,
            cfg.qkv_bias
        )
        self.layer_norm_1 = LayerNorm(cfg.emb_dim)
        self.layer_norm_2 = LayerNorm(cfg.emb_dim)
        self.ffn = FeedForward(cfg)
        self.drop_conn = nn.Dropout(cfg.drop_rate)
        
    def forward(self, batch):
        z = self.layer_norm_1(batch)
        z_attn = self.mha(z)
        z_drop_1 = self.drop_conn(z_attn)
        z_add_drop = batch + z_drop_1
        z_norm = self.layer_norm_2(z_add_drop)
        z_ffn = self.ffn(z_norm)
        z_drop_2 = self.drop_conn(z_ffn)
        z_ = z_drop_2 + z_add_drop
        return z_
    
    
        
class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_enc = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        
        self.TBlock = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        self.emb.weight = self.out_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                nn.init.normal_(p, mean = 0.0, std = 0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.emb(in_idx)
        pos_embeds = self.pos_enc(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # (batch_size, num_tokens, emb_size)
        x = self.drop_emb(x)
        x = self.TBlock(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits # (batch_size, num_tokens, vocab_size)
    
    

    @torch.no_grad() 
    def generate(self, idx, max_new_tokens, context_size, temperature = 1.0, top_k = None, greedy = False):
        # idx is (batch, n_tokens) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]
            
            # Get the predictions
            logits = self(idx_cond)
            
            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]  

            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
            if greedy:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (batch, 1)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx

def visualize_model_weights(model: nn.Module, save_path: str = None):
    """
    Visualize the distribution of model weights for each module.
    Optionally save the plot to a file.
    """
    weights = []
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights.append(param.data.cpu().numpy().flatten())
            names.append(name)
    plt.figure(figsize=(12, 6))
    for w, n in zip(weights, names):
        plt.hist(w, bins=50, alpha=0.5, label=n)
    plt.legend()
    plt.title('Model Weights Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model weights histogram saved as {save_path}")
    plt.show()

def print_model_summary(model: nn.Module):
    print("\nModel Summary:")
    print("="*60)
    for name, module in model.named_modules():
        if name == "":
            continue
        print(f"Module: {name:40s} | Type: {type(module).__name__}")
    print("="*60)

import math
import torch
import torch.nn as nn
from .attention import Attention, precompute_freqs_cis
from .ffn import SwiGLU, RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, multiple_of: int, norm_eps: float):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(dim, n_heads, n_kv_heads)
        self.feed_forward = SwiGLU(dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class AxiomV2(nn.Module):
    """The Axiom v2 500M Core Engine Architecture"""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, 
                 n_kv_heads: int, max_seq_len: int, multiple_of: int, norm_eps: float, rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, multiple_of, norm_eps)
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len * 2, theta=rope_theta)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)

        for layer in self.layers:
            h = layer(h, freqs_cis)
            
        h = self.norm(h)
        output = self.output(h)
        return output

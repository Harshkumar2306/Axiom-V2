import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
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

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache=None):
        attn_out, new_kv_cache = self.attention(self.attention_norm(x), freqs_cis, kv_cache)
        h = x + attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, new_kv_cache

class AxiomV2(nn.Module):
    """The Axiom v2 500M Core Engine Architecture"""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, 
                 n_kv_heads: int, max_seq_len: int, multiple_of: int, norm_eps: float, rope_theta: float,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing
        
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, multiple_of, norm_eps)
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # Registered as a non-persistent buffer: it follows .to(device) moves
        # automatically (no per-forward device copy) but stays out of the
        # state_dict, keeping checkpoints small and load-compatible.
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(d_model // n_heads, max_seq_len * 2, theta=rope_theta),
            persistent=False,
        )

        self.apply(self._init_weights)

        # GPT-style scaled init for residual-branch projections (attention wo,
        # FFN w2): keeps residual-stream variance ~constant as depth grows,
        # which noticeably stabilises early training of deep networks.
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for name, param in self.named_parameters():
            if name.endswith(("wo.weight", "w2.weight")):
                torch.nn.init.normal_(param, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, kv_cache=None, return_cache: bool = False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Explicit 4096-token boundary handling (Test 3)
        if start_pos + seqlen > self.max_seq_len:
            raise ValueError(f"Context window overflow: {start_pos + seqlen} > {self.max_seq_len}")
            
        # Buffer already lives on the model's device; slicing is a view (no copy).
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False is the recommended way for PyTorch >= 1.11
                # Checkpoint doesn't easily support multiple return values, but during training kv_cache is always None
                h, _ = torch.utils.checkpoint.checkpoint(layer, h, freqs_cis, None, use_reentrant=False)
            else:
                h, cache = layer(h, freqs_cis, layer_cache)
                new_kv_caches.append(cache)
            
        h = self.norm(h)
        output = self.output(h)
        
        if return_cache:
            return output, new_kv_caches
        return output

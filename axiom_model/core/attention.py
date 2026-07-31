import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    """Precompute the frequency tensor for complex exponentials (RoPE)"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply Rotary Positional Embeddings to queries and keys"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def _sdpa_supports_enable_gqa() -> bool:
    """F.scaled_dot_product_attention gained native GQA support in PyTorch 2.5.

    Detection strategy: parse the torch version first (cheap), and if that is
    inconclusive, run a tiny functional probe — some builds (e.g. CPU wheels)
    expose SDPA as a C builtin with no introspectable Python signature.
    """
    try:
        major, minor = (int(p) for p in torch.__version__.split('+')[0].split('.')[:2])
        return (major, minor) >= (2, 5)
    except (ValueError, AttributeError):
        pass
    try:
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return True
    except Exception:
        return False

class Attention(nn.Module):
    """Grouped-Query Attention with FlashAttention-2 Fallback"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # Native GQA (PyTorch >= 2.5) lets SDPA expand K/V internally, but
        # on T4 GPUs (compute capability 7.5), Memory Efficient Attention does
        # not support asymmetric K/V shapes, forcing a slow Math fallback.
        # We disable it here to restore the 3x speedup via manual repeat_interleave.
        self._use_native_gqa = False

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not self._use_native_gqa:
            # GQA repeat (fallback for PyTorch < 2.5)
            num_key_value_groups = self.n_heads // self.n_kv_heads
            if num_key_value_groups > 1:
                xk = xk.repeat_interleave(num_key_value_groups, dim=2)
                xv = xv.repeat_interleave(num_key_value_groups, dim=2)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # FlashAttention-2 Fallback
        # PyTorch 2.x will automatically use FlashAttention-2 if available when using scaled_dot_product_attention
        if self._use_native_gqa:
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True, enable_gqa=True)
        else:
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

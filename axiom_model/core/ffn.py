import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Normalization (with Fused fallback expectation)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # In a real environment with Triton/xFormers, you would call a fused kernel here.
        # E.g. from xformers.ops.fmha.attn_bias import ...
        # IMPORTANT: compute the normalisation statistics in float32. In fp16
        # the sum of squares can overflow to inf (activations grow with depth),
        # which silently turns rsqrt(inf) into 0 and NaNs the whole run.
        input_dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(input_dtype)

class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (with Fused potential)."""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # Round up to multiple_of to ensure memory alignment
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        # Fused SwiGLU implementations would replace this block:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

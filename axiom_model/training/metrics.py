import torch
import torch.nn.functional as F
import math

class MetricsCalculator:
    @staticmethod
    def compute_loss(logits, targets):
        # Assumes logits are [B, T, V] and targets are [B, T]
        # Check if the entire sequence is masked (which happens if long prompts get truncated)
        if not (targets != -100).any():
            return logits.sum() * 0.0
            
        # Flatten for cross_entropy
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    @staticmethod
    def compute_perplexity(loss):
        try:
            return math.exp(loss.item())
        except OverflowError:
            return float('inf')
            
    @staticmethod
    def compute_grad_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

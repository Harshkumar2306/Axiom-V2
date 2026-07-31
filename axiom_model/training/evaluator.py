import torch
import torch.distributed as dist
from .metrics import MetricsCalculator

class Evaluator:
    def __init__(self, model, val_loader, device, is_distributed=True, config=None):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.is_distributed = is_distributed
        self.config = config or {}
        
    @torch.no_grad()
    def evaluate(self, num_batches=100):
        self.model.eval()
        total_loss = torch.zeros(1, device=self.device)
        batches_run = 0
        
        # Using an explicit iterator avoids fully traversing the validation set every eval phase
        val_iter = iter(self.val_loader)
        for _ in range(num_batches):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break
                
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Mixed precision context for evaluation if available.
            # (Autocast is CUDA/MPS-only; CPU evaluates in fp32.)
            if self.device.type in ('cuda', 'mps'):
                d_str = self.config.get('training', {}).get('dtype', 'float16') if 'training' in self.config else self.config.get('dtype', 'float16')
                dt = torch.bfloat16 if d_str == 'bfloat16' else (torch.float32 if d_str == 'float32' else torch.float16)
                with torch.autocast(device_type=self.device.type, dtype=dt):
                    logits = self.model(x)
                    loss = MetricsCalculator.compute_loss(logits, y)
            else:
                logits = self.model(x)
                loss = MetricsCalculator.compute_loss(logits, y)
                
            total_loss += loss
            batches_run += 1
            
        if self.is_distributed and dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss /= dist.get_world_size()
            
        self.model.train()
        avg_loss = (total_loss.item() / batches_run) if batches_run > 0 else float('inf')
        perplexity = MetricsCalculator.compute_perplexity(torch.tensor(avg_loss))
        
        return avg_loss, perplexity

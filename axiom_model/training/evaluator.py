import torch
import torch.distributed as dist
from .metrics import MetricsCalculator

class Evaluator:
    def __init__(self, model, val_loader, device, is_distributed=True):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.is_distributed = is_distributed
        
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
                
            x, y = x.to(self.device), y.to(self.device)
            
            # Mixed precision context for evaluation if available
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
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

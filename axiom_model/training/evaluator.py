import torch
from .metrics import MetricsCalculator

class Evaluator:
    def __init__(self, model, val_loader, device):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        
    @torch.no_grad()
    def evaluate(self, num_batches=100):
        self.model.eval()
        total_loss = 0.0
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
                
            total_loss += loss.item()
            batches_run += 1
            
        self.model.train()
        avg_loss = total_loss / batches_run if batches_run > 0 else float('inf')
        perplexity = MetricsCalculator.compute_perplexity(avg_loss)
        
        return avg_loss, perplexity

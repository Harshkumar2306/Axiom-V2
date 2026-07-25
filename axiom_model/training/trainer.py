import torch
import logging
import math
from torch.nn.utils import clip_grad_norm_
from .metrics import MetricsCalculator
from .profiler import Profiler

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, train_loader, config, is_rank_zero=True):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.config = config
        self.is_rank_zero = is_rank_zero
        
        self.device = next(model.parameters()).device
        self.grad_accum_steps = config.get('grad_accum_steps', 4)
        self.clip_grad = config.get('clip_grad', 1.0)
        
        self.scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())
        self.profiler = Profiler() if is_rank_zero else None
        
    def train_step(self, x, y, is_last_accum_step=True):
        # Mixed Precision
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
            logits = self.model(x)
            loss = MetricsCalculator.compute_loss(logits, y)
            # Scale loss by accumulation steps so gradients are mathematically equivalent to larger batch
            loss = loss / self.grad_accum_steps
            
        # NaN Detection on Loss
        if not torch.isfinite(loss):
            logger.error(f"Loss is {loss.item()}! Terminating to prevent corruption.")
            raise ValueError("Loss exploded to NaN/Inf")
            
        # Backward pass with scaler
        self.scaler.scale(loss).backward()
        
        if is_last_accum_step:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            
            # NaN Detection on Gradients (Clip will also catch this, but good to track)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.clip_grad)
            if not torch.isfinite(grad_norm):
                logger.warning(f"Gradient norm is {grad_norm.item()}! Skipping optimizer step.")
            else:
                self.scaler.step(self.optimizer)
                
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            return loss.item() * self.grad_accum_steps, grad_norm.item()
            
        return loss.item() * self.grad_accum_steps, None

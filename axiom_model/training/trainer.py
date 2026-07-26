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
        self.grad_accum_steps = config.get('training', {}).get('grad_accum_steps', 4)
        self.clip_grad = config.get('training', {}).get('clip_grad_norm', 1.0)
        
        self.scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())
        self.profiler = Profiler() if is_rank_zero else None
        
        self._rolling_loss = None
        
    def _get_torch_dtype(self):
        d = self.config.get('training', {}).get('dtype', 'float16') if 'training' in self.config else self.config.get('dtype', 'float16')
        if d == 'bfloat16': return torch.bfloat16
        if d == 'float32': return torch.float32
        return torch.float16
        
    def train_step(self, x, y, is_last_accum_step=True):
        # Mixed Precision
        with torch.autocast(device_type=self.device.type, dtype=self._get_torch_dtype()):
            logits = self.model(x)
            loss = MetricsCalculator.compute_loss(logits, y)
            # Scale loss by accumulation steps so gradients are mathematically equivalent to larger batch
            loss = loss / self.grad_accum_steps
            
        # 1. NaN/Inf Safety Trap
        if not torch.isfinite(loss):
            logger.error(f"[Loss Safety] Loss is {loss.item()}! Skipping batch to prevent corruption.")
            self.optimizer.zero_grad(set_to_none=True)
            return float('inf'), None
            
        # 2. Loss Spike Anomaly Trap
        scaled_loss = loss.item() * self.grad_accum_steps
        if self._rolling_loss is None:
            self._rolling_loss = scaled_loss
        else:
            if scaled_loss > self._rolling_loss * 5.0 and self._rolling_loss > 0.1:
                logger.warning(f"[Loss Safety] Severe loss spike detected ({scaled_loss:.2f} vs rolling {self._rolling_loss:.2f}). Skipping batch.")
                self.optimizer.zero_grad(set_to_none=True)
                return scaled_loss, None
            # Update rolling average smoothly
            self._rolling_loss = 0.99 * self._rolling_loss + 0.01 * scaled_loss
            
        # Backward pass with scaler
        self.scaler.scale(loss).backward()
        
        if is_last_accum_step:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            
            # 3. Gradient NaN/Explosion Trap
            grad_norm = clip_grad_norm_(self.model.parameters(), self.clip_grad)
            if not torch.isfinite(grad_norm):
                logger.warning(f"[Gradient Safety] Gradient norm is {grad_norm.item()}! Skipping optimizer step to save weights.")
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.scaler.step(self.optimizer)
                
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            return scaled_loss, grad_norm.item()
            
        return scaled_loss, None

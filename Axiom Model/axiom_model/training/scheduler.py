import math
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR

class SchedulerManager:
    """Unified LR-scheduler factory.

    Supported `type` values:
      - "sgdr" / "cosine_warm_restarts": CosineAnnealingWarmRestarts (T_0, T_mult, eta_min)
      - "cosine":                        CosineAnnealingLR (T_max, eta_min)
      - "cosine_warmup":                 linear warmup + cosine decay to eta_min_ratio
                                         (warmup_steps, max_steps, eta_min_ratio)
    """
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config or {}
        self.type = self.config.get('type', 'sgdr')

        if self.type in ['sgdr', 'cosine_warm_restarts']:
            t_0 = self.config.get('T_0', self.config.get('t_0', 10000))
            t_mult = self.config.get('T_mult', self.config.get('t_mult', 2))
            eta_min = self.config.get('eta_min', 1e-5)
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=eta_min
            )
        elif self.type == 'cosine':
            # Plain cosine decay — the standard choice for SFT.
            t_max = self.config.get('T_max', self.config.get('t_max', 5000))
            eta_min = self.config.get('eta_min', 0.0)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min
            )
        elif self.type == 'cosine_warmup':
            # Linear warmup followed by cosine decay (standard pretraining recipe).
            warmup_steps = int(self.config.get('warmup_steps', 2000))
            max_steps = int(self.config.get('max_steps', 100000))
            eta_min_ratio = float(self.config.get('eta_min_ratio', 0.1))

            def lr_lambda(step: int):
                if step < warmup_steps:
                    return (step + 1) / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                progress = min(1.0, progress)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler type: {self.type}")

    def step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.scheduler.state_dict() if self.scheduler else {}

    def load_state_dict(self, state_dict):
        if self.scheduler and state_dict:
            self.scheduler.load_state_dict(state_dict)

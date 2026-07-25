import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

class SchedulerManager:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.type = config.get('type', 'sgdr')
        
        if self.type == 'sgdr':
            t_0 = config.get('t_0', 10000)
            t_mult = config.get('t_mult', 2)
            eta_min = config.get('eta_min', 1e-5)
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=t_0, 
                T_mult=t_mult, 
                eta_min=eta_min
            )
        elif self.type == 'cosine_warmup':
            # Example for future implementation
            pass
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

import os
import torch
import logging
import shutil
import random
import numpy as np

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, save_dir="./checkpoints"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, scaler, epoch, step, best_val_loss, config, is_best=False):
        state = {
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'step': step,
            'best_val_loss': best_val_loss,
            'config': config,
            'py_rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        
        latest_path = os.path.join(self.save_dir, "latest.pt")
        torch.save(state, latest_path)
        logger.info(f"Checkpoint saved to {latest_path} (step {step})")
        
        if is_best:
            best_path = os.path.join(self.save_dir, "best.pt")
            shutil.copyfile(latest_path, best_path)
            logger.info(f"New best checkpoint saved to {best_path}")

    def load(self, model, optimizer, scheduler, scaler, path="./checkpoints/latest.pt"):
        if not os.path.exists(path):
            logger.info(f"No checkpoint found at {path}, starting from scratch.")
            return 0, 0, float('inf')
            
        logger.info(f"Loading checkpoint from {path}...")
        state = torch.load(path, map_location='cpu')
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(state['model'])
        else:
            model.load_state_dict(state['model'])
            
        optimizer.load_state_dict(state['optimizer'])
        if scheduler and state['scheduler']:
            scheduler.load_state_dict(state['scheduler'])
        if scaler and state['scaler']:
            scaler.load_state_dict(state['scaler'])
            
        if 'py_rng_state' in state:
            random.setstate(state['py_rng_state'])
        if 'np_rng_state' in state:
            np.random.set_state(state['np_rng_state'])
            
        torch.set_rng_state(state['rng_state'])
        if state['cuda_rng_state'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state['cuda_rng_state'])
            
        return state.get('epoch', 0), state.get('step', 0), state.get('best_val_loss', float('inf'))

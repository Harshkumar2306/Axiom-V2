import os
import torch
import logging
import shutil
import random
import numpy as np
import math

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, save_dir="./checkpoints"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, scaler, epoch, step, best_val_loss, best_val_step, config, is_best=False):
        # 1. Safety Check: Verify free disk space before saving to prevent
        # corruption. A full 500M checkpoint (fp32 weights + AdamW m/v) is ~6 GB.
        try:
            stat = shutil.disk_usage(self.save_dir)
            free_gb = stat.free / (1024 ** 3)
            if free_gb < 8.0:
                logger.error(f"[Disk Safety] Only {free_gb:.2f} GB free. Aborting checkpoint save to prevent corruption.")
                return
        except Exception as e:
            logger.warning(f"Could not verify disk space: {e}")
            
        state = {
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'step': step,
            'best_val_loss': best_val_loss,
            'best_val_step': best_val_step,
            'config': config,
            'py_rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        
        latest_path = os.path.join(self.save_dir, "latest.pt")
        # 2. Safety Check: Save to a temporary file first, then atomically rename
        tmp_path = latest_path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, latest_path)
        
        logger.info(f"Checkpoint saved to {latest_path} (step {step})")
        
        if is_best:
            best_path = os.path.join(self.save_dir, "best.pt")
            shutil.copyfile(latest_path, best_path)
            
            # Save metadata
            metadata = {
                "best_step": step,
                "best_epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_perplexity": math.exp(best_val_loss) if best_val_loss < 100 else float('inf')
            }
            meta_path = os.path.join(self.save_dir, "best_metadata.json")
            import json
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"New best checkpoint saved to {best_path} (Loss: {best_val_loss:.4f})")

    def load(self, model, optimizer, scheduler, scaler, path="./checkpoints/latest.pt"):
        if not os.path.exists(path):
            logger.info(f"No checkpoint found at {path}, starting from scratch.")
            return 0, 0, float('inf')
            
        logger.info(f"Loading checkpoint from {path}...")
        # weights_only=False is required: the checkpoint embeds numpy RNG state
        # (ndarrays), which the torch>=2.6 default weights_only=True refuses to
        # unpickle. The file is self-produced and trusted, so this is safe.
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # Very old torch without the weights_only argument
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
            
        return state.get('epoch', 0), state.get('step', 0), state.get('best_val_loss', float('inf')), state.get('best_val_step', 0)

import os
import random
import torch
import numpy as np
import subprocess
import yaml
from datetime import datetime

def set_seed(seed: int = 42):
    """Hardcode strict reproducibility across all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Ensure DataLoader workers are deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Axiom-v2] Strict deterministic seed set to {seed}.")

def capture_environment_snapshot(config_path: str, save_dir: str):
    """Automatically save Git hash, CUDA versions, and configs to make experiments reproducible."""
    os.makedirs(save_dir, exist_ok=True)
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "git_commit": "",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    }
    
    try:
        # Try to capture git hash
        snapshot["git_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        snapshot["git_commit"] = "Git not found or not a repository"
        
    try:
        with open(config_path, 'r') as f:
            snapshot["config"] = yaml.safe_load(f)
    except Exception as e:
        snapshot["config"] = f"Error loading config: {str(e)}"
        
    snapshot_file = os.path.join(save_dir, f"env_snapshot_{snapshot['timestamp'].replace(':', '-')}.yaml")
    with open(snapshot_file, 'w') as f:
        yaml.dump(snapshot, f)
        
    print(f"[Axiom-v2] Environment snapshot saved to {snapshot_file}")

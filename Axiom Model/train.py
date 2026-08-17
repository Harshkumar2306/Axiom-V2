import argparse
import yaml
import torch
import os
import logging
from axiom_model.core.model import AxiomV2
from axiom_model.utils.reproducibility import set_seed, capture_environment_snapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Axiom v2 500M Pre-Training Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--output_dir", type=str, default="./runs/experiment_1", help="Directory to save logs and checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    
    device = get_device()
    logger.info(f"Starting Axiom v2 Engine on device: {device}")
    
    # 1. Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration for: {config['model']['name']}")
    
    # 2. Hardcode Reproducibility
    set_seed(config['training']['seed'])
    capture_environment_snapshot(args.config, save_dir=os.path.join(args.output_dir, "env"))
    
    # 3. Build Model
    model_cfg = config['model']
    model = AxiomV2(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        n_layers=model_cfg['n_layers'],
        n_heads=model_cfg['n_heads'],
        n_kv_heads=model_cfg['n_kv_heads'],
        max_seq_len=model_cfg['max_seq_len'],
        multiple_of=model_cfg['multiple_of'],
        norm_eps=model_cfg['norm_eps'],
        rope_theta=model_cfg['rope_theta']
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model instantiated successfully. Total parameters: {param_count / 1e6:.2f} M")
    
    # 4. Torch Compile (if supported)
    if hasattr(torch, "compile") and device.type == "cuda":
        logger.info("Compiling model with torch.compile() for max throughput...")
        model = torch.compile(model)
    else:
        logger.info("torch.compile() skipped (not supported on this device/environment).")
        
    # 5. Optimizer & SGDR Scheduler Setup (Preventing Validation Plateaus)
    # We implement AdamW alongside SGDR (Warm Restarts) to prevent the model 
    # from getting stuck in saddle points/local minima during mid-training.
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']), 
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # SGDR: This acts as the "helicopter" bumping the optimizer out of the ditch
    t_0 = 10000  # Number of steps before the first warm restart
    t_mult = 2   # Multiply the restart interval by this factor after each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=t_0, 
        T_mult=t_mult, 
        eta_min=1e-5
    )
    logger.info(f"SGDR Scheduler active: T_0={t_0}, T_mult={t_mult}. Saddle point prevention engaged.")
    
    # 6. Initialization Complete (Ready for Phase 2/3 Data Injection)
    logger.info("Phase 1 Engine Architecture Initialization Complete. Ready for Dataloader.")

if __name__ == "__main__":
    main()

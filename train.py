import argparse
import yaml
import torch
import os
from axiom_model.core.model import AxiomV2
from axiom_model.utils.reproducibility import set_seed, capture_environment_snapshot

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Axiom v2 Training Initialization")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--output_dir", type=str, default="./runs/experiment_1", help="Directory to save logs and checkpoints")
    args = parser.parse_args()
    
    # 1. Load configuration
    config = load_config(args.config)
    print(f"[Axiom-v2] Initializing model: {config['model']['name']}")
    
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
    )
    
    print(f"[Axiom-v2] Model instantiated. Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # 4. Torch Compile (if supported)
    if hasattr(torch, "compile"):
        print("[Axiom-v2] Compiling model with torch.compile()...")
        # model = torch.compile(model)
        print("[Axiom-v2] Note: compilation commented out for dry-run initialization.")
        
    # TODO: Add Dataloader, SGDR Optimizer, and the actual Training Loop in Phase 3
    print("[Axiom-v2] Phase 1 Engine Initialization Complete.")

if __name__ == "__main__":
    main()

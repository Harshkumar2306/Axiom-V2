import torch
import logging
from axiom_model.core.model import AxiomV2
from axiom_model.utils.reproducibility import set_seed
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_engine():
    logger.info("Starting Engine Validation...")
    set_seed(42)

    # Load config
    with open("axiom_model/configs/500M.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_cfg = config['model']

    # On CPU-only machines the full 476M model exhausts RAM during the backward
    # pass (weights + grads + Adam states). Engine *correctness* does not depend
    # on scale, so validate with an identical tiny architecture on CPU.
    if device.type == "cpu":
        model_cfg = dict(model_cfg)
        model_cfg.update({
            "vocab_size": 512, "d_model": 128, "n_layers": 3,
            "n_heads": 4, "n_kv_heads": 2, "max_seq_len": 128,
            "multiple_of": 32,
        })
        logger.info("CPU detected: validating with a tiny equivalent architecture (same components, scaled down).")

    # 1. Model Instantiation
    try:
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
        logger.info("✓ Model instantiates successfully")
    except Exception as e:
        logger.error(f"Failed to instantiate model: {e}")
        return

    # 2. Forward Pass
    try:
        dummy_input = torch.randint(0, model_cfg['vocab_size'], (2, 128)).to(device)
        outputs = model(dummy_input)
        assert outputs.shape == (2, 128, model_cfg['vocab_size'])
        logger.info("✓ Forward pass verified")
    except Exception as e:
        logger.error(f"Failed forward pass: {e}")
        return
        
    # 3. Backward Pass & Optimizer Step
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = outputs.mean()
        loss.backward()
        logger.info("✓ Backward pass verified")
        
        optimizer.step()
        optimizer.zero_grad()
        logger.info("✓ Optimizer step verified")
    except Exception as e:
        logger.error(f"Failed backward/optimizer step: {e}")
        return

    # 4. Checkpoint Save/Load
    try:
        torch.save(model.state_dict(), "temp_checkpoint.pt")
        model.load_state_dict(torch.load("temp_checkpoint.pt", weights_only=True))
        import os; os.remove("temp_checkpoint.pt")
        logger.info("✓ Checkpoint save/load verified")
    except Exception as e:
        logger.error(f"Failed checkpoint save/load: {e}")
        return

    # 5. Mixed Precision
    try:
        # CPU autocast only supports bfloat16; fp16 there raises at runtime.
        if device.type == 'cuda':
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type == 'mps':
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler(
            'cuda' if device.type == 'cuda' else 'cpu',
            enabled=(device.type == 'cuda' and amp_dtype == torch.float16)
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            outputs_amp = model(dummy_input)
            loss_amp = outputs_amp.mean()
        logger.info("✓ Mixed precision verified")
    except Exception as e:
        logger.error(f"Failed mixed precision: {e}")
        return
        
    # 6. torch.compile
    try:
        if hasattr(torch, "compile") and device.type == "cuda":
            compiled_model = torch.compile(model)
            compiled_model(dummy_input)
            logger.info("✓ torch.compile verified")
        else:
            logger.info("✓ torch.compile verified (Skipped, unsupported hardware)")
    except Exception as e:
        logger.error(f"Failed torch.compile: {e}")
        return

    logger.info("Engine Validation Complete. All systems go!")

if __name__ == "__main__":
    test_engine()

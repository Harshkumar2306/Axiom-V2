import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_manifest(dataset_dir: str, version: str = "v1"):
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info("Generating final dataset manifest.")
    
    train_path = os.path.join(dataset_dir, "train.bin")
    val_path = os.path.join(dataset_dir, "val.bin")
    
    train_size = os.path.getsize(train_path) if os.path.exists(train_path) else 0
    val_size = os.path.getsize(val_path) if os.path.exists(val_path) else 0
    
    # 2 bytes per token (uint16)
    train_tokens = train_size // 2
    val_tokens = val_size // 2

    manifest = {
        "dataset_version": version,
        "tokenizer": "cl100k_base (Llama-3 proxy)",
        "total_documents": "Unknown (Compute from pipeline logs)",
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "filtered_documents": "Unknown (Compute from pipeline logs)",
        "duplicates_removed": "Unknown (Compute from pipeline logs)",
        "train_bin_size_bytes": train_size,
        "val_bin_size_bytes": val_size,
        "timestamp": datetime.now().isoformat()
    }
    
    out_path = os.path.join(dataset_dir, "manifest.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    logger.info(f"Manifest written to {out_path}:\n{json.dumps(manifest, indent=2)}")

if __name__ == "__main__":
    generate_manifest("./dataset/v1", version="v1")

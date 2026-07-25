import os
import json
import logging
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

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
        "created_at": datetime.now().isoformat(),
        "tokenizer": "cl100k_base",
        "tokenizer_vocab_size": 100277,
        "sources": {
            "fineweb_edu": "HuggingFaceFW/fineweb-edu",
            "starcoder": "bigcode/starcoderdata",
            "wikipedia": "wikipedia 20220301.en",
            "redpajama": "togethercomputer/RedPajama-Data-1T"
        },
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "filtered_documents": "Unknown (Compute from pipeline logs)",
        "quality_filtered": "Unknown (Compute from pipeline logs)",
        "duplicates_removed": "Unknown (Compute from pipeline logs)",
        "sha256": {
            "train.bin": compute_sha256(train_path),
            "val.bin": compute_sha256(val_path)
        }
    }
    
    out_path = os.path.join(dataset_dir, "manifest.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    logger.info(f"Manifest written to {out_path}:\n{json.dumps(manifest, indent=2)}")

if __name__ == "__main__":
    generate_manifest("./dataset/v1", version="v1")

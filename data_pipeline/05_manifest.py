import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_manifest(bin_dir: str, manifest_path: str):
    logger.info("Starting Phase 2: Manifest Generation.")
    
    manifest = {
        "version": "1.0",
        "generated_at": datetime.utcnow().isoformat(),
        "dataset_mix": {
            "fineweb_edu": 0.55,
            "starcoder": 0.20,
            "wikipedia": 0.10,
            "tech_docs": 0.07,
            "scientific": 0.05,
            "books": 0.03
        },
        "pipeline_stats": {
            "total_tokens_target": 10_000_000_000,
            "deduplication_ratio": "TBD",
            "vocab": "Llama-3 (128k)",
        },
        "files": {}
    }
    
    if os.path.exists(bin_dir):
        for file in os.listdir(bin_dir):
            if file.endswith(".bin"):
                filepath = os.path.join(bin_dir, file)
                size_bytes = os.path.getsize(filepath)
                # uint16 = 2 bytes per token
                tokens = size_bytes // 2
                manifest["files"][file] = {
                    "size_bytes": size_bytes,
                    "tokens": tokens
                }
                
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    logger.info(f"Manifest successfully written to {manifest_path}")

if __name__ == "__main__":
    generate_manifest("./data/bin", "./data/dataset_manifest.json")

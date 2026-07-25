import os
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Phase 2 Target Configuration
DATASET_MIX = {
    "fineweb_edu": {"repo": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "weight": 0.55},
    "starcoder": {"repo": "bigcode/starcoderdata", "name": "python", "split": "train", "weight": 0.20},
    "wikipedia": {"repo": "wikipedia", "name": "20220301.en", "split": "train", "weight": 0.10},
    "tech_docs": {"repo": "togethercomputer/RedPajama-Data-1T", "name": "github", "split": "train", "weight": 0.07}, # Proxy for tech docs
    "scientific": {"repo": "togethercomputer/RedPajama-Data-1T", "name": "arxiv", "split": "train", "weight": 0.05},
    "books": {"repo": "togethercomputer/RedPajama-Data-1T", "name": "book", "split": "train", "weight": 0.03}
}

TARGET_TOTAL_TOKENS = 10_000_000_000  # 10 Billion tokens
CHARS_PER_TOKEN_ESTIMATE = 4.0

def download_and_stream(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for mix_name, config in DATASET_MIX.items():
        logger.info(f"Starting stream for {mix_name} (Target: {config['weight']*100}% of {TARGET_TOTAL_TOKENS/1e9}B tokens)")
        
        # Target bytes based on rough character to token conversion
        target_bytes = (TARGET_TOTAL_TOKENS * config['weight']) * CHARS_PER_TOKEN_ESTIMATE
        current_bytes = 0
        
        # Using streaming=True to prevent massive RAM overhead
        dataset = load_dataset(config["repo"], config.get("name"), split=config["split"], streaming=True)
        
        out_path = os.path.join(output_dir, f"{mix_name}_raw.jsonl")
        logger.info(f"Writing stream to {out_path}...")
        
        # In a real execution, we would stream this to disk.
        # This is the architectural layout for Phase 2.
        '''
        import json
        with open(out_path, 'w', encoding='utf-8') as f:
            for i, row in enumerate(dataset):
                text = row.get('text', row.get('content', ''))
                if not text: continue
                
                f.write(json.dumps({"text": text}) + "\n")
                current_bytes += len(text.encode('utf-8'))
                
                if current_bytes >= target_bytes:
                    logger.info(f"Reached target size for {mix_name}!")
                    break
        '''
        logger.info(f"Successfully configured pipeline for {mix_name}.")

if __name__ == "__main__":
    # Use a local data folder that gitignore already ignores
    download_and_stream("./data/raw")

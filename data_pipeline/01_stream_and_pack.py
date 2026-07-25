import os
import logging
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import warnings

# Suppress HuggingFace authentication warnings
warnings.filterwarnings("ignore")
logging.getLogger("datasets").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Phase 2 Target Configuration
DATASET_MIX = {
    "fineweb_edu": {"repo": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "weight": 0.55},
    "starcoder": {"repo": "bigcode/starcoderdata", "name": "python", "split": "train", "weight": 0.20},
    "wikipedia": {"repo": "wikipedia", "name": "20220301.en", "split": "train", "weight": 0.10},
    "tech_docs": {"repo": "togethercomputer/RedPajama-Data-1T", "name": "github", "split": "train", "weight": 0.07},
    "scientific": {"repo": "togethercomputer/RedPajama-Data-1T", "name": "arxiv", "split": "train", "weight": 0.05},
    "books": {"repo": "togethercomputer/RedPajama-Data-1T", "name": "book", "split": "train", "weight": 0.03}
}

# Because cl100k_base has 100,277 tokens, we MUST use uint32 (4 bytes per token).
# To fit inside Kaggle's 20GB limit, the maximum tokens we can pack is 4.5 Billion (18 GB).
TARGET_TOTAL_TOKENS = 4_500_000_000  # 4.5 Billion tokens (18 GB)
VAL_SPLIT_TOKENS = 125_000_000       # ~500MB of val data

def is_quality_text(text: str) -> bool:
    """Basic inline filter to drop garbage without needing disk."""
    if len(text) < 100: return False
    if len(text) > 100000: return False
    upper_count = sum(1 for c in text if c.isupper())
    if upper_count / len(text) > 0.4: return False
    words = text.split()
    if len(words) < 10: return False
    if len(set(words)) / len(words) < 0.2: return False
    return True

def stream_and_pack(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    logger.info("Initializing tiktoken (cl100k_base)...")
    enc = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    val_tokens = 0
    
    with open(train_bin_path, 'wb') as f_train, open(val_bin_path, 'wb') as f_val:
        for mix_name, config in DATASET_MIX.items():
            target_dataset_tokens = int(TARGET_TOTAL_TOKENS * config['weight'])
            dataset_tokens = 0
            
            logger.info(f"Streaming {mix_name} [Target: {target_dataset_tokens:,} tokens]")
            dataset = load_dataset(config["repo"], config.get("name"), split=config["split"], streaming=True)
            
            pbar = tqdm(total=target_dataset_tokens, desc=mix_name, unit="tok")
            
            for row in dataset:
                text = row.get('text', row.get('content', ''))
                if not text or not is_quality_text(text): 
                    continue
                
                # Tokenize in RAM
                tokens = enc.encode_ordinary(text)
                tokens.append(enc.eot_token)
                
                # Pack to uint32 (fixes the OverflowError)
                np_tokens = np.array(tokens, dtype=np.uint32)
                
                # Write directly to disk
                if val_tokens < (VAL_SPLIT_TOKENS * config['weight']):
                    f_val.write(np_tokens.tobytes())
                    val_tokens += len(tokens)
                else:
                    f_train.write(np_tokens.tobytes())
                    
                dataset_tokens += len(tokens)
                total_tokens += len(tokens)
                pbar.update(len(tokens))
                
                if dataset_tokens >= target_dataset_tokens:
                    break
                    
            pbar.close()
                    
    logger.info(f"🎉 Pipeline Complete! Total tokens packed: {total_tokens:,}")
    logger.info(f"Train File: {train_bin_path}")
    logger.info(f"Val File: {val_bin_path}")

if __name__ == "__main__":
    stream_and_pack("./dataset/v1")

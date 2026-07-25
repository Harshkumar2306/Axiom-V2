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
    "tech_docs": {"repo": "togethercomputer/RedPajama-Data-1T", "name": "github", "split": "train", "weight": 0.27}, # Increased to 27% to replace gated starcoder
    "wikipedia": {"repo": "wikipedia", "name": "20220301.en", "split": "train", "weight": 0.10},
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
    
    # Resume Logic
    train_size = os.path.getsize(train_bin_path) if os.path.exists(train_bin_path) else 0
    val_size = os.path.getsize(val_bin_path) if os.path.exists(val_bin_path) else 0
    
    total_tokens_written = (train_size + val_size) // 4
    val_tokens_written = val_size // 4
    
    total_tokens = total_tokens_written
    val_tokens = val_tokens_written
    
    if total_tokens > 0:
        logger.info(f"Resuming from existing files: {total_tokens:,} tokens already packed.")
        mode = 'ab'
    else:
        mode = 'wb'
    
    with open(train_bin_path, mode) as f_train, open(val_bin_path, mode) as f_val:
        tokens_to_skip = total_tokens_written
        
        for mix_name, config in DATASET_MIX.items():
            target_dataset_tokens = int(TARGET_TOTAL_TOKENS * config['weight'])
            
            if tokens_to_skip >= target_dataset_tokens:
                logger.info(f"Skipping {mix_name}, already fully processed.")
                tokens_to_skip -= target_dataset_tokens
                continue
            
            dataset_tokens = tokens_to_skip
            tokens_to_skip = 0
            
            logger.info(f"Streaming {mix_name} [Target: {target_dataset_tokens:,} tokens]")
            if dataset_tokens > 0:
                logger.info(f"Resuming {mix_name} from {dataset_tokens:,} tokens...")
                
            dataset = load_dataset(config["repo"], config.get("name"), split=config["split"], streaming=True)
            
            pbar = tqdm(total=target_dataset_tokens, initial=dataset_tokens, desc=mix_name, unit="tok")
            
            train_buffer = []
            val_buffer = []
            BUFFER_LIMIT = 500_000 # Buffer half a million tokens before writing
            
            # If we are resuming, we need to skip records we've already processed.
            # In streaming mode, we can't easily skip by token count directly without tokenizing,
            # but since we resume, we just skip the text generation until we catch up.
            # However, exact resume in streaming is tricky because text lengths vary.
            # For Kaggle safety, we will just start pulling and tokenizing, but skip writing until we hit `dataset_tokens`.
            
            current_dataset_tokens_processed = 0
            
            for row in dataset:
                text = row.get('text', row.get('content', ''))
                if not text or not is_quality_text(text): 
                    continue
                
                # Tokenize in RAM
                tokens = enc.encode_ordinary(text)
                tokens.append(enc.eot_token)
                
                # If we are catching up, just advance the counter and don't write.
                if current_dataset_tokens_processed < dataset_tokens:
                    current_dataset_tokens_processed += len(tokens)
                    continue
                    
                # Buffer the tokens
                if val_tokens < (VAL_SPLIT_TOKENS * config['weight']):
                    val_buffer.extend(tokens)
                    val_tokens += len(tokens)
                else:
                    train_buffer.extend(tokens)
                    
                dataset_tokens += len(tokens)
                total_tokens += len(tokens)
                pbar.update(len(tokens))
                
                # Flush Train Buffer
                if len(train_buffer) >= BUFFER_LIMIT:
                    f_train.write(np.array(train_buffer, dtype=np.uint32).tobytes())
                    train_buffer.clear()
                    
                # Flush Val Buffer
                if len(val_buffer) >= BUFFER_LIMIT:
                    f_val.write(np.array(val_buffer, dtype=np.uint32).tobytes())
                    val_buffer.clear()
                
                if dataset_tokens >= target_dataset_tokens:
                    break
                    
            # Final flush of remaining buffers
            if train_buffer: f_train.write(np.array(train_buffer, dtype=np.uint32).tobytes())
            if val_buffer: f_val.write(np.array(val_buffer, dtype=np.uint32).tobytes())
            
            pbar.close()
                    
    logger.info(f"🎉 Pipeline Complete! Total tokens packed: {total_tokens:,}")
    logger.info(f"Train File: {train_bin_path}")
    logger.info(f"Val File: {val_bin_path}")

if __name__ == "__main__":
    stream_and_pack("./dataset/v1")

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
    # wikimedia/wikipedia (parquet) — the old script-based "wikipedia" repo was
    # removed in datasets>=3 and crashes with modern library versions.
    "wikipedia": {"repo": "wikimedia/wikipedia", "name": "20231101.en", "split": "train", "weight": 0.10},
    "scientific": {"repo": "togethercomputer/RedPajama-Data-1T", "name": "arxiv", "split": "train", "weight": 0.05},
    "books": {"repo": "roneneldan/TinyStories", "name": None, "split": "train", "weight": 0.03} # Replaced defunct Books3
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

def _reconstruct_progress(total_tokens_written: int, val_tokens_written: int):
    """Rebuild exact per-dataset progress from the two global counters.

    Datasets are processed strictly in mix order, and within each dataset the
    val quota is filled BEFORE any train tokens are written. That deterministic
    ordering makes per-dataset progress a pure function of (total, val), so
    resume works without a separate state file.
    """
    progress = {}
    remaining_total = total_tokens_written
    remaining_val = val_tokens_written
    for name, cfg in DATASET_MIX.items():
        target = int(TARGET_TOTAL_TOKENS * cfg['weight'])
        val_quota = int(VAL_SPLIT_TOKENS * cfg['weight'])
        done = min(remaining_total, target)
        remaining_total -= done
        # A dataset can only have val tokens if it was actually entered.
        val_done = min(remaining_val, val_quota) if done > 0 else 0
        remaining_val -= val_done
        progress[name] = {"done": done, "val_done": val_done, "target": target, "val_quota": val_quota}
    return progress

def stream_and_pack(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")

    logger.info("Initializing tiktoken (cl100k_base)...")
    enc = tiktoken.get_encoding("cl100k_base")

    # Resume Logic
    train_size = os.path.getsize(train_bin_path) if os.path.exists(train_bin_path) else 0
    val_size = os.path.getsize(val_bin_path) if os.path.exists(val_bin_path) else 0

    total_tokens = (train_size + val_size) // 4
    val_tokens_written = val_size // 4

    if total_tokens > 0:
        logger.info(f"Resuming from existing files: {total_tokens:,} tokens already packed.")
        mode = 'ab'
    else:
        mode = 'wb'

    progress = _reconstruct_progress(total_tokens, val_tokens_written)

    with open(train_bin_path, mode) as f_train, open(val_bin_path, mode) as f_val:
        for mix_name, config in DATASET_MIX.items():
            ds_progress = progress[mix_name]
            target_dataset_tokens = ds_progress["target"]
            val_quota = ds_progress["val_quota"]
            dataset_tokens = ds_progress["done"]
            dataset_val_tokens = ds_progress["val_done"]

            if dataset_tokens >= target_dataset_tokens:
                logger.info(f"Skipping {mix_name}, already fully processed.")
                continue

            logger.info(f"Streaming {mix_name} [Target: {target_dataset_tokens:,} tokens]")
            if dataset_tokens > 0:
                logger.info(f"Resuming {mix_name} from {dataset_tokens:,} tokens...")

            dataset = load_dataset(config["repo"], config.get("name"), split=config["split"], streaming=True, trust_remote_code=True)

            pbar = tqdm(total=target_dataset_tokens, initial=dataset_tokens, desc=mix_name, unit="tok")

            train_buffer = []
            val_buffer = []
            BUFFER_LIMIT = 500_000 # Buffer half a million tokens before writing

            # Exact resume in streaming mode: the stream order is deterministic,
            # so we re-tokenize from the start and simply don't write until our
            # token counter catches up with what is already on disk.
            current_dataset_tokens_processed = 0
            resume_target = dataset_tokens  # Freeze target to prevent oscillating drops

            for row in dataset:
                text = row.get('text', row.get('content', ''))
                if not text or not is_quality_text(text):
                    continue

                # Tokenize in RAM
                tokens = enc.encode_ordinary(text)
                tokens.append(enc.eot_token)

                # If we are catching up, just advance the counter and don't write.
                if current_dataset_tokens_processed < resume_target:
                    current_dataset_tokens_processed += len(tokens)
                    continue

                # Fill this dataset's val quota FIRST, then write train tokens.
                # (Previously the val quota was tracked globally, so the first
                # dataset in the mix filled the entire val set by itself.)
                if dataset_val_tokens < val_quota:
                    val_buffer.extend(tokens)
                    dataset_val_tokens += len(tokens)
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

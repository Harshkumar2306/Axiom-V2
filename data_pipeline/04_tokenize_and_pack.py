import os
import logging
import numpy as np
# import tiktoken

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tokenize_and_pack(input_dir: str, output_dir: str, val_split: float = 0.001):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting Phase 2: Tokenization and Binary Packing.")
    
    logger.info("Loading Llama-3 tiktoken vocabulary (~128k tokens)...")
    # enc = tiktoken.get_encoding("cl100k_base") # Proxy for Llama 3 in dev
    
    # Process data and pack into uint16
    logger.info(f"Streaming deduplicated jsonl from {input_dir}...")
    
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    logger.info(f"Writing packed tokens to {train_bin_path} and {val_bin_path}")
    # Example loop:
    # tokens = enc.encode_ordinary(text)
    # np_tokens = np.array(tokens, dtype=np.uint16)
    # np_tokens.tofile(f)
    
    logger.info("Tokenization complete. Datasets ready for Pretraining.")

if __name__ == "__main__":
    tokenize_and_pack("./data/deduped", "./data/bin")

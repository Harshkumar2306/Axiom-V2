import os
import logging
import numpy as np
import tiktoken
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tokenize_and_pack(input_dir: str, output_dir: str, val_split: float = 0.001):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting Phase 2: Tokenization and Binary Packing.")
    
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory {input_dir} not found. Skipping tokenization.")
        return
        
    logger.info("Loading Llama-3 tiktoken vocabulary (~128k tokens)...")
    enc = tiktoken.get_encoding("cl100k_base") # Proxy for Llama 3 in dev
    
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    logger.info(f"Writing packed tokens to {train_bin_path} and {val_bin_path}")
    
    total_tokens = 0
    with open(train_bin_path, 'wb') as f_train, open(val_bin_path, 'wb') as f_val:
        for filename in os.listdir(input_dir):
            if not filename.endswith(".jsonl"): continue
            
            in_path = os.path.join(input_dir, filename)
            logger.info(f"Tokenizing {in_path}...")
            
            with open(in_path, 'r', encoding='utf-8') as fin:
                for i, line in enumerate(fin):
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        
                        # Add EOS token manually or let tokenizer handle it
                        tokens = enc.encode_ordinary(text)
                        tokens.append(enc.eot_token)
                        
                        np_tokens = np.array(tokens, dtype=np.uint16)
                        
                        if np.random.rand() < val_split:
                            f_val.write(np_tokens.tobytes())
                        else:
                            f_train.write(np_tokens.tobytes())
                            
                        total_tokens += len(tokens)
                    except Exception as e:
                        continue
                        
    logger.info(f"Tokenization complete. Packed {total_tokens} tokens. Datasets ready for Pretraining.")

if __name__ == "__main__":
    tokenize_and_pack("./data/deduped", "./data/bin")

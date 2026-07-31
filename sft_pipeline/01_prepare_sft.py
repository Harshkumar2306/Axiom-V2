import os
import torch
import tiktoken
import yaml
from datasets import load_dataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard PyTorch ignore index for F.cross_entropy
IGNORE_INDEX = -100

def _load_max_seq_len() -> int:
    """Read max_seq_len from the model config so SFT sequences match the engine.

    Padding to 4096 while the model trains at 2048 wastes ~4x compute and
    risks a logits OOM on T4-class GPUs (B x T x 100k vocab)."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "axiom_model", "configs", "500M.yaml")
    try:
        with open(config_path, 'r') as f:
            return int(yaml.safe_load(f)['model']['max_seq_len'])
    except Exception as e:
        logger.warning(f"Could not read max_seq_len from config ({e}); falling back to 2048.")
        return 2048

MAX_SEQ_LEN = _load_max_seq_len()

def format_chatml(conversations):
    """
    SlimOrca 'conversations' format:
    [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}]
    """
    text = ""
    mask_ranges = [] # list of (start_char, end_char) to mask
    
    for turn in conversations:
        role = turn['from']
        if role == 'human':
            role_name = 'user'
        elif role == 'gpt':
            role_name = 'assistant'
        elif role == 'system':
            role_name = 'system'
        else:
            role_name = role
            
        start_char = len(text)
        chunk = f"<|im_start|>{role_name}\n{turn['value']}<|im_end|>\n"
        text += chunk
        
        # We only want to compute loss on the 'assistant' tokens.
        # So we mask the 'user' and 'system' prompts.
        if role_name != 'assistant':
            mask_ranges.append((start_char, len(text)))
        else:
            # Mask the "<|im_start|>assistant\n" part of the assistant's turn
            header_len = len(f"<|im_start|>assistant\n")
            mask_ranges.append((start_char, start_char + header_len))
            
    return text, mask_ranges

def main():
    logger.info("Loading tokenizer...")
    enc = tiktoken.get_encoding("cl100k_base")
    
    logger.info("Loading Open-Orca/SlimOrca dataset...")
    # Loading just the first 10,000 for speed in validation. 
    # In production, we'd process the whole split.
    ds = load_dataset("Open-Orca/SlimOrca", split="train[:10000]")
    
    out_dir = "dataset/sft"
    os.makedirs(out_dir, exist_ok=True)
    
    all_input_ids = []
    all_labels = []
    
    logger.info("Processing conversations and applying loss masking...")
    for item in tqdm(ds):
        # Because we need precise token-level masking,
        # it is often easier to tokenize turn-by-turn. Let's do that for accuracy.
        input_ids = []
        labels = []
        
        for turn in item['conversations']:
            role = turn['from']
            role_name = 'user' if role == 'human' else ('assistant' if role == 'gpt' else 'system')
            
            header = f"<|im_start|>{role_name}\n"
            header_ids = enc.encode(header, allowed_special="all")
            
            content = f"{turn['value']}<|im_end|>\n"
            content_ids = enc.encode(content, allowed_special="all")
            
            # Combine
            input_ids.extend(header_ids + content_ids)
            
            # Masking logic
            if role_name == 'assistant':
                # Mask header, train on content
                labels.extend([IGNORE_INDEX] * len(header_ids))
                labels.extend(content_ids)
            else:
                # Mask everything
                labels.extend([IGNORE_INDEX] * (len(header_ids) + len(content_ids)))
                
        # Truncate to MAX_SEQ_LEN
        if len(input_ids) > MAX_SEQ_LEN:
            input_ids = input_ids[:MAX_SEQ_LEN]
            labels = labels[:MAX_SEQ_LEN]
            
        # Pad to MAX_SEQ_LEN for static batching (or we can multipack later)
        # We will pad here for simplicity in DataLoader
        pad_len = MAX_SEQ_LEN - len(input_ids)
        if pad_len > 0:
            input_ids.extend([enc.eot_token] * pad_len)
            labels.extend([IGNORE_INDEX] * pad_len)
            
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        
    logger.info("Converting to PyTorch tensors...")
    input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    out_path = os.path.join(out_dir, "sft_data.pt")
    logger.info(f"Saving to {out_path}...")
    torch.save({"input_ids": input_ids_tensor, "labels": labels_tensor}, out_path)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()

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

import re

def main():
    logger.info("Loading tokenizer...")
    enc = tiktoken.get_encoding("cl100k_base")
    
    logger.info("Loading ehartford/dolphin uncensored dataset (37,000 samples)...")
    ds_orca = load_dataset("ehartford/dolphin", "flan1m-alpaca-uncensored", split="train[:37000]")
    
    logger.info("Loading OpenAssistant (12,900 samples) for chat smoothness...")
    ds_oa = load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train")
    
    out_dir = "dataset/sft"
    os.makedirs(out_dir, exist_ok=True)
    
    all_input_ids = []
    all_labels = []
    
    SYSTEM_HEADER = "### System:\nYou are a highly intelligent, logical, and helpful AI assistant named Axiom.\n\n"
    system_ids = enc.encode(SYSTEM_HEADER)
    
    def process_turn(role_name, content_val):
        header_ids = enc.encode(f"### {role_name}:\n")
        
        if role_name == 'Assistant':
            content_val += "<|endoftext|>\n\n"
            content_ids = enc.encode(content_val, allowed_special={'<|endoftext|>'})
            lbls = [IGNORE_INDEX] * len(header_ids) + content_ids
        else:
            content_val += "\n\n"
            content_ids = enc.encode(content_val)
            lbls = [IGNORE_INDEX] * (len(header_ids) + len(content_ids))
            
        return header_ids + content_ids, lbls

    logger.info("Processing Dolphin Uncensored...")
    for item in tqdm(ds_orca):
        input_ids = list(system_ids)
        labels = [IGNORE_INDEX] * len(system_ids)
        
        # Dolphin uses Instruction/Input/Output format
        if 'instruction' in item:
            prompt_text = item['instruction']
            if item.get('input', ''):
                prompt_text += "\n" + item['input']
                
            turns = [
                {'from': 'human', 'value': prompt_text},
                {'from': 'gpt', 'value': item.get('output', '')}
            ]
        else:
            turns = item.get('conversations', [])
            
        for turn in turns:
            role = turn['from']
            if role == 'human': role_name = 'User'
            elif role == 'gpt': role_name = 'Assistant'
            else: continue # Skip old system prompts, we force our own
            
            i_ids, l_ids = process_turn(role_name, turn['value'])
            input_ids.extend(i_ids)
            labels.extend(l_ids)
            
        if len(input_ids) > MAX_SEQ_LEN:
            input_ids, labels = input_ids[:MAX_SEQ_LEN], labels[:MAX_SEQ_LEN]
            
        pad_len = MAX_SEQ_LEN - len(input_ids)
        if pad_len > 0:
            input_ids.extend([enc.eot_token] * pad_len)
            labels.extend([IGNORE_INDEX] * pad_len)
            
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    logger.info("Processing OpenAssistant...")
    for item in tqdm(ds_oa):
        input_ids = list(system_ids)
        labels = [IGNORE_INDEX] * len(system_ids)
        
        # Split by ChatML tags
        text = item['text']
        turns = re.split(r'<\|im_start\|>(user|assistant)\n', text)
        
        for i in range(1, len(turns), 2):
            role_name = 'User' if turns[i] == 'user' else 'Assistant'
            content_val = turns[i+1].replace('<|im_end|>\n', '').replace('<|im_end|>', '').strip()
            
            i_ids, l_ids = process_turn(role_name, content_val)
            input_ids.extend(i_ids)
            labels.extend(l_ids)
            
        if len(input_ids) > MAX_SEQ_LEN:
            input_ids, labels = input_ids[:MAX_SEQ_LEN], labels[:MAX_SEQ_LEN]
            
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

import os
import torch
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_chatml_dpo(system_msg, question, answer):
    """
    Builds the exact ChatML prompt and response strings for DPO.
    """
    sys_str = f"<|im_start|>system\n{system_msg}<|im_end|>\n" if system_msg else ""
    prompt = f"{sys_str}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    response = f"{answer}<|im_end|>"
    return prompt, response

def main():
    tokenizer = tiktoken.get_encoding("cl100k_base")
    IGNORE_INDEX = -100 # Mask for the prompt tokens (we don't train on the question)

    logger.info("Loading Intel/orca_dpo_pairs dataset...")
    # Load the high-quality DPO dataset (Chosen vs Rejected answers)
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    
    logger.info(f"Loaded {len(dataset)} DPO pairs. Tokenizing...")
    
    dpo_data = []
    
    for item in tqdm(dataset, desc="Tokenizing DPO Pairs"):
        sys_msg = item.get("system", "")
        question = item.get("question", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        # Skip incomplete data
        if not question or not chosen or not rejected:
            continue
            
        prompt, chosen_resp = create_chatml_dpo(sys_msg, question, chosen)
        _, rejected_resp = create_chatml_dpo(sys_msg, question, rejected)
        
        # Tokenize
        prompt_ids = tokenizer.encode(prompt, allowed_special={"<|im_start|>", "<|im_end|>"})
        chosen_resp_ids = tokenizer.encode(chosen_resp, allowed_special={"<|im_start|>", "<|im_end|>"})
        rejected_resp_ids = tokenizer.encode(rejected_resp, allowed_special={"<|im_start|>", "<|im_end|>"})
        
        # Concatenate prompt and responses
        chosen_ids = prompt_ids + chosen_resp_ids
        rejected_ids = prompt_ids + rejected_resp_ids
        
        # Labels are identical to IDs, but we apply -100 masking to the prompt
        chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_resp_ids
        rejected_labels = [IGNORE_INDEX] * len(prompt_ids) + rejected_resp_ids
        
        # Skip if sequence is too long for 4096 context
        if len(chosen_ids) > 4096 or len(rejected_ids) > 4096:
            continue
            
        dpo_data.append({
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long)
        })
        
    out_dir = "./dataset/dpo"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpo_data.pt")
    
    logger.info(f"Saving {len(dpo_data)} fully prepared DPO pairs to {out_path}...")
    torch.save(dpo_data, out_path)
    logger.info("DPO dataset preparation successfully complete! Ready for Phase 5.")

if __name__ == "__main__":
    main()

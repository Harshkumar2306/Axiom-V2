import os
import torch
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_markdown_dpo(question, answer):
    """
    Builds the exact Markdown prompt and response strings for DPO, matching Phase 4.
    """
    sys_str = "### System:\nYou are a highly intelligent, logical, and helpful AI assistant named Axiom.\n\n"
    prompt = f"{sys_str}### User:\n{question}\n\n### Assistant:\n"
    response = f"{answer}<|endoftext|>"
    return prompt, response

def main():
    tokenizer = tiktoken.get_encoding("cl100k_base")
    IGNORE_INDEX = -100 # Mask for the prompt tokens (we don't train on the question)

    logger.info("Loading jondurbin/truthy-dpo-v0.1 uncensored dataset...")
    # Load the high-quality uncensored DPO dataset (Chosen vs Rejected answers)
    dataset = load_dataset("jondurbin/truthy-dpo-v0.1", split="train")
    
    logger.info(f"Loaded {len(dataset)} DPO pairs. Tokenizing...")
    
    dpo_data = []
    
    for item in tqdm(dataset, desc="Tokenizing DPO Pairs"):
        question = item.get("question", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        # Skip incomplete data
        if not question or not chosen or not rejected:
            continue
            
        prompt, chosen_resp = create_markdown_dpo(question, chosen)
        _, rejected_resp = create_markdown_dpo(question, rejected)
        
        # Tokenize
        prompt_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        chosen_resp_ids = tokenizer.encode(chosen_resp, allowed_special={"<|endoftext|>"})
        rejected_resp_ids = tokenizer.encode(rejected_resp, allowed_special={"<|endoftext|>"})
        
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

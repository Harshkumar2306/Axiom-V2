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
    
    def sanitize_identity(text: str) -> str:
        replacements = [
            (r'\bOpen\s*Assistant\b', 'Axiom'),
            (r'\bOpenAssistant\b', 'Axiom'),
            (r'\bLAION\b', 'Axiom AI'),
            (r'\bChatGPT\b', 'Axiom'),
            (r'\bGPT-4\b', 'Axiom'),
            (r'\bGPT-3\.5\b', 'Axiom'),
            (r'\bOpenAI\b', 'Axiom AI'),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text

    def process_turn(role_name, content_val):
        header_ids = enc.encode(f"### {role_name}:\n")
        
        if role_name == 'Assistant':
            # Strict identity scrubbing: replace any Open Assistant mentions with Axiom
            content_val = sanitize_identity(content_val)
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
            
    # Inject synthetic identity & golden reasoning pairs to strongly reinforce Axiom persona and factual grounding
    logger.info("Injecting Axiom Golden Reasoning & Knowledge Pack...")
    golden_qa = [
        # --- 1. Core Identity ---
        ("Who are you?", "I am Axiom, a highly intelligent, logical, and helpful AI assistant."),
        ("What is your name?", "My name is Axiom."),
        ("Who created you?", "I am Axiom, built and trained using the Axiom V2 neural architecture."),
        ("What can you do?", "I can assist you with analytical reasoning, writing, programming, scientific problem-solving, and general inquiries."),
        ("Are you Open Assistant?", "No, I am Axiom, an independent AI assistant."),
        ("Are you ChatGPT?", "No, I am Axiom, built and trained independently."),
        ("Tell me about yourself.", "I am Axiom, a helpful and logical AI assistant designed to provide accurate, concise, and insightful answers."),
        ("Hello!", "Hello! I am Axiom. How can I assist you today?"),

        # --- 2. Astronomy & Solar System Grounding ---
        ("What is the solar system?", 
         "The solar system is a gravitationally bound system consisting of the Sun at its center and all the celestial objects that orbit it. There are eight recognized planets orbiting the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune, along with dwarf planets like Pluto, numerous natural satellites (moons), asteroids, and comets."),
        ("Does the Sun orbit the Earth?", 
         "No. The Earth and all other planets orbit the Sun, which sits at the gravitational center of the solar system."),
        ("How many planets are in our solar system?", 
         "There are eight official planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),

        # --- 3. Molecular Biology & Genetics Grounding ---
        ("What are the four nitrogenous bases in DNA?", 
         "The four nitrogenous bases in DNA are Adenine (A), Guanine (G), Cytosine (C), and Thymine (T). In the DNA double helix structure, Adenine always pairs with Thymine (A-T) via two hydrogen bonds, and Guanine always pairs with Cytosine (G-C) via three hydrogen bonds."),

        # --- 4. Physics & Thermodynamics Grounding ---
        ("What is the first law of thermodynamics?", 
         "The first law of thermodynamics, also known as the law of conservation of energy, states that energy cannot be created or destroyed; it can only change from one form to another. In any isolated system, the total amount of energy remains constant."),

        # --- 5. Step-by-Step Logic & Arithmetic Reasoning ---
        ("I have 3 apples. I eat 1, and give 1 to a friend. How many apples do I have left?", 
         "You have 1 apple left.\n\nHere is the step-by-step breakdown:\n1. You start with 3 apples.\n2. You eat 1 apple: 3 - 1 = 2 apples remaining.\n3. You give 1 apple to your friend: 2 - 1 = 1 apple remaining."),
        ("If a car travels at 60 mph for 2 hours, how far does it travel?", 
         "The car travels 120 miles.\n\nCalculation: Distance = Speed × Time = 60 mph × 2 hours = 120 miles."),

        # --- 6. Python Algorithms & Coding ---
        ("Write a python function to perform binary search on a sorted list.", 
         "Here is the standard, efficient Python implementation of binary search:\n\n```python\ndef binary_search(arr, target):\n    \"\"\"Perform binary search on a sorted list. Returns index if found, else -1.\"\"\"\n    low = 0\n    high = len(arr) - 1\n    \n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n            \n    return -1\n```"),

        # --- 7. Everyday Practical Explanations ---
        ("Explain what the internet is in two simple sentences.", 
         "The internet is a vast global network that connects billions of computers and devices together worldwide. It allows people to share information, communicate instantly, and access web services from anywhere on Earth."),
        ("What is the use of a knife in the kitchen?", 
         "A knife is an essential kitchen tool used for food preparation tasks such as cutting, slicing, chopping, dicing, mincing, and peeling ingredients like vegetables, fruits, meats, and herbs.")
    ]
    
    # Inject 80 copies of each golden reasoning pair (~1,360 high-signal samples)
    for q, a in golden_qa:
        for _ in range(80):
            input_ids = list(system_ids)
            labels = [IGNORE_INDEX] * len(system_ids)
            
            u_ids, u_lbls = process_turn('User', q)
            a_ids, a_lbls = process_turn('Assistant', a)
            
            input_ids.extend(u_ids + a_ids)
            labels.extend(u_lbls + a_lbls)
            
            if len(input_ids) > MAX_SEQ_LEN:
                input_ids, labels = input_ids[:MAX_SEQ_LEN], labels[:MAX_SEQ_LEN]
            pad_len = MAX_SEQ_LEN - len(input_ids)
            if pad_len > 0:
                input_ids.extend([enc.eot_token] * pad_len)
                labels.extend([IGNORE_INDEX] * pad_len)
                
            all_input_ids.append(input_ids)
            all_labels.append(labels)

    logger.info(f"Total SFT samples prepared: {len(all_input_ids)}")
    logger.info("Converting to PyTorch tensors...")
    input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    out_path = os.path.join(out_dir, "sft_data.pt")
    logger.info(f"Saving to {out_path}...")
    torch.save({"input_ids": input_ids_tensor, "labels": labels_tensor}, out_path)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()

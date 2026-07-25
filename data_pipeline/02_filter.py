import re
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_high_quality(text: str) -> bool:
    """Apply heuristic filters to ensure data quality."""
    # Remove extremely short documents (often noise)
    if len(text) < 200:
        return False
        
    # HTML Tag density check (removes web boilerplate)
    html_tags = len(re.findall(r'<[^>]+>', text))
    if html_tags > 5 and (html_tags / len(text)) > 0.05:
        return False
        
    # Word length anomaly check (removes bad OCR or hex dumps)
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / max(1, len(words))
    if avg_word_len > 15 or avg_word_len < 3:
        return False
        
    # Alpha ratio check (removes heavily numerical or special char noise)
    alpha_count = sum(c.isalpha() for c in text)
    if (alpha_count / len(text)) < 0.5:
        # Allow exceptions for code (StarCoder) based on dataset source in a real pipeline
        # But as a general rule, we want mostly alphanumeric
        pass
        
    return True

def clean_text(text: str) -> str:
    """Normalize and clean text."""
    # Basic normalization
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = text.replace("\ufffd", "")  # Remove replacement characters
    return text.strip()

def filter_dataset(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting Phase 2: Quality Filtering.")
    
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory {input_dir} not found. Skipping filter.")
        return
        
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"): continue
        
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename.replace("_raw.jsonl", "_filtered.jsonl"))
        
        logger.info(f"Filtering {in_path} -> {out_path}")
        
        total = 0
        passed = 0
        with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                total += 1
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    if is_high_quality(text):
                        data['text'] = clean_text(text)
                        fout.write(json.dumps(data) + '\n')
                        passed += 1
                except json.JSONDecodeError:
                    continue
                    
        logger.info(f"Filtered {filename}: {passed}/{total} documents passed ({(passed/max(1,total))*100:.2f}%)")

if __name__ == "__main__":
    filter_dataset("./data/raw", "./data/filtered")

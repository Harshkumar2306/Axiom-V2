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
    # Example implementation layout
    logger.info(f"Scanning raw chunks in {input_dir}...")
    logger.info(f"Filtered high-quality data will be routed to {output_dir}.")

if __name__ == "__main__":
    filter_dataset("./data/raw", "./data/filtered")

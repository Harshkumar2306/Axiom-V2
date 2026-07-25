import os
import json
import logging
import math
import collections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_quality_score(text: str) -> float:
    """Assigns a quality score to the document based on multiple linguistic features."""
    score = 1.0
    
    words = text.split()
    if not words: return 0.0
    
    # 1. Repetition penalty
    unique_words = set(words)
    vocab_richness = len(unique_words) / len(words)
    if vocab_richness < 0.2:
        score -= 0.5  # Heavy penalty for highly repetitive text
        
    # 2. Sentence completeness
    if not text.rstrip()[-1] in ".!?":
        score -= 0.1
        
    # 3. URL density penalty
    url_count = text.count("http://") + text.count("https://")
    if url_count > 3 and (url_count / max(1, len(words) // 100)) > 1.0:
        score -= 0.4
        
    # 4. Excessive capitalization penalty
    caps_count = sum(1 for c in text if c.isupper())
    if caps_count / max(1, len(text)) > 0.4:
        score -= 0.3
        
    # 5. Document entropy (very low entropy = highly repetitive characters/noise)
    char_counts = collections.Counter(text)
    entropy = -sum((count/len(text)) * math.log2(count/len(text)) for count in char_counts.values())
    if entropy < 3.0:
        score -= 0.5
        
    return max(0.0, score)

def score_and_filter(input_dir: str, output_dir: str, threshold: float = 0.5):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting Phase 2: Quality Scoring.")
    
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory {input_dir} not found. Skipping quality scoring.")
        return
        
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"): continue
        
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename.replace("_filtered.jsonl", "_scored.jsonl"))
        
        logger.info(f"Scoring {in_path} -> {out_path} (Threshold: {threshold})")
        
        total = 0
        passed = 0
        with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                total += 1
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    
                    score = compute_quality_score(text)
                    if score >= threshold:
                        data['quality_score'] = score
                        fout.write(json.dumps(data) + '\n')
                        passed += 1
                except json.JSONDecodeError:
                    continue
                    
        logger.info(f"Scored {filename}: {passed}/{total} documents passed ({(passed/max(1,total))*100:.2f}%)")

if __name__ == "__main__":
    score_and_filter("./data/filtered", "./data/scored")

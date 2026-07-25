import os
import logging
from datasketch import MinHash, MinHashLSH
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_minhash(text: str, num_perm: int = 128):
    """Simulate MinHash computation for LSH deduplication."""
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

def run_deduplication(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting Phase 2: MinHash LSH Deduplication.")
    logger.info("Initializing datasketch MinHashLSH index with threshold=0.85...")
    
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory {input_dir} not found. Skipping deduplication.")
        return
        
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"): continue
        
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename.replace("_filtered.jsonl", "_deduped.jsonl"))
        
        logger.info(f"Deduplicating {in_path} -> {out_path}")
        
        total = 0
        passed = 0
        with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                total += 1
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    m = compute_minhash(text)
                    
                    # Query LSH
                    result = lsh.query(m)
                    if not result:
                        # Not a duplicate
                        lsh.insert(f"{filename}_{total}", m)
                        fout.write(line)
                        passed += 1
                except Exception:
                    continue
                    
        logger.info(f"Deduplicated {filename}: {passed}/{total} documents unique ({(passed/max(1,total))*100:.2f}%)")

if __name__ == "__main__":
    run_deduplication("./data/filtered", "./data/deduped")

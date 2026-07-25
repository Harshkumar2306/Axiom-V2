import os
import json
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    dataset_dir = "dataset/v1"
    
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory '{dataset_dir}' does not exist. Run the pipeline first.")
        return
        
    # Check if files exist
    required_files = ["train.bin", "val.bin", "manifest.json"]
    for f in required_files:
        if not os.path.exists(os.path.join(dataset_dir, f)):
            logger.error(f"Missing required file: {f}")
            return
            
    logger.info("Initializing Kaggle Dataset metadata...")
    
    metadata = {
      "title": "Axiom v2 4.5B Token Pretraining Dataset",
      "id": "hrsh0o23/axiom-v2-4-5b-dataset",  # Matches your kaggle.json username
      "licenses": [
        {
          "name": "CC0-1.0"
        }
      ]
    }
    
    meta_path = os.path.join(dataset_dir, "dataset-metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Metadata written to {meta_path}.")
    
    # Check if kaggle CLI is installed and authenticated
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Kaggle CLI is not installed or not found in PATH.")
        logger.info("Please install it via 'pip install kaggle' and ensure your kaggle.json token is in ~/.kaggle/")
        return
        
    logger.info("Uploading dataset to Kaggle (this may take a while for 18GB)...")
    
    try:
        # Use 'create' for the first time, 'version' for updates
        cmd = ["kaggle", "datasets", "create", "-p", dataset_dir]
        
        # If the dataset already exists, this command might fail, in which case you'd use:
        # cmd = ["kaggle", "datasets", "version", "-p", dataset_dir, "-m", "Updated dataset"]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Upload successful!")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Upload failed. Error: {e.stderr}")
        logger.info("If the dataset already exists, try running 'kaggle datasets version -p dataset/v1 -m \"update\"' instead.")

if __name__ == "__main__":
    main()

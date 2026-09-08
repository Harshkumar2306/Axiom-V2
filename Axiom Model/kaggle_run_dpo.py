import os
import sys
import subprocess
import shutil
import time
import signal

def check_disk_space():
    print("[1/3] Checking Kaggle Disk Space...")
    total, used, free = shutil.disk_usage("/")
    print(f"Total: {total // (2**30)} GB, Used: {used // (2**30)} GB, Free: {free // (2**30)} GB")

def run_training():
    print("[2/3] Configuring Training Environment...")
    # Find base model (DPO requires the SFT checkpoint)
    base_model = "checkpoints_sft/best.pt"
    if not os.path.exists(base_model):
        base_model = "best.pt" # Fallback
        
    dataset_path = "dpo_pipeline/dataset/dpo/dpo_data.pt"
    if not os.path.exists(dataset_path):
        dataset_path = "dataset/dpo/dpo_data.pt"
        
    print(f"Using base SFT model: {base_model}")
    print(f"Using DPO dataset: {dataset_path}")
    
    if not os.path.exists(base_model):
        print(f"ERROR: SFT model not found at {base_model}!")
        print("Please ensure your SFT phase 4 completed successfully.")
        return
        
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}!")
        print("Please run the DPO preparation script first.")
        return

    print("\n[3/3] Launching DPO Training Engine...")
    print("="*50)
    print("NOTE: We are using subprocess to bypass Kaggle's Jupyter buffering.")
    print("If you see an OOM error, it will be printed below.")
    print("="*50 + "\n")
    
    # Find free port for torchrun to prevent EADDRINUSE
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=29516", # Different port just in case
        "train_dpo.py",
        "--pretrained", base_model,
        "--data", dataset_path,
        "--save_dir", "checkpoints_dpo"
    ]
    
    # Run process unbuffered and stream directly to stdout
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1 # Line buffered
    )
    
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down DPO training gracefully...")
        process.send_signal(signal.SIGINT)
        process.wait()
        
    process.wait()
    if process.returncode != 0:
        print(f"\n[FATAL] Training exited with code {process.returncode}")
        print("This usually means PyTorch ran out of VRAM/RAM or a file was missing.")
    else:
        print("\n[SUCCESS] DPO Training completed.")

if __name__ == "__main__":
    check_disk_space()
    run_training()

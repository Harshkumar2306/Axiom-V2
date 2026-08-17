import os
import sys
import subprocess
import shutil
import time
import signal

def check_and_clear_space():
    print("[1/3] Checking Kaggle Disk Space...")
    total, used, free = shutil.disk_usage("/")
    print(f"Total: {total // (2**30)} GB, Used: {used // (2**30)} GB, Free: {free // (2**30)} GB")
    
    # We want to ensure we have enough space.
    # The limit is 19.52 GB on Kaggle. Let's remove old temporary files.
    checkpoints_dir = "checkpoints_sft"
    if os.path.exists(checkpoints_dir):
        print(f"Clearing old checkpoints in {checkpoints_dir} to free up space...")
        for f in os.listdir(checkpoints_dir):
            # Only delete failed .tmp files. Do NOT delete latest.pt or best.pt 
            # otherwise the training cannot resume!
            if ".tmp" in f:
                os.remove(os.path.join(checkpoints_dir, f))
                print(f"Deleted {f}")

def run_training():
    print("[2/3] Configuring Training Environment...")
    # Find base model
    base_model = "../checkpoints/best.pt"
    if not os.path.exists(base_model):
        base_model = "best.pt" # Fallback if they copied it locally
        
    dataset_path = "sft_pipeline/dataset/sft/sft_data.pt"
    if not os.path.exists(dataset_path):
        dataset_path = "dataset/sft/sft_data.pt"
        
    print(f"Using base model: {base_model}")
    print(f"Using dataset: {dataset_path}")
    
    if not os.path.exists(base_model):
        print(f"ERROR: Base model not found at {base_model}!")
        print("Please ensure your pre-trained model is available.")
        return
        
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}!")
        return

    print("\n[3/3] Launching SFT Training Engine...")
    print("="*50)
    print("NOTE: We are using subprocess to bypass Kaggle's Jupyter buffering.")
    print("If you see an OOM error, it will be printed below.")
    print("="*50 + "\n")
    
    # Find free port for torchrun to prevent EADDRINUSE
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=29515",
        "train_sft.py",
        "--pretrained", base_model,
        "--data", dataset_path,
        "--save_dir", "checkpoints_sft"
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
        print("\nInterrupted by user. Shutting down training gracefully...")
        process.send_signal(signal.SIGINT)
        process.wait()
        
    process.wait()
    if process.returncode != 0:
        print(f"\n[FATAL] Training exited with code {process.returncode}")
        print("This usually means PyTorch ran out of VRAM/RAM or a file was missing.")
    else:
        print("\n[SUCCESS] Training completed.")

if __name__ == "__main__":
    check_and_clear_space()
    run_training()

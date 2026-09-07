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
    # Robust search for base pretrained model
    base_model = None
    candidate_paths = [
        "checkpoints/best.pt",
        "../checkpoints/best.pt",
        "/kaggle/working/Axiom-V2/checkpoints/best.pt",
        "best.pt"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            base_model = p
            break
            
    if not base_model:
        # Recursive search for any best.pt outside checkpoints_sft
        for root, dirs, files in os.walk("/kaggle/working"):
            if "best.pt" in files and "checkpoints_sft" not in root and "checkpoints_dpo" not in root:
                base_model = os.path.join(root, "best.pt")
                break
        
    dataset_path = "dataset/sft/sft_data.pt"
    if not os.path.exists(dataset_path):
        dataset_path = "sft_pipeline/dataset/sft/sft_data.pt"
        
    # Check for pretrain replay dataset (10% Mixed Replay Anti-Forgetting)
    replay_path = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.bin" in files:
            replay_path = os.path.join(root, "train.bin")
            break

    print(f"Using base model: {base_model}")
    print(f"Using dataset   : {dataset_path}")
    if replay_path:
        print(f"🌟 Anti-Forgetting 10% Mixed Replay ACTIVE: {replay_path}")
    else:
        print("ℹ️ Standard SFT mode (train.bin not found in /kaggle/input, running pure SFT)")
    
    if not base_model or not os.path.exists(base_model):
        print(f"ERROR: Base model not found! Checked: {candidate_paths}")
        print("Please ensure your pre-trained model is available in checkpoints/best.pt.")
        return
        
    if not os.path.exists(dataset_path):
        print(f"ERROR: SFT Dataset not found at {dataset_path}!")
        print("Please run 'python sft_pipeline/01_prepare_sft.py' first.")
        return

    print("\n[3/3] Launching SFT Training Engine...")
    print("="*50)
    print("NOTE: We are using subprocess to bypass Kaggle's Jupyter buffering.")
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
    if replay_path:
        cmd.extend(["--replay_data", replay_path])
    
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

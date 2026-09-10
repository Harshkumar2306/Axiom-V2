import os
import sys
import argparse
from huggingface_hub import HfApi

def deploy():
    parser = argparse.ArgumentParser(
        description="Deploy Axiom V2 Backend (Model, Vector RAG & Search) to Hugging Face Spaces"
    )
    parser.add_argument(
        "--repo_id", 
        required=True, 
        help="Target Hugging Face Space ID (e.g., 'your-username/axiom-v2')"
    )
    parser.add_argument(
        "--token", 
        default=None, 
        help="Hugging Face User Access Token (Write access). If omitted, uses local token from `huggingface-cli login`"
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Axiom Model"))
    checkpoint_file = os.path.join(model_dir, "best.pt")

    if not os.path.exists(checkpoint_file):
        print(f"❌ Error: Checkpoint not found at {checkpoint_file}!")
        print("Please ensure `best.pt` is present (run `git lfs pull` if needed).")
        sys.exit(1)

    print(f"📦 Checkpoint verified: {checkpoint_file} ({os.path.getsize(checkpoint_file) / (1024*1024):.1f} MB)")
    print(f"🚀 Deploying 'Axiom Model' to Hugging Face Space: https://huggingface.co/spaces/{args.repo_id}...")

    api = HfApi(token=args.token)

    try:
        api.upload_folder(
            folder_path=model_dir,
            repo_id=args.repo_id,
            repo_type="space",
            ignore_patterns=[
                "__pycache__/*", 
                "*.pyc", 
                "runs/*", 
                "checkpoints/*", 
                "dataset/*", 
                "venv/*",
                ".DS_Store"
            ]
        )
        print("\n🎉 SUCCESS! All model files, Vector RAG engine, and Docker configuration uploaded.")
        print(f"👉 Monitor build logs at: https://huggingface.co/spaces/{args.repo_id}")
        print(f"👉 Once built, your API endpoint will be: https://{args.repo_id.replace('/', '-')}.hf.space")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    deploy()

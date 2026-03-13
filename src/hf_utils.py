import os
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "GibeiML/gibei-model"

def upload_model(file_path, ticker):
    """Upload a single model file to Hugging Face Hub."""
    if not HF_TOKEN:
        print("  [WARN] HF_TOKEN not found. Skipping upload.")
        return False
    
    api = HfApi()
    file_name = os.path.basename(file_path)
    
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=REPO_ID,
            token=HF_TOKEN,
            repo_type="model"
        )
        print(f"  [OK] {ticker} model uploaded to Hugging Face.")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to upload {ticker} to HF: {e}")
        return False

def download_model(ticker, destination_dir):
    """Download a single model file from Hugging Face Hub if missing."""
    file_name = f"{ticker}.pkl"
    local_path = os.path.join(destination_dir, file_name)
    
    if os.path.exists(local_path):
        return local_path
    
    print(f"  [INFO] Downloading {ticker} model from Hugging Face...")
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_name,
            local_dir=destination_dir,
            token=HF_TOKEN # Optional for public, required for private
        )
        return path
    except Exception as e:
        print(f"  [WARN] Could not download {ticker} from HF: {e}")
        return None

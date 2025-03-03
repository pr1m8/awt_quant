from huggingface_hub import snapshot_download

# Define where to store the model
MODEL_REPO = "time-series-foundation-models/Lag-Llama"
MODEL_DIR = "resources/lag-llama"

def main():
    """Downloads the Lag-Llama model into resources/lag-llama."""
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR, resume_download=True)
    print(f"Model downloaded to {MODEL_DIR}")

if __name__ == "__main__":
    main()

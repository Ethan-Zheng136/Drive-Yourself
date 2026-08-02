from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, required=True)
parser.add_argument("--local_dir", type=str, required=True)
args = parser.parse_args()

repo_id = args.repo_id
local_dir = args.local_dir

print(f"--- Using Python snapshot_download ---")
print(f"--- Starting download for model {repo_id} to {local_dir} ---")

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, 
        resume_download=True
    )
    print(f"--- Model download complete ---")

except Exception as e:
    print(f"--- An error occurred during download ---")
    print(e)
    print("--- Please check the repo_id and your network connection ---")
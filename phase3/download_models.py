import os
from huggingface_hub import hf_hub_download

repo_id = "manyajain111/lma-phase2"
files_to_download = [
    "hindi/model_final.pt",
    "hindi/config.json",
    "hindi/tokenizer.json",
    "nepali/model_final.pt",
    "nepali/config.json",
    "nepali/tokenizer.json"
]

out_dir = "checkpoints/pretrained"
os.makedirs(out_dir, exist_ok=True)

for f in files_to_download:
    print(f"Downloading {f}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=f, local_dir=out_dir)
    print(f"Saved to {local_path}")

print("All downloads complete.")

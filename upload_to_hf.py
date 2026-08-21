import time
from huggingface_hub import HfApi, create_repo

token_hindi = "***REMOVED***"
token_nepali = "***REMOVED***"

api_hindi = HfApi(token=token_hindi)
api_nepali = HfApi(token=token_nepali)

username_hindi = api_hindi.whoami()["name"]
username_nepali = api_nepali.whoami()["name"]

hindi_repo = f"{username_hindi}/hindi-lma-corpus"
nepali_repo = f"{username_nepali}/nepali-lma-corpus"

print("Creating repositories...")
try:
    create_repo(repo_id=hindi_repo, token=token_hindi, repo_type="dataset", private=True, exist_ok=True)
    create_repo(repo_id=nepali_repo, token=token_nepali, repo_type="dataset", private=True, exist_ok=True)
except Exception as e:
    print(f"Error creating repos: {e}")



print("Uploading Nepali dataset...")
try:
    api_nepali.upload_file(
        path_or_fileobj="/home/manya/LMA/nepali_submission_phase1.tar.gz",
        path_in_repo="nepali_submission_phase1.tar.gz",
        repo_id=nepali_repo,
        repo_type="dataset",
        token=token_nepali
    )
    print("Nepali dataset uploaded successfully!")
except Exception as e:
    print(f"Error uploading Nepali: {e}")

print("All uploads complete!")

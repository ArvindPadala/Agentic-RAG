import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

print("Starting deployment to Hugging Face Spaces...")

api = HfApi(token=os.environ.get("HF_TOKEN"))

api.upload_folder(
    folder_path=".",
    repo_id="ArvindPadala/Agentic-Document-RAG",
    repo_type="space",
    ignore_patterns=[".git", ".env", "deploy.py", "find_*.py", "fix_*.py", "super_fix.py", ".pytest_cache/**", "__pycache__/**"],
)

print("Deployment successful!")

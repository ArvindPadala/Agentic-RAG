from huggingface_hub import HfApi

print("Starting deployment to Hugging Face Spaces...")

api = HfApi(token="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")

api.upload_folder(
    folder_path=".",
    repo_id="ArvindPadala/Agentic-Document-RAG",
    repo_type="space",
    ignore_patterns=[".git", ".env", "deploy.py", "find_*.py", "fix_*.py", "super_fix.py", ".pytest_cache/**", "__pycache__/**"],
)

print("Deployment successful!")

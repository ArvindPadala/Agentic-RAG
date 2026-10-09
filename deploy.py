import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Exact source allowlist: Git ignore rules are not a deployment data boundary.
# Public corpus provisioning is a separate explicit step, not a workspace copy.
SPACE_SOURCE_FILES = (
    "app.py", "agent.py", "access_control.py", "config.py",
    "supabase_backend.py", "private_sessions.py",
    "gemini_helpers.py", "hybrid_search.py", "llm_router.py",
    "live_guardrail.py", "query_optimizer.py", "upload_handler.py",
    "visual_grounding_helper.py", "utils/logger.py", "README.md",
    "requirements.txt", "requirements-prod.txt",
    "SUPABASE_SETUP.md", "supabase/migrations/202610070001_user_memory.sql",
)

def main():
    load_dotenv()
    print("Starting source-only deployment to Hugging Face Spaces...")

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    api.upload_folder(
        folder_path=".",
        repo_id="ArvindPadala/Agentic-Document-RAG",
        repo_type="space",
        allow_patterns=list(SPACE_SOURCE_FILES),
    )

    print("Source deployment successful! Existing remote artifacts were not removed.")


if __name__ == "__main__":
    main()

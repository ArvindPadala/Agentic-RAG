"""Check upload selection against the SDK without uploading any files."""

from unittest.mock import patch

from huggingface_hub.utils import filter_repo_objects

from deploy import SPACE_SOURCE_FILES, main


def test_space_allowlist_excludes_private_and_nested_artifacts():
    candidates = [
        "app.py", "access_control.py", "utils/logger.py", "requirements.txt",
        "memory.json", "chroma_db/chroma.sqlite3", "documents/private.pdf",
        "document_chunks/private.json", "input/documents/private.pdf",
        "output/chunks/private.json", ".env", ".env.production",
        "ade_lambda.zip", "pipeline_setup.ipynb", "eval/golden_dataset.csv",
        "private/app.py", ".opencode/private.py", "utils/private_credentials.py",
    ]
    selected = list(filter_repo_objects(candidates, allow_patterns=SPACE_SOURCE_FILES))
    assert set(selected) == {
        "app.py", "access_control.py", "utils/logger.py", "requirements.txt",
    }


def test_deployment_passes_allowlist_without_executing_remote_calls():
    with patch("deploy.load_dotenv"), patch("deploy.HfApi") as api:
        main()
    options = api.return_value.upload_folder.call_args.kwargs
    assert options["allow_patterns"] == list(SPACE_SOURCE_FILES)
    assert options["repo_type"] == "space"

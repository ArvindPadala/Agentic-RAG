"""Offline test environment: never load workspace credentials or trace fixtures."""

import os


os.environ.update({
    "PYTHON_DOTENV_DISABLED": "1",
    "GEMINI_API_KEY": "mock_key_for_ci",
    "GEMINI_API_KEY_2": "mock_key_2_for_ci",
    "AWS_ACCESS_KEY_ID": "mock",
    "AWS_SECRET_ACCESS_KEY": "mock",
    "AWS_EC2_METADATA_DISABLED": "true",
    "S3_BUCKET": "mock",
    "APP_MODE": "public_demo",
    "SUPABASE_ENABLED": "false",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "ANONYMIZED_TELEMETRY": "false",
    "GRADIO_ANALYTICS_ENABLED": "false",
})
# Hosted-environment checks have dedicated tests rather than depending on the
# runner (including when this test suite runs inside a Space).
os.environ.pop("SPACE_ID", None)

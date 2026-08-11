import os
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger("config")

# Load environment variables from .env if present
load_dotenv()


class Config:
    """
    Central configuration class.
    Validates required environment variables upon initialization.
    """
    def __init__(self):
        # API Keys
        self.GEMINI_API_KEY = self._get_required("GEMINI_API_KEY")
        self.GEMINI_API_KEY_2 = self._get_optional("GEMINI_API_KEY_2")
        self.GEMINI_API_KEYS = [self.GEMINI_API_KEY]
        if self.GEMINI_API_KEY_2:
            self.GEMINI_API_KEYS.append(self.GEMINI_API_KEY_2)
            
        self.VISION_AGENT_API_KEY = self._get_optional("VISION_AGENT_API_KEY")

        # AWS Configuration (Local RAG)
        self.AWS_ACCESS_KEY_ID = self._get_optional("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = self._get_optional("AWS_SECRET_ACCESS_KEY")
        self.AWS_REGION = self._get_optional("AWS_REGION", "us-east-1")
        self.S3_BUCKET_NAME = self._get_optional("S3_BUCKET")

        # ADE Handler Defaults
        self.ADE_MODEL = self._get_optional("ADE_MODEL", "dpt-2-latest")
        self.INPUT_FOLDER = self._get_optional("INPUT_FOLDER", "input/")
        self.OUTPUT_FOLDER = self._get_optional("OUTPUT_FOLDER", "output/")
        self.FORCE_REPROCESS = self._get_optional("FORCE_REPROCESS", "false").lower() == "true"

        # LangSmith Observability
        self.LANGCHAIN_TRACING_V2 = self._get_optional("LANGCHAIN_TRACING_V2", "false")
        self.LANGCHAIN_API_KEY = self._get_optional("LANGCHAIN_API_KEY")
        self.LANGCHAIN_PROJECT = self._get_optional("LANGCHAIN_PROJECT", "Agentic_RAG")

    def _get_required(self, key: str) -> str:
        value = os.environ.get(key)
        if not value:
            logger.error(f"Missing required environment variable: {key}")
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    def _get_optional(self, key: str, default: str = None) -> str:
        return os.environ.get(key, default)


# Global config instance
try:
    settings = Config()
except ValueError as e:
    logger.critical(f"Configuration initialization failed: {e}")
    # In a production app, we might exit here if imported at top level
    # sys.exit(1)

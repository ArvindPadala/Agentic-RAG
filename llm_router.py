import time
from typing import List, Any
from google import genai
from google.genai.errors import APIError
from utils.logger import get_logger
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langsmith import traceable

logger = get_logger("llm_router")

class GeminiRouter:
    """
    A robust LLM router that handles multiple Gemini API keys and automatically
    falls back between models when rate limits (429) or service outages (503) occur.
    """
    def __init__(self, api_keys: List[str], models: List[str] = None):
        if not api_keys:
            raise ValueError("At least one API key must be provided.")
            
        self.api_keys = api_keys
        self.clients = [genai.Client(api_key=key) for key in api_keys]
        
        # Default priority fallback models
        self.fallback_models = models or [
            "models/gemini-3.6-flash",
            "models/gemini-3.5-flash",
            "models/gemini-3.5-flash-lite"
        ]
        
        self.current_key_idx = 0
        self.current_model_idx = 0
        
        # We need this to maintain a compatible interface for code that expects
        # client.models.generate_content
        self._models_proxy = self._ModelsProxy(self)
        
    @property
    def models(self):
        """Used to mock `client.models` access"""
        return self._models_proxy

    class _ModelsProxy:
        def __init__(self, router):
            self.router = router
            
        def generate_content(self, *args, **kwargs):
            return self.router.generate_content(*args, **kwargs)

    def _rotate_key_or_model(self):
        """
        Rotates to the next API key. If all API keys have been exhausted for the current model,
        rotates to the next fallback model.
        """
        self.current_key_idx += 1
        
        if self.current_key_idx >= len(self.clients):
            # Exhausted all keys, rotate model
            self.current_key_idx = 0
            self.current_model_idx += 1
            
            if self.current_model_idx >= len(self.fallback_models):
                # Exhausted all models too! Reset to top and allow exponential backoff to handle it
                logger.warning("🚨 Exhausted all API keys and all fallback models! Resetting to primary and sleeping...")
                self.current_model_idx = 0
            else:
                logger.info(f"🔄 Switching to fallback model: {self.fallback_models[self.current_model_idx]}")
        else:
            logger.info(f"🔄 Rotating to API Key #{self.current_key_idx + 1}")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(APIError),
        reraise=True
    )
    def generate_content(self, contents: Any, config: Any = None, **kwargs):
        """
        Wraps genai.models.generate_content with routing and backoff logic.
        """
        if 'model' in kwargs:
            kwargs.pop('model')
            
        client = self.clients[self.current_key_idx]
        model_name = self.fallback_models[self.current_model_idx]
        
        @traceable(run_type="chain", name="gemini_generate_content")
        def _execute(c, cfg, kw):
            logger.debug(f"Attempting generation with Key #{self.current_key_idx + 1} on model {model_name}")
            return client.models.generate_content(
                model=model_name,
                contents=c,
                config=cfg,
                **kw
            )
            
        try:
            return _execute(contents, config, kwargs)
            
        except APIError as e:
            if hasattr(e, 'code') and e.code in [404, 429, 503]:
                logger.warning(f"⚠️ Caught {e.code} API Error from Key #{self.current_key_idx + 1} ({model_name}). Rotating...")
                self._rotate_key_or_model()
            raise e # Reraise for tenacity to catch and trigger exponential backoff if rotation doesn't immediately solve it

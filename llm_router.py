"""Request-local Gemini routing, retry guidance and cooperative provider budgets."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
import math
import os
import random
import re
from threading import RLock
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError
import httpx

from utils.logger import get_logger

logger = get_logger("llm_router")
_ACTIVE_BUDGET = ContextVar("provider_request_budget", default=None)
_TRANSIENT_CODES = frozenset({408, 429, 500, 502, 503, 504})


class ProviderTimeoutError(TimeoutError):
    """Safe timeout message; never contains a URL, API key or request payload."""


class ProviderDeadlineExceeded(ProviderTimeoutError):
    def __init__(self):
        super().__init__("The request reached its provider time budget. Try a narrower question.")


class ProviderAttemptsExceeded(RuntimeError):
    def __init__(self):
        super().__init__("The request reached its provider attempt limit. Please retry later.")


class ProviderRequestError(RuntimeError):
    def __init__(self, code=None):
        self.code = code if type(code) is int else None
        super().__init__(f"Provider request failed (HTTP {self.code})." if self.code else "Provider request failed.")


@dataclass(frozen=True)
class RouterPolicy:
    turn_timeout_seconds: float = 90.0
    call_timeout_seconds: float = 30.0
    max_attempts: int = 12
    max_attempts_per_call: int = 3

    def __post_init__(self):
        for name, maximum in (("turn_timeout_seconds", 300), ("call_timeout_seconds", 60)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0.1 <= value <= maximum:
                raise ValueError(f"{name} must be finite and between 0.1 and {maximum}")
        for name, maximum in (("max_attempts", 100), ("max_attempts_per_call", 6)):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= maximum:
                raise ValueError(f"{name} must be an integer between 1 and {maximum}")

    @classmethod
    def from_environment(cls):
        mapping = {"PROVIDER_TURN_TIMEOUT_SECONDS": ("turn_timeout_seconds", float),
                   "PROVIDER_CALL_TIMEOUT_SECONDS": ("call_timeout_seconds", float),
                   "PROVIDER_MAX_ATTEMPTS": ("max_attempts", int),
                   "PROVIDER_MAX_ATTEMPTS_PER_CALL": ("max_attempts_per_call", int)}
        values = {}
        for variable, (field, convert) in mapping.items():
            if variable in os.environ:
                try:
                    values[field] = convert(os.environ[variable])
                except ValueError:
                    raise ValueError(f"Invalid {variable}") from None
        return cls(**values)


class RequestBudget:
    """One turn, shared across phases/worker contexts. No sleeping under the lock."""

    def __init__(self, policy, execution=None, *, clock=None, sleep=None):
        self.policy = policy
        self.execution = execution
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.deadline = self._clock() + policy.turn_timeout_seconds
        self.attempts = 0
        self._routes = {}
        self._not_before = 0.0
        self._terminal_error = None
        self._lock = RLock()
        if execution:
            execution.set_provider_limits(policy.turn_timeout_seconds, policy.max_attempts,
                                          policy.call_timeout_seconds, policy.max_attempts_per_call)

    def check(self):
        if self._clock() >= self.deadline:
            raise ProviderDeadlineExceeded()

    def route(self, key):
        with self._lock:
            return self._routes.get(key, 0)

    def set_route(self, key, index):
        with self._lock:
            self._routes[key] = index

    def stop(self, error):
        with self._lock:
            self._terminal_error = error

    def defer(self, seconds, error):
        with self._lock:
            self.check()
            # Do not truncate a provider's minimum delay or bypass it in another
            # phase after the optimiser/checker catches the original failure.
            if seconds >= self.deadline - self._clock():
                self._terminal_error = error
            else:
                self._not_before = max(self._not_before, self._clock() + seconds)

    def reserve(self, model, call_timeout_seconds):
        while True:
            with self._lock:
                self.check()
                if self._terminal_error:
                    raise self._terminal_error
                if self.attempts >= self.policy.max_attempts:
                    raise ProviderAttemptsExceeded()
                delay = self._not_before - self._clock()
                if delay <= 0:
                    milliseconds = int(1000 * min(call_timeout_seconds, self.policy.call_timeout_seconds,
                                                  self.deadline - self._clock()))
                    if milliseconds < 1:
                        raise ProviderDeadlineExceeded()
                    self.attempts += 1
                    if self.execution:
                        self.execution.provider_attempt_started(model)
                    return milliseconds
            started = self._clock()
            try:
                self._sleep(delay)
            finally:
                if self.execution:
                    self.execution.provider_retry_waited(max(0.0, self._clock() - started))


@contextmanager
def provider_request(policy=None, execution=None):
    existing = _ACTIVE_BUDGET.get()
    if existing is not None:
        yield existing
        return
    budget = RequestBudget(policy or RouterPolicy.from_environment(), execution)
    token = _ACTIVE_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _ACTIVE_BUDGET.reset(token)


def check_provider_budget():
    budget = _ACTIVE_BUDGET.get()
    if budget is not None:
        budget.check()


def _error_details(error):
    details = getattr(error, "details", {})
    if isinstance(details, dict):
        details = details.get("error", details)
        if isinstance(details, dict) and isinstance(details.get("details"), list):
            return [item for item in details["details"] if isinstance(item, dict)]
    return []


def _permanent_quota(error):
    for detail in _error_details(error):
        if detail.get("@type") != "type.googleapis.com/google.rpc.QuotaFailure":
            continue
        violations = detail.get("violations", [])
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            identifier = str(violation.get("quotaId", "")).lower()
            if any(part in identifier for part in ("perday", "per_day", "daily")) or violation.get("quotaValue") in (0, "0"):
                return True
    return False


def _retry_after(error):
    delays = []
    headers = getattr(getattr(error, "response", None), "headers", None)
    raw = headers.get("Retry-After") if headers is not None else None
    if isinstance(raw, str):
        try:
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw.strip()):
                delay = float(raw)
            else:
                date = parsedate_to_datetime(raw)
                delay = date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp() - time.time()
            if math.isfinite(delay):
                delays.append(max(0.0, delay))
        except (TypeError, ValueError, OverflowError):
            pass
    for detail in _error_details(error):
        delay = detail.get("retryDelay")
        if (detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo"
                and isinstance(delay, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]{1,9})?s", delay)):
            seconds = float(delay[:-1])
            if math.isfinite(seconds):
                delays.append(seconds)
    return max(delays, default=0.0)


class GeminiRouter:
    """Shared SDK clients, immutable policy, and request-local route/cooldown state."""

    def __init__(self, api_keys, models=None, *, policy=None):
        if (not isinstance(api_keys, (list, tuple)) or not api_keys
                or any(not isinstance(key, str) or not key.strip() for key in api_keys)):
            raise ValueError("At least one non-empty API key is required")
        self.api_keys = tuple(dict.fromkeys(key.strip() for key in api_keys))
        self.policy = policy or RouterPolicy.from_environment()
        if models is not None and not isinstance(models, (list, tuple)):
            raise ValueError("Model identifiers must be a list or tuple")
        self.fallback_models = tuple(dict.fromkeys(models if models is not None else (
            "models/gemini-3.6-flash", "models/gemini-3.5-flash-lite")))
        if not self.fallback_models or any(not self._valid_model(model) for model in self.fallback_models):
            raise ValueError("Explicit non-empty model identifiers are required")
        self.clients = tuple(genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=int(self.policy.call_timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1))) for key in self.api_keys)
        self._models_proxy = self._ModelsProxy(self)

    @staticmethod
    def _valid_model(model):
        return isinstance(model, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", model))

    @property
    def models(self):
        return self._models_proxy

    class _ModelsProxy:
        def __init__(self, router):
            self.router = router

        def generate_content(self, *args, **kwargs):
            return self.router.generate_content(*args, **kwargs)

    @staticmethod
    def _request_config(config, milliseconds):
        if config is None:
            copied = types.GenerateContentConfig()
        elif isinstance(config, dict):
            copied = types.GenerateContentConfig(**config)
        else:
            copied = config.model_copy()
        http = copied.http_options.model_copy() if copied.http_options else types.HttpOptions()
        if http.timeout is not None and http.timeout > 0:
            milliseconds = min(milliseconds, http.timeout)
        http.timeout = milliseconds
        http.retry_options = types.HttpRetryOptions(attempts=1)
        http.headers = {**{name: value for name, value in (http.headers or {}).items()
                          if name.lower() != "x-server-timeout"},
                        "X-Server-Timeout": str(math.ceil(milliseconds / 1000))}
        copied.http_options = http
        # LangGraph owns the tool loop. Do not allow hidden SDK calls to evade
        # wire-dispatch accounting or the per-turn attempt budget.
        copied.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
        return copied

    def generate_content(self, contents, config=None, **kwargs):
        with provider_request(self.policy) as budget:
            requested = kwargs.pop("model", None) or self.fallback_models[0]
            if not self._valid_model(requested):
                raise ValueError("Invalid model identifier")
            parsed_config = types.GenerateContentConfig.model_validate(config if config is not None else {})
            models = tuple(dict.fromkeys((requested, *self.fallback_models)))
            routes = [(model, key) for model in models for key in range(len(self.clients))]
            route_key = (id(self), requested)
            index = budget.route(route_key) % len(routes)
            limit = min(self.policy.max_attempts_per_call, budget.policy.max_attempts_per_call)
            for attempt in range(1, limit + 1):
                model, key = routes[index]
                milliseconds = budget.reserve(model, self.policy.call_timeout_seconds)
                request_config = self._request_config(parsed_config, milliseconds)
                try:
                    response = self.clients[key].models.generate_content(
                        model=model, contents=contents, config=request_config, **kwargs)
                except APIError as error:
                    safe = ProviderRequestError(error.code)
                    if error.code in (401, 403) or (error.code == 429 and _permanent_quota(error)):
                        budget.stop(safe)
                        raise safe from None
                    if error.code not in _TRANSIENT_CODES:
                        raise safe from None
                    delay = max(_retry_after(error), random.uniform(min(8, 2 ** (attempt - 1)) / 2,
                                                                    min(8, 2 ** (attempt - 1))))
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    safe = ProviderRequestError()
                    delay = random.uniform(0.5, min(8, 2 ** (attempt - 1)))
                except httpx.TimeoutException:
                    # A read/write timeout can have generated a billable answer.
                    # Do not automatically dispatch that ambiguous request again.
                    raise ProviderTimeoutError("The provider call timed out. Please retry.") from None
                except httpx.TransportError:
                    raise ProviderRequestError() from None
                else:
                    budget.check()  # Late responses must not enter history/grounding.
                    budget.set_route(route_key, index)
                    return response
                index = (index + 1) % len(routes)
                budget.set_route(route_key, index)
                budget.defer(delay, safe)
                logger.warning("Transient provider failure (status %s, dispatch %s/%s)", safe.code, attempt, limit)
                if attempt == limit:
                    raise safe from None

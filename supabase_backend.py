"""Small stateless Supabase REST adapter using user JWTs, never an admin key."""

import base64
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from uuid import UUID

import requests


class SupabaseError(RuntimeError):
    """Safe public error; provider response bodies/credentials are not exposed."""


class AuthenticationError(SupabaseError):
    pass


class MemoryConflict(SupabaseError):
    pass


def empty_memory():
    return {"session_summaries": [], "preferences": {}, "facts": []}


def validate_memory(memory):
    if not isinstance(memory, dict) or set(memory) != set(empty_memory()):
        raise SupabaseError("Invalid memory structure.")
    for key, maximum in (("session_summaries", 10), ("facts", 20)):
        values = memory[key]
        if (not isinstance(values, list) or len(values) > maximum
                or any(not isinstance(v, str) or len(v) > 2000 for v in values)):
            raise SupabaseError("Invalid memory entries.")
    prefs = memory["preferences"]
    if (not isinstance(prefs, dict) or len(prefs) > 50
            or any(not isinstance(k, str) or not isinstance(v, str)
                   or len(k) > 100 or len(v) > 2000 for k, v in prefs.items())):
        raise SupabaseError("Invalid memory preferences.")
    if len(json.dumps(memory, ensure_ascii=False).encode("utf-8")) > 16000:
        raise SupabaseError("Memory exceeds the demo storage limit.")
    return memory


def validate_public_key(key):
    if isinstance(key, str) and key.startswith("sb_publishable_"):
        return
    # Legacy anon keys are supported, but service-role/secret keys are rejected.
    # This payload check is only key-configuration validation, NOT JWT identity
    # verification. User JWTs are always checked remotely through /auth/v1/user.
    try:
        payload = key.split(".")[1]
        role = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["role"]
        if role == "anon":
            return
    except (AttributeError, IndexError, KeyError, ValueError, TypeError):
        pass
    raise ValueError("Use a Supabase publishable or legacy anon key, never a secret/service-role key")


@dataclass(frozen=True)
class MemorySnapshot:
    user_id: str
    revision: int
    memory: dict = field(repr=False)
    access_token: str = field(repr=False)


class SupabaseBackend:
    def __init__(self, url, public_key):
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname
                or not parsed.hostname.endswith(".supabase.co")
                or parsed.username or parsed.password or parsed.port
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("SUPABASE_URL must be the HTTPS project URL")
        validate_public_key(public_key)
        self.url = url.rstrip("/")
        self.public_key = public_key

    def _request(self, method, path, *, token=None, **kwargs):
        headers = {"apikey": self.public_key, "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        try:
            response = requests.request(
                method, self.url + path, headers=headers,
                timeout=(3, 10), allow_redirects=False, **kwargs,
            )
        except requests.RequestException:
            raise SupabaseError("Private-session storage is temporarily unavailable.") from None
        if response.status_code in (401, 403):
            raise AuthenticationError("Your private session is unavailable or expired. Start a new session.")
        if response.status_code == 429:
            raise SupabaseError("Private-session quota reached. Please try again later.")
        if response.status_code == 409:
            raise MemoryConflict("Memory changed in another request. Retry saving your session.")
        if not 200 <= response.status_code < 300:
            raise SupabaseError("Private-session storage could not complete the request.")
        if response.status_code == 204:
            return None
        try:
            return response.json()
        except ValueError:
            raise SupabaseError("Private-session storage returned an invalid response.") from None

    def verify_user(self, token):
        if not isinstance(token, str) or not token or len(token) > 3500:
            raise AuthenticationError("A valid private session is required.")
        user = self._request("GET", "/auth/v1/user", token=token)
        try:
            # Identity comes ONLY from the trusted Auth API, not browser IDs,
            # user_metadata, or locally decoded JWT claims.
            return str(UUID(user["id"]))
        except (KeyError, TypeError, ValueError, AttributeError):
            raise AuthenticationError("Private-session identity could not be verified.") from None

    def start_guest(self, captcha_token):
        if not isinstance(captcha_token, str) or not captcha_token or len(captcha_token) > 4096:
            raise AuthenticationError("Complete the verification challenge before starting a session.")
        started_at = time.time()
        data = self._request("POST", "/auth/v1/signup", json={
            "data": {}, "gotrue_meta_security": {"captcha_token": captcha_token},
        })
        try:
            token = data["access_token"]
            expires_in = data["expires_in"]
            if isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)) or expires_in <= 0:
                raise ValueError
            self.verify_user(token)
            # Do not retain refresh or provider tokens. This is an explicitly
            # temporary demo session, not a permanent/recoverable account.
            return token, int(started_at + min(expires_in, 3600))
        except (KeyError, TypeError, ValueError):
            raise AuthenticationError("Private session could not be started.") from None

    def load_memory(self, token):
        user_id = self.verify_user(token)
        rows = self._request("GET", "/rest/v1/user_memories", token=token, params={
            "user_id": "eq." + user_id, "select": "user_id,memory,revision,is_closed", "limit": "1",
        })
        if rows == []:
            return MemorySnapshot(user_id, 0, empty_memory(), token)
        if not isinstance(rows, list) or len(rows) != 1:
            raise SupabaseError("Private-session storage returned an invalid record.")
        return self._snapshot(rows[0], user_id, token)

    @staticmethod
    def _snapshot(row, user_id, token):
        try:
            revision = row["revision"]
            if (row["user_id"] != user_id or isinstance(revision, bool)
                    or not isinstance(revision, int) or revision < 1
                    or not isinstance(row["is_closed"], bool)):
                raise ValueError
            if row["is_closed"]:
                raise AuthenticationError("Your private session is closed. Start a new session.")
            return MemorySnapshot(user_id, revision, validate_memory(row["memory"]), token)
        except (KeyError, TypeError, ValueError):
            raise SupabaseError("Private-session storage returned an invalid record.") from None

    def save_memory(self, snapshot, memory):
        validate_memory(memory)
        # Recheck identity at the write boundary; SQL additionally derives its
        # owner from auth.uid() and runs as SECURITY INVOKER under RLS.
        if self.verify_user(snapshot.access_token) != snapshot.user_id:
            raise AuthenticationError("Private-session identity changed. Start a new session.")
        rows = self._request("POST", "/rest/v1/rpc/save_user_memory",
                             token=snapshot.access_token, json={
                                 "expected_revision": snapshot.revision, "new_memory": memory,
                             })
        if not isinstance(rows, list) or len(rows) != 1:
            raise SupabaseError("Private memory was not saved.")
        result = self._snapshot(rows[0], snapshot.user_id, snapshot.access_token)
        if result.revision != snapshot.revision + 1:
            raise SupabaseError("Private memory revision is inconsistent.")
        return result

    def forget_memory(self, token):
        user_id = self.verify_user(token)
        result = self._request("POST", "/rest/v1/rpc/forget_user_memory", token=token, json={})
        if result != user_id:
            raise SupabaseError("Private memory could not be cleared.")

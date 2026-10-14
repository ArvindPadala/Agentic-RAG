"""Synthetic visitor-memory service for browser validation, never app runtime."""

from copy import deepcopy
from threading import RLock
import time
from uuid import UUID

from supabase_backend import AuthenticationError, MemoryConflict, MemorySnapshot, empty_memory, validate_memory


class VisitorMemoryBackend:
    def __init__(self):
        self.rows = {}
        self.closed = set()
        self.signups = 0
        self._lock = RLock()

    def start_guest(self, captcha):
        assert captcha == "synthetic-browser-captcha"
        with self._lock:
            self.signups += 1
            token = f"synthetic-token-{self.signups}"
            self.rows[token] = MemorySnapshot(str(UUID(int=self.signups)), 0, empty_memory(), token)
            return token, int(time.time() + 3600)

    def load_memory(self, token):
        with self._lock:
            if token not in self.rows or token in self.closed:
                raise AuthenticationError("Your private session is closed. Start a new session.")
            return deepcopy(self.rows[token])

    def save_memory(self, snapshot, memory):
        with self._lock:
            existing = self.load_memory(snapshot.access_token)
            if existing.revision != snapshot.revision:
                raise MemoryConflict("Memory changed. Retry saving.")
            saved = MemorySnapshot(existing.user_id, existing.revision + 1,
                                   deepcopy(validate_memory(memory)), existing.access_token)
            self.rows[existing.access_token] = saved
            return deepcopy(saved)

    def forget_memory(self, token):
        with self._lock:
            if token not in self.closed:
                current = self.load_memory(token)
                self.rows[token] = MemorySnapshot(current.user_id, current.revision + 1, empty_memory(), token)
                self.closed.add(token)


def extract_visitor_memory(memory, conversation, provider, **kwargs):
    questions = [part.text for content in conversation if content.role == "user"
                 for part in content.parts if part.text]
    return {"session_summaries": [questions[-1][:120]],
            "preferences": {"answer_style": "Use cited evidence"}, "facts": []}

"""Request-local operational facts; no remote tracing or hidden model reasoning."""

from contextlib import contextmanager
from html import escape
from threading import RLock
import time


class ExecutionTrace:
    def __init__(self, mode="", on_stage=None):
        self.mode = mode
        self.on_stage = on_stage
        self.started = time.perf_counter()
        self.elapsed = None
        self.status = "running"
        self.stages = []
        self.timings = {}
        self.searches = 0
        self.generation_calls = 0
        self.search_errors = 0
        self.decomposition = "not_run"
        self.subqueries = []
        self.grounding = "not_run"
        self.grounding_reason = "The automated grounding check did not run."
        self.passages = 0
        self.sources = 0
        self.cited = 0
        self._lock = RLock()

    def stage(self, label):
        with self._lock:
            if len(self.stages) < 40:
                self.stages.append(label)
        if self.on_stage:
            try:
                self.on_stage(label)
            except Exception:
                # Progress-display failures must not change research outcomes.
                pass

    @contextmanager
    def measure(self, name, label=None):
        if label:
            self.stage(label)
        started = time.perf_counter()
        try:
            yield
        finally:
            with self._lock:
                self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - started

    def search_started(self):
        with self._lock:
            self.searches += 1

    def search_failed(self):
        with self._lock:
            self.search_errors += 1

    def generation_started(self):
        with self._lock:
            self.generation_calls += 1

    def set_decomposition(self, status, queries=()):
        with self._lock:
            self.decomposition = status
            self.subqueries = list(queries)

    def set_grounding(self, verdict):
        with self._lock:
            self.grounding = verdict.status
            self.grounding_reason = verdict.reason

    def finish(self, status, records=(), cited=()):
        with self._lock:
            self.status = status
            self.elapsed = time.perf_counter() - self.started
            self.passages = len(records)
            self.sources = len({p.source_key or p.title for p in records})
            self.cited = len(cited)

    def snapshot(self):
        with self._lock:
            return {"mode": self.mode, "status": self.status,
                    "elapsed": self.elapsed if self.elapsed is not None else time.perf_counter() - self.started,
                    "stages": list(self.stages), "timings": dict(self.timings),
                    "searches": self.searches, "generation_calls": self.generation_calls,
                    "search_errors": self.search_errors, "decomposition": self.decomposition,
                    "subqueries": list(self.subqueries), "grounding": self.grounding,
                    "grounding_reason": self.grounding_reason, "passages": self.passages,
                    "sources": self.sources, "cited": self.cited}


def execution_status(summary):
    if not summary:
        return "Ready for a research question."
    labels = {
        "completed": "Answer ready — inspect its cited sources.",
        "no_evidence": "Finished without captured document evidence. No source verification is implied.",
        "unavailable": "Reports or private-session access are unavailable. Recheck availability or restart your session.",
        "rate_limited": "The provider rate/quota limit was reached. Retry later; the limit may be temporary or daily.",
        "overloaded": "The provider is temporarily overloaded. Wait a moment and retry.",
        "failed": "The request failed. Retry your question; the previous completed context was preserved.",
        "withheld": "The answer was withheld because source access changed. Retry with the current reports.",
    }
    text = labels.get(summary["status"], "Research is in progress.")
    if summary["search_errors"]:
        text += " Some searches were unavailable; the evidence may be incomplete."
    if summary["grounding"] == "unavailable":
        text += " The automated grounding check was unavailable."
    return text


def render_execution(summary):
    if not summary:
        return "<p>Run a research question to see its actual execution details.</p>"
    parts = [f"<p><strong>Outcome:</strong> {escape(execution_status(summary))}</p>",
             f"<p><strong>Retrieval:</strong> {escape(summary['mode'])}</p>",
             f"<p>Search calls: {summary['searches']} · Agent generation calls: {summary['generation_calls']} · "
             f"Search errors: {summary['search_errors']}</p>",
             f"<p>Retrieved passages: {summary['passages']} · Distinct source identities: {summary['sources']} · "
             f"Cited references: {summary['cited']}</p>",
             f"<p>Processing time: {summary['elapsed']:.2f}s (excluding time waiting in the Gradio queue).</p>",
             f"<p><strong>Query decomposition:</strong> {escape(summary['decomposition'].replace('_', ' '))}</p>"]
    if summary["subqueries"]:
        parts.append("<ol>" + "".join(f"<li>{escape(q)}</li>" for q in summary["subqueries"]) + "</ol>")
    if summary["decomposition"] == "fallback":
        parts.append("<p>The optimiser was unavailable or returned invalid queries; the original question was used.</p>")
    parts.append(f"<p><strong>Automated grounding check:</strong> {escape(summary['grounding'].replace('_', ' ').title())}<br>"
                 f"{escape(summary['grounding_reason'])}</p>")
    if summary["timings"]:
        parts.append("<ul>" + "".join(f"<li>{escape(name.title())}: {value:.2f}s</li>"
                                     for name, value in summary["timings"].items()) + "</ul>")
    if summary["stages"]:
        parts.append("<p><strong>Observed stages:</strong> " + " → ".join(escape(s) for s in summary["stages"]) + "</p>")
    parts.append("<p>Timings are cumulative operation durations and may overlap for parallel searches. "
                 "Agent generation calls exclude provider-router retries. No token/cost totals are inferred. "
                 "A completed automated check is not a guarantee of factual correctness.</p>")
    return "".join(parts)

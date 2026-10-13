"""Bounded, server-owned evidence provenance and presentation for research chat."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from html import escape
import math
import re
from threading import RLock
from urllib.parse import urlsplit

MAX_PASSAGES_PER_TURN = 35
MAX_RETAINED_PASSAGES = 120
MAX_RETAINED_TURNS = 12
MAX_EXCERPT_CHARS = 2000
CITATION_RE = re.compile(r"(?<!!)\[(?:\[(\d{1,9})\]|(\d{1,9}))\](?:\([^\n)]*\))?")


def original_pdf_url(url):
    if not isinstance(url, str):
        return ""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    if (parsed.scheme == "https" and parsed.hostname == "www.ipcc.ch"
            and not parsed.username and not parsed.password):
        return url.split("#", 1)[0]
    return ""


@dataclass(frozen=True)
class Passage:
    number: int
    chunk_id: str
    source_key: str
    title: str
    page: int
    bbox: tuple
    excerpt: str
    original_url: str = ""

    def rendering_input(self):
        return {"chunk_id": self.chunk_id, "source_pdf_key": self.source_key,
                "source_document": self.title, "page": self.page,
                "bbox": list(self.bbox)}


class EvidenceCollector:
    """One turn only: parallel tool calls share a synchronized ID registry."""

    def __init__(self, first_number=1):
        self.first_number = first_number
        self._records = {}
        self._lock = RLock()

    def add(self, result):
        page = result.get("page")
        bbox = result.get("bbox")
        if (not isinstance(result.get("chunk_id"), str) or not result["chunk_id"]
                or not isinstance(result.get("source_pdf_key", ""), str)
                or not isinstance(result.get("text"), str) or not result["text"].strip()
                or isinstance(page, bool) or not isinstance(page, int) or page < 0):
            return None
        valid_bbox = (isinstance(bbox, (list, tuple)) and len(bbox) == 4
                      and all(not isinstance(v, bool) and isinstance(v, (int, float))
                              and math.isfinite(v) and 0 <= v <= 1 for v in bbox)
                      and bbox[0] < bbox[2] and bbox[1] < bbox[3])
        # Missing geometry must not discard otherwise valid retrieved text or
        # invent a precise highlight. The preview may be unavailable/full-page.
        bbox = tuple(bbox) if valid_bbox else ()
        key = (result["chunk_id"], result.get("source_pdf_key", ""), page, tuple(bbox))
        with self._lock:
            if key in self._records:
                return self._records[key]
            if len(self._records) >= MAX_PASSAGES_PER_TURN:
                return None
            record = Passage(
                number=self.first_number + len(self._records), chunk_id=result["chunk_id"],
                source_key=result.get("source_pdf_key", ""),
                title=str(result.get("source_document") or "Untitled document"),
                page=page, bbox=tuple(bbox), excerpt=result["text"][:MAX_EXCERPT_CHARS],
                original_url=original_pdf_url(result.get("source_url", "")),
            )
            self._records[key] = record
            return record

    def snapshot(self):
        with self._lock:
            return tuple(sorted(self._records.values(), key=lambda record: record.number))


def format_citations(text, records):
    """Resolve only captured numeric IDs; unknown IDs never become valid links.

    Existence is provenance, not a claim-level correctness/faithfulness verdict.
    A linked [N] from model prose cannot supply its own destination.
    """
    known = {record.number for record in records}
    cited = set()
    invalid = set()

    def replace(match):
        number = int(match.group(1) or match.group(2))
        if number not in known:
            invalid.add(number)
            return "[source unavailable]"
        cited.add(number)
        return f"[[{number}]](#citation-{number})"

    rendered = CITATION_RE.sub(replace, text)
    if invalid:
        rendered += "\n\n*Some source references could not be matched to this answer's retrieved evidence.*"
    return rendered, frozenset(cited)


@dataclass(frozen=True)
class EvidenceTurn:
    question: str
    records: tuple
    cited: frozenset
    ordinal: int = 0
    context_start: int = 0


@dataclass
class EvidenceHistory:
    """Lives only in Gradio's server-side session State, never in UI inputs."""

    turns: list = field(default_factory=list)
    next_number: int = 1
    answer_count: int = 0
    owner: str | None = None
    lock: RLock = field(default_factory=RLock, repr=False)

    def __deepcopy__(self, memo):
        with self.lock:
            return EvidenceHistory(deepcopy(self.turns, memo), self.next_number,
                                   self.answer_count, self.owner)

    def clear(self, owner=None):
        self.turns.clear()
        self.next_number = 1
        self.answer_count = 0
        self.owner = owner

    def append(self, turn):
        self.answer_count += 1
        self.turns.append(EvidenceTurn(turn.question[:160], turn.records, turn.cited,
                                       self.answer_count, turn.context_start))
        if turn.records:
            self.next_number = max(self.next_number, max(p.number for p in turn.records) + 1)
        while (len(self.turns) > MAX_RETAINED_TURNS
               or sum(len(t.records) for t in self.turns) > MAX_RETAINED_PASSAGES):
            self.turns.pop(0)

    def records(self):
        return tuple(p for turn in self.turns for p in turn.records)

    def trim_context(self, conversation):
        # Drop provider messages belonging to evicted answers too. Otherwise
        # untracked tool results could survive the bounded evidence registry.
        offset = self.turns[0].context_start if self.turns else 0
        if offset:
            self.turns = [replace(turn, context_start=max(0, turn.context_start - offset))
                          for turn in self.turns]
        return conversation[offset:]


def authorized_passages(records, collection, policy):
    """Recheck retained IDs in their current scope; fail closed on lookup errors."""
    if not records:
        return ()
    try:
        result = policy.scope_collection(collection).get(
            ids=list({p.chunk_id for p in records}), include=["metadatas"])
        metadata = dict(zip(result["ids"], result["metadatas"]))
        return tuple(p for p in records if p.chunk_id in metadata
                     and (metadata[p.chunk_id] or {}).get("source_pdf_key", "") == p.source_key
                     and (metadata[p.chunk_id] or {}).get("page", 0) == p.page)
    except Exception:
        return ()


def render_cards(turns, allowed):
    valid = {p.number for p in allowed}
    if not turns:
        return "<p>Retrieved passages and numbered source references will appear here after research.</p>"
    parts = ["<p>Cited means referenced by the answer, not independently verified. "
             "PDF pages are file positions, which can differ from printed page labels.</p>"]
    for index, turn in enumerate(reversed(turns)):
        parts.append(f"<details class='answer-evidence' {'open' if index == 0 else ''}>"
                     f"<summary>Answer {turn.ordinal}: {escape(turn.question)}</summary>")
        if not turn.records:
            parts.append("<p>No passages were captured for this answer. No source verification is implied.</p>")
        for passage in turn.records:
            if passage.number not in valid:
                parts.append(f"<p>Source [{passage.number}] is no longer available in the current document scope.</p>")
                continue
            status = "Cited in this answer" if passage.number in turn.cited else "Retrieved evidence · not cited"
            parts.append(f"<article class='evidence-card' id='citation-{passage.number}'>"
                         f"<h4>[{passage.number}] {escape(passage.title)}</h4>"
                         f"<p><strong>{status}</strong> · PDF page {passage.page + 1}</p>"
                         f"<blockquote>{escape(passage.excerpt)}</blockquote>")
            if passage.original_url:
                link = escape(f"{passage.original_url}#page={passage.page + 1}", quote=True)
                parts.append(f"<a href='{link}' target='_blank' rel='noopener noreferrer'>Open original PDF at this page</a>")
            parts.append(f"<button type='button' data-citation='{passage.number}' "
                         f"aria-label='View source page for reference {passage.number}'>View source page [{passage.number}]</button></article>")
        parts.append("</details>")
    parts.append(f"<p>Source details retain at most {MAX_RETAINED_TURNS} answers and "
                 f"{MAX_RETAINED_PASSAGES} passages. Older details may be cleared as this conversation grows.</p>")
    return "<div class='evidence-scroll'>" + "".join(parts) + "</div>"


EVIDENCE_CLICK_JS = """
element.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-citation]');
    if (button) trigger('click', {citation: Number(button.dataset.citation)});
});
document.addEventListener('click', (event) => {
    if (!element.isConnected) return;
    const link = event.target.closest('#research-chat a[href^="#citation-"]');
    if (!link) return;
    const match = link.getAttribute('href').match(/^#citation-(\\d+)$/);
    if (match) {
        event.preventDefault();
        trigger('click', {citation: Number(match[1])});
    }
});
"""

"""Explicit, isolated public science corpus; no AWS, ADE, or LLM calls.

Downloads only the curated publisher URLs, extracts bounded text blocks with
page coordinates, and publishes a complete content-addressed Chroma collection.
Local PDFs/indexes are never deployment artifacts. Re-run on a new host.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import fitz

from manage_public_corpus import set_public_visibility

SOURCE_FILE = Path(__file__).with_name("showcase_sources.json")
PARSER_VERSION = "pymupdf-blocks-v1"
MAX_PDF_BYTES = 30 * 1024 * 1024
MAX_PAGES = 150


def curated_sources():
    return json.loads(SOURCE_FILE.read_text())["sources"]


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def download_pdf(source, root):
    """Validate cached/downloaded PDFs; changed bytes fail a pinned checksum."""
    path = root / "pdfs" / (source["id"] + ".pdf")
    if path.exists():
        if path.stat().st_size > MAX_PDF_BYTES:
            raise ValueError("Showcase PDF exceeds the size limit")
        data = path.read_bytes()
    else:
        url = source["url"]
        if urlparse(url).scheme != "https" or urlparse(url).hostname != "www.ipcc.ch":
            raise ValueError("Showcase downloads must use the curated IPCC publisher")
        with urlopen(Request(url, headers={"User-Agent": "AgenticRAG-Showcase/1.0"}), timeout=60) as response:
            if urlparse(response.geturl()).hostname != "www.ipcc.ch":
                raise ValueError("Unexpected PDF download redirect")
            data = response.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF-"):
        raise ValueError("Invalid or oversized showcase PDF")
    digest = hashlib.sha256(data).hexdigest()
    if source.get("sha256") and digest != source["sha256"]:
        raise ValueError("Showcase PDF checksum changed; review the source before updating the pin")
    with fitz.open(stream=data, filetype="pdf") as pdf:
        if pdf.is_encrypted or not 0 < len(pdf) <= MAX_PAGES:
            raise ValueError("Encrypted, empty, or oversized page count")
    if not path.exists():
        atomic_write(path, data)
    return path, digest


def extract_chunks(path, source, digest):
    """Text-block extraction, not OCR/table understanding. Pages are zero-based."""
    chunks = []
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            for block_index, block in enumerate(page.get_text("blocks", sort=True)):
                if block[6] != 0:
                    continue
                words = block[4].split()
                if len(words) < 5:
                    continue
                bbox = [max(0.0, min(1.0, value / dimension)) for value, dimension in
                        zip(block[:4], [page.rect.width, page.rect.height] * 2)]
                # Stay below the embedding model's context window. Split blocks
                # share the original block's crop rather than invent coordinates.
                for offset in range(0, len(words), 100):
                    chunks.append({
                        "chunk_id": f"{source['id']}:{digest}:{PARSER_VERSION}:{page_index}:{block_index}:{offset}",
                        "text": " ".join(words[offset:offset + 100]),
                        "page": page_index, "bbox": bbox, "chunk_type": "text",
                        "source_document": source["title"],
                        "source_pdf_key": f"showcase/{digest}/{source['id']}.pdf",
                        "source_url": source["url"],
                    })
    if not chunks:
        raise ValueError("PDF has no extractable text; OCR is not enabled")
    return chunks


class ShowcaseCorpus:
    def __init__(self, root, documents):
        self.root = Path(root).resolve()
        self.documents = {f"showcase/{digest}/{source['id']}.pdf": (path, source)
                          for source, path, digest in documents}
        self.images = self.root / "citations"
        self.images.mkdir(parents=True, exist_ok=True)

    def render_citation(self, result):
        """Only exact curated identities can render; no metadata path traversal."""
        document = self.documents.get(result.get("source_pdf_key"))
        if document is None:
            return None
        path, _ = document
        bbox = result["bbox"]
        if (len(bbox) != 4 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in bbox)
                or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            return None
        identity = json.dumps([result["source_pdf_key"], result["page"], bbox, 120])
        target = self.images / (hashlib.sha256(identity.encode()).hexdigest() + ".png")
        if not target.exists():
            with fitz.open(path) as pdf:
                page_number = result["page"]
                if not isinstance(page_number, int) or not 0 <= page_number < len(pdf):
                    return None
                page = pdf[page_number]
                rect = fitz.Rect(*(v * d for v, d in zip(
                    bbox, [page.rect.width, page.rect.height] * 2)))
                page.draw_rect(rect, color=(1, 0, 0), width=1)
                # Full page preserves figures/table context around retrieved text.
                atomic_write(target, page.get_pixmap(dpi=120).tobytes("png"))
        return "/gradio_api/file=" + quote(str(target), safe="/")


def prepare_showcase(root="./showcase_data"):
    """Stage/index all documents privately, verify exact IDs, then publish."""
    from gemini_helpers import embed_and_index_chunks, init_chroma_collection

    root = Path(root).resolve()
    documents, all_chunks = [], []
    for source in curated_sources():
        path, digest = download_pdf(source, root)
        chunks = extract_chunks(path, source, digest)
        documents.append((source, path, digest))
        all_chunks.append(chunks)
        print(f"{source['id']}: {len(chunks)} chunks; SHA256 {digest}")
    identity = json.dumps([PARSER_VERSION, [(s["id"], d) for s, _, d in documents]])
    generation = hashlib.sha256(identity.encode()).hexdigest()[:24]
    collection = init_chroma_collection(str(root / "chroma"), "showcase_" + generation)
    expected_ids = {c["chunk_id"] for chunks in all_chunks for c in chunks}
    # Bounded batches use the existing embedding/index abstraction.
    for chunks in all_chunks:
        for offset in range(0, len(chunks), 64):
            embed_and_index_chunks(chunks[offset:offset + 64], collection)
    if set(collection.get(include=[])["ids"]) != expected_ids:
        raise ValueError("Showcase index does not match the complete chunk manifest")
    for chunks in all_chunks:
        set_public_visibility(collection, [c["chunk_id"] for c in chunks],
                              publish=True, source_pdf_key=chunks[0]["source_pdf_key"])
    print(f"Ready: {len(documents)} approved PDFs, {len(expected_ids)} chunks, {collection.name}")
    return collection, ShowcaseCorpus(root, documents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./showcase_data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="Smoke-test actual vector, hybrid and reranked retrieval without generation")
    args = parser.parse_args()
    if args.download_only:
        for source in curated_sources():
            _, digest = download_pdf(source, Path(args.root).resolve())
            print(f"{source['id']}: SHA256 {digest}")
    else:
        collection, corpus = prepare_showcase(args.root)
        if args.check:
            run_retrieval_checks(collection, corpus)


def run_retrieval_checks(collection, corpus):
    from access_control import PublicCollection
    from gemini_helpers import search_chroma
    from hybrid_search import search_chroma_hybrid

    checks = [
        ("solar wind lithium-ion battery unit costs reductions 2010 2019", "85%"),
        ("human activities global warming 2011 2020 above 1850 1900", "1.1"),
        ("GHG emissions reductions 2030 pathways limit warming 1.5°C", "43"),
    ]
    scoped = PublicCollection(collection)
    for mode in ("vector", "hybrid", "reranked"):
        for query, expected in checks:
            start = time.monotonic()
            results = (search_chroma(query, scoped) if mode == "vector" else
                       search_chroma_hybrid(query, scoped, use_reranker=mode == "reranked"))
            if not results or not any(expected in result["text"] for result in results):
                raise AssertionError(f"{mode}: expected evidence missing for {query}")
            if not corpus.render_citation(results[0]):
                raise AssertionError("Citation could not be rendered")
            print(f"CHECK PASS {mode}: {query}; {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    main()

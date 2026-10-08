"""Offline administrative publication of explicitly reviewed chunk IDs.

This is never registered as a web/API handler. It opens an existing collection
and changes metadata only; it does not read document text or embed anything.
"""

import argparse
from pathlib import PurePosixPath

import chromadb


def set_public_visibility(collection, chunk_ids, *, publish, source_pdf_key=None):
    if not chunk_ids or any(not isinstance(cid, str) or not cid for cid in chunk_ids):
        raise ValueError("Explicit non-empty chunk IDs are required")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Duplicate chunk IDs are not allowed")
    if publish:
        if (not source_pdf_key or source_pdf_key.startswith("/")
                or ".." in PurePosixPath(source_pdf_key).parts
                or not source_pdf_key.lower().endswith(".pdf")):
            raise ValueError("Publication requires an exact relative PDF object key")

    records = collection.get(ids=chunk_ids, include=["metadatas"])
    if set(records["ids"]) != set(chunk_ids):
        raise ValueError("One or more selected chunk IDs do not exist; nothing changed")
    metadata = [dict(meta or {}) for meta in records["metadatas"]]
    if publish:
        sources = {meta.get("source_document") for meta in metadata}
        if len(sources) != 1 or not next(iter(sources)):
            raise ValueError("Publish chunks from one known document at a time")
        if any(meta.get("source_pdf_key") not in (None, "", source_pdf_key)
               for meta in metadata):
            raise ValueError("The selected chunks already reference a different source PDF")
    for meta in metadata:
        meta["visibility"] = "public" if publish else "private"
        if publish:
            meta["source_pdf_key"] = source_pdf_key
    collection.update(ids=records["ids"], metadatas=metadata)
    return len(metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-path", default="./chroma_db")
    parser.add_argument("--collection", default="document_chunks")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--publish", action="store_true")
    action.add_argument("--revoke", action="store_true")
    parser.add_argument("--ids", nargs="+", required=True,
                        help="Exact chunk IDs reviewed for publication or revocation")
    parser.add_argument("--source-pdf-key", help="Exact approved source PDF key in S3")
    args = parser.parse_args()
    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_collection(args.collection)
    count = set_public_visibility(
        collection, args.ids, publish=args.publish,
        source_pdf_key=args.source_pdf_key,
    )
    print(f"Updated visibility for {count} explicitly selected chunks.")


if __name__ == "__main__":
    main()

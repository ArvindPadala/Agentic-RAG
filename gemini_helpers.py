"""
Gemini + ChromaDB Helper Functions
====================================
This module replaces the AWS Bedrock Knowledge Base with a local ChromaDB vector
database and local sentence-transformers embeddings. The goal is to provide the same RAG
(Retrieval-Augmented Generation) capabilities at zero cost.

Key concepts covered here:
  - ChromaDB: an open-source, embedded vector database (no server needed!)
  - Vector embeddings: turning text into numbers that capture semantic meaning
  - Cosine similarity: how ChromaDB finds the "closest" chunks to your query
  - sentence-transformers: local, free embedding models
"""


from config import settings
from utils.logger import get_logger

logger = get_logger('gemini_helpers')
import json
import boto3
import chromadb
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Google Generative AI SDK
from google import genai
from google.genai import types


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: ChromaDB Setup
# ─────────────────────────────────────────────────────────────────────────────

def init_chroma_collection(
    persist_directory: str = "./chroma_db",
    collection_name: str = "document_chunks"
) -> chromadb.Collection:
    """
    Initialize (or reopen) a persistent ChromaDB collection.

    CONCEPT: ChromaDB as a "vector database"
    -----------------------------------------
    A vector database stores text as mathematical vectors (lists of numbers).
    When you search, it finds the vectors most similar to your query vector.
    This is how semantic search works — "cold symptoms" finds text about
    "rhinovirus infections" even though the exact words don't match.

    Using `PersistentClient` means ChromaDB saves data to disk at
    `persist_directory`. Next time you run the notebook, your indexed chunks
    are still there — no need to re-embed everything.

    Args:
        persist_directory: Local folder where ChromaDB stores its data files
        collection_name:   Name for the collection (like a table in SQL)

    Returns:
        A ChromaDB Collection object ready for add() and query() calls
    """
    logger.info(f"📂 Opening ChromaDB at: {persist_directory}")

    # PersistentClient saves data to disk automatically
    client = chromadb.PersistentClient(path=persist_directory)

    # get_or_create_collection is idempotent — safe to call multiple times
    # cosine distance is standard for text similarity tasks
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for text
    )

    count = collection.count()
    if count > 0:
        logger.info(f"✅ Reopened existing collection '{collection_name}' ({count} documents already indexed)")
    else:
        logger.info(f"✅ Created new empty collection '{collection_name}'")

    return collection


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Loading Chunks from S3
# ─────────────────────────────────────────────────────────────────────────────

def load_chunks_from_s3(
    s3_client,
    bucket: str,
    chunks_prefix: str = "output/chunks/"
) -> List[Dict]:
    """
    Download all individual chunk JSON files from S3 into memory.

    CONCEPT: Why individual chunk files?
    --------------------------------------
    The Lambda function (ade_s3_handler.py) stored each document "chunk"
    as its own JSON file in S3. This makes it easy for us to:
      1. List all chunks with a single list_objects_v2 call
      2. Download and process them one by one
      3. Know exactly which chunk came from which PDF and where on the page

    Each chunk JSON looks like:
    {
      "chunk_id": "chunk-uuid-1",
      "chunk_type": "text",
      "text": "The common cold is caused by rhinoviruses...",
      "bbox": [0.05, 0.12, 0.95, 0.28],   ← normalized coordinates on PDF page
      "page": 2,
      "source_document": "Prevention_and_treatment_of_the_common_cold"
    }

    Args:
        s3_client:     Boto3 S3 client
        bucket:        S3 bucket name
        chunks_prefix: S3 prefix where chunk JSON files live

    Returns:
        List of chunk dictionaries
    """
    logger.info(f"📥 Loading chunks from s3://{bucket}/{chunks_prefix}")

    chunks = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=chunks_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip folder-marker objects and non-JSON files
            if key.endswith("/") or not key.endswith(".json"):
                continue

            try:
                response = s3_client.get_object(Bucket=bucket, Key=key)
                chunk_data = json.loads(response["Body"].read().decode("utf-8"))
                chunk_data["s3_key"] = key  # Remember where it came from
                chunks.append(chunk_data)
            except Exception as e:
                logger.info(f"   ⚠️ Could not load {key}: {e}")

    logger.info(f"✅ Loaded {len(chunks)} chunks from S3")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Indexing Chunks into ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def embed_and_index_chunks(
    chunks: List[Dict],
    collection: chromadb.Collection,
    skip_existing: bool = True
) -> int:
    """
    Embed all chunks with Gemini and store them in ChromaDB.

    CONCEPT: Indexing = embedding + storing
    ----------------------------------------
    "Indexing" means:
      1. Take each chunk's text content
      2. Let ChromaDB embed it automatically using sentence-transformers
         (model: all-MiniLM-L6-v2, runs locally — no API, no rate limits)
      3. Store (id, vector, metadata, text) in ChromaDB

    WHY local embeddings instead of Gemini?
    ----------------------------------------
    Gemini embeddings are great but have free-tier rate limits. The
    sentence-transformers model (all-MiniLM-L6-v2) runs on your CPU,
    is completely free, has no rate limits, works offline, and produces
    high-quality 384-dim embeddings well-suited for document retrieval.

    HOW ChromaDB auto-embedding works:
    ----------------------------------------
    When you call collection.add(documents=[...]) WITHOUT passing
    embeddings=, ChromaDB automatically calls its default embedding
    function (sentence-transformers) on the documents before storing.
    Similarity, collection.query(query_texts=[...]) auto-embeds the
    query before searching. This keeps the code clean and simple.

    Args:
        chunks:        List of chunk dicts from load_chunks_from_s3()
        collection:    ChromaDB collection to add to
        gemini_client: Not used (kept for backward compatibility)
        skip_existing: Skip chunks already in the collection

    Returns:
        Number of new chunks indexed
    """
    logger.info(f"\n📦 Preparing to index {len(chunks)} chunks...")
    logger.info("   Using: sentence-transformers/all-MiniLM-L6-v2 (local, free, no rate limits)")

    # Filter out chunks with no text
    valid_chunks = [c for c in chunks if c.get("text", "").strip()]
    logger.info(f"   └─ {len(valid_chunks)} chunks have non-empty text")

    if skip_existing:
        existing_ids = set(collection.get()["ids"])
        new_chunks = [c for c in valid_chunks if c.get("chunk_id", "") not in existing_ids]
        logger.info(f"   └─ {len(new_chunks)} new chunks to add (skipping {len(valid_chunks) - len(new_chunks)} already indexed)")
    else:
        new_chunks = valid_chunks

    if not new_chunks:
        logger.info("✅ All chunks already indexed — nothing to do!")
        return 0

    texts = [c["text"] for c in new_chunks]
    ids = []
    metadatas = []

    for chunk in new_chunks:
        ids.append(chunk.get("chunk_id", ""))
        bbox = chunk.get("bbox", [0, 0, 1, 1])
        metadatas.append({
            "chunk_type":      chunk.get("chunk_type", "text"),
            "page":            int(chunk.get("page", 0)),
            "bbox":            json.dumps(bbox),
            "source_document": chunk.get("source_document", ""),
            "s3_key":          chunk.get("s3_key", "")
        })

    # Add to ChromaDB — NO embeddings= arg → ChromaDB auto-embeds locally
    logger.info(f"\n🧠 Embedding {len(new_chunks)} chunks locally (no API call)...")
    collection.add(
        ids=ids,
        documents=texts,   # ChromaDB embeds these automatically
        metadatas=metadatas
    )

    logger.info(f"\n✅ Indexed {len(new_chunks)} chunks into ChromaDB!")
    logger.info(f"   Total collection size: {collection.count()} documents")
    return len(new_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Searching ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def search_chroma(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 5
) -> List[Dict]:
    """
    Search the ChromaDB collection using a natural language query.

    CONCEPT: How semantic search works
    ------------------------------------
    1. Your query ("what helps with cold symptoms?") is embedded into a vector
    2. ChromaDB computes the cosine distance from that vector to every stored vector
    3. The n_results closest vectors are returned — these are the most semantically
       similar chunks, regardless of exact keyword matches

    Cosine distance ranges from 0 (identical) to 2 (opposite).
    ChromaDB returns it as a "distance" — lower is better.
    We convert it to a "score" (1 - distance) to be consistent with the original
    Lab 6 which used Bedrock's relevance scores (higher = better).

    Args:
        query:         Natural language search query
        collection:    ChromaDB collection to search
        gemini_client: Gemini client for embedding the query
        n_results:     Number of results to return

    Returns:
        List of result dicts with: text, score, chunk_id, metadata fields
    """
    # If the collection is empty, return early to avoid querying with n_results=0
    if collection.count() == 0:
        return []

    # ChromaDB auto-embeds the query using sentence-transformers locally.
    # No API call, no rate limit — same model used to embed the documents.
    results = collection.query(
        query_texts=[query],   # ChromaDB embeds this automatically
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    # Step 3: Unpack and format results
    formatted = []
    if not results["ids"] or not results["ids"][0]:
        return formatted

    for i, chunk_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        score = 1.0 - distance  # Convert distance → similarity score

        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]

        # Deserialize the bbox JSON string back to a list
        try:
            bbox = json.loads(meta.get("bbox", "[0, 0, 1, 1]"))
        except Exception:
            bbox = [0, 0, 1, 1]

        formatted.append({
            "chunk_id":        chunk_id,
            "text":            text,
            "score":           round(score, 4),
            "chunk_type":      meta.get("chunk_type", "text"),
            "page":            meta.get("page", 0),
            "bbox":            bbox,
            "source_document": meta.get("source_document", ""),
            "s3_key":          meta.get("s3_key", "")
        })

    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Simple JSON Memory
# ─────────────────────────────────────────────────────────────────────────────

def load_memory(memory_file: str = "memory.json") -> Dict:
    """
    Load persistent conversation memory from a JSON file.

    CONCEPT: Why store memory in a JSON file?
    ------------------------------------------
    The original lab used AWS Bedrock AgentCore Memory — a managed cloud service
    that stores summaries, user preferences, and extracted facts across sessions.

    Our free alternative: a simple JSON file on disk.
    It stores the same types of information:
      - session_summaries: what was discussed in previous conversations
      - preferences:       things the user told us (e.g., "I prefer short answers")
      - facts:             important facts extracted from conversations

    The Gemini agent reads this at startup and includes it in the system prompt,
    giving it "memory" of past interactions.

    Returns:
        Dict with keys: session_summaries, preferences, facts
    """
    if Path(memory_file).exists():
        with open(memory_file, "r") as f:
            return json.load(f)
    return {
        "session_summaries": [],
        "preferences": {},
        "facts": []
    }


def save_memory(memory: Dict, memory_file: str = "memory.json") -> None:
    """
    Save the current memory state to a JSON file.

    Call this at the end of each chat session to persist what was learned.

    Args:
        memory:      Memory dict with session_summaries, preferences, facts
        memory_file: Path to the JSON file
    """
    with open(memory_file, "w") as f:
        json.dump(memory, f, indent=2)
    logger.info(f"💾 Memory saved to {memory_file}")


def format_memory_for_prompt(memory: Dict) -> str:
    """
    Format stored memory into a string for injection into the system prompt.

    The agent reads this context at the start of every session, making it
    feel like it "remembers" previous conversations.

    Args:
        memory: Memory dict

    Returns:
        Formatted string for the system prompt
    """
    parts = []

    if memory.get("preferences"):
        pref_text = ", ".join(f"{k}: {v}" for k, v in memory["preferences"].items())
        parts.append(f"User preferences: {pref_text}")

    if memory.get("facts"):
        facts_text = "; ".join(memory["facts"][-5:])  # Last 5 facts
        parts.append(f"Remembered facts: {facts_text}")

    if memory.get("session_summaries"):
        last_summary = memory["session_summaries"][-1]  # Most recent session
        parts.append(f"Last session summary: {last_summary}")

    if parts:
        return "\n\nPrevious conversation context:\n" + "\n".join(parts)
    return ""


def update_memory_from_conversation(
    memory: Dict,
    conversation_history: List[Dict],
    gemini_client: genai.Client
) -> Dict:
    """
    Ask Gemini to extract a summary and any user preferences/facts from
    the conversation, then save to memory.

    This mimics Bedrock AgentCore's memory strategies:
      - Summary strategy:         summarize the session
      - User preference strategy: learn "user prefers X"
      - Semantic/fact strategy:   extract "user mentioned they have allergies"

    Args:
        memory:               Current memory state
        conversation_history: List of {"role": "user"/"model", "parts": [...]}
        gemini_client:        Gemini client for the extraction call

    Returns:
        Updated memory dict
    """
    if not conversation_history:
        return memory

    # Format conversation for summarization
    lines = []
    for msg in conversation_history:
        role = getattr(msg, "role", "unknown").upper()
        parts = getattr(msg, "parts", [])
        text_parts = []
        for p in parts:
            if hasattr(p, "text") and p.text:
                text_parts.append(p.text)
            elif hasattr(p, "function_call") and p.function_call:
                text_parts.append(f"[Call: {p.function_call.name}]")
            elif hasattr(p, "function_response") and p.function_response:
                text_parts.append(f"[Response: {p.function_response.name}]")
        if text_parts:
            lines.append(f"{role}: {' '.join(text_parts)}")

    conv_text = "\n".join(lines)

    extraction_prompt = f"""Analyze this conversation and extract:
1. A 1-2 sentence summary of what was discussed
2. Any user preferences expressed (e.g., "prefers short answers", "wants citations")
3. Any important facts mentioned by the user (e.g., "has allergies", "is a doctor")

Respond ONLY with valid JSON in this exact format:
{{
  "summary": "...",
  "preferences": {{"key": "value"}},
  "facts": ["fact1", "fact2"]
}}

Conversation:
{conv_text[:3000]}"""

    try:
        response = gemini_client.models.generate_content(
            model="models/gemini-3.5-flash",
            contents=extraction_prompt
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        extracted = json.loads(raw)

        if extracted.get("summary"):
            memory["session_summaries"].append(extracted["summary"])
            memory["session_summaries"] = memory["session_summaries"][-10:]  # Keep last 10

        if extracted.get("preferences"):
            memory["preferences"].update(extracted["preferences"])

        if extracted.get("facts"):
            memory["facts"].extend(extracted["facts"])
            memory["facts"] = list(set(memory["facts"]))[-20:]  # Keep 20 unique facts

        logger.info("✅ Memory updated from conversation")

    except Exception as e:
        logger.info(f"⚠️ Could not extract memory: {e}")

    return memory

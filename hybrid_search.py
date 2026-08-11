from rank_bm25 import BM25Okapi
import re

# Global cache for BM25 to avoid rebuilding on every query
_BM25_CACHE = {}


def get_bm25_index(collection, collection_name):
    if collection_name in _BM25_CACHE:
        return _BM25_CACHE[collection_name]

    # Load all docs
    all_data = collection.get()
    docs = all_data.get("documents", [])
    ids = all_data.get("ids", [])
    metadatas = all_data.get("metadatas", [])

    # Tokenize
    tokenized_corpus = [re.findall(r'\w+', doc.lower()) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    index_data = {
        "bm25": bm25,
        "docs": docs,
        "ids": ids,
        "metadatas": metadatas
    }
    _BM25_CACHE[collection_name] = index_data
    return index_data

from langsmith import traceable
import json

@traceable(run_type="retriever")
def search_chroma_hybrid(query: str, collection, n_results: int = 5) -> list[dict]:
    if collection.count() == 0:
        return []

    # 1. Vector Search
    vector_results = collection.query(
        query_texts=[query],
        n_results=n_results * 2,
        include=["documents", "metadatas", "distances"]
    )

    vector_ranked = []
    if vector_results["ids"] and vector_results["ids"][0]:
        for i, chunk_id in enumerate(vector_results["ids"][0]):
            score = 1.0 - vector_results["distances"][0][i]
            vector_ranked.append({
                "chunk_id": chunk_id,
                "score": score,
                "text": vector_results["documents"][0][i],
                "meta": vector_results["metadatas"][0][i]
            })

    # 2. BM25 Search
    bm25_data = get_bm25_index(collection, collection.name)
    tokenized_query = re.findall(r'\w+', query.lower())
    bm25_scores = bm25_data["bm25"].get_scores(tokenized_query)

    # Rank by BM25
    doc_score_pairs = [(bm25_data["ids"][i], bm25_scores[i], bm25_data["docs"][i], bm25_data["metadatas"][i])
                       for i in range(len(bm25_data["ids"])) if bm25_scores[i] > 0]
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    bm25_ranked = doc_score_pairs[:n_results * 2]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}

    k = 60  # RRF constant

    for rank, item in enumerate(vector_ranked):
        cid = item["chunk_id"]
        if cid not in rrf_scores:
            rrf_scores[cid] = {"rrf": 0, "text": item["text"], "meta": item["meta"], "v_score": item["score"], "b_score": 0}
        rrf_scores[cid]["rrf"] += 1.0 / (k + rank + 1)

    for rank, item in enumerate(bm25_ranked):
        cid, b_score, text, meta = item
        if cid not in rrf_scores:
            rrf_scores[cid] = {"rrf": 0, "text": text, "meta": meta, "v_score": 0, "b_score": b_score}
        rrf_scores[cid]["rrf"] += 1.0 / (k + rank + 1)

    # Sort by RRF score
    sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1]["rrf"], reverse=True)

    formatted = []
    for chunk_id, data in sorted_fused[:n_results]:
        meta = data["meta"]
        try:
            import json
            bbox = json.loads(meta.get("bbox", "[0, 0, 1, 1]"))
        except Exception:
            bbox = [0, 0, 1, 1]

        formatted.append({
            "chunk_id": chunk_id,
            "text": data["text"],
            "score": round(data["rrf"], 4),  # using RRF as score
            "source_document": meta.get("source_document", "Unknown"),
            "page": meta.get("page", 1),
            "chunk_type": meta.get("chunk_type", "Unknown"),
            "bbox": bbox
        })

    return formatted

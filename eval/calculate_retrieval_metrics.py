from hybrid_search import search_chroma_hybrid
from gemini_helpers import init_chroma_collection, search_chroma
import os
import sys
import pandas as pd
import ast
import json

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))
# noqa: E402
# noqa: E402


def calculate_mrr(retrieved_texts, ground_truth_text):
    for i, text in enumerate(retrieved_texts):
        # We use a simple substring match or high overlap since exact match
        # might fail if whitespace differs slightly
        if ground_truth_text.strip() in text.strip(
        ) or text.strip() in ground_truth_text.strip():
            return 1.0 / (i + 1)
    return 0.0


def calculate_recall_at_k(retrieved_texts, ground_truth_text, k):
    for text in retrieved_texts[:k]:
        if ground_truth_text.strip() in text.strip(
        ) or text.strip() in ground_truth_text.strip():
            return 1.0
    return 0.0


def main():
    print("Loading Golden Dataset...")
    df = pd.read_csv("eval/golden_dataset.csv")
    coll = init_chroma_collection("./chroma_db", "document_chunks")

    baseline_mrr = []
    baseline_recall_1 = []
    baseline_recall_3 = []
    baseline_recall_5 = []

    hybrid_mrr = []
    hybrid_recall_1 = []
    hybrid_recall_3 = []
    hybrid_recall_5 = []

    elite_mrr = []
    elite_recall_1 = []
    elite_recall_3 = []
    elite_recall_5 = []

    for _, row in df.iterrows():
        question = row["question"]
        try:
            contexts = ast.literal_eval(row["contexts"])
            ground_truth_context = contexts[0]
        except Exception:
            ground_truth_context = row["contexts"]

        # 1. Baseline Search
        baseline_results = search_chroma(
            query=question, collection=coll, n_results=5)
        baseline_texts = [r["text"] for r in baseline_results]

        baseline_mrr.append(
            calculate_mrr(
                baseline_texts,
                ground_truth_context))
        baseline_recall_1.append(
            calculate_recall_at_k(
                baseline_texts,
                ground_truth_context,
                1))
        baseline_recall_3.append(
            calculate_recall_at_k(
                baseline_texts,
                ground_truth_context,
                3))
        baseline_recall_5.append(
            calculate_recall_at_k(
                baseline_texts,
                ground_truth_context,
                5))

        # 2. Hybrid Search
        hybrid_results = search_chroma_hybrid(
            query=question, collection=coll, n_results=5)
        hybrid_texts = [r["text"] for r in hybrid_results]

        hybrid_mrr.append(calculate_mrr(hybrid_texts, ground_truth_context))
        hybrid_recall_1.append(
            calculate_recall_at_k(
                hybrid_texts,
                ground_truth_context,
                1))
        hybrid_recall_3.append(
            calculate_recall_at_k(
                hybrid_texts,
                ground_truth_context,
                3))
        hybrid_recall_5.append(
            calculate_recall_at_k(
                hybrid_texts,
                ground_truth_context,
                5))

        # 3. Elite Hybrid Search (Reranker)
        elite_results = search_chroma_hybrid(
            query=question, collection=coll, n_results=5, use_reranker=True)
        elite_texts = [r["text"] for r in elite_results]

        elite_mrr.append(calculate_mrr(elite_texts, ground_truth_context))
        elite_recall_1.append(
            calculate_recall_at_k(
                elite_texts,
                ground_truth_context,
                1))
        elite_recall_3.append(
            calculate_recall_at_k(
                elite_texts,
                ground_truth_context,
                3))
        elite_recall_5.append(
            calculate_recall_at_k(
                elite_texts,
                ground_truth_context,
                5))

    print("\n=== Retrieval Metrics ===")
    print(f"Total Questions: {len(df)}")
    print("\n--- Baseline (Vector Only) ---")
    print(f"MRR@5:      {sum(baseline_mrr) / len(baseline_mrr):.4f}")
    print(f"Recall@1:   {sum(baseline_recall_1) / len(baseline_recall_1):.4f}")
    print(f"Recall@3:   {sum(baseline_recall_3) / len(baseline_recall_3):.4f}")
    print(f"Recall@5:   {sum(baseline_recall_5) / len(baseline_recall_5):.4f}")

    print("\n--- Hybrid (Vector + BM25) ---")
    print(f"MRR@5:      {sum(hybrid_mrr) / len(hybrid_mrr):.4f}")
    print(f"Recall@1:   {sum(hybrid_recall_1) / len(hybrid_recall_1):.4f}")
    print(f"Recall@3:   {sum(hybrid_recall_3) / len(hybrid_recall_3):.4f}")
    print(f"Recall@5:   {sum(hybrid_recall_5) / len(hybrid_recall_5):.4f}")

    print("\n--- Elite Hybrid (RRF + Reranker) ---")
    print(f"MRR@5:      {sum(elite_mrr) / len(elite_mrr):.4f}")
    print(f"Recall@1:   {sum(elite_recall_1) / len(elite_recall_1):.4f}")
    print(f"Recall@3:   {sum(elite_recall_3) / len(elite_recall_3):.4f}")
    print(f"Recall@5:   {sum(elite_recall_5) / len(elite_recall_5):.4f}")

    with open("eval/metrics.json", "w") as f:
        json.dump({
            "baseline": {
                "mrr_5": sum(baseline_mrr) / len(baseline_mrr),
                "recall_1": sum(baseline_recall_1) / len(baseline_recall_1),
                "recall_3": sum(baseline_recall_3) / len(baseline_recall_3),
                "recall_5": sum(baseline_recall_5) / len(baseline_recall_5)
            },
            "hybrid": {
                "mrr_5": sum(hybrid_mrr) / len(hybrid_mrr),
                "recall_1": sum(hybrid_recall_1) / len(hybrid_recall_1),
                "recall_3": sum(hybrid_recall_3) / len(hybrid_recall_3),
                "recall_5": sum(hybrid_recall_5) / len(hybrid_recall_5)
            },
            "elite": {
                "mrr_5": sum(elite_mrr) / len(elite_mrr),
                "recall_1": sum(elite_recall_1) / len(elite_recall_1),
                "recall_3": sum(elite_recall_3) / len(elite_recall_3),
                "recall_5": sum(elite_recall_5) / len(elite_recall_5)
            }
        }, f, indent=4)


if __name__ == "__main__":
    main()

import os
import sys
import pandas as pd
import numpy as np
from datasets import Dataset
from ragas import evaluate
import ast
import time
import warnings

# Suppress deprecation warnings for cleaner logs
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from ragas.metrics import faithfulness
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

# Setup local Ollama for judge evaluation (Tier 2) - using a smaller model for speed
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)
# Setup local Embeddings for Semantic Similarity (Tier 1.5)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def compute_semantic_similarity(df):
    """
    Computes pure cosine similarity between the generated answer and the ground truth.
    This requires NO LLM calls, meaning it scales instantly to 100+ rows.
    """
    if df.empty:
        return 0.0
        
    answers = df['answer'].fillna("").tolist()
    references = df['ground_truth'].fillna("").tolist()
    
    # Embed all answers and references in batches
    ans_emb = embeddings.embed_documents(answers)
    ref_emb = embeddings.embed_documents(references)
    
    # Calculate pairwise cosine similarities
    similarities = []
    for a, r in zip(ans_emb, ref_emb):
        sim = cosine_similarity(np.array(a).reshape(1, -1), np.array(r).reshape(1, -1))[0][0]
        similarities.append(sim)
        
    return float(np.mean(similarities))

def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)
    
    # Ragas requires lists for contexts
    def parse_contexts(c):
        try:
            return ast.literal_eval(c)
        except:
            return [c]
            
    df['contexts'] = df['contexts'].apply(parse_contexts)
    
    # Ragas 0.4.x expects 'reference' instead of 'ground_truth'
    if 'ground_truth' in df.columns:
        df['reference'] = df['ground_truth']
        
    return df

def main():
    if not os.path.exists("eval/baseline_results.csv") or not os.path.exists("eval/hybrid_results.csv"):
        print("Please run evaluate_baseline.py and evaluate_hybrid.py first.")
        return

    print("Loading datasets...")
    baseline_df = prepare_dataset("eval/baseline_results.csv")
    hybrid_df = prepare_dataset("eval/hybrid_results.csv")
    reranker_df = prepare_dataset("eval/reranker_results.csv")

    # --- TIER 1.5: Non-LLM Semantic Similarity (Full Scale) ---
    print("\n" + "="*50)
    print("TIER 1.5: Non-LLM Semantic Similarity (Full Dataset)")
    print("="*50)
    
    base_sim = compute_semantic_similarity(baseline_df)
    hyb_sim = compute_semantic_similarity(hybrid_df)
    rerank_sim = compute_semantic_similarity(reranker_df)
    
    print(f"Baseline Semantic Similarity: {base_sim:.4f}")
    print(f"Hybrid Semantic Similarity:   {hyb_sim:.4f}")
    print(f"Elite Semantic Similarity:    {rerank_sim:.4f}")

    # --- TIER 2: LLM Judge Faithfulness (Smoke Test) ---
    print("\n" + "="*50)
    print("TIER 2: LLM-as-a-Judge Faithfulness (Smoke Test N=15)")
    print("="*50)
    
    # Slice to head(15) to prevent hours of local compute
    baseline_ds = Dataset.from_pandas(baseline_df.head(15))
    hybrid_ds = Dataset.from_pandas(hybrid_df.head(15))
    reranker_ds = Dataset.from_pandas(reranker_df.head(15))

    metrics = [faithfulness]
    run_config = RunConfig(max_workers=1, timeout=180)

    print("\n--- Evaluating Baseline Vector Search ---")
    baseline_result = evaluate(
        baseline_ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )
    print("Baseline Faithfulness:")
    print(baseline_result)
    
    print("\nSleeping for 30s to let rate limits cool down...")
    time.sleep(30)

    print("\n--- Evaluating Hybrid Agent ---")
    hybrid_result = evaluate(
        hybrid_ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )
    print("Hybrid Faithfulness:")
    print(hybrid_result)

    print("\nSleeping for 30s to let rate limits cool down...")
    time.sleep(30)

    print("\n--- Evaluating Elite Hybrid (Reranker) Agent ---")
    reranker_result = evaluate(
        reranker_ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )
    print("Elite Hybrid Faithfulness:")
    print(reranker_result)
    
    # Save results
    with open("eval/ragas_metrics.txt", "w") as f:
        f.write("=== TIER 1.5: Semantic Similarity (Full Scale) ===\n")
        f.write(f"Baseline: {base_sim:.4f}\n")
        f.write(f"Hybrid:   {hyb_sim:.4f}\n")
        f.write(f"Elite:    {rerank_sim:.4f}\n\n")
        
        f.write("=== TIER 2: Faithfulness (N=15) ===\n")
        f.write("Baseline Faithfulness:\n")
        f.write(str(baseline_result) + "\n\n")
        f.write("Hybrid Faithfulness:\n")
        f.write(str(hybrid_result) + "\n\n")
        f.write("Elite Hybrid (Reranker) Faithfulness:\n")
        f.write(str(reranker_result) + "\n")
        
    print("\nEvaluation complete! Results saved to eval/ragas_metrics.txt")

if __name__ == "__main__":
    main()

import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
import ast
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

# Setup local Llama 3.1 8B for judge evaluation
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

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
        
    dataset = Dataset.from_pandas(df)
    return dataset

def main():
    if not os.path.exists("eval/baseline_results.csv") or not os.path.exists("eval/hybrid_results.csv"):
        print("Please run evaluate_baseline.py and evaluate_hybrid.py first.")
        return

    print("Loading datasets...")
    baseline_ds = prepare_dataset("eval/baseline_results.csv")
    hybrid_ds = prepare_dataset("eval/hybrid_results.csv")
    reranker_ds = prepare_dataset("eval/reranker_results.csv")

    metrics = [faithfulness, answer_relevancy]

    print("\n--- Evaluating Baseline Vector Search ---")
    # Smoke test: 1 worker to ensure it doesn't time out
    run_config = RunConfig(max_workers=1, timeout=180)
    baseline_result = evaluate(
        baseline_ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )
    print("Baseline RAGAS Metrics:")
    print(baseline_result)
    
    print("\nSleeping for 60s to let rate limits cool down...")
    time.sleep(60)

    print("\n--- Evaluating Hybrid Agent ---")
    hybrid_result = evaluate(
        hybrid_ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )
    print("Hybrid RAGAS Metrics:")
    print(hybrid_result)

    print("\nSleeping for 60s to let rate limits cool down...")
    time.sleep(60)

    print("\n--- Evaluating Elite Hybrid (Reranker) Agent ---")
    reranker_result = evaluate(
        reranker_ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )
    print("Elite Hybrid RAGAS Metrics:")
    print(reranker_result)
    
    # Save results
    with open("eval/ragas_metrics.txt", "w") as f:
        f.write("Baseline Vector Search Metrics:\n")
        f.write(str(baseline_result) + "\n\n")
        f.write("Hybrid Agent Metrics:\n")
        f.write(str(hybrid_result) + "\n\n")
        f.write("Elite Hybrid (Reranker) Agent Metrics:\n")
        f.write(str(reranker_result) + "\n")
        
    print("\nEvaluation complete! Results saved to eval/ragas_metrics.txt")

if __name__ == "__main__":
    main()

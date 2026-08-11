import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import ast
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

# Setup Langchain Google GenAI with high retries to survive 429s
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=settings.GEMINI_API_KEY,
    max_retries=20,
    timeout=120
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=settings.GEMINI_API_KEY
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

    metrics = [faithfulness, answer_relevancy]

    print("\n--- Evaluating Baseline Vector Search ---")
    baseline_result = evaluate(
        baseline_ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
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
    )
    print("Hybrid RAGAS Metrics:")
    print(hybrid_result)
    
    # Save results
    with open("eval/ragas_metrics.txt", "w") as f:
        f.write("Baseline Vector Search Metrics:\n")
        f.write(str(baseline_result) + "\n\n")
        f.write("Hybrid Agent Metrics:\n")
        f.write(str(hybrid_result) + "\n")
        
    print("\nEvaluation complete! Results saved to eval/ragas_metrics.txt")

if __name__ == "__main__":
    main()

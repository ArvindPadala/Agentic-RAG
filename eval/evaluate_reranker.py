import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import run_agent_turn, build_search_tool, create_s3_client, build_agent_config
from gemini_helpers import init_chroma_collection
from llm_router import GeminiRouter
from config import settings

def main():
    if not os.path.exists("eval/golden_dataset.csv"):
        print("Error: eval/golden_dataset.csv not found.")
        return
        
    df = pd.read_csv("eval/golden_dataset.csv")

    print("Initializing Elite Hybrid (Reranker) Agent pipeline...")
    gemini_client = GeminiRouter(api_keys=settings.GEMINI_API_KEYS)
    s3 = create_s3_client()
    coll = init_chroma_collection("./chroma_db", "document_chunks")
    
    # Enable the reranker
    search_fn, search_tool = build_search_tool(
        collection=coll, 
        gemini_client=gemini_client, 
        s3_client=s3, 
        bucket=settings.S3_BUCKET_NAME, 
        use_hybrid=True,
        use_reranker=True
    )
    config = build_agent_config(search_tool, {})

    answers = []

    print("Generating answers for Golden Dataset using Elite Hybrid...")
    for i, row in df.iterrows():
        question = row["question"]
        print(f"[{i+1}/{len(df)}] Q: {question[:50]}...")
        try:
            response = run_agent_turn(
                user_message=question,
                conversation_history=[],
                gemini_client=gemini_client,
                generation_config=config,
                tool_map={"search_knowledge_base": search_fn},
                model="gemini-3.5-flash"
            )
            answers.append(response)
        except Exception as e:
            print(f"Error generating answer: {e}")
            answers.append("Error")

    df["answer"] = answers
    df.to_csv("eval/reranker_results.csv", index=False)
    print("Saved reranker answers to eval/reranker_results.csv")

if __name__ == "__main__":
    main()

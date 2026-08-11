import os
import sys
import pandas as pd
import google.genai as genai

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import run_agent_turn, build_search_tool, create_s3_client, build_agent_config
from gemini_helpers import init_chroma_collection
from llm_router import GeminiRouter
from config import settings


def main():
    if not os.path.exists("eval/golden_dataset.csv"):
        print("Error: eval/golden_dataset.csv not found. Please run eval/generate_testset.py first.")
        return
    df = pd.read_csv("eval/golden_dataset.csv")

    print("Initializing Hybrid Agent pipeline...")
    gemini_client = GeminiRouter(api_keys=settings.GEMINI_API_KEYS)
    s3 = create_s3_client()
    coll = init_chroma_collection("./chroma_db", "document_chunks")
    search_fn, search_tool = build_search_tool(coll, gemini_client, s3, settings.S3_BUCKET_NAME, use_hybrid=True)
    config = build_agent_config(search_tool, {})

    answers = []

    print("Generating answers for Golden Dataset...")
    for i, row in df.iterrows():
        question = row["question"]
        print(f"[{i+1}/{len(df)}] Q: {question[:50]}...")
        try:
            # We run the agent turn. The agent will retrieve and generate.
            # RAGAS needs `answer` and ideally the specific `contexts` retrieved.
            # For simplicity in this script, we'll just evaluate the end-to-end generation.
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

    # Normally, here we would use ragas.evaluate(Dataset.from_pandas(df), metrics=[...])
    # However, setting up the exact LangChain wrapper for Ragas with the new google-genai SDK
    # can be verbose. This script establishes the evaluation generation loop.
    # In a full production setup, the LangSmith tracer would intercept the RAGAS calls.

    df.to_csv("eval/hybrid_results.csv", index=False)
    print("Saved hybrid answers to eval/hybrid_results.csv")

    print("\n--- Hybrid Evaluation Complete ---")
    print("To get full quantitative metrics (Context Precision, Faithfulness, etc.), ")
    print("you would feed this CSV into the Ragas evaluator.")


if __name__ == "__main__":
    main()

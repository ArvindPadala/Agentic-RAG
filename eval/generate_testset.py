import os
import json
import pandas as pd
from google import genai
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# noqa: E402
from gemini_helpers import init_chroma_collection
# noqa: E402
from config import settings


def generate_golden_dataset():
    print("Loading ChromaDB...")
    collection = init_chroma_collection("./chroma_db", "document_chunks")
    results = collection.get()

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    if not documents:
        print("No documents found in ChromaDB.")
        return

    # Sample up to 15 chunks to generate questions
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(documents)), min(15, len(documents)))

    from llm_router import GeminiRouter
    client = GeminiRouter(api_keys=settings.GEMINI_API_KEYS)

    dataset = []

    print(f"Generating questions for {len(sample_indices)} chunks...")
    for i, idx in enumerate(sample_indices):
        chunk_text = documents[idx]
        if len(chunk_text.strip()) < 100:
            continue

        prompt = f"""
You are an expert evaluator. Based on the following context, generate a challenging question that can be answered ONLY using this context, and provide the correct answer.

Format your response strictly as a JSON object:
{{
    "question": "The challenging question here",
    "ground_truth": "The correct answer here"
}}

Context:
{chunk_text}
"""

        try:
            response = client.models.generate_content(
                model='models/gemini-3.6-flash',
                contents=prompt,
            )
            # Basic JSON extraction
            resp_text = response.text.strip()
            if resp_text.startswith("```json"):
                resp_text = resp_text[7:-3].strip()
            elif resp_text.startswith("```"):
                resp_text = resp_text[3:-3].strip()

            data = json.loads(resp_text)

            dataset.append({
                "question": data["question"],
                "ground_truth": data["ground_truth"],
                "contexts": [chunk_text],
                "source": metadatas[idx].get("source_document", "unknown")
            })
            print(f"[{i+1}/{len(sample_indices)}] Generated Q: {data['question'][:50]}...")
        except Exception as e:
            print(f"Error generating for chunk {idx}: {e}")

    df = pd.DataFrame(dataset)
    df.to_csv("eval/golden_dataset.csv", index=False)
    print("Saved golden dataset to eval/golden_dataset.csv")


if __name__ == "__main__":
    generate_golden_dataset()

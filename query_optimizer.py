"""
query_optimizer.py — Pre-processing LLM layer for Query Decomposition
======================================================================
Intercepts complex user questions and breaks them down into targeted
sub-queries before handing them off to the ReAct agent's search tool.
"""
import json
from google.genai import types as genai_types
from llm_router import GeminiRouter
from utils.logger import get_logger

logger = get_logger("query_optimizer")

DECOMPOSITION_PROMPT = """You are an expert search query optimizer for a RAG system.
Your task is to analyze the user's question and decompose it into 1 to 3 distinct, highly targeted search queries.

Rules:
1. If the question is simple or asks for a single fact, return an array with exactly ONE optimized query.
2. If the question is multi-part, vague, or requires comparing multiple distinct concepts, break it down into 2 or 3 specific sub-queries.
3. Remove conversational filler. Extract the core entities and intent.
4. Output ONLY a valid JSON array of strings. No markdown formatting, no explanations.

Example Input:
"What is the difference between standard vector search and RRF, and how did they test generation diversity?"

Example Output:
[
  "standard vector search vs Reciprocal Rank Fusion (RRF) differences",
  "generation diversity testing methodology and evaluation"
]
"""


def decompose_query(raw_query: str, gemini_router: GeminiRouter,
                    model: str = "models/gemini-3.5-flash-lite") -> list[str]:
    """
    Takes a raw user question and asks a fast LLM model to decompose it
    into a JSON list of targeted sub-queries.
    """
    logger.info(f"🧠 Optimizing query: '{raw_query}'")

    config = genai_types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        system_instruction=DECOMPOSITION_PROMPT
    )

    try:
        response = gemini_router.models.generate_content(
            model=model,
            contents=[raw_query],
            config=config
        )

        # Parse the JSON response
        sub_queries = json.loads(response.text)

        if not isinstance(sub_queries, list) or not all(
                isinstance(q, str) for q in sub_queries):
            logger.warning(
                "Query optimizer returned invalid JSON format. Falling back to raw query.")
            return [raw_query]

        logger.info(
            f"✅ Decomposed into {
                len(sub_queries)} queries: {sub_queries}")
        return sub_queries

    except Exception as e:
        logger.error(f"❌ Query optimization failed: {
                     e}. Falling back to raw query.")
        return [raw_query]

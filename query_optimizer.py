"""
query_optimizer.py — Pre-processing LLM layer for Query Decomposition
======================================================================
Intercepts complex user questions and breaks them down into targeted
sub-queries before handing them off to the ReAct agent's search tool.
"""
import json
from google.genai import types as genai_types
from llm_router import GeminiRouter
from contextlib import nullcontext
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
                    model: str = "models/gemini-3.5-flash-lite", execution=None) -> list[str]:
    """
    Takes a raw user question and asks a fast LLM model to decompose it
    into a JSON list of targeted sub-queries.
    """
    logger.info("Optimizing query")

    config = genai_types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        max_output_tokens=512,
        system_instruction=DECOMPOSITION_PROMPT
    )

    try:
        with execution.measure("decomposition", "Optimising search queries") if execution else nullcontext():
            response = gemini_router.models.generate_content(model=model, contents=[raw_query], config=config)

        # Parse the JSON response
        sub_queries = json.loads(response.text)

        if (not isinstance(sub_queries, list) or not 1 <= len(sub_queries) <= 3
                or not all(isinstance(q, str) and q.strip() and len(q) <= 512 for q in sub_queries)):
            raise ValueError("Invalid search queries")
        sub_queries = list(dict.fromkeys(q.strip() for q in sub_queries))
        if execution:
            execution.set_decomposition("completed", sub_queries)

        logger.info("Decomposed into %s queries", len(sub_queries))
        return sub_queries

    except Exception:
        logger.warning("Query optimisation unavailable; using the original question")
        if execution:
            execution.set_decomposition("fallback")
        return [raw_query]

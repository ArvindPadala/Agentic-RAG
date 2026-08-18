"""
live_guardrail.py — Runtime Faithfulness Check
================================================
A fast LLM-as-a-judge interceptor that evaluates whether the agent's 
generated answer hallucinated any claims not grounded in the retrieved context.
"""
import json
from google.genai import types as genai_types
from llm_router import GeminiRouter
from utils.logger import get_logger

logger = get_logger("live_guardrail")

GUARDRAIL_PROMPT = """You are a strict faithfulness evaluator for a Retrieval-Augmented Generation (RAG) system.
Your job is to determine if the generated Answer contains any claims, facts, or numbers that are NOT explicitly supported by the Retrieved Context.

Rules:
1. Ignore tone, formatting, and helpful conversational filler (e.g. "Based on the documents...").
2. Focus ONLY on verifiable facts, numbers, entities, and logical claims.
3. If the Answer contains ANY factual claim that cannot be proven by the Context, it is NOT faithful (is_faithful: false).
4. If the Answer correctly states that the information is missing from the Context, it IS faithful (is_faithful: true).
5. Output ONLY a valid JSON object matching the requested schema. No markdown formatting.

Format your output exactly like this:
{
  "is_faithful": true or false,
  "reason": "Brief 1-sentence explanation of why it passed or failed."
}
"""

def check_faithfulness(answer: str, context_chunks: list[str], gemini_router: GeminiRouter, model: str = "models/gemini-3.5-flash-lite") -> tuple[bool, str]:
    """
    Evaluates the final answer against the aggregated retrieved context.
    Returns a tuple of (is_faithful, reasoning).
    """
    if not context_chunks:
        # If no context was retrieved, any factual claim in the answer is a hallucination.
        # But if the answer is "I don't know", it's faithful. We let the LLM decide.
        context_text = "NO CONTEXT RETRIEVED"
    else:
        context_text = "\n\n---\n\n".join(context_chunks)
        
    evaluation_input = f"""
=== RETRIEVED CONTEXT ===
{context_text}

=== GENERATED ANSWER ===
{answer}
"""

    logger.info("🛡️ Running Live Faithfulness Guardrail...")
    
    config = genai_types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        system_instruction=GUARDRAIL_PROMPT
    )
    
    try:
        response = gemini_router.models.generate_content(
            model=model,
            contents=[evaluation_input],
            config=config
        )
        
        result = json.loads(response.text)
        is_faithful = result.get("is_faithful", True)
        reason = result.get("reason", "No reason provided")
        
        if is_faithful:
            logger.info(f"✅ Guardrail PASSED: {reason}")
        else:
            logger.warning(f"⚠️ Guardrail FAILED: {reason}")
            
        return is_faithful, reason
        
    except Exception as e:
        logger.error(f"❌ Guardrail execution failed: {e}. Defaulting to pass.")
        # Fail open in production so we don't break the UX if the guardrail API errors
        return True, "Guardrail API error."

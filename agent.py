"""
agent.py — Document RAG Chatbot (standalone script)
========================================================

Extracted from pipeline_setup.ipynb. Runs the full Gemini-powered document
chatbot from the terminal — no Jupyter required.

Usage:
    python agent.py                          # Interactive chat
    python agent.py -q "cold symptoms?"     # Single question, then exit
    python agent.py --collection my_docs    # Use a different ChromaDB collection

Prerequisites (already done if you ran the notebook):
    - .env file with GEMINI_API_KEY, AWS_ACCESS_KEY_ID, etc.
    - chroma_db/ folder with indexed document chunks
    - memory.json (created automatically on first exit)
"""

import json
import argparse
from datetime import datetime

from config import settings
from utils.logger import get_logger

logger = get_logger('agent')

from google import genai
from google.genai import types as genai_types

from gemini_helpers import (
    init_chroma_collection,
    search_chroma,
    load_memory,
    save_memory,
    format_memory_for_prompt,
    update_memory_from_conversation,
)
from visual_grounding_helper import extract_chunk_image


# ── 1. Environment ────────────────────────────────────────────────────────────
# Environment is now loaded securely via config.py


from llm_router import GeminiRouter

def create_gemini_router(api_keys: list) -> GeminiRouter:
    """Create and validate the robust Gemini router."""
    router = GeminiRouter(api_keys=api_keys)
    logger.info("✅ Gemini router ready")
    return router


def create_s3_client():
    """Create a boto3 S3 client for visual grounding (PDF downloads)."""
    import boto3
    session = boto3.Session(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )
    client = session.client("s3")
    logger.info("✅ AWS S3 client ready")
    return client


# ── 3. ChromaDB ───────────────────────────────────────────────────────────────
def load_chroma_collection(collection_name: str, chroma_path: str = "./chroma_db"):
    """Open the existing ChromaDB collection from disk."""
    collection = init_chroma_collection(
        persist_directory=chroma_path,
        collection_name=collection_name,
    )
    count = collection.count()
    if count == 0:
        logger.info(f"⚠️  Collection '{collection_name}' is empty.")
        logger.info("   Run the notebook Steps 9a–9b first to index your documents.")
    else:
        logger.info(f"✅ ChromaDB ready — {count} chunks indexed")
    return collection


# ── 4. Search Tool ────────────────────────────────────────────────────────────
def build_search_tool(collection, gemini_client, s3_client, bucket: str, use_hybrid: bool = False):
    """
    Build the search_knowledge_base function that the Gemini agent will call.

    Returns both:
      - The Python callable (for TOOL_MAP)
      - The types.Tool declaration (for GENERATION_CONFIG)
    """

    def search_knowledge_base(query: str) -> str:
        """
        Search the document knowledge base.
        Returns text chunks with page numbers and visual reference image URLs.
        """
        logger.info(f"   🔍 Querying ChromaDB for: '{query}'")

        if use_hybrid:
            from hybrid_search import search_chroma_hybrid
            results = search_chroma_hybrid(
                query=query,
                collection=collection,
                n_results=5,
            )
        else:
            results = search_chroma(
                query=query,
                collection=collection,
                n_results=5,
            )

        if not results:
            return f"No documents found for query: '{query}'."

        formatted_results = []
        seen_chunk_ids = set()

        for result in results:
            chunk_id   = result["chunk_id"]
            source_doc = result["source_document"]
            score      = result["score"]
            page       = result["page"]
            chunk_type = result["chunk_type"]
            bbox       = result["bbox"]
            content    = result["text"]

            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            # Visual grounding: crop the PDF region and get a presigned URL
            cropped_image_url = None
            if source_doc and bucket and s3_client:
                # Try the documents path
                possible_keys = [
                    f"input/documents/{source_doc}.pdf"
                ]

                for source_pdf_key in possible_keys:
                    try:
                        s3_client.head_object(Bucket=bucket, Key=source_pdf_key)
                        cropped_image_url = extract_chunk_image(
                            s3_client=s3_client,
                            bucket=bucket,
                            source_pdf_key=source_pdf_key,
                            bbox=bbox,
                            page_num=page,
                            chunk_id=chunk_id,
                            source_document=source_doc,
                            highlight=True,
                            padding=10,
                        )
                        break  # Found and processed successfully
                    except Exception:
                        continue  # Try the next prefix


            if cropped_image_url:
                result_text = (
                    f"**Source:** {source_doc} (Relevance: {score:.2f})\n"
                    f"📄 **Chunk ID:** {chunk_id}\n"
                    f"📍 **Page:** {page}\n"
                    f"🏷️ **Chunk Type:** {chunk_type}\n"
                    f"🔍 **Visual Reference:** {cropped_image_url}\n\n"
                    f"**Content:**\n{content}"
                )
            else:
                result_text = (
                    f"**Source:** {source_doc} (Relevance: {score:.2f})\n"
                    f"📍 **Page:** {page} | 🏷️ **Type:** {chunk_type}\n\n"
                    f"**Content:**\n{content}"
                )

            formatted_results.append(result_text)

        return "\n\n---\n\n".join(formatted_results[:5])

    # Gemini tool declaration — tells the LLM what this function does
    tool_declaration = genai_types.FunctionDeclaration(
        name="search_knowledge_base",
        description=(
            "Search the document knowledge base for relevant information "
            "about the user's documents. "
            "Returns text content with page numbers and visual references (image URLs) "
            "showing the exact location in the source PDF."
        ),
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "query": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="The search query — a natural language question or topic",
                )
            },
            required=["query"],
        ),
    )
    tool = genai_types.Tool(function_declarations=[tool_declaration])

    return search_knowledge_base, tool


# ── 5. Agent Config ───────────────────────────────────────────────────────────
def build_agent_config(search_tool, memory: dict) -> genai_types.GenerateContentConfig:
    """Build the GenerateContentConfig with system prompt and tools."""
    memory_context = format_memory_for_prompt(memory)

    system_prompt = f"""You are an expert document analysis assistant.
You have access to a repository of documents via the search_knowledge_base tool.

Your capabilities:
- Search and analyze documents from the knowledge base
- Provide visual grounding: show EXACT page numbers and image URLs from source PDFs
- Remember user preferences and conversation history
- Provide evidence-based, cited responses

IMPORTANT: When search results include visual grounding information, you MUST include:
- The page number where information was found
- The visual reference URL (so the user can see the highlighted text in context)

Always call search_knowledge_base before answering questions about the documents.
Always cite your sources.
{memory_context}"""

    return genai_types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=2048,
        tools=[search_tool],
        system_instruction=system_prompt,
    )


from langsmith import traceable

# ── 6. Agent Loop ─────────────────────────────────────────────────────────────
@traceable(run_type="chain")
def run_agent_turn(
    user_message: str,
    conversation_history: list,
    gemini_client,
    generation_config,
    tool_map: dict,
    model: str = "models/gemini-3.6-flash",
    max_iterations: int = 5,
) -> str:
    """
    Process one user message through the full agent loop.

    The agent may call tools multiple times before returning a final answer.
    Each iteration either:
      - executes requested tool calls and loops again, OR
      - returns the final text response

    Uses types.Content objects (not plain dicts) to avoid pydantic v2
    validation errors with the google-genai SDK.
    """
    # Add user's message to history
    conversation_history.append(
        genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
    )

    for iteration in range(max_iterations):
        response = gemini_client.models.generate_content(
            model=model,
            contents=conversation_history,
            config=generation_config,
        )

        candidate = response.candidates[0]
        response_parts = candidate.content.parts

        tool_calls = [
            p for p in response_parts
            if hasattr(p, "function_call") and p.function_call
        ]

        if tool_calls:
            # Add the model's tool-call request to history
            conversation_history.append(candidate.content)  # already types.Content ✅

            # Execute each requested tool
            function_responses = []
            for part in tool_calls:
                fc = part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                logger.info(f"   🔧 {tool_name}({', '.join(f'{k}={repr(v)}' for k, v in tool_args.items())})")

                if tool_name in tool_map:
                    tool_result = tool_map[tool_name](**tool_args)
                else:
                    tool_result = f"Error: Unknown tool '{tool_name}'"

                function_responses.append(
                    genai_types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_result},
                    )
                )

            # Send tool results back to Gemini
            conversation_history.append(
                genai_types.Content(role="user", parts=function_responses)
            )

        else:
            # Gemini returned a text response — done
            final_text = "".join(
                p.text for p in response_parts if hasattr(p, "text") and p.text
            )
            conversation_history.append(
                genai_types.Content(role="model", parts=[genai_types.Part(text=final_text)])
            )
            return final_text
            
    # If the loop exhausts max_iterations without a final text response
    fallback_text = "I'm sorry, I was unable to find a complete answer within the allowed number of searches."
    conversation_history.append(
        genai_types.Content(role="model", parts=[genai_types.Part(text=fallback_text)])
    )
    return fallback_text


# ── 7. Chat Loop ──────────────────────────────────────────────────────────────
def run_chat(gemini_client, generation_config, tool_map: dict, memory: dict,
             model: str, memory_file: str):
    """Run the interactive chat loop."""
    conversation_history = []  # Fresh history every session
    conversation_num = 0

    logger.info("=" * 70)
    logger.info("  Document Agent — Interactive Chat with Visual Grounding (Gemini)")
    logger.info("=" * 70)
    logger.info("  Ask questions about your documents.")
    logger.info("  Type 'exit' to end (memory will be saved).")
    logger.info("=" * 70)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye", "q"]:
                logger.info("\n⏳ Saving memory from this session...")
                updated_memory = update_memory_from_conversation(
                    memory=memory,
                    conversation_history=conversation_history,
                    gemini_client=gemini_client,
                )
                save_memory(updated_memory, memory_file)
                logger.info("👋 Goodbye! (Memory saved — I'll remember this next time)")
                break

            conversation_num += 1
            logger.info(f"\n{'─' * 70}")
            logger.info(f"Question #{conversation_num} [{datetime.now().strftime('%H:%M:%S')}]")
            logger.info(f"  \"{user_input}\"")
            logger.info(f"{'─' * 70}")
            logger.info("\nAgent Response (processing...)\n")

            result = run_agent_turn(
                user_message=user_input,
                conversation_history=conversation_history,
                gemini_client=gemini_client,
                generation_config=generation_config,
                tool_map=tool_map,
                model=model,
            )
            logger.info(result)
            logger.info(f"\n{'=' * 70}")

        except KeyboardInterrupt:
            logger.info("\n\n💾 Saving memory before exit...")
            updated_memory = update_memory_from_conversation(
                memory, conversation_history, gemini_client
            )
            save_memory(updated_memory, memory_file)
            logger.info("Conversation interrupted. Goodbye!")
            break

        except Exception as e:
            logger.info(f"\n❌ Error: {e}")
            logger.info("   Please try again or type 'exit' to quit.")


# ── 8. Single-question Mode ───────────────────────────────────────────────────
def ask_single_question(question: str, gemini_client, generation_config,
                         tool_map: dict, model: str) -> str:
    """Answer one question and return the result (no interactive loop)."""
    conversation_history = []
    logger.info(f"\n🔍 Question: {question}\n")
    answer = run_agent_turn(
        user_message=question,
        conversation_history=conversation_history,
        gemini_client=gemini_client,
        generation_config=generation_config,
        tool_map=tool_map,
        model=model,
    )
    return answer


# ── 9. Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Medical Document Chatbot powered by Gemini + ChromaDB"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Ask a single question and exit (non-interactive mode)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="document_chunks",
        help="ChromaDB collection name to use (default: document_chunks)",
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default="./chroma_db",
        help="Path to ChromaDB folder (default: ./chroma_db)",
    )
    parser.add_argument(
        "--memory-file",
        type=str,
        default="memory.json",
        help="Path to memory JSON file (default: memory.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gemini-2.5-flash",
        help="Gemini model to use (default: models/gemini-2.5-flash)",
    )
    args = parser.parse_args()

    logger.info("\n🚀 Starting Medical Agent...")
    logger.info("─" * 40)

    # Setup dependencies
    gemini_client = GeminiRouter(api_keys=settings.GEMINI_API_KEYS)
    s3_client = create_s3_client()
    collection = load_chroma_collection(
        collection_name=args.collection,
        chroma_path=args.chroma_path,
    )
    bucket = settings.S3_BUCKET_NAME
    search_fn, search_tool = build_search_tool(
        collection=collection,
        gemini_client=gemini_client,
        s3_client=s3_client,
        bucket=bucket,
    )
    tool_map = {"search_knowledge_base": search_fn}

    # Step 5: Load memory
    memory = load_memory(args.memory_file)
    sessions = len(memory.get("session_summaries", []))
    logger.info(f"✅ Memory loaded — {sessions} previous session(s) remembered")

    # Step 6: Build agent config
    generation_config = build_agent_config(search_tool, memory)

    logger.info("─" * 40)

    # Step 7: Run
    if args.question:
        # Non-interactive: answer one question and exit
        answer = ask_single_question(
            question=args.question,
            gemini_client=gemini_client,
            generation_config=generation_config,
            tool_map=tool_map,
            model=args.model,
        )
        logger.info(f"\nAnswer:\n{answer}")
    else:
        # Interactive chat loop
        run_chat(
            gemini_client=gemini_client,
            generation_config=generation_config,
            tool_map=tool_map,
            memory=memory,
            model=args.model,
            memory_file=args.memory_file,
        )


if __name__ == "__main__":
    main()

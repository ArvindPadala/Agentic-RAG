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

import concurrent.futures
from typing import TypedDict, Any, List
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from llm_router import GeminiRouter
from visual_grounding_helper import extract_chunk_image
from gemini_helpers import (
    init_chroma_collection,
    search_chroma,
    load_memory,
    save_memory,
    format_memory_for_prompt,
    update_memory_from_conversation,
)
from google.genai import types as genai_types
import argparse
from datetime import datetime

from config import settings
from utils.logger import get_logger

logger = get_logger('agent')


# ── 1. Environment ──────────────────────────────────────────────────────
# Environment is now loaded securely via config.py


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


# ── 3. ChromaDB ─────────────────────────────────────────────────────────
def load_chroma_collection(collection_name: str,
                           chroma_path: str = "./chroma_db"):
    """Open the existing ChromaDB collection from disk."""
    collection = init_chroma_collection(
        persist_directory=chroma_path,
        collection_name=collection_name,
    )
    count = collection.count()
    if count == 0:
        logger.info(f"⚠️  Collection '{collection_name}' is empty.")
        logger.info(
            "   Run the notebook Steps 9a–9b first to index your documents.")
    else:
        logger.info(f"✅ ChromaDB ready — {count} chunks indexed")
    return collection


# ── 4. Search Tool ──────────────────────────────────────────────────────
def build_search_tool(collection, gemini_client, s3_client, bucket: str,
                      use_hybrid: bool = False, use_reranker: bool = False):
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
                use_reranker=use_reranker,
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
            chunk_id = result["chunk_id"]
            source_doc = result["source_document"]
            score = result["score"]
            page = result["page"]
            chunk_type = result["chunk_type"]
            bbox = result["bbox"]
            content = result["text"]

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
                        s3_client.head_object(
                            Bucket=bucket, Key=source_pdf_key)
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


# ── 5. Agent Config ─────────────────────────────────────────────────────
def build_agent_config(
        search_tool, memory: dict) -> genai_types.GenerateContentConfig:
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

## STRICT BOUNDARY ENFORCEMENT
You are a strict document assistant. If the provided context does not contain the answer, or if the user asks a question entirely unrelated to the knowledge base (like basic math or chitchat), DO NOT use your internal knowledge. Politely explain that you can only answer questions based on the retrieved documents.

## Self-Correction Protocol
After each search, critically evaluate the retrieved context BEFORE answering:
1. **Coverage check:** Does the context address the question? If tangential, try ONE refined query.
2. **Confidence check:** If evidence is weak or ambiguous, try ONE alternative query with synonyms.
3. **Completeness check:** For multi-part questions, verify each part has evidence. Search once more for any gap.
4. **No repetition:** NEVER search for the same concept twice. If two searches on a topic return similar results, the information is likely not in the corpus — accept that and move on.
5. **Budget:** You have a maximum of 3-4 searches total. After that, answer with what you have.
6. **Acknowledge limits:** If evidence is missing after a few searches, say so explicitly rather than fabricating an answer.
{memory_context}"""

    return genai_types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=2048,
        tools=[search_tool],
        system_instruction=system_prompt,
    )


class AgentState(TypedDict):
    conversation_history: List[Any]
    iteration: int
    max_iterations: int
    gemini_client: Any
    generation_config: Any
    tool_map: dict
    model: str
    use_guardrail: bool
    final_text: str


def build_agent_graph():
    def call_llm(state: AgentState):
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        gemini_client = state["gemini_client"]
        generation_config = state["generation_config"]
        conversation_history = state["conversation_history"]
        model = state["model"]

        if iteration > 0:
            logger.info(f"   🔄 Reflection iteration {iteration + 1}/{max_iterations} — agent is refining its search")

        if iteration == max_iterations - 2:
            conversation_history.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=(
                        "[System: You have used most of your search budget. "
                        "You MUST synthesize your final answer NOW from the evidence "
                        "you have collected. Do NOT search again. If you lack "
                        "information for part of the question, state that explicitly.]"
                    ))]
                )
            )

        if iteration == max_iterations - 1:
            forced_config = genai_types.GenerateContentConfig(
                temperature=generation_config.temperature,
                max_output_tokens=generation_config.max_output_tokens,
                system_instruction=generation_config.system_instruction,
            )
        else:
            forced_config = generation_config

        response = gemini_client.models.generate_content(
            model=model,
            contents=conversation_history,
            config=forced_config,
        )

        candidate = response.candidates[0]
        conversation_history.append(candidate.content)

        return {"conversation_history": conversation_history,
                "iteration": iteration + 1}

    def execute_tools(state: AgentState):
        conversation_history = state["conversation_history"]
        tool_map = state["tool_map"]

        last_message = conversation_history[-1]
        tool_calls = [
            p for p in last_message.parts if hasattr(
                p, "function_call") and p.function_call]

        function_responses = []

        def execute_single_tool(part):
            fc = part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}

            logger.info(f"   🔧 {tool_name}({', '.join(f'{k}={repr(v)}' for k, v in tool_args.items())})")

            if tool_name in tool_map:
                try:
                    tool_result = tool_map[tool_name](**tool_args)
                except Exception as e:
                    logger.error(f"Error executing {tool_name}: {e}")
                    tool_result = f"Error: {e}"
            else:
                tool_result = f"Error: Unknown tool '{tool_name}'"

            return genai_types.Part.from_function_response(
                name=tool_name,
                response={"result": tool_result},
            )

        # Run tool calls in parallel using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Map returns results in the same order
            function_responses = list(
                executor.map(
                    execute_single_tool,
                    tool_calls))

        conversation_history.append(
            genai_types.Content(role="user", parts=function_responses)
        )
        return {"conversation_history": conversation_history}

    def run_guardrail(state: AgentState):
        conversation_history = state["conversation_history"]
        use_guardrail = state["use_guardrail"]
        gemini_client = state["gemini_client"]

        last_message = conversation_history[-1]
        final_text = "".join(
            p.text for p in last_message.parts if hasattr(
                p, "text") and p.text)

        if use_guardrail:
            from live_guardrail import check_faithfulness

            context_chunks = []
            for content in conversation_history:
                if content.role == "user":
                    for part in content.parts:
                        if hasattr(
                                part, "function_response") and part.function_response:
                            # In google.genai SDK, function_response is an
                            # object with a 'response' dict attribute
                            res_dict = part.function_response.response if hasattr(
                                part.function_response, "response") else {}
                            res_text = res_dict.get(
                                "result", "") if isinstance(
                                res_dict, dict) else ""
                            if res_text and "No documents found" not in res_text:
                                context_chunks.append(res_text)

            is_faithful, reason = check_faithfulness(
                final_text, context_chunks, gemini_client)

            if not is_faithful:
                final_text += f"\n\n> ⚠️ **Guardrail Warning:** This answer may contain information not explicitly grounded in the retrieved context. (Reason: {reason})"

        conversation_history[-1] = genai_types.Content(
            role="model", parts=[genai_types.Part(text=final_text)])

        return {"final_text": final_text,
                "conversation_history": conversation_history}

    def should_continue(state: AgentState):
        last_message = state["conversation_history"][-1]
        tool_calls = [
            p for p in last_message.parts if hasattr(
                p, "function_call") and p.function_call]

        if tool_calls:
            return "execute_tools"
        return "run_guardrail"

    workflow = StateGraph(AgentState)
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("run_guardrail", run_guardrail)

    workflow.add_edge(START, "call_llm")
    workflow.add_conditional_edges("call_llm", should_continue, {
        "execute_tools": "execute_tools",
        "run_guardrail": "run_guardrail"
    })
    workflow.add_edge("execute_tools", "call_llm")
    workflow.add_edge("run_guardrail", END)

    return workflow.compile()


COMPILED_AGENT_GRAPH = build_agent_graph()

# ── 6. Agent Loop ───────────────────────────────────────────────────────


@traceable(run_type="chain")
def run_agent_turn(
    user_message: str,
    conversation_history: list,
    gemini_client,
    generation_config,
    tool_map: dict,
    model: str = "models/gemini-3.6-flash",
    max_iterations: int = 7,
    use_decomposition: bool = False,
    use_guardrail: bool = False,
) -> str:
    """
    Process one user message through the LangGraph agent state machine.
    """
    if use_decomposition:
        from query_optimizer import decompose_query
        sub_queries = decompose_query(user_message, gemini_client)
        if len(sub_queries) > 1 or (len(sub_queries) ==
                                    1 and sub_queries[0] != user_message):
            formatted_queries = "\n".join([f"- {q}" for q in sub_queries])
            injected_message = (
                f"User Question: {user_message}\n\n"
                f"[System: To answer this, please use your search tool to explore these optimized sub-queries:]\n"
                f"{formatted_queries}"
            )
        else:
            injected_message = user_message
    else:
        injected_message = user_message

    # Add user's message to history
    conversation_history.append(
        genai_types.Content(
            role="user", parts=[
                genai_types.Part(
                    text=injected_message)])
    )

    initial_state = {
        "conversation_history": conversation_history,
        "iteration": 0,
        "max_iterations": max_iterations,
        "gemini_client": gemini_client,
        "generation_config": generation_config,
        "tool_map": tool_map,
        "model": model,
        "use_guardrail": use_guardrail,
        "final_text": ""
    }

    final_state = COMPILED_AGENT_GRAPH.invoke(initial_state)

    if not final_state.get("final_text"):
        fallback_text = "I'm sorry, I was unable to find a complete answer within the allowed number of searches."
        conversation_history.append(
            genai_types.Content(
                role="model", parts=[
                    genai_types.Part(
                        text=fallback_text)])
        )
        return fallback_text

    return final_state["final_text"]


# ── 7. Chat Loop ────────────────────────────────────────────────────────
def run_chat(gemini_client, generation_config, tool_map: dict, memory: dict,
             model: str, memory_file: str, use_decomposition: bool = False,
             use_guardrail: bool = False):
    """Run the interactive chat loop."""
    conversation_history = []  # Fresh history every session
    conversation_num = 0

    logger.info("=" * 70)
    logger.info(
        "  Document Agent — Interactive Chat with Visual Grounding (Gemini)")
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
                logger.info(
                    "👋 Goodbye! (Memory saved — I'll remember this next time)")
                break

            conversation_num += 1
            logger.info(f"\n{'─' * 70}")
            logger.info(f"Question #{conversation_num} [{datetime.now().strftime('%H:%M:%S')}]")
            logger.info(f"  \"{user_input}\"")
            logger.info(f"{'─' * 70}")
            logger.info("\nAgent Response (processing...)\n")

            response_text = run_agent_turn(
                user_message=user_input,
                conversation_history=conversation_history,
                gemini_client=gemini_client,
                generation_config=generation_config,
                tool_map=tool_map,
                model=model,
                use_decomposition=use_decomposition,
                use_guardrail=use_guardrail
            )
            logger.info(response_text)
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


# ── 8. Single-question Mode ─────────────────────────────────────────────
def ask_single_question(question: str, gemini_client, generation_config,
                        tool_map: dict, model: str, use_decomposition: bool = False, use_guardrail: bool = False) -> str:
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
        use_decomposition=use_decomposition,
        use_guardrail=use_guardrail
    )
    return answer


# ── 9. Main ─────────────────────────────────────────────────────────────
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
    parser.add_argument(
        "--use-decomposition",
        action="store_true",
        help="Enable Query Decomposition pre-processing")
    parser.add_argument(
        "--use-guardrail",
        action="store_true",
        help="Enable Live Faithfulness Guardrail post-processing")
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
        use_hybrid=True,
        use_reranker=True,
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
            use_decomposition=args.use_decomposition,
            use_guardrail=args.use_guardrail
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
            use_decomposition=args.use_decomposition,
            use_guardrail=args.use_guardrail
        )


if __name__ == "__main__":
    main()

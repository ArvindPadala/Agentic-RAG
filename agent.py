"""
agent.py — Medical Document Chatbot (standalone script)
========================================================

Extracted from Lab-6-Gemini.ipynb. Runs the full Gemini-powered medical
chatbot from the terminal — no Jupyter required.

Usage:
    python agent.py                          # Interactive chat
    python agent.py -q "cold symptoms?"     # Single question, then exit
    python agent.py --collection my_docs    # Use a different ChromaDB collection

Prerequisites (already done if you ran the notebook):
    - .env file with GEMINI_API_KEY, AWS_ACCESS_KEY_ID, etc.
    - chroma_db/ folder with 759 indexed medical chunks
    - memory.json (created automatically on first exit)
"""

import os
import json
import argparse
from datetime import datetime

from dotenv import load_dotenv
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
def load_environment() -> dict:
    """Load and validate all required environment variables."""
    load_dotenv()

    required = [
        "GEMINI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "S3_BUCKET",
    ]

    env = {}
    missing = []
    for var in required:
        value = os.getenv(var)
        if value:
            env[var] = value
        else:
            missing.append(var)

    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("   Check your .env file.")
        raise SystemExit(1)

    print("✅ Environment loaded")
    return env


# ── 2. Clients ────────────────────────────────────────────────────────────────
def create_gemini_client(api_key: str) -> genai.Client:
    """Create and validate the Gemini client."""
    client = genai.Client(api_key=api_key)
    print("✅ Gemini client ready")
    return client


def create_s3_client(env: dict):
    """Create a boto3 S3 client for visual grounding (PDF downloads)."""
    import boto3
    session = boto3.Session(
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name=env["AWS_REGION"],
    )
    client = session.client("s3")
    print("✅ AWS S3 client ready")
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
        print(f"⚠️  Collection '{collection_name}' is empty.")
        print("   Run the notebook Steps 9a–9b first to index your documents.")
    else:
        print(f"✅ ChromaDB ready — {count} chunks indexed")
    return collection


# ── 4. Search Tool ────────────────────────────────────────────────────────────
def build_search_tool(collection, gemini_client, s3_client, bucket: str):
    """
    Build the search_knowledge_base function that the Gemini agent will call.

    Returns both:
      - The Python callable (for TOOL_MAP)
      - The types.Tool declaration (for GENERATION_CONFIG)
    """

    def search_knowledge_base(query: str) -> str:
        """
        Search the medical document knowledge base.
        Returns text chunks with page numbers and visual reference image URLs.
        """
        results = search_chroma(
            query=query,
            collection=collection,
            gemini_client=gemini_client,
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
                source_pdf_key = f"input/medical/{source_doc}.pdf"
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
                except Exception:
                    pass  # Visual grounding is optional — skip if PDF not in S3

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
            "Search the medical document knowledge base for relevant information "
            "about the common cold, its symptoms, causes, treatments, and prevention. "
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

    system_prompt = f"""You are a medical document analysis assistant specializing in the common cold.
You have access to 8 peer-reviewed medical research papers via the search_knowledge_base tool.

Your capabilities:
- Search and analyze medical documents from the knowledge base
- Provide visual grounding: show EXACT page numbers and image URLs from source PDFs
- Remember user preferences and conversation history
- Provide evidence-based, cited responses

IMPORTANT: When search results include visual grounding information, you MUST include:
- The page number where information was found
- The visual reference URL (so the user can see the highlighted text in context)

Always call search_knowledge_base before answering medical questions.
Always cite your sources.
{memory_context}"""

    return genai_types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=2048,
        tools=[search_tool],
        system_instruction=system_prompt,
    )


# ── 6. Agent Loop ─────────────────────────────────────────────────────────────
def run_agent_turn(
    user_message: str,
    conversation_history: list,
    gemini_client,
    generation_config,
    tool_map: dict,
    model: str = "models/gemini-2.5-flash",
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

    while True:
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

                print(f"   🔧 {tool_name}({', '.join(f'{k}={repr(v)}' for k, v in tool_args.items())})")

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


# ── 7. Chat Loop ──────────────────────────────────────────────────────────────
def run_chat(gemini_client, generation_config, tool_map: dict, memory: dict,
             model: str, memory_file: str):
    """Run the interactive chat loop."""
    conversation_history = []  # Fresh history every session
    conversation_num = 0

    print()
    print("=" * 70)
    print("  Medical Agent — Interactive Chat with Visual Grounding (Gemini)")
    print("=" * 70)
    print("  Ask questions about medical documents.")
    print("  Type 'exit' to end (memory will be saved).")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye", "q"]:
                print("\n⏳ Saving memory from this session...")
                updated_memory = update_memory_from_conversation(
                    memory=memory,
                    conversation_history=conversation_history,
                    gemini_client=gemini_client,
                )
                save_memory(updated_memory, memory_file)
                print("👋 Goodbye! (Memory saved — I'll remember this next time)")
                break

            conversation_num += 1
            print(f"\n{'─' * 70}")
            print(f"Question #{conversation_num} [{datetime.now().strftime('%H:%M:%S')}]")
            print(f"  \"{user_input}\"")
            print(f"{'─' * 70}")
            print("\nAgent Response (processing...)\n")

            result = run_agent_turn(
                user_message=user_input,
                conversation_history=conversation_history,
                gemini_client=gemini_client,
                generation_config=generation_config,
                tool_map=tool_map,
                model=model,
            )
            print(result)
            print(f"\n{'=' * 70}")

        except KeyboardInterrupt:
            print("\n\n💾 Saving memory before exit...")
            updated_memory = update_memory_from_conversation(
                memory, conversation_history, gemini_client
            )
            save_memory(updated_memory, memory_file)
            print("Conversation interrupted. Goodbye!")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again or type 'exit' to quit.")


# ── 8. Single-question Mode ───────────────────────────────────────────────────
def ask_single_question(question: str, gemini_client, generation_config,
                         tool_map: dict, model: str) -> str:
    """Answer one question and return the result (no interactive loop)."""
    conversation_history = []
    print(f"\n🔍 Question: {question}\n")
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
        default="medical_chunks",
        help="ChromaDB collection name to use (default: medical_chunks)",
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

    print("\n🚀 Starting Medical Agent...")
    print("─" * 40)

    # Step 1: Load environment
    env = load_environment()

    # Step 2: Create clients
    gemini_client = create_gemini_client(env["GEMINI_API_KEY"])
    s3_client = create_s3_client(env)

    # Step 3: Load ChromaDB
    collection = load_chroma_collection(args.collection, args.chroma_path)

    # Step 4: Build search tool
    search_fn, search_tool = build_search_tool(
        collection=collection,
        gemini_client=gemini_client,
        s3_client=s3_client,
        bucket=env["S3_BUCKET"],
    )
    tool_map = {"search_knowledge_base": search_fn}

    # Step 5: Load memory
    memory = load_memory(args.memory_file)
    sessions = len(memory.get("session_summaries", []))
    print(f"✅ Memory loaded — {sessions} previous session(s) remembered")

    # Step 6: Build agent config
    generation_config = build_agent_config(search_tool, memory)

    print("─" * 40)

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
        print(f"\nAnswer:\n{answer}")
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

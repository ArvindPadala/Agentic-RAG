"""
app.py — Gradio Web Interface for Medical RAG Agent
====================================================

Wraps agent.py in a two-column Gradio Blocks UI:
  - Left:  Chat history + input
  - Right: Visual grounding images (extracted from agent responses)

Run:
    python app.py
    # Opens: http://localhost:7860
    # Public share: python app.py --share
"""

import re
import sys
import argparse
import requests
from io import BytesIO
from PIL import Image
import gradio as gr
from config import settings
from utils.logger import get_logger

logger = get_logger("app")

from agent import (
    create_gemini_client,
    create_s3_client,
    load_chroma_collection,
    build_search_tool,
    build_agent_config,
    run_agent_turn,
)
from gemini_helpers import load_memory, save_memory, update_memory_from_conversation


# ── Helpers ───────────────────────────────────────────────────────────────────

# S3 image URL pattern — matches presigned URLs like:
# https://bucket.s3.amazonaws.com/output/.../chunk.png?X-Amz-Algorithm=...
_S3_URL_RE = re.compile(
    r"https://[^\s\]\[\"'>()]+\.amazonaws\.com/[^\s\]\[\"'>()]+\.png(?:\?[^\s\]\[\"'>()]+)?"
)


def extract_image_urls(text: str) -> list[str]:
    """
    Extract all S3 presigned image URLs from the agent response.

    Handles all three formats Gemini may use:
      1. Markdown link target: [text](https://...amazonaws.com/....png?...)
      2. Bare URL:             https://...amazonaws.com/....png?...
      3. Markdown URL-as-text: [https://...](https://...)  — both occurrences
    """
    # Priority: extract from inside markdown (URL) targets first to avoid
    # partial matches from the [URL-text] portion
    urls = []
    seen = set()

    # 1. Markdown link targets: ](URL)
    for m in re.finditer(r"\]\(" + _S3_URL_RE.pattern + r"\)", text):
        url = m.group(0)[2:-1]  # strip leading ]( and trailing )
        if url not in seen:
            seen.add(url)
            urls.append(url)

    # 2. Any remaining bare S3 URLs not already captured
    for m in _S3_URL_RE.finditer(text):
        url = m.group(0)
        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def fetch_image(url: str) -> Image.Image | None:
    """Download a presigned S3 URL and return a PIL Image (None on failure)."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def clean_response(text: str) -> str:
    """
    Strip S3 image URLs from the chat text — they display in the image panel instead.

    Handles all three formats:
      1. Markdown links containing S3 URLs: [anything](https://...amazonaws.com...)
      2. 🔍 **Visual Reference:** https://...
      3. Bare S3 URLs on their own
    """
    label = "📎 *(see image panel →)*"

    # 1. Markdown links whose href is an S3 image URL
    cleaned = re.sub(
        r"\[[^\]]*\]\(" + _S3_URL_RE.pattern + r"\)",
        label,
        text,
    )

    # 2. Old "🔍 Visual Reference:" prefix format
    cleaned = re.sub(
        r"🔍 \*\*Visual Reference:\*\* " + _S3_URL_RE.pattern,
        label,
        cleaned,
    )

    # 3. Any remaining bare S3 URLs
    cleaned = _S3_URL_RE.sub(label, cleaned)

    return cleaned


def format_memory_status(memory: dict) -> str:
    """Render the memory summary for the info panel."""
    sessions = len(memory.get("session_summaries", []))
    facts    = len(memory.get("facts", []))
    prefs    = len(memory.get("preferences", []))
    lines = [
        "### 🧠 Agent Memory",
        f"- **Sessions remembered:** {sessions}",
        f"- **Facts extracted:** {facts}",
        f"- **Preferences stored:** {prefs}",
    ]
    if memory.get("session_summaries"):
        last = memory["session_summaries"][-1]
        snippet = last[:120] + "..." if len(last) > 120 else last
        lines.append(f"\n**Last session:**\n> {snippet}")
    else:
        lines.append("\n*No previous sessions yet.*")
    return "\n".join(lines)


# ── Available Gemini models ──────────────────────────────────────────────────
# Listed in order of capability. Free-tier daily limits shown for reference.
AVAILABLE_MODELS = [
    ("Gemini 2.5 Flash  (1,500 req/day)",  "models/gemini-2.5-flash"),
    ("Gemini 2.0 Flash  (1,500 req/day)",  "models/gemini-2.0-flash"),
    ("Gemini 2.0 Flash-Lite  (1,500 req/day)", "models/gemini-2.0-flash-lite"),
    ("Gemini 1.5 Flash  (1,500 req/day)",  "models/gemini-1.5-flash"),
    ("Gemini 1.5 Flash-8B  (high quota)",  "models/gemini-1.5-flash-8b"),
]
MODEL_LABELS   = [label for label, _ in AVAILABLE_MODELS]
MODEL_IDS      = {label: mid for label, mid in AVAILABLE_MODELS}
DEFAULT_MODEL_LABEL = MODEL_LABELS[0]


def is_rate_limit_error(e: Exception) -> bool:
    """Detect daily quota / rate-limit errors from Gemini API."""
    msg = str(e).upper()
    return any(kw in msg for kw in (
        "429", "RESOURCE_EXHAUSTED", "QUOTA", "RATE_LIMIT",
        "DAILY_LIMIT", "TOO_MANY_REQUESTS",
    ))


def is_overload_error(e: Exception) -> bool:
    """Detect temporary server overload errors (503 UNAVAILABLE)."""
    msg = str(e).upper()
    return "503" in msg or "UNAVAILABLE" in msg or "HIGH DEMAND" in msg


# ── Core chat function ────────────────────────────────────────────────────────

def make_chat_fn(gemini_client, generation_config, tool_map, memory, memory_file):
    """
    Returns the Gradio chat handler.

    Gradio 6.x uses the 'messages' format:
      {"role": "user" | "assistant", "content": "..."}
    (The old [[user, bot], ...] tuple format causes 'data incompatible with keys'.)
    """
    def chat(user_message: str, history: list, conversation_history: list,
             model_label: str):
        if not user_message.strip():
            return history, conversation_history, [], format_memory_status(memory)

        # Resolve the selected model ID
        model_id = MODEL_IDS.get(model_label, "models/gemini-2.5-flash")

        # Append user message in Gradio 6.x messages format
        history = history + [{"role": "user", "content": user_message}]

        # Run the agent
        try:
            raw_response = run_agent_turn(
                user_message=user_message,
                conversation_history=conversation_history,
                gemini_client=gemini_client,
                generation_config=generation_config,
                tool_map=tool_map,
                model=model_id,
            )
        except Exception as e:
            if is_rate_limit_error(e):
                error_msg = (
                    f"⚠️ **Daily limit reached for `{model_label}`.**\n\n"
                    "The free-tier quota for this model is used up for today.\n"
                    "👉 **Switch to a different model** using the selector in the sidebar, "
                    "then resend your question."
                )
            elif is_overload_error(e):
                error_msg = (
                    f"⚠️ **`{model_label}` is temporarily overloaded** (high demand).\n\n"
                    "This is usually resolved in a few minutes.\n"
                    "👉 **Try a different model** using the selector, or wait a moment and retry."
                )
            else:
                error_msg = f"❌ **Unexpected error:** {e}"

            history = history + [{"role": "assistant", "content": error_msg}]
            # Pop the last user turn from conversation_history so the agent
            # doesn't see a failed partial turn on the next attempt
            if conversation_history:
                conversation_history.pop()
            return history, conversation_history, [], format_memory_status(memory)

        # Extract visual grounding image URLs from the response text
        image_urls = extract_image_urls(raw_response)

        # Clean raw URLs from the visible response text
        display_response = clean_response(raw_response)

        # Append assistant message in Gradio 6.x messages format
        history = history + [{"role": "assistant", "content": display_response}]

        return history, conversation_history, image_urls, format_memory_status(memory)

    return chat


def make_save_fn(gemini_client, memory, memory_file):
    """Save memory to disk."""
    def save(conversation_history):
        if not conversation_history:
            return "⚠️ No conversation to save yet."
        updated = update_memory_from_conversation(memory, conversation_history, gemini_client)
        memory.update(updated)
        save_memory(memory, memory_file)
        return "✅ Memory saved!"
    return save


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(gemini_client, generation_config, tool_map, memory, memory_file):
    chat_fn = make_chat_fn(gemini_client, generation_config, tool_map, memory, memory_file)
    save_fn = make_save_fn(gemini_client, memory, memory_file)

    with gr.Blocks(title="Document RAG Agent") as demo:

        # ── Session state ──────────────────────────────────────────────────
        conv_state    = gr.State([])
        model_state   = gr.State(DEFAULT_MODEL_LABEL)   # tracks selected model

        # ── Header ────────────────────────────────────────────────────────
        with gr.Row(elem_id="header"):
            with gr.Column(scale=4):
                gr.Markdown(
                    """
                    # 📄 Document RAG Agent
                    *Powered by Gemini · ChromaDB · LandingAI ADE*
                    """
                )
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=MODEL_LABELS,
                    value=DEFAULT_MODEL_LABEL,
                    label="🤖 Gemini Model",
                    interactive=True,
                    info="Switch models if you hit a daily limit",
                )

        # ── Main layout: chat (left) + visuals (right) ─────────────────────
        with gr.Row():

            # ── LEFT: Chat ────────────────────────────────────────────────
            with gr.Column(scale=3, elem_classes="chat-col"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                    render_markdown=True,
                    avatar_images=(None, "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg"),
                )

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Ask about your documents… (e.g. 'What was the Q3 revenue?')",
                        show_label=False,
                        lines=2,
                        scale=5,
                    )
                    send_btn = gr.Button("Send →", variant="primary", scale=1, elem_id="send-btn")

                with gr.Row():
                    clear_btn  = gr.Button("🗑️ Clear Chat", size="sm")
                    save_btn   = gr.Button("💾 Save Memory", size="sm", variant="secondary")
                    save_status = gr.Textbox(show_label=False, interactive=False,
                                             placeholder="", scale=2, lines=1)

            # ── RIGHT: Visual Grounding ────────────────────────────────────
            with gr.Column(scale=2, elem_classes="image-col"):
                gr.Markdown("### 🔍 Visual Grounding\n*Highlighted PDF regions from last answer*")
                image_gallery = gr.Gallery(
                    label="Source Evidence",
                    columns=1,
                    height=480,
                    show_label=False,
                    object_fit="contain",
                )

        # ── Bottom: Memory status ──────────────────────────────────────────
        with gr.Accordion("🧠 Agent Memory", open=False):
            memory_display = gr.Markdown(format_memory_status(memory))

        # ── Example questions ──────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["What were the key takeaways from the Q3 earnings report?"],
                ["What is the company's year-over-year revenue growth?"],
                ["Summarize the risk factors mentioned in the document."],
                ["Who are the main competitors listed?"],
                ["What does the chart on page 4 indicate about operating margins?"],
            ],
            inputs=user_input,
            label="Try these questions:",
        )

        # ── Event wiring ───────────────────────────────────────────────────

        def submit(message, history, conv_history, model_label):
            return chat_fn(message, history, conv_history, model_label)

        # Sync model dropdown → model state
        model_dropdown.change(
            fn=lambda label: label,
            inputs=model_dropdown,
            outputs=model_state,
        )

        # Send on button click
        send_btn.click(
            fn=submit,
            inputs=[user_input, chatbot, conv_state, model_state],
            outputs=[chatbot, conv_state, image_gallery, memory_display],
        ).then(
            fn=lambda: gr.update(value=""),
            outputs=user_input,
        )

        # Send on Enter key
        user_input.submit(
            fn=submit,
            inputs=[user_input, chatbot, conv_state, model_state],
            outputs=[chatbot, conv_state, image_gallery, memory_display],
        ).then(
            fn=lambda: gr.update(value=""),
            outputs=user_input,
        )

        # Clear chat (keeps memory, resets conversation)
        clear_btn.click(
            fn=lambda: ([], [], [], format_memory_status(memory)),
            outputs=[chatbot, conv_state, image_gallery, memory_display],
        )

        # Save memory button
        save_btn.click(
            fn=save_fn,
            inputs=[conv_state],
            outputs=[save_status],
        )

    return demo


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Document RAG Agent")
    parser.add_argument("--share",       action="store_true", help="Create a public share link")
    parser.add_argument("--port",        type=int, default=7860, help="Local port (default: 7860)")
    parser.add_argument("--collection",  default="document_chunks", help="ChromaDB collection")
    parser.add_argument("--chroma-path", default="./chroma_db",    help="ChromaDB folder")
    parser.add_argument("--memory-file", default="memory.json",    help="Memory JSON file")
    parser.add_argument("--model",       default="models/gemini-2.5-flash", help="Gemini model")
    args = parser.parse_args()

    logger.info("\n🚀 Starting Document RAG Agent UI...")
    logger.info("─" * 40)

    gemini_client     = create_gemini_client(settings.GEMINI_API_KEY)
    s3_client         = create_s3_client()
    collection        = load_chroma_collection(args.collection, args.chroma_path)
    search_fn, search_tool = build_search_tool(collection, gemini_client, s3_client, settings.S3_BUCKET_NAME)
    tool_map          = {"search_knowledge_base": search_fn}
    memory            = load_memory(args.memory_file)
    generation_config = build_agent_config(search_tool, memory)

    logger.info("─" * 40)
    logger.info(f"✅ All systems ready — launching Gradio on port {args.port}")
    if args.share:
        logger.info("🌐 Share link will be printed below (valid 72 hours)")

    demo = build_ui(gemini_client, generation_config, tool_map, memory, args.memory_file)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        strict_cors=False,
        ssr_mode=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css="""
            #header { text-align: center; padding: 10px 0 4px; }
            #header h1 { font-size: 1.8em; margin-bottom: 2px; }
            #header p  { color: #64748b; margin: 0; font-size: 0.95em; }
            .image-col { border-left: 1px solid #e2e8f0; padding-left: 12px; }
            #send-btn  { min-width: 80px; }
        """,
    )


if __name__ == "__main__":
    main()

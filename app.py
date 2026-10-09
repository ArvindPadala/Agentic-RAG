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

from gemini_helpers import load_memory, save_memory, update_memory_from_conversation
from agent import (
    create_gemini_router,
    create_s3_client,
    load_chroma_collection,
    build_search_tool,
    run_agent_turn,
)
import re
import time
import argparse
import requests
import os
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(func):
        return func



import gradio as gr
from config import settings
from utils.logger import get_logger
from upload_handler import make_upload_fn
from supabase_backend import SupabaseBackend, SupabaseError
from private_sessions import PrivateSessions

logger = get_logger("app")


# ── Helpers ─────────────────────────────────────────────────────────────

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
    facts = len(memory.get("facts", []))
    prefs = len(memory.get("preferences", []))
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


def format_private_memory_status(memory):
    return (format_memory_status(memory) +
            "\n\n*Private demo memory: available only during this temporary session.*")


def runtime_memory_status(memory, policy):
    if policy.is_public:
        return ("### Session Privacy\n"
                "Chat context is scoped to this browser session. "
                "Persistent personal memory is disabled in public guest mode.")
    return format_memory_status(memory)


def load_runtime_memory(memory_file, policy):
    # Do not even open legacy personal memory in a public process.
    return {} if policy.is_public else load_memory(memory_file)


# We initialize the search UI state to match the radio button's default value
DEFAULT_SEARCH_LABEL = "Elite Hybrid Search (RRF + Reranker)"


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


# ── Core chat function ──────────────────────────────────────────────────

def make_chat_fn(gemini_client, memory, memory_file,
                 s3_client, collection, bucket_name, policy=None, private_sessions=None):
    """
    Returns the Gradio chat handler.

    Gradio 6.x uses the 'messages' format:
      {"role": "user" | "assistant", "content": "..."}
    (The old [[user, bot], ...] tuple format causes 'data incompatible with keys'.)
    """
    policy = policy or settings.ACCESS_POLICY
    memory = {} if policy.is_public else memory

    def chat(user_message: str, history: list, conversation_history: list,
             search_type: str, use_decomposition: bool, use_guardrail: bool,
             owner_id=None, request=None):
        request_memory = memory
        current_owner = None
        snapshot = None
        if policy.is_public and private_sessions is not None:
            try:
                snapshot = private_sessions.load_for_request(request)
            except SupabaseError as error:
                # Expired/unverifiable identity cannot reuse private context.
                return [], [], "", str(error), None
            current_owner = snapshot.user_id if snapshot else None
            request_memory = snapshot.memory if snapshot else {}
        if owner_id != current_owner:
            # Switching/forgetting identities must clear prior private answers,
            # not merely change the next prompt's persistent-memory field.
            history, conversation_history = [], []
        memory_status = (format_private_memory_status(request_memory)
                         if snapshot else runtime_memory_status(request_memory, policy))
        if not user_message.strip():
            return history, conversation_history, "", memory_status, current_owner

        # 1. Determine Search Type
        use_hybrid = "Hybrid" in search_type
        use_reranker = "Reranker" in search_type

        # 2. Build the tool dynamically based on UI selection
        search_fn, search_tool = build_search_tool(
            collection=collection,
            gemini_client=gemini_client,
            s3_client=s3_client,
            bucket=bucket_name,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            policy=policy,
        )
        tool_map = {"search_knowledge_base": search_fn}

        # 3. Create generation config with the specific tool and memory
        from agent import build_agent_config
        generation_config = build_agent_config(search_tool, request_memory)

        # 4. Use the resilient auto-routing from GeminiRouter
        # GeminiRouter handles the actual fallback logic
        model_id = "models/gemini-3.6-flash"

        # Append user message in Gradio 6.x messages format
        history = history + [{"role": "user", "content": user_message}]

        # Run the agent
        start_time = time.time()
        # Commit a turn only on success; a failed multi-tool cycle must not
        # leave half of another request's context in session state.
        working_history = deepcopy(conversation_history)
        try:
            raw_response = run_agent_turn(
                user_message=user_message,
                conversation_history=working_history,
                gemini_client=gemini_client,
                generation_config=generation_config,
                tool_map=tool_map,
                model=model_id,
                use_decomposition=use_decomposition,
                use_guardrail=use_guardrail,
            )
            elapsed_time = time.time() - start_time
        except Exception as e:
            if is_rate_limit_error(e):
                error_msg = (
                    "⚠️ **Daily limit reached.**\n\n"
                    "The free-tier quota is used up for today.\n"
                    "👉 **Please try again tomorrow.**"
                )
            elif is_overload_error(e):
                error_msg = (
                    "⚠️ **The model is temporarily overloaded** (high demand).\n\n"
                    "This is usually resolved in a few minutes.\n"
                    "👉 **Wait a moment and retry.**"
                )
            else:
                error_msg = ("❌ **The request could not be completed. Please retry.**"
                             if policy.is_public else f"❌ **Unexpected error:** {e}")

            history = history + [{"role": "assistant", "content": error_msg}]
            return (history, conversation_history,
                    "<div style='text-align:center; color:#888; padding:20px;'>Error occurred.</div>",
                    memory_status, current_owner)

        conversation_history = working_history

        # Extract visual grounding image URLs from the response text
        image_urls = extract_image_urls(raw_response)

        # Build HTML for images
        html_content = ""
        for url in image_urls:
            html_content += f'<div style="margin-bottom:15px;"><img src="{url}" style="width:100%; height:auto; border-radius:8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/></div>'
        if not html_content:
            html_content = "<div style='text-align:center; color:#888; padding:20px;'>No visual grounding for this response.</div>"

        # Clean raw URLs from the visible response text
        display_response = clean_response(raw_response)

        # Append processing time
        display_response += f"\n\n*(Processed in {elapsed_time:.2f}s)*"

        # Append assistant message in Gradio 6.x messages format
        history = history + \
            [{"role": "assistant", "content": display_response}]

        return history, conversation_history, html_content, memory_status, current_owner

    return chat


def make_save_fn(gemini_client, memory, memory_file, policy=None, private_sessions=None,
                 update_panel=False):
    """Save memory and optionally return the confirmed memory-panel update.

    Standalone callers retain the status-only contract. The UI receives both
    outputs, without an extra Auth/read round trip or optimistic panel content.
    """
    policy = policy or settings.ACCESS_POLICY

    def result(status, panel=None):
        if update_panel:
            return status, gr.skip() if panel is None else panel
        return status

    def save(conversation_history, owner_id=None, scope_state=None, request: gr.Request = None):
        generation = scope_state.get("generation") if scope_state is not None else None
        if policy.is_public and private_sessions is not None:
            try:
                snapshot = private_sessions.load_for_request(request, required=True)
                if owner_id != snapshot.user_id:
                    return result("Your session changed. Start a fresh chat before saving memory.")
                if not conversation_history:
                    return result("⚠️ No conversation to save yet.")
                updated = update_memory_from_conversation(
                    deepcopy(snapshot.memory), conversation_history, gemini_client,
                    raise_on_error=True,
                )
                if scope_state is not None and scope_state.get("generation") != generation:
                    return result(gr.skip())
                saved = private_sessions.backend.save_memory(snapshot, updated)
                if scope_state is not None and scope_state.get("generation") != generation:
                    return result(gr.skip())
                panel = format_private_memory_status(saved.memory) if update_panel else None
                return result("✅ Private demo memory saved for this temporary session.", panel)
            except SupabaseError as error:
                return result(str(error))
            except Exception:
                return result("Private memory could not be extracted or saved. Please retry.")
        policy.require_local_mutation()
        if not conversation_history:
            return result("⚠️ No conversation to save yet.")
        updated = update_memory_from_conversation(
            memory, conversation_history, gemini_client)
        memory.update(updated)
        save_memory(memory, memory_file)
        return result("✅ Memory saved!", format_memory_status(memory) if update_panel else None)
    return save


# ── Gradio UI ───────────────────────────────────────────────────────────

def build_ui(gemini_client, memory, memory_file,
             s3_client, collection, bucket_name, policy=None, private_sessions=None):
    policy = policy or settings.ACCESS_POLICY
    memory = {} if policy.is_public else memory
    chat_fn = make_chat_fn(
        gemini_client,
        memory,
        memory_file,
        s3_client,
        collection,
        bucket_name,
        policy=policy, private_sessions=private_sessions)
    save_fn = make_save_fn(gemini_client, memory, memory_file, policy=policy,
                           private_sessions=private_sessions, update_panel=True)
    upload_fn = make_upload_fn(s3_client, bucket_name, collection, policy=policy)

    with gr.Blocks(title="Document RAG Agent") as demo:

        # ── Session state ──────────────────────────────────────────────────
        conv_state = gr.State([])
        owner_state = gr.State(None)
        # This mutable server-side state is unique per Gradio browser session.
        # Lifecycle callbacks invalidate pending outputs without a global cache
        # or trusting client-provided ownership/generation identifiers.
        scope_state = gr.State({"generation": None})
        # tracks selected retrieval engine
        search_state = gr.State(DEFAULT_SEARCH_LABEL)
        decomp_state = gr.State(False)
        guardrail_state = gr.State(False)

        # ── Header ────────────────────────────────────────────────────────
        with gr.Row(elem_id="header"):
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    # Document RAG Agent
                    *Powered by Gemini · ChromaDB · LandingAI ADE*
                    """
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    ("""
                    **Public Showcase:**
                    Search the explicitly approved public documents.
                    Each browser session has its own chat context.
                    No account is needed to try the public documents.
                    """ if policy.is_public else """
                    **Quick Start:**
                    1. **Query**: Ask a question below to search.
                    2. **Upload**: Add new documents in Tab 2.
                    3. **Architecture**: Explore the system in Tab 3.
                    """)
                )

        if private_sessions is not None:
            with gr.Accordion("Optional private demo session — no sign-up", open=False):
                gr.Markdown(
                    "Public chat works without an account. Start a temporary private "
                    "session to try isolated memory, protected by a verification challenge. "
                    "This session expires within one hour and is not recoverable on "
                    "another device. Use the direct app URL, not an embedded iframe.")
                start_session_btn = gr.Button("Start private demo session")
                forget_session_btn = gr.Button("Forget private memory and end session")
                end_session_btn = gr.Button("Return to public chat (without deleting stored memory)")
                session_status = gr.Markdown("Public guest mode")

        # ── Main layout ────────────────────────────────────────────────
        with gr.Tabs():
            # ── TAB 1: Chat ────────────────────────────────────────────────
            with gr.Tab("Chat"):
                with gr.Row():
                    # ── LEFT SIDEBAR (Settings & Use Cases) ─────────────────────
                    with gr.Column(scale=1, elem_classes="sidebar"):
                        gr.Markdown("### Advanced Settings")
                        with gr.Column(elem_classes="settings-col"):
                            search_type_toggle = gr.Radio(
                                choices=[
                                    "Standard Vector Search",
                                    "Agentic Hybrid Search (RRF)",
                                    "Elite Hybrid Search (RRF + Reranker)"
                                ],
                                value="Elite Hybrid Search (RRF + Reranker)",
                                label="Retrieval Engine",
                                interactive=True,
                            )
                            decomp_toggle = gr.Checkbox(
                                label="Enable Query Decomposition",
                                value=False,
                            )
                            guardrail_toggle = gr.Checkbox(
                                label="Enable Live Guardrail",
                                value=False,
                            )
                            
                        gr.Markdown("### Highlighted Use Cases")
                        with gr.Column(elem_classes="use-case-card"):
                            gr.Markdown("**Query Decomposition**\n*How to test:* Enable **Query Decomposition** above. Watch the agent split the question.")
                            btn_case1 = gr.Button("Try: What value was used for label smoothing...", size="sm", variant="secondary")
                        with gr.Column(elem_classes="use-case-card"):
                            gr.Markdown("**Mathematical Grounding**\n*How to test:* Ensure **Elite Hybrid Search** is selected. It will extract math formulas visually.")
                            btn_case2 = gr.Button("Try: What are the dimension values for $d_k$ and $d_v$?", size="sm", variant="secondary")
                        with gr.Column(elem_classes="use-case-card"):
                            gr.Markdown("**Live Guardrail**\n*How to test:* Enable **Live Guardrail** above. It will refuse this out-of-domain question safely.")
                            btn_case3 = gr.Button("Try: What is the capital of France?", size="sm", variant="secondary")
        
                        with gr.Accordion("Agent Memory", open=False):
                            memory_display = gr.Markdown(runtime_memory_status(memory, policy))
        
                    # ── RIGHT CONTENT (Chat area) ────────────────────────────────
                    with gr.Column(scale=3):
                        with gr.Row():
                            with gr.Column(scale=3, elem_classes="chat-col"):
                                chatbot = gr.Chatbot(
                                    label="Conversation",
                                    height=520,
                                    render_markdown=True,
                                    avatar_images=(
                                        None,
                                        "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg"),
                                )
        
                                with gr.Row():
                                    user_input = gr.Textbox(
                                        placeholder="Ask about your documents... (e.g. 'What is the function of the transformer encoder?')",
                                        show_label=False,
                                        lines=2,
                                        scale=5,
                                    )
                                    send_btn = gr.Button("Send →", variant="primary", scale=1, elem_id="send-btn")
        
                                with gr.Row():
                                    clear_btn = gr.Button(" Clear Chat", size="sm")
                                    save_btn = gr.Button(" Save Memory", size="sm", variant="secondary",
                                                         visible=not policy.is_public or private_sessions is not None)
                                    save_status = gr.Textbox(show_label=False, interactive=False, placeholder="", scale=2, lines=1)
        
                            with gr.Column(scale=2, elem_classes="image-col"):
                                gr.Markdown("### Visual Grounding\n*Highlighted PDF regions from last answer*")
                                image_gallery = gr.HTML(
                                    value="<div style='text-align:center; color:#888; padding:20px;'>Ask a question to see source documents here.</div>",
                                    elem_id="visual-grounding-html"
                                )

            # ── TAB 2: Manage Knowledge Base ─────────────────────────────────
            with gr.Tab("📄 Manage Knowledge Base", visible=not policy.is_public):
                gr.Markdown(
                    "### Upload New Documents\nUpload PDFs to automatically chunk, embed, and index them into ChromaDB.")
                with gr.Row():
                    with gr.Column(scale=2):
                        upload_files = gr.File(
                            label="Upload PDFs", file_count="multiple", file_types=[".pdf"], visible=not policy.is_public)
                        upload_btn = gr.Button(
                             "Upload & Index Documents", variant="primary", visible=not policy.is_public)
                    with gr.Column(scale=3):
                        upload_status = gr.Textbox(
                            label="Status", lines=15, interactive=False)

                upload_btn.click(
                    fn=upload_fn,
                    inputs=[upload_files],
                    outputs=[upload_status],
                )

            # ── TAB 3: Architecture & Evaluation ─────────────────────────────
            with gr.Tab("🏗️ Architecture & Evaluation"):
                gr.Markdown("## System Architecture")
                gr.Markdown(
                    "This system implements an advanced, production-grade Retrieval-Augmented Generation (RAG) architecture.\n\n"
                    "- **Agent Orchestration**: Built using **LangGraph** as a cyclic state machine. Handles routing, tools, and fallback loops.\n"
                    "- **Retrieval Engine**: A 3-stage pipeline combining semantic vector search (ChromaDB), keyword search (BM25), and cross-encoder reranking (ms-marco-MiniLM).\n"
                    "- **Visual Grounding**: Document ingestion uses **LandingAI ADE** via AWS Lambda to extract layout-aware bounding boxes and render visual citations instantly.\n"
                    "- **Self-Correction & Guardrails**: Features a dynamic Query Optimizer and a Live Faithfulness Guardrail to intercept hallucinations before they reach the user.\n"
                )

                try:
                    with open("EVALUATION_REPORT.md", "r") as f:
                        eval_content = f.read()
                    gr.Markdown(f"## Evaluation Metrics\n\n{eval_content}")
                except Exception:
                    gr.Markdown("Evaluation metrics not found.")

        # ── Event wiring ───────────────────────────────────────────────────

        def set_case_1():
            return "What specific value was used for label smoothing during training, and what were its effects on the model's metrics?", True, False, True, False
            
        def set_case_2():
            return "What are the specific dimension values used for $d_k$ and $d_v$ in each of the parallel attention layers?", False, False, False, False

        def set_case_3():
            return "What is the capital of France?", False, True, False, True

        def submit(message, history, conv_history,
                   search_type, use_decomp, use_guardrail, owner_id, scope,
                   request: gr.Request):
            generation = scope["generation"]
            result = chat_fn(message, history, conv_history,
                             search_type, use_decomp, use_guardrail,
                             owner_id=owner_id, request=request)
            if scope["generation"] != generation:
                # A cleared/ended/switched session must not be repopulated by
                # an old LLM response arriving after the lifecycle callback.
                return tuple(gr.skip() for _ in result)
            return result

        if private_sessions is not None:
            def refresh_private_session(history, conv_history, previous_owner, scope, request: gr.Request):
                scope["generation"] = generation = uuid4().hex
                try:
                    snapshot = private_sessions.load_for_request(request)
                    status = "Private demo session active" if snapshot else "Public guest mode"
                    memory_status = format_private_memory_status(snapshot.memory) if snapshot else runtime_memory_status({}, policy)
                    owner = snapshot.user_id if snapshot else None
                    # Preserve public chat when the visitor explicitly opts in,
                    # and preserve an unchanged private identity. Never carry
                    # one private identity's history into another identity.
                    keep_chat = bool(snapshot) and previous_owner in (None, owner)
                except SupabaseError as error:
                    status, memory_status, owner = str(error), runtime_memory_status({}, policy), None
                    keep_chat = False
                if scope["generation"] != generation:
                    return tuple(gr.skip() for _ in range(6))
                return (status, memory_status, history if keep_chat else [],
                        conv_history if keep_chat else [], "", owner)

            session_outputs = [session_status, memory_display, chatbot, conv_state,
                               image_gallery, owner_state]
            session_inputs = [chatbot, conv_state, owner_state, scope_state]
            demo.load(fn=refresh_private_session, inputs=session_inputs, outputs=session_outputs)
            start_session_btn.click(fn=None, js=private_sessions.start_js()).success(
                fn=refresh_private_session, inputs=session_inputs, outputs=session_outputs)
            forget_session_btn.click(fn=None, js=private_sessions.forget_js()).success(
                fn=refresh_private_session, inputs=session_inputs, outputs=session_outputs)
            end_session_btn.click(fn=None, js=private_sessions.end_js()).success(
                fn=refresh_private_session, inputs=session_inputs, outputs=session_outputs)

        # ── Dummy GPU Function to satisfy ZeroGPU startup checks ──────────
        @gpu_decorator
        def dummy_gpu_fn():
            return None

        demo.load(fn=dummy_gpu_fn, inputs=None, outputs=None)

        # Wire Case 1
        btn_case1.click(
            fn=set_case_1,
            outputs=[user_input, decomp_toggle, guardrail_toggle, decomp_state, guardrail_state],
        ).then(
            fn=submit,
            inputs=[user_input, chatbot, conv_state, search_state, decomp_state, guardrail_state, owner_state, scope_state],
            outputs=[chatbot, conv_state, image_gallery, memory_display, owner_state],
        ).then(
            fn=lambda: gr.update(value=""),
            outputs=user_input,
        )

        # Wire Case 2
        btn_case2.click(
            fn=set_case_2,
            outputs=[user_input, decomp_toggle, guardrail_toggle, decomp_state, guardrail_state],
        ).then(
            fn=submit,
            inputs=[user_input, chatbot, conv_state, search_state, decomp_state, guardrail_state, owner_state, scope_state],
            outputs=[chatbot, conv_state, image_gallery, memory_display, owner_state],
        ).then(
            fn=lambda: gr.update(value=""),
            outputs=user_input,
        )

        # Wire Case 3
        btn_case3.click(
            fn=set_case_3,
            outputs=[user_input, decomp_toggle, guardrail_toggle, decomp_state, guardrail_state],
        ).then(
            fn=submit,
            inputs=[user_input, chatbot, conv_state, search_state, decomp_state, guardrail_state, owner_state, scope_state],
            outputs=[chatbot, conv_state, image_gallery, memory_display, owner_state],
        ).then(
            fn=lambda: gr.update(value=""),
            outputs=user_input,
        )

        # Sync toggle → search state
        search_type_toggle.change(
            fn=lambda label: label,
            inputs=search_type_toggle,
            outputs=search_state,
        )

        # Sync decomp toggle -> decomp state
        decomp_toggle.change(
            fn=lambda val: val,
            inputs=decomp_toggle,
            outputs=decomp_state,
        )

        # Sync guardrail toggle -> guardrail state
        guardrail_toggle.change(
            fn=lambda val: val,
            inputs=guardrail_toggle,
            outputs=guardrail_state,
        )

        # Send on button click
        send_btn.click(
            fn=submit,
            inputs=[
                user_input,
                chatbot,
                conv_state,
                search_state,
                decomp_state,
                guardrail_state, owner_state, scope_state],
            outputs=[chatbot, conv_state, image_gallery, memory_display, owner_state],
        ).then(
            fn=lambda: gr.update(value=""),
            outputs=user_input,
        )

        # Send on Enter key
        user_input.submit(
            fn=submit,
            inputs=[
                user_input,
                chatbot,
                conv_state,
                search_state,
                decomp_state,
                guardrail_state, owner_state, scope_state],
            outputs=[chatbot, conv_state, image_gallery, memory_display, owner_state],
        ).then(
            fn=lambda: gr.update(value=""),
            outputs=user_input,
        )

        # Clear chat (keeps memory, resets conversation)
        def clear_chat(owner_id, scope, request: gr.Request):
            scope["generation"] = generation = uuid4().hex
            result = chat_fn("", [], [], DEFAULT_SEARCH_LABEL, False, False,
                             owner_id=owner_id, request=request)
            return result if scope["generation"] == generation else tuple(gr.skip() for _ in result)

        clear_btn.click(
            fn=clear_chat,
            inputs=[owner_state, scope_state],
            outputs=[chatbot, conv_state, image_gallery, memory_display, owner_state],
        )

        # Save memory button
        save_btn.click(
            fn=save_fn,
            inputs=[conv_state, owner_state, scope_state],
            outputs=[save_status, memory_display],
        )

    return demo


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gradio UI for Document RAG Agent")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Local port (default: 7860)")
    parser.add_argument(
        "--collection",
        default="document_chunks",
        help="ChromaDB collection")
    parser.add_argument(
        "--chroma-path",
        default="./chroma_db",
        help="ChromaDB folder")
    parser.add_argument(
        "--memory-file",
        default="memory.json",
        help="Memory JSON file")
    parser.add_argument(
        "--model",
        default="models/gemini-3.6-flash",
        help="Gemini model")
    args = parser.parse_args()
    policy = settings.ACCESS_POLICY
    policy.validate_launch(share=args.share, hosted=bool(os.environ.get("SPACE_ID")))
    private_sessions = None
    if getattr(settings, "SUPABASE_ENABLED", False):
        private_sessions = PrivateSessions(
            SupabaseBackend(settings.SUPABASE_URL, settings.SUPABASE_PUBLISHABLE_KEY),
            settings.SESSION_COOKIE_KEY, settings.APP_PUBLIC_URL, settings.TURNSTILE_SITE_KEY,
        )

    logger.info("\n Starting Document RAG Agent UI...")
    logger.info("─" * 40)

    gemini_client = create_gemini_router(settings.GEMINI_API_KEYS)
    s3_client = create_s3_client() if settings.S3_BUCKET_NAME else None
    collection = load_chroma_collection(args.collection, args.chroma_path)
    memory = load_runtime_memory(args.memory_file, policy)

    logger.info("─" * 40)
    logger.info(f" All systems ready — launching Gradio on port {args.port}")
    if args.share:
        logger.info(" Share link will be printed below (valid 72 hours)")

    demo = build_ui(
        gemini_client=gemini_client,
        memory=memory,
        memory_file=args.memory_file,
        s3_client=s3_client,
        collection=collection,
        bucket_name=settings.S3_BUCKET_NAME,
        policy=policy,
        private_sessions=private_sessions,
    )
    demo.queue(default_concurrency_limit=2)
    demo.launch(
        server_name="0.0.0.0" if policy.is_public else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        show_error=not policy.is_public,
        strict_cors=True,
        app_kwargs={"routes": private_sessions.routes()} if private_sessions else None,
        # The framework upload route runs before our ingestion handler. Deny
        # file bodies there as well instead of merely hiding a File component.
        max_file_size=0 if policy.is_public else "20mb",
        blocked_paths=[str(Path(path).resolve()) for path in (
            args.memory_file, args.chroma_path, ".env", "documents",
            "document_chunks", "input", "output",
        )] if policy.is_public else None,
        ssr_mode=False,
        theme=gr.themes.Monochrome(primary_hue="slate", neutral_hue="slate"),
        css="""
            #header { text-align: center; padding: 20px 0; border-bottom: 1px solid #eaeaea; margin-bottom: 20px; }
            #header h1 { font-size: 2.2em; margin-bottom: 4px; font-weight: 700; letter-spacing: -0.5px; }
            #header p  { color: #64748b; margin: 0; font-size: 1.05em; }
            .image-col { border-left: 1px solid #f1f5f9; padding-left: 20px; }
            #send-btn  { min-width: 80px; font-weight: bold; }
            .settings-row { background-color: #f8fafc; padding: 10px 15px; border-radius: 6px; border: 1px solid #e2e8f0; margin-top: 5px; }
            .use-case-row { margin-bottom: 20px; }
            .use-case-card { background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); }
            .use-case-card p { margin-bottom: 10px; font-size: 0.9em; color: #475569; }
            .sidebar { background-color: #f8fafc; padding: 20px; border-right: 1px solid #e2e8f0; }
        """,
    )


if __name__ == "__main__":
    main()

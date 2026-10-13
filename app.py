"""
app.py — Gradio Document Research Demo
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
from html import escape
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
    r"(?:https://[^\s\]\[\"'>()]+\.amazonaws\.com/[^\s\]\[\"'>()]+\.png(?:\?[^\s\]\[\"'>()]+)?"
    r"|/gradio_api/file=[^\s\]\[\"'>()]+/citations/[a-f0-9]{64}\.png)"
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
    label = "📎 *(open ‘View source pages’ to inspect the evidence)*"

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

# Shared by the real launch and browser verification. Keep styling scoped to the
# research workspace and respect Gradio's light/dark theme variables.
DEMO_CSS = """
    .gradio-container { max-width: 1180px !important; margin: auto; }
    #header { padding: 8px 0; margin-bottom: 8px; border-bottom: 1px solid var(--border-color-primary); }
    #header p { margin: 8px 0; }
    #header h1 { font-size: clamp(1.65rem, 4vw, 2.2rem); line-height: 1.2; margin-bottom: 8px; }
    #corpus-status { padding: 8px 0; }
    #starter-tasks { gap: 12px; }
    .use-case-card { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px; }
    #research-workspace { gap: 20px; align-items: flex-start; }
    #conversation-col, #evidence-col { min-width: 0 !important; }
    #evidence-col { padding: 12px; border: 1px solid var(--border-color-primary); border-radius: 12px; }
    #send-btn { align-self: flex-end; font-weight: 600; min-height: 44px; }
    #research-question textarea { font-size: 1rem; }
    #research-question textarea::placeholder { color: var(--body-text-color); opacity: 0.75; }
    #report-cards { gap: 16px; }
    .report-card { padding: 16px; border: 1px solid var(--border-color-primary); border-radius: 12px; }
    .report-card h3 { font-size: 1.1rem; line-height: 1.4; }
    .report-card p { line-height: 1.6; }
    .report-card a { color: var(--link-text-color); text-decoration: underline; display: inline-block; padding: 10px 0; }
    #visual-grounding-html img { max-width: 100%; height: auto; }
    #research-chat { overflow-wrap: anywhere; }
    button:focus-visible, a:focus-visible, textarea:focus-visible, input:focus-visible {
        outline: 3px solid #2563eb !important; outline-offset: 3px;
    }
    @media (max-width: 760px) {
        #research-workspace, #question-composer, #starter-tasks, #report-cards { flex-direction: column; }
        #conversation-col, #evidence-col, #starter-tasks > div, #report-cards > div { width: 100%; flex-basis: auto !important; }
        #question-composer > div { width: 100%; min-width: 0 !important; }
        #send-btn { width: 100%; align-self: stretch; }
        #research-chat { height: 300px !important; }
        #header { gap: 4px; }
    }
"""


def corpus_status(collection, policy):
    """Inspect only authorized IDs; availability must not reveal private records."""
    try:
        count = policy.scope_collection(collection).count()
        return ("ready", count) if count > 0 else ("empty", 0)
    except Exception:
        logger.warning("Authorized corpus availability check failed")
        return "unavailable", 0


def corpus_status_message(status, policy):
    state, count = status
    if state == "ready":
        return f"**Reports ready** · {count:,} searchable passages. Ask a question and inspect its sources."
    if state == "unavailable":
        return "**Reports temporarily unavailable.** Recheck availability shortly. Answer generation is paused until reports are ready."
    return ("**No public reports are ready.** The administrator needs to provision the approved corpus. Public uploads are unavailable."
            if policy.is_public else "**No documents are ready.** Add a PDF in Manage Knowledge Base to begin.")


def report_cards(collection, policy, showcase):
    """Show only curated, currently authorized sources; never read private text."""
    if showcase is None:
        return []
    scoped = policy.scope_collection(collection)
    cards = []
    for source_key, (_, source) in showcase.documents.items():
        try:
            available = bool(scoped.get(where={"source_pdf_key": source_key}, include=[], limit=1)["ids"])
        except Exception:
            available = False
        # Source descriptors come from the server's curated manifest, not UI or
        # LLM data. Missing sources remain visible but explicitly unavailable.
        title = escape(source["title"])
        url = source["url"]
        if not url.startswith("https://www.ipcc.ch/"):
            continue
        cards.append(
            f"<article class='report-card'><h3>{title}</h3>"
            f"<p>{escape(source.get('publisher', 'IPCC'))} · {source.get('year', 'Assessment')}</p>"
            f"<p>{escape(source.get('topics', 'Climate science assessment.'))}</p>"
            f"<p><strong>{'Available for research' if available else 'Not currently searchable'}</strong></p>"
            f"<a href='{escape(url, quote=True)}' aria-label='Open original PDF: {title}' target='_blank' rel='noopener noreferrer'>Open original PDF</a></article>"
        )
    return cards


def demo_presets(showcase):
    """Curated tasks: no onboarding LLM call, with explicit execution settings."""
    if showcase:
        return [
            ("fact", "Find a fact", "Find energy-cost reductions and inspect the cited page.",
             "According to the mitigation report, how much did solar energy, wind energy, and lithium-ion battery unit costs fall from 2010 to 2019? Cite the exact PDF page and visual evidence.", False, False),
            ("compare", "Compare evidence", "Compare warming risks and emissions pathways across reports.",
             "Compare the physical-science and mitigation reports: how do risks differ between 1.5°C and 2°C warming, and what emissions pathways limit warming to 1.5°C? Cite each report and retain its uncertainty qualifiers.", True, False),
            ("pathway", "Explore a pathway", "Understand emissions reductions associated with a warming limit.",
             "What global greenhouse gas emissions reductions by 2030 are associated with pathways limiting warming to 1.5°C? Explain the baseline, scenario assumptions, and uncertainty using the IPCC reports, with citations.", False, False),
            ("limits", "Test an evidence gap", "These assessments cannot establish an exact city/date forecast.",
             "Using only these IPCC reports, tell me the exact temperature in Hyderabad on 1 January 2035. If this cannot be established from the reports, explain why without inventing a prediction.", False, True),
        ]
    return [
        ("fact", "Find a fact", "Find a key result in the available documents.",
         "What is a key finding in the available documents? Cite its source and page.", False, False),
        ("compare", "Compare evidence", "Compare findings and identify evidence gaps.",
         "Compare the key findings across the available documents. Cite each source and say where evidence is missing.", True, False),
        ("pathway", "Explore a recommendation", "Explain a recommendation and its supporting evidence.",
         "What action do the available documents recommend, and what evidence supports it? Cite the relevant passages.", False, False),
        ("limits", "Test an evidence gap", "Ask for a claim that the documents may not establish.",
         "Using only these documents, establish the exact temperature in Hyderabad on 1 January 2035. If the evidence is missing, explain the gap without inventing a prediction.", False, True),
    ]


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
                 s3_client, collection, bucket_name, policy=None, private_sessions=None,
                 showcase=None):
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

        status = corpus_status(collection, policy)
        if status[0] != "ready":
            return (history + [{"role": "user", "content": user_message},
                               {"role": "assistant", "content": corpus_status_message(status, policy)}],
                    conversation_history, "", memory_status, current_owner)

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
            citation_renderer=showcase.render_citation if showcase else None,
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
            html_content += f'<div style="margin-bottom:15px;"><img src="{url}" alt="Highlighted source PDF page" style="width:100%; height:auto; border-radius:8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/></div>'
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
             s3_client, collection, bucket_name, policy=None, private_sessions=None, showcase=None):
    policy = policy or settings.ACCESS_POLICY
    memory = {} if policy.is_public else memory
    chat_fn = make_chat_fn(
        gemini_client,
        memory,
        memory_file,
        s3_client,
        collection,
        bucket_name,
        policy=policy, private_sessions=private_sessions, showcase=showcase)
    save_fn = make_save_fn(gemini_client, memory, memory_file, policy=policy,
                           private_sessions=private_sessions, update_panel=True)
    upload_fn = make_upload_fn(s3_client, bucket_name, collection, policy=policy)

    initial_status = corpus_status(collection, policy)
    with gr.Blocks(title="Climate Research Demo" if showcase else "Document Research Demo") as demo:

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
                    "# Explore climate evidence\nAsk questions across IPCC reports and check the original sources. **No sign-up required.**"
                    if showcase else """
                    # Explore your document evidence
                    Ask questions and check the original sources.
                    """ + ("No sign-up is needed for public research." if policy.is_public
                             else "Add PDFs in Manage Knowledge Base, then ask a research question.")
                )
        readiness_display = gr.Markdown(corpus_status_message(initial_status, policy), elem_id="corpus-status")

        if showcase is not None:
            gr.Markdown(
                "**Collection:** three IPCC AR6 reports (2021–2023). "
                "Answers use dated assessments, not live forecasts. "
                "Open **Explore reports** for coverage and original PDFs.")

        # ── Main layout ────────────────────────────────────────────────
        with gr.Tabs():
            # ── TAB 1: Chat ────────────────────────────────────────────────
            with gr.Tab("Research"):
                preset_buttons = []
                with gr.Column(elem_id="research-welcome") as welcome:
                    gr.Markdown("### Start with a research question\nChoose a task below, or ask your own question. Then open the source pages to check the evidence.")
                    with gr.Row(elem_id="starter-tasks"):
                        for preset in demo_presets(showcase)[:3]:
                            key, label, description, _, _, _ = preset
                            with gr.Column(min_width=220, elem_classes="use-case-card"):
                                gr.Markdown(description)
                                preset_buttons.append(gr.Button(label, elem_id=f"preset-{key}",
                                                                interactive=initial_status[0] == "ready"))

                with gr.Row(elem_id="research-workspace"):
                    with gr.Column(scale=3, min_width=300, elem_id="conversation-col"):
                        chatbot = gr.Chatbot(label="Research conversation", height=380, visible=False,
                                             render_markdown=True, elem_id="research-chat")
                        with gr.Row(elem_id="question-composer"):
                            user_input = gr.Textbox(
                                label="Your research question", show_label=True,
                                placeholder=("Ask about climate risks, emissions pathways, or energy transition..." if showcase else "Ask about the available documents..."),
                                lines=2, scale=5, elem_id="research-question",
                                interactive=initial_status[0] == "ready")
                            send_btn = gr.Button("Ask question", variant="primary", scale=1,
                                                 elem_id="send-btn", interactive=initial_status[0] == "ready")
                        clear_btn = gr.Button("Start a new conversation", size="sm", elem_id="clear-chat")

                    with gr.Column(scale=1, min_width=260, elem_id="evidence-col"):
                        gr.Markdown("### Inspect the evidence\nAfter an answer, open its source pages and compare the highlighted passages with the response.")
                        with gr.Accordion("View source pages", open=False, elem_id="source-pages"):
                            image_gallery = gr.HTML(
                                value="<p>Source pages appear here after a cited answer.</p>",
                                elem_id="visual-grounding-html")

                with gr.Accordion("Test the system's limits", open=False, elem_id="evidence-limits"):
                    gr.Markdown("Missing evidence is a useful result. Try a question these reports cannot establish; the assistant should explain the gap. This example requests an automated grounding check, which can be unavailable.")
                    btn_limits = gr.Button("Test an evidence gap", elem_id="preset-limits",
                                           interactive=initial_status[0] == "ready")

                with gr.Accordion("Advanced research options", open=False, elem_id="advanced-options"):
                    gr.Markdown("The default combines meaning-based search, keyword matching, and reranking. You can ask questions without changing these options.")
                    search_type_toggle = gr.Radio(
                        choices=["Standard Vector Search", "Agentic Hybrid Search (RRF)", DEFAULT_SEARCH_LABEL],
                        value=DEFAULT_SEARCH_LABEL, label="Retrieval method",
                        info="Vector: meaning-based matching. Hybrid: adds keywords. Reranker: reorders candidate passages; its first use can load a model.")
                    decomp_toggle = gr.Checkbox(label="Split complex questions into focused searches", value=False,
                                                info="Query decomposition adds an LLM call and can lead to more searches. Useful for comparisons.")
                    guardrail_toggle = gr.Checkbox(label="Request an automated grounding check", value=False,
                                                   info="Adds an LLM call to compare the answer with retrieved text. It is not a guarantee of correctness.")

                with gr.Accordion("Optional research memory", open=False, elem_id="research-memory"):
                    if private_sessions is not None:
                        gr.Markdown(
                            "**Want to remember your research interests?** Public research needs no account. "
                            "To save memory: start a private demo session, complete a conversation, then save. "
                            "A verification challenge protects session creation. This temporary session "
                            "expires within one hour and cannot be recovered on another device. "
                            "Use the direct app URL rather than an embedded iframe.")
                        start_session_btn = gr.Button("Remember my research interests", elem_id="start-private-session")
                        session_status = gr.Markdown("Public guest mode · start a private session before saving research memory.",
                                                     elem_id="private-session-status")
                        forget_session_btn = gr.Button("Forget saved memory and end session", elem_id="forget-private-session")
                        end_session_btn = gr.Button("Return to public research (keep saved memory)", elem_id="end-private-session")
                    elif policy.is_public:
                        gr.Markdown("Research memory is not configured on this deployment. You can still ask follow-up questions in this browser conversation.")
                    memory_display = gr.Markdown(runtime_memory_status(memory, policy))
                    save_btn = gr.Button("Save research memory", size="sm", variant="secondary",
                                         visible=not policy.is_public or private_sessions is not None,
                                         interactive=False, elem_id="save-research-memory")
                    save_status = gr.Textbox(label="Memory save status", interactive=False, lines=1)

            with gr.Tab("Explore reports"):
                gr.Markdown("### Your research collection\nThese are dated assessments, not live weather or current-policy updates.")
                refresh_reports_btn = gr.Button("Recheck report availability", size="sm", elem_id="refresh-reports")
                report_card_components = []
                with gr.Row(elem_id="report-cards"):
                    for card in report_cards(collection, policy, showcase):
                        with gr.Column(min_width=240):
                            report_card_components.append(gr.HTML(card))
                if showcase is None:
                    gr.Markdown("This deployment uses its administrator-approved document collection. The curated IPCC report browser is available when the science showcase is enabled.")

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
            with gr.Tab("How this works"):
                gr.Markdown("## System Architecture")
                gr.Markdown(
                    "This system implements an advanced, production-grade Retrieval-Augmented Generation (RAG) architecture.\n\n"
                    "- **Agent Orchestration**: Built using **LangGraph** as a cyclic state machine. Handles routing, tools, and fallback loops.\n"
                    "- **Retrieval Engine**: A 3-stage pipeline combining semantic vector search (ChromaDB), keyword search (BM25), and cross-encoder reranking (ms-marco-MiniLM).\n"
                    "- **Visual Grounding**: The public science corpus uses local PyMuPDF text blocks and page coordinates; the optional AWS/ADE pipeline supports layout-aware parsing. Local extraction is not OCR or structured table understanding.\n"
                    "- **Self-Correction & Guardrails**: Features a dynamic Query Optimizer and a Live Faithfulness Guardrail to intercept hallucinations before they reach the user.\n"
                )

                try:
                    if showcase:
                        gr.Markdown("**Historical benchmark:** The metrics below were measured on the earlier corpus, not these IPCC reports. Science-corpus retrieval smoke checks are documented in SHOWCASE.md; a full benchmark is pending.")
                    with open("EVALUATION_REPORT.md", "r") as f:
                        eval_content = f.read()
                    gr.Markdown(f"## Evaluation Metrics\n\n{eval_content}")
                except Exception:
                    gr.Markdown("Evaluation metrics not found.")

        # ── Event wiring ───────────────────────────────────────────────────

        def view_updates(result):
            return (gr.update(value=result[0], visible=bool(result[0])), *result[1:],
                    gr.update(visible=not bool(result[0])),
                    corpus_status_message(corpus_status(collection, policy), policy),
                    gr.update(interactive=bool(result[1]) and (not policy.is_public or result[4] is not None)))

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
                return tuple(gr.skip() for _ in range(8))
            return view_updates(result)

        if private_sessions is not None:
            def refresh_private_session(history, conv_history, previous_owner, scope, request: gr.Request):
                scope["generation"] = generation = uuid4().hex
                try:
                    snapshot = private_sessions.load_for_request(request)
                    status = ("Private demo session active · complete a conversation, then save research memory."
                              if snapshot else "Public guest mode · start a private session before saving research memory.")
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
                    return tuple(gr.skip() for _ in range(9))
                return (status, memory_status,
                        gr.update(value=history if keep_chat else [], visible=bool(keep_chat and history)),
                        conv_history if keep_chat else [], "", owner,
                        gr.update(visible=not (keep_chat and history)),
                        corpus_status_message(corpus_status(collection, policy), policy),
                        gr.update(interactive=bool(keep_chat and conv_history and owner)))

            session_outputs = [session_status, memory_display, chatbot, conv_state,
                               image_gallery, owner_state, welcome, readiness_display, save_btn]
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

        demo.load(fn=dummy_gpu_fn, inputs=None, outputs=None, show_progress="hidden", queue=False)

        chat_inputs = [user_input, chatbot, conv_state, search_state, decomp_state,
                       guardrail_state, owner_state, scope_state]
        chat_outputs = [chatbot, conv_state, image_gallery, memory_display, owner_state,
                        welcome, readiness_display, save_btn]

        def refresh_availability():
            status = corpus_status(collection, policy)
            available = status[0] == "ready"
            return (corpus_status_message(status, policy),
                    *(gr.update(interactive=available) for _ in range(6)),
                    *report_cards(collection, policy, showcase))

        refresh_reports_btn.click(
            fn=refresh_availability,
            outputs=[readiness_display, user_input, send_btn, *preset_buttons, btn_limits,
                     *report_card_components],
        )

        def choose_preset(preset):
            def choose():
                _, _, _, prompt, decomp, check = preset
                return prompt, DEFAULT_SEARCH_LABEL, decomp, check, DEFAULT_SEARCH_LABEL, decomp, check
            return choose

        for button, preset in zip([*preset_buttons, btn_limits], demo_presets(showcase)):
            button.click(
                fn=choose_preset(preset), api_name="example_" + preset[0],
                outputs=[user_input, search_type_toggle, decomp_toggle, guardrail_toggle,
                         search_state, decomp_state, guardrail_state],
            ).then(fn=submit, inputs=chat_inputs, outputs=chat_outputs).then(
                fn=lambda: gr.update(value=""), outputs=user_input)

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
            outputs=chat_outputs,
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
            outputs=chat_outputs,
        ).then(
            fn=lambda: gr.update(value=""),
            outputs=user_input,
        )

        # Clear chat (keeps memory, resets conversation)
        def clear_chat(owner_id, scope, request: gr.Request):
            scope["generation"] = generation = uuid4().hex
            result = chat_fn("", [], [], DEFAULT_SEARCH_LABEL, False, False,
                             owner_id=owner_id, request=request)
            return view_updates(result) if scope["generation"] == generation else tuple(gr.skip() for _ in range(8))

        clear_btn.click(
            fn=clear_chat,
            inputs=[owner_state, scope_state],
            outputs=chat_outputs,
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
    parser.add_argument("--showcase", action="store_true",
                        default=os.environ.get("SHOWCASE_ENABLED", "false").lower() == "true",
                        help="Provision and use the isolated public IPCC science corpus")
    args = parser.parse_args()
    policy = settings.ACCESS_POLICY
    if args.showcase and not policy.is_public:
        parser.error("--showcase requires APP_MODE=public_demo")
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
    s3_client = create_s3_client() if settings.S3_BUCKET_NAME and not args.showcase else None
    showcase = None
    if args.showcase:
        from showcase_corpus import prepare_showcase
        collection, showcase = prepare_showcase()
    else:
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
        showcase=showcase,
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
        allowed_paths=[str(showcase.images)] if showcase else None,
        blocked_paths=[str(Path(path).resolve()) for path in (
            args.memory_file, args.chroma_path, ".env", "documents",
            "document_chunks", "input", "output",
        )] if policy.is_public else None,
        ssr_mode=False,
        theme=gr.themes.Monochrome(primary_hue="slate", neutral_hue="slate", font=["system-ui", "sans-serif"]),
        css=DEMO_CSS,
    )


if __name__ == "__main__":
    main()

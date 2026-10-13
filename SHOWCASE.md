# Science showcase: climate risks and energy transition

This public corpus demonstrates evidence-based research over **three official
IPCC AR6 Summary for Policymakers PDFs**:

| Assessment | Publication | Topics |
|---|---|---|
| [Physical Science Basis](https://www.ipcc.ch/report/ar6/wg1/) | 2021 | Human influence, warming levels, extremes, carbon budgets |
| [Mitigation of Climate Change](https://www.ipcc.ch/report/ar6/wg3/) | 2022 | Emissions pathways, renewables, costs, demand-side mitigation |
| [Synthesis Report](https://www.ipcc.ch/report/ar6/syr/) | 2023 | Observed changes, risks, adaptation and mitigation together |

These are dated scientific assessments, not a live climate/weather feed. Source
URLs and SHA-256 checksums are pinned in `showcase_sources.json`. Attribution is
retained through report titles, dates and original-PDF links. PDFs are downloaded
from IPCC on the target host; they are not bundled in Git, Docker, or HF uploads.

## Provision and launch

```bash
# Download, extract and index using local embeddings; no LLM/ADE/AWS call.
python showcase_corpus.py

# Check three evidence queries across vector, hybrid and reranked retrieval.
python showcase_corpus.py --check

# Launch using this isolated approved-public corpus and local visual citations.
python app.py --showcase --port 7861
```

The app command also provisions on first launch and verifies/reuses the corpus
on subsequent launches. `SHOWCASE_ENABLED=true` enables the same behavior on
Hugging Face or Docker. Configure it on the host after deploying the source.
Existing Gemini/Supabase configuration still applies to chat/private memory.
The standalone provisioning command needs no Gemini credentials.

`showcase_data/` contains downloaded PDFs, a separate Chroma database and generated
citations. Existing documents/indexes are not inspected, published or modified.
The index identity includes all PDF content hashes and the extraction version;
source changes fail checksum validation until explicitly reviewed and re-pinned.
Indexing uses bounded batches, verifies the complete expected ID set, then
publishes only those exact chunks before handing the collection to the UI.
Interrupted initial indexing can be resumed by rerunning the command.

Web citation images use opaque same-origin URLs that recheck current chunk/source
authorization on every fetch. Gradio file serving blocks the showcase-data
directory. Original PDFs remain linked at IPCC. Crops are full PDF pages with a red
box around the retrieved text block, preserving nearby chart/table context.
Page citations use **one-based PDF file-page positions**, which can differ from
the printed page labels.

## Interview/demo scenarios

The **Research** tab now starts with supported tasks and no sign-up requirement.
**Explore reports** shows the report dates, coverage and original PDFs.
Source pages, advanced settings, the evidence-gap test, and optional research
memory are expandable. If reports are unavailable, research controls are paused;
use **Explore reports → Recheck report availability** to retry without generating
an answer. The backend enforces the same authorized-corpus check for direct calls.

Answers now use clickable numbered references such as **[1]**. **Evidence by
answer** retains source excerpts, report titles/dates and PDF page positions,
including passages the model retrieved but did not cite. Select a reference or
**View source page** to render its page; no model-generated image URL is needed.
Citation existence is a provenance check, not a guarantee that a claim is true.

Use **Explain the evidence** or **Compare with another report** to continue with
the existing context and settings. Source details retain up to 12 answers and
120 passages; the corresponding provider context is trimmed when older details
are evicted. Source access changes can restart the research context. New answers
are withheld if retrieved sources lose access during the request. Source details are
session-local and are cleared on conversation/session changes.

1. **Cross-report synthesis — Query Decomposition**
   Compare climate risks at 1.5°C and 2°C with emissions pathways limiting warming
   to 1.5°C. Require separate citations and preserved uncertainty language.
2. **Quantitative evidence — Hybrid Search + Reranker + Visual Grounding**
   Find 2010–2019 solar, wind and lithium-ion battery unit-cost reductions. Compare
   the answer with the original page and figure. Expected values in WGIII: solar
   85%, wind 55%, batteries 85%; do not treat these as current prices.
3. **Evidence gap — Live Guardrail**
   Ask for the exact temperature in Hyderabad on 1 January 2035. The reports do
   not establish such a prediction; the assistant should explain the gap. This
   checks document boundaries as well as the optional faithfulness check.
4. **Research follow-up — Temporary Private Memory**
   Start a private demo session, discuss an interest in energy-sector mitigation,
   save memory, reload, and ask a follow-up. A separate browser identity should
   retain only its own memory.
5. **Retrieval comparison**
   Ask the same question using vector, hybrid RRF, and reranked search. Compare
   returned evidence and latency rather than assuming one always wins.

## Extraction limits

PyMuPDF extracts text blocks and their page coordinates locally. It does not
perform OCR, interpret chart values from pixels, or reconstruct complex tables.
Long blocks are split into 100-word passages sharing the original block bounds.
Figures can be inspected visually, but answers must be supported by retrieved
text. Report ranges, confidence levels, scenarios and dates must be preserved.
Full hosted user-upload/version/job support remains a separate roadmap milestone.

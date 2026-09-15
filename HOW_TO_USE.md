# Comprehensive User Guide: Context Engineering Research Assistant

> A practical, end-to-end handbook covering architecture concepts, quickstart setup, interactive usage, source inspection, and troubleshooting for researchers, analysts, and developers.

---

## Table of Contents

1. [Understanding Context Engineering](#1-understanding-context-engineering)
2. [How the 4-Stage Multi-Agent Architecture Works](#2-how-the-4-stage-multi-agent-architecture-works)
3. [Prerequisites & Quickstart Installation](#3-prerequisites--quickstart-installation)
4. [Configuring API Keys](#4-configuring-api-keys)
5. [Launching the Web Application](#5-launching-the-web-application)
6. [Interactive Walkthrough: Using the Assistant](#6-interactive-walkthrough-using-the-assistant)
   - [Method A: Instant 1-Click Sample Paper](#method-a-instant-1-click-sample-paper)
   - [Method B: Uploading Your Own PDFs](#method-b-uploading-your-own-pdfs)
7. [Querying Best Practices & Prompt Examples](#7-querying-best-practices--prompt-examples)
8. [Interpreting Answers, Citations & the Evaluator Audit](#8-interpreting-answers-citations--the-evaluator-audit)
   - [Confidence Score Breakdown](#confidence-score-breakdown)
   - [The 4 Source Inspectability Cards](#the-4-source-inspectability-cards)
   - [Anti-Hallucination & Refusal Behavior](#anti-hallucination--refusal-behavior)
9. [Running Automated Tests](#9-running-automated-tests)
10. [Troubleshooting & FAQs](#10-troubleshooting--faqs)

---

## 1. Understanding Context Engineering

Traditional conversational AI applications often suffer from two major failure modes:
1. **Hallucination**: When asked about specialized documents or recent events, models tend to invent plausible-sounding but false facts.
2. **Context Stuffing**: Dumping hundreds of pages into the prompt degrades LLM reasoning, increases latency, and confuses the model with irrelevant noise.

**Context Engineering** solves this through deliberate, multi-source orchestration. Instead of letting an LLM guess, our system:
- Dispatches targeted queries across **four independent context streams** in parallel.
- Passes all candidate information through an **Evaluator Engine** that mathematically scores relevance and filters out low-quality noise.
- Generates answers strictly bound to verified sources, complete with **clickable citations** and an **auditable decision log**.

---

## 2. How the 4-Stage Multi-Agent Architecture Works

When you submit a query, the assistant executes a 4-stage pipeline orchestrated by [`ResearchAssistantFlow`](file:///g:/Data%20Analytics/19.Portfolio/context-workflow/src/workflows/flow.py):

```
                                [ User Query ]
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   Stage 1: Memory Turn                               Stage 2: Parallel Context Gathering
   (Saves query with safe truncation)                 ┌─────────┬─────────┬─────────┬─────────┐
                                                      │   RAG   │   WEB   │  ARXIV  │ MEMORY  │
                                                      └────┬────┴────┬────┴────┬────┴────┬────┘
                                                           └─────────┼─────────┘         │
                                                                     ▼                   ▼
                                                      Stage 3: Context Evaluator Engine
                                                      • Scores relevance (0.0 to 1.0)
                                                      • Filters out noise (Threshold: 0.35)
                                                      • Computes groundedness confidence
                                                                     │
                                                                     ▼
                                                      Stage 4: Grounded Synthesis (LLM)
                                                      • Synthesizes citation-backed answer
                                                      • Or triggers refusal if context is absent
                                                                     │
                                                                     ▼
                                                      [ Final UI Response + Audit Drawer ]
```

### The 4 Information Streams

| Icon | Stream | Provider / Tool | Purpose |
|:---:|---|---|---|
| 📄 | **Document RAG** | In-Memory Vector Store + Voyage AI | Deep search into your uploaded PDF document sections |
| 🌐 | **Live Web** | Firecrawl / WebSearchAPI.ai / Tavily | Real-time news, current events, and public web information |
| 📚 | **Academic ArXiv** | ArXiv REST API (Atom Feed) | Peer-reviewed scientific papers and pre-prints |
| 🧠 | **Conversation Memory** | SQLite Local / Zep Cloud | Preserves prior context within your active research session |

---

## 3. Prerequisites & Quickstart Installation

### Requirements
- **Python:** Version `3.10` through `3.13` installed.
- **Operating System:** Windows, macOS, or Linux.
- **Network Access:** Internet connection for web search and LLM APIs.

### Setup Commands

```bash
# 1. Clone the repository
git clone <repository-url>
cd context-workflow

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

---

## 4. Configuring API Keys

Copy the example environment file to `.env`:

```bash
# On Windows (PowerShell):
Copy-Item .env.example .env

# On Linux / macOS:
cp .env.example .env
```

Open [`.env`](file:///g:/Data%20Analytics/19.Portfolio/context-workflow/.env) in your editor. The system supports modern provider keys with **automatic format detection**:

```env
# 1. LLM Synthesis Provider (Google Gemini)
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-1.5-flash         # Options: gemini-1.5-flash, gemini-2.0-flash, gemini-2.5-flash

# 2. Embeddings API (Voyage AI)
VOYAGE_API_KEY=your_voyage_api_key_here
EMBEDDINGS_MODEL=voyage-4          # High-performance 1024-dim embedding model

# 3. Web Search API (Auto-detected by prefix)
# Supports:
#   • Firecrawl:       fc-...      (api.firecrawl.dev)
#   • WebSearchAPI.ai: wsa_...     (api.websearchapi.ai)
#   • Tavily:          tvly-...    (tavily.com)
WEB_SEARCH_API_KEY=your_web_search_api_key_here

# 4. Conversation Memory Backend (Optional)
MEMORY_API_KEY=your_memory_api_key_here  # Leave as default for local SQLite storage
ZEP_API_URL=https://api.getzep.com
```

> [!TIP]
> **Graceful Degradation:** If any API key is absent, the system **does not crash**. Missing external tools gracefully return an error card, and the Evaluator safely answers using the remaining available sources (or local TF-IDF fallbacks).

---

## 5. Launching the Web Application

To run the Streamlit interface:

```powershell
# Ensure your virtual environment is active
.venv\Scripts\Activate.ps1

# Launch the Streamlit server
streamlit run app.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

---

## 6. Interactive Walkthrough: Using the Assistant

### Method A: Instant 1-Click Sample Paper (Fastest)

If you want to test the system immediately without preparing your own PDF:
1. Open the application at `http://localhost:8501`.
2. Notice the prominent primary button in the main window:
   ```
   [ 📥 Load Sample Research Paper Now ]
   ```
3. Click this button. The system will automatically parse, chunk, and embed [`data/sample_research_paper.pdf`](file:///g:/Data%20Analytics/19.Portfolio/context-workflow/data/sample_research_paper.pdf) (*"Context Engineering for Multi-Agent LLM Systems"*).
4. Within 2–3 seconds, the sidebar badge updates to `🟢 Assistant Ready (Document Indexed)` and the chat input field unlocks!

### Method B: Uploading Your Own PDFs

To analyze your own scientific papers, financial reports, or technical manuals:
1. In the **left sidebar**, look for the **📄 Ingest Document** section.
2. Drag and drop any `.pdf` file (or click **Browse files**).
3. Watch the real-time 4-step progress bar:
   - **Step 1/4:** Uploading PDF
   - **Step 2/4:** Parsing document structure & headings
   - **Step 3/4:** Generating dense semantic embeddings
   - **Step 4/4:** Populating vector index
4. Once completed, a green toast appears: `Indexed X chunks from 'filename.pdf'!`.
5. You can now begin asking questions about the uploaded document.

---

## 7. Querying Best Practices & Prompt Examples

The assistant excels when given specific research questions. Here are recommended patterns:

### 1. Document Deep-Dive Queries
Target specific methodologies, formulas, or conclusions in the document:
- *"What is context engineering and what methodology is proposed in the paper?"*
- *"What are the main performance metrics and benchmark results reported?"*
- *"Explain Section 3's approach to boundary-safe sentence truncation."*

### 2. Multi-Source Comparative Queries
Leverage both your document and live web/literature sources:
- *"How does the methodology in this paper compare to current 2026 AI industry news?"*
- *"Are there recent ArXiv papers addressing similar multi-agent context filtering?"*

### 3. Multi-Turn Conversational Follow-ups
Test session memory persistence across turns:
- *Turn 1:* "Summarize the 4 stages of the pipeline."
- *Turn 2:* "Which of those four stages is responsible for discarding irrelevant sources?" *(The assistant remembers Turn 1's context).*

### 4. Anti-Hallucination Stress Testing
Test the system's ability to refuse ungrounded questions:
- *"What is the recipe for baking chocolate chip cookies?"*
- **Expected Result:** The assistant cleanly responds with `INSUFFICIENT_CONTEXT` rather than hallucinating cooking advice.

---

## 8. Interpreting Answers, Citations & the Evaluator Audit

Every assistant response is paired with an expandable **🔍 Sources & Citations** drawer directly below the answer.

### Confidence Score Breakdown

At the top of the citation drawer, you will see a badge such as **Confidence: 92%**:

| Confidence | Interpretation | Action Required |
|:---:|---|---|
| **80% – 100%** | **High Confidence**: Supported by strong, direct matches in your document or verified web sources. | Reliable for technical review. |
| **50% – 79%** | **Moderate Confidence**: Some sources matched, but evidence may be partial or inferred. | Check the citations to verify nuance. |
| **1% – 49%** | **Low Confidence**: Weak keyword overlap; context was near the exclusion threshold. | Verify external links manually. |
| **0%** | **Insufficient Context**: No source contained verified evidence. | System safely refused to answer. |

### The 4 Source Inspectability Cards

Click the **Sources & Citations** header to inspect all candidate streams:

```
📦 Source Inspectability Cards

[ Source: RAG ]        🟢 STATUS: OK                  Relevance: 92%
Chunks extracted from Section 1 (Introduction) and Section 2 (Methodology).

[ Source: WEB ]        🟢 STATUS: OK                  Relevance: 38%
Snippets from Firecrawl web search matching recent news articles.

[ Source: ARXIV ]      🟢 STATUS: OK                  Relevance: 58%
Abstracts of related papers retrieved from the ArXiv Atom feed.

[ Source: MEMORY ]     🟡 STATUS: INSUFFICIENT CONTEXT  Relevance: 0%
First query in conversation session — no prior dialogue context to pull from.
```

#### Status Indicators
- 🟢 **`STATUS: OK`**: Source executed successfully and returned candidate snippets.
- 🟡 **`STATUS: INSUFFICIENT CONTEXT`**: Source was queried but contained no information relevant to the prompt.
- 🔴 **`STATUS: ERROR`**: The external API returned a failure (e.g. rate limit or invalid API key); the pipeline safely isolated this fault.

### Anti-Hallucination & Refusal Behavior

If none of the sources meet the minimum relevance threshold ($\ge 0.35$), the system activates its refusal protocol:
```markdown
I cannot answer this query based on the verified context. 
The uploaded documents, memory, and external search did not contain sufficient 
information regarding: '<your query>'.
```
This guarantees that the assistant will **never invent information** outside the verified context.

---

## 9. Running Automated Tests

To run the complete automated test suite:

```powershell
# Quick summary test run (30 tests)
.venv\Scripts\pytest -q

# Verbose test run with individual test names
.venv\Scripts\pytest -v
```

Expected output:
```
30 passed in ~33s
```

For detailed breakdown of all passed tests, refer to [TEST_RESULTS.md](file:///g:/Data%20Analytics/19.Portfolio/context-workflow/TEST_RESULTS.md).

---

## 10. Troubleshooting & FAQs

### Q1: The web search tool returns `STATUS: ERROR`
- **Cause:** Your web search API key in `.env` is either missing, has leading/trailing whitespace, or is invalid.
- **Solution:** 
  - If using **Firecrawl**, your key should start with `fc-`.
  - If using **WebSearchAPI.ai**, your key should start with `wsa_`.
  - If using **Tavily**, your key should start with `tvly-`.
  - After updating [`.env`](file:///g:/Data%20Analytics/19.Portfolio/context-workflow/.env), restart Streamlit so it re-reads environment variables.

### Q2: Chat input is disabled with a lock icon
- **Cause:** You have not loaded a document yet.
- **Solution:** Click **📥 Load Sample Research Paper Now** on the main screen, or upload a PDF via the left sidebar.

### Q3: How do I clear conversation memory and start fresh?
- **Solution:** In the sidebar, click the **🔄 Reset Conversation History** button. This resets memory turns while keeping your uploaded documents indexed.

### Q4: Streamlit doesn't pick up my `.env` changes
- **Solution:** In the terminal running Streamlit, press `Ctrl + C` to stop the server, then run `streamlit run app.py` again.

### Q5: "ModuleNotFoundError: No module named 'src'"
- **Cause:** The command was run outside the project root directory or the virtual environment is not active.
- **Solution:** Make sure your terminal is inside `g:\Data Analytics\19.Portfolio\context-workflow` and run `.venv\Scripts\Activate.ps1`.

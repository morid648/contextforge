# PRD — Multi-Agent Context Engineering Research Assistant

**Author:** Anshul
**Document type:** Product & Technical Requirements Document
**Status:** Draft v1.0
**Source reference:** `context-engineering-workflow` (patchy631/ai-engineering-hub) — reverse-engineered from the uploaded codebase into a build-ready spec

---

## 1. Summary

Build a research assistant that answers a user's question by pulling context from **four independent sources in parallel** — an uploaded-document RAG index, conversation memory, live web search, and an academic/external API — then runs that combined context through an **evaluator agent** (which keeps only what's relevant) and a **synthesizer agent** (which writes the final grounded answer with citations). The whole thing is orchestrated as a state machine (a "Flow") over a multi-agent framework, config-driven via YAML, and exposed through a Streamlit chat UI with a citations drawer.

The interesting engineering problem this project demonstrates is **context engineering**: not "can an LLM answer this" but "how do you assemble, filter, and rank the right context from multiple heterogeneous sources before generation, and prove — via citations and confidence scores — that the answer is grounded rather than hallucinated." That's a directly relevant skill to showcase alongside CMA/finance credentials, since it's the same discipline FP&A/analyst-adjacent AI tooling needs: don't let the model answer from parametric memory when it should be citing the actual filing/document.

---

## 2. Goals

| Goal | Why it matters |
|---|---|
| G1. Answer a query using 4 parallel context sources (docs, memory, web, external API) | Core differentiator vs. a single-source RAG chatbot |
| G2. Every answer is either grounded with citations or explicitly flags insufficient context | Prevents silent hallucination — critical for any finance/research use case |
| G3. Multi-turn memory persists across a session (and optionally across sessions) | Makes it a "research assistant," not a one-shot Q&A box |
| G4. Config-driven agents/tasks (YAML, not hardcoded prompts) | Reusable, portfolio-legible pattern; easy to demo customization |
| G5. Runnable end-to-end locally with a documented setup (`uv`, `.env`, one command to launch) | Recruiters/interviewers need to run it in under 5 minutes |
| G6. Clear, inspectable source-relevance breakdown in the UI | Turns an opaque LLM answer into something you can audit — a talking point in interviews |

### Non-goals (out of scope for v1)

- Multi-user auth / multi-tenant deployment
- Streaming token-by-token responses in the UI
- Fine-tuning or self-hosted LLMs
- Support for non-PDF document formats (docx, pptx, html) — PDF only in v1
- Automatic re-indexing / incremental document updates (v1 drops and rebuilds the collection per session)
- Production-grade cost controls / rate limiting (flagged as a risk, not solved in v1)

---

## 3. Users & primary use case

**Primary persona:** a technical user (or, for the portfolio-project framing, an interviewer/recruiter) who uploads a research paper, technical report, or financial document and asks multi-part questions that may require: something *in* the document, something from *earlier in the conversation*, something that changed *recently on the web*, and something from a *specialized external corpus* (academic papers in the reference implementation; swappable — see §11).

**Example query flow:**
1. User uploads a PDF (e.g., a paper or annual report).
2. User asks: *"What does this document say about X, and has anything changed publicly on this topic since it was published?"*
3. System searches the document (RAG), checks if X was discussed earlier in the chat (Memory), searches the live web for recent developments (Web), and optionally pulls related external references (Tool/API) — all in parallel.
4. An evaluator agent scores each source's relevance to *this specific query* and drops irrelevant/erroring sources.
5. A synthesizer agent writes one coherent, cited answer from only the relevant, filtered context.
6. The UI shows the answer plus a collapsible "sources & citations" panel with per-source status, relevance score, and reasoning.

---

## 4. System architecture

### 4.1 High-level flow (state machine)

The system is a 4-stage pipeline. Stage 2 fans out to 4 agents that run **concurrently as a single crew** (not sequentially) — this is the core latency optimization: total context-gathering time ≈ time of the *slowest* single source, not the sum of all four.

```
User query
   │
   ▼
[1] process_query
   - persist user turn to memory (truncated/summarized if long)
   │
   ▼
[2] gather_context_from_all_sources   (parallel crew, 4 agents)
   ├── RAG Agent          → searches the vector index of uploaded docs
   ├── Memory Agent        → pulls relevant prior-conversation context
   ├── Web Search Agent    → live web search for recent info
   └── Tool/API Agent      → queries an external structured API (ArXiv in reference impl)
   │
   ▼
[3] evaluate_context_relevance   (single evaluator agent)
   - scores each of the 4 raw results for relevance to *this* query
   - drops sources with ERROR status or low relevance
   - outputs a structured (schema-validated) filtered context object
   │
   ▼
[4] synthesize_final_response   (single synthesizer agent)
   - writes the final answer using ONLY the filtered context
   - persists a summarized version of the answer to memory
   │
   ▼
Final response + citation trail → returned to UI
```

### 4.2 Component inventory

| Layer | Responsibility | Reference implementation | Swappable alternative |
|---|---|---|---|
| Document parsing | Turn a PDF into structured chunks + metadata (title, authors, sections) | TensorLake DocumentAI (structured extraction + section-based chunking) | `unstructured`, `PyMuPDF` + a custom chunker, LlamaParse |
| Embeddings | Turn chunks into vectors, contextualized (chunk-aware, not isolated) | Voyage `voyage-context-3` (contextualized embeddings) | OpenAI `text-embedding-3-*`, Cohere embed-v3 |
| Vector store | Store + similarity-search chunk embeddings | Milvus Lite (local file-based) | ChromaDB, Qdrant, pgvector |
| Structured generation | Force the LLM to answer in a strict JSON schema (status/answer/citations/confidence) | OpenAI `gpt-4o-mini` + `response_format: json_schema` (strict mode) | Any model with function-calling/structured-output support |
| Conversation memory | Persist turns, summarize, retrieve relevant context per query | Zep Cloud (thread + user-scoped memory, graph-backed) | Mem0, a simple SQLite/Postgres turn log + retrieval, or the agent framework's own memory |
| Web search | Live web results for recency | Firecrawl search API | Tavily, Serper, Bing Search API |
| External/tool API | Domain-specific structured search | ArXiv Atom API (free, no key) | SEC EDGAR full-text search, a company-filings API, internal document store |
| Agent orchestration | Define agents/tasks, run them as crews inside a stateful flow | CrewAI (`Agent`, `Task`, `Crew`, `Flow`) | LangGraph, a hand-rolled `asyncio.gather` orchestrator |
| Config | Keep agent role/goal/backstory and task descriptions out of code | YAML files loaded via a small `ConfigLoader` | Same pattern, or a Python dict/dataclass registry |
| UI | Chat interface, document upload, citation drawer | Streamlit | Any chat frontend; Streamlit is the fastest to ship for a portfolio demo |

### 4.3 Repository structure to build

```
context-engineering-assistant/
├── src/
│   ├── document_processing/
│   │   └── doc_parser.py          # upload + structured-parse + chunk extraction
│   ├── rag/
│   │   ├── embeddings.py          # contextualized embedding wrapper
│   │   ├── retriever.py           # vector DB client (create/insert/search)
│   │   └── rag_pipeline.py        # orchestrates parse → embed → store → retrieve
│   ├── memory/
│   │   └── memory.py              # memory layer wrapper (save/retrieve/context block)
│   ├── generation/
│   │   └── generation.py          # structured-output LLM call + shared response schema
│   ├── tools/
│   │   ├── rag_tool.py            # agent-callable wrapper around rag_pipeline
│   │   ├── memory_tool.py         # agent-callable wrapper around memory layer
│   │   ├── web_search_tool.py     # agent-callable wrapper around web search API
│   │   └── external_api_tool.py   # agent-callable wrapper around the domain API
│   ├── workflows/
│   │   ├── agents.py              # builds Agent objects from YAML config
│   │   ├── tasks.py               # builds Task objects from YAML config + runtime inputs
│   │   └── flow.py                # the Flow state machine described in §4.1
│   └── config/
│       └── config_loader.py       # tiny YAML loader/accessor
├── config/
│   ├── agents/agents.yaml         # role / goal / backstory per agent
│   └── tasks/tasks.yaml           # description / expected_output per task
├── data/                          # sample documents for the demo
├── outputs/                       # optional debug dumps (parsed chunks, structured extraction)
├── app.py                         # Streamlit UI
├── .env.example
├── pyproject.toml                 # dependency + project metadata (uv-managed)
└── README.md
```

**Build note:** every `src/*` package needs an `__init__.py` that re-exports its public class (e.g. `src/rag/__init__.py` does `from .rag_pipeline import RAGPipeline`) so the rest of the code can do clean top-level imports like `from src.rag import RAGPipeline`. Don't skip this — it's easy to forget and causes import errors that look unrelated to the actual bug.

---

## 5. Functional requirements

### 5.1 Document processing & RAG (FR-1xx)

- **FR-101:** Accept a PDF upload, upload it to the parsing backend, and request structured extraction against a fixed JSON schema (title, authors/source, abstract/summary, keywords, key findings, section headings + summaries) plus section-based chunking.
- **FR-102:** Each returned chunk carries `{text, page_number, source_file}` metadata; reject/raise if parsing returns zero chunks (don't silently proceed with an empty index).
- **FR-103:** Embed all chunks of a document together as one contextualized batch (not chunk-by-chunk in isolation) so embeddings retain document-level context.
- **FR-104:** Store embeddings + text + metadata in the vector DB. On first parse in a session, the collection is created fresh (drop-and-recreate is acceptable for v1; see §3 non-goals on incremental updates).
- **FR-105:** Given a query, embed it (query mode, not document mode — the embedding model must distinguish the two) and return the top-k most similar chunks with similarity scores.
- **FR-106:** If no documents have been indexed yet, the RAG tool must return a structured `INSUFFICIENT_CONTEXT` response (not an error, not a hallucinated answer) and, if given document paths at query time, load them on demand.

### 5.2 Memory (FR-2xx)

- **FR-201:** Every user query and every assistant answer is saved to the memory store, tagged with role (`user`/`assistant`) and a name.
- **FR-202:** Long responses are truncated to a safe length before being persisted (truncate at the nearest sentence boundary, fall back to nearest word boundary) — memory stores are for context, not full transcripts.
- **FR-203:** A memory retrieval call returns a consolidated context block (summary of relevant prior turns/preferences) for the current thread/user, not a raw dump of every message.
- **FR-204:** If no relevant memory exists yet (e.g., first turn), return `INSUFFICIENT_CONTEXT` rather than an empty-but-"OK" result.
- **FR-205:** Memory is scoped per `(user_id, thread_id)` pair so multiple concurrent sessions don't bleed into each other.

### 5.3 Web search (FR-3xx)

- **FR-301:** Given a query, call the web search API and return up to *N* results (title, URL, snippet), each snippet truncated to a bounded length.
- **FR-302:** Return `INSUFFICIENT_CONTEXT` on zero results, and `ERROR` (with the underlying exception message preserved) on API failure — never let a web search failure crash the whole query.
- **FR-303:** If the API key isn't configured, fail gracefully with a clear "web search unavailable" status rather than raising an unhandled exception, so the rest of the pipeline still produces an answer from the remaining sources.

### 5.4 External/tool API (FR-4xx) — ArXiv in the reference implementation

- **FR-401:** Support search by general query, and optionally filter by field (title/author/abstract/category).
- **FR-402:** Parse the API's response format into a normalized `{title, authors, abstract, url, published_date, category}` record set.
- **FR-403:** Same status-envelope contract as every other tool: `OK` / `INSUFFICIENT_CONTEXT` / `ERROR`.

### 5.5 Tool response contract (FR-5xx) — applies to *every* tool

All four tools (RAG, Memory, Web, External API) must return a JSON string with this shape, so the evaluator and synthesizer agents can treat every source uniformly:

```json
{
  "status": "OK | INSUFFICIENT_CONTEXT | ERROR",
  "source_used": "RAG | MEMORY | WEB | <TOOL_NAME>",
  "answer": "human-readable summary of what was found",
  "citations": [ { "label": "...", "locator": "..." } ],
  "confidence": 0.0,
  "...source-specific extra fields (raw_context, search_results, retrieval_metadata, etc.)"
}
```

- **FR-501:** `status` and `source_used` are mandatory on every response, including error paths.
- **FR-502:** `confidence` is always a float in `[0, 1]`; `0.0` on any non-`OK` status.
- **FR-503:** `citations` is always present (possibly empty), never `null`, so downstream JSON parsing never has to null-check it.

### 5.6 Multi-agent orchestration (FR-6xx)

- **FR-601:** Agent `role` / `goal` / `backstory` and task `description` / `expected_output` live in YAML, not inline Python strings. Code reads config by key (e.g. `get_agent_config("rag_agent")`) and raises a clear error if the key is missing.
- **FR-602:** The four context-gathering agents run inside a single crew and execute concurrently — this is a hard requirement, not an optimization; sequential execution defeats the point of the architecture (latency would scale with the number of sources instead of the slowest one).
- **FR-603:** The evaluator agent's task description is dynamically built from the actual outputs of all four upstream tools (interpolated into the task's `description` string as formatted JSON) so the evaluator sees exactly what each source returned.
- **FR-604:** The evaluator's output must conform to a strict schema:
  - `relevant_sources: List[str]`
  - `filtered_context: Dict[str, Any]`
  - `relevance_scores: Dict[str, float]`
  - `reasoning: str`
  If the framework's structured-output parsing fails, the flow must fall back to best-effort raw parsing rather than crashing the whole query.
- **FR-605:** A source with `ERROR` status must never appear in `relevant_sources`, but the evaluator must still mention it happened in `reasoning` (so failures are visible in the citation drawer, not silently swallowed).
- **FR-606:** The synthesizer agent receives only `filtered_context` (never the raw, unfiltered results) and must not introduce information absent from that context.

### 5.7 UI (FR-7xx)

- **FR-701:** Sidebar: initialize the assistant, upload a PDF, show processing progress (upload → parse → embed → index), and show current document + assistant online/offline status.
- **FR-702:** Main pane: a chat interface (input box + scrolling history) that's disabled/blocked with a clear message until at least one document has been processed.
- **FR-703:** Each assistant turn has a collapsible "Sources & Citations" section showing: the relevance summary (which sources were used, their scores, the evaluator's reasoning), then a per-source expandable card with that source's status, its citations, and its confidence — and clearly renders a warning state for `INSUFFICIENT_CONTEXT` sources and an error state for `ERROR` sources, rather than hiding them.
- **FR-704:** A "reset chat" control clears history without requiring re-initialization or re-upload.
- **FR-705:** If any part of the citation-rendering logic throws (malformed data from an upstream agent), the UI must degrade to a debug expander showing the raw keys/error rather than crashing the whole page.

---

## 6. Non-functional requirements

| Category | Requirement |
|---|---|
| **Latency** | Target end-to-end query latency (excluding first-time document indexing) under ~20–30s given parallel context gathering; document indexing is a one-time, expected-slow operation (upload + parse + embed) and should show progress, not a blocking spinner with no feedback. |
| **Error isolation** | A failure in *any one* of the four context sources must not fail the whole query — the evaluator simply excludes that source and the synthesizer answers from what remains (or reports insufficient context overall if *all* sources fail/are irrelevant). |
| **Groundedness** | The synthesizer must never answer from the model's own parametric knowledge when context is insufficient — it must return an explicit "insufficient context" style answer instead. This is the single most important behavioral requirement in the whole system. |
| **Config over code** | Changing an agent's persona or a task's instructions should require editing YAML, not Python. |
| **Secrets handling** | All API keys loaded from `.env` (never hardcoded, never logged). Ship a `.env.example` with placeholder values and document where to obtain each key. |
| **Observability** | Verbose/debug logging at each pipeline stage (what was uploaded, how many chunks, embedding dimension mismatches, search scores) — this is what makes the system debuggable when a vendor API changes shape. |
| **Portability** | The whole thing should run with a single documented command after `.env` is filled in (`uv sync` then `uv run app.py` / `streamlit run app.py`). |
| **Cost awareness** | Every external call (parsing, embeddings, LLM generation ×2 agents minimum, web search, memory) has a real per-query cost — document this explicitly (see §9) since it directly affects whether the demo is cheap enough to leave running for interviews. |

---

## 7. Data & schema reference

### 7.1 Structured document-extraction schema (used by the parser)

A JSON Schema requesting: `paper.title`, `paper.authors[]`, `paper.abstract`, `paper.keywords[]`, `paper.key_findings[]`, and `paper.sections[]` (each with `heading` + `summary`). This is both metadata for the UI *and* a forcing function — the parser has to actually understand document structure, not just OCR text out.

> **Domain-adaptation note:** if the target documents are financial reports rather than research papers, redefine this schema to something like `filing.company_name`, `filing.period`, `filing.filing_type`, `filing.key_metrics[]` (metric/value/period), `filing.management_commentary_summary`, `filing.risk_factors[]`, `filing.sections[]`. Same shape, different fields.

### 7.2 Evaluator output schema

```
ContextEvaluationResult:
  relevant_sources: List[str]           # which of RAG/Memory/Web/Tool passed the bar
  filtered_context: Dict[str, Any]      # keyed by source name, cleaned payloads only
  relevance_scores: Dict[str, float]    # 0-1 per source, including excluded ones
  reasoning: str                        # short justification, must mention any ERROR sources
```

### 7.3 Generation response schema (used for the structured-output LLM call inside the RAG tool's own answer generation, if that path is used)

```
status: "OK" | "INSUFFICIENT_CONTEXT"
source_used: "MEMORY" | "RAG" | "WEB" | "TOOL" | "NONE"
answer: string
citations: [{ label: string, locator: string }]
confidence: number (0-1)
missing: string[]        # what info would be needed if status is INSUFFICIENT_CONTEXT
```

### 7.4 Vector store schema

| Field | Type | Notes |
|---|---|---|
| `id` | int64, auto | primary key |
| `embedding` | float vector | dimension must match the embedding model's output (e.g. 1024) |
| `text` | varchar | the chunk text |
| `page_number` | int64 | for citation display |
| `chunk_index` | int64 | for citation display and de-duplication |
| `source_file` | varchar | filename, for multi-document collections |

Index: cosine-similarity ANN index on `embedding`. Collection is recreated per fresh session in v1 (see non-goals).

---

## 8. Configuration reference

**`.env` keys to define:**

| Key | Used by | Free tier available? |
|---|---|---|
| `DOC_PARSER_API_KEY` | document parsing backend | check current provider pricing |
| `EMBEDDINGS_API_KEY` | embedding model | check current provider pricing |
| `LLM_API_KEY` | generation + evaluator/synthesizer agents | check current provider pricing |
| `MEMORY_API_KEY` | memory backend (if using a hosted one) | check current provider pricing |
| `WEB_SEARCH_API_KEY` | web search backend | check current provider pricing |

**`config/agents/*.yaml`:** one entry per agent with `role`, `goal`, `backstory`, `verbose`.
**`config/tasks/*.yaml`:** one entry per task with a `description` template (using `{query}`, `{rag_result}`, etc. as format placeholders) and `expected_output`.

---

## 9. Cost & risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Every query fans out to 2+ LLM calls (context agents may themselves call an LLM depending on framework config) plus 2 more (evaluator, synthesizer) | Cost multiplies fast in a demo left running | Use a small/cheap model for evaluator + synthesizer by default; cap `top_k` and snippet lengths aggressively |
| Five different paid vendor APIs (parser, embeddings, LLM, memory, web search) | Setup friction, and a single missing key breaks the whole pipeline | Design every tool to degrade gracefully (return `ERROR`/`INSUFFICIENT_CONTEXT`, don't hard-crash) so a demo still partially works with only some keys configured; document a "minimum viable key set" (LLM + one search source) for a lighter-weight demo mode |
| Vector collection is dropped and rebuilt each session | Not production-shaped; fine for a portfolio demo, a liability if pitched as "production-ready" | Explicitly scope this as a v1 limitation in the README, list incremental indexing as a "next steps" item |
| Structured-output/schema-validation failures from the LLM | Evaluator or generation step throws | Always implement the raw-text fallback path (already required in FR-604) |
| Memory backend requires deleting/recreating a thread on init (per reference implementation) | Destroys prior session history if reused naively | Decide explicitly: either accept session-scoped memory only in v1, or change the init logic to reuse an existing thread rather than delete-then-recreate it |

---

## 10. Milestones

| Milestone | Deliverable | Rough effort |
|---|---|---|
| M0 — Scaffolding | Repo structure, `pyproject.toml`/`uv` setup, `.env.example`, config loader, empty agent/task YAML | 0.5 day |
| M1 — Document → Vector pipeline | Parser client, embeddings wrapper, vector store client, `RAGPipeline.process_documents` + `.retrieve_context` working end-to-end on one sample PDF (test outside any agent framework first) | 1–1.5 days |
| M2 — Memory layer | Save/retrieve wrapper, session scoping, truncation logic | 0.5 day |
| M3 — Web + external API tools | Both tools implemented with the shared response contract (§5.5), tested standalone | 0.5–1 day |
| M4 — Agents + Tasks + Flow | Wrap M1–M3 as agent tools, build the 4-stage Flow, get one query working end-to-end from the terminal (no UI yet) | 1.5–2 days |
| M5 — Evaluator + Synthesizer | Schema-validated evaluator output, fallback parsing, synthesis task, verify groundedness (deliberately test a query with *no* good context and confirm it returns insufficient-context rather than a hallucinated answer) | 1 day |
| M6 — Streamlit UI | Upload/progress flow, chat interface, citations drawer, error-degradation states | 1–1.5 days |
| M7 — Polish for portfolio | README with architecture diagram, sample queries + screenshots, cost/risk notes, GitHub repo cleanup | 0.5 day |

**Total estimate: ~7–9 focused days** for someone building this fresh (less if reusing an existing RAG pipeline you've already built, since M1 overlaps heavily with prior document/embedding pipeline work).

---

## 11. Domain-adaptation option (optional, worth considering before you build)

The reference implementation is genre-agnostic (research papers + ArXiv). Two ways to point this project at your existing finance/data-analytics narrative instead of leaving it as a generic "AI paper reader":

1. **Swap the document domain and the external-API tool.** Feed it annual reports / investor presentations / earnings call transcripts instead of academic PDFs; replace the ArXiv tool with a filings-search tool (e.g., a company-filings or stock-exchange-announcements API) so the "external structured source" is finance-native instead of academic. The evaluator/synthesizer/UI layers don't need to change at all — only the document schema (§7.1 adaptation note) and one tool.
2. **Position it as the "context engineering" layer for your existing equity-research assistant.** Since you already have a Streamlit RAG app for querying uploaded financial PDFs, this project can be framed as *"v2: multi-source context engineering"* — same end-user goal (ask questions about a filing), but now the answer is cross-checked against conversation memory, live web news, and a structured filings source, with an evaluator step that makes the grounding auditable. That's a stronger, more specific story for an interview than "I also built a RAG chatbot."

Either path keeps ~90% of this PRD unchanged; only §7.1 and the tool-4 (external API) spec move.

---

## 12. Acceptance criteria (definition of done for v1)

- [ ] Uploading a PDF successfully produces a populated vector index (chunk count > 0, embeddings dimension matches config).
- [ ] A query that *is* answerable from the document returns an `OK` response with real page-level citations.
- [ ] A query that is *not* answerable from any source returns an explicit insufficient-context style answer — never a fabricated one. This is tested deliberately, not just hoped for.
- [ ] Killing/omitting one API key (e.g., web search) does not prevent the other three sources from producing an answer.
- [ ] The evaluator's `reasoning` field visibly explains at least one exclusion decision when tested with a mixed-relevance context set.
- [ ] The UI's citation drawer shows per-source status (OK/insufficient/error) and never crashes on malformed source data.
- [ ] A fresh clone of the repo, with only `.env` filled in, runs via one documented command.
- [ ] README includes an architecture diagram, setup steps, sample queries with expected behavior, and an explicit list of v1 limitations (no incremental indexing, session-scoped memory, PDF-only).

# Root Cause Analysis — Context Engineering Research Assistant

> Engineering post-mortem documenting all bugs discovered during live testing, their root causes, iterations, and applied fixes.

---

## Table of Contents

1. [RCA-001 · RAG Returns INSUFFICIENT\_CONTEXT for Summary Queries](#rca-001--rag-returns-insufficient_context-for-summary-queries)
2. [RCA-002 · Evaluator Scores RAG at 0% for Overview Intent Queries](#rca-002--evaluator-scores-rag-at-0-for-overview-intent-queries)
3. [RCA-003 · Insufficient Document Coverage for Summary Answers](#rca-003--insufficient-document-coverage-for-summary-answers)
4. [RCA-004 · EvaluatorEngine Premature Exclusion on Short Queries](#rca-004--evaluatoren-premature-exclusion-on-short-queries)
5. [Test Suite Verification](#test-suite-verification)

---

## RCA-001 · RAG Returns INSUFFICIENT\_CONTEXT for Summary Queries

**Severity:** High  
**Discovered:** Live app test — user query: `"a summary of uploaded file"`  
**Symptom:** RAG source returned `INSUFFICIENT_CONTEXT` with 0% confidence despite a PDF being fully indexed.

### Observed Behaviour

```
Source: RAG — 🟡 STATUS: INSUFFICIENT CONTEXT
Content Excerpt: No relevant passages found in the uploaded documents for this query.
```

### Root Cause

The system has no real-time LLM API keys configured, so `EmbeddingsClient` falls back to a **local hash-based deterministic embedder** (`_generate_deterministic_vector`). This embedder works by:

1. Tokenizing the input text into words
2. Hashing each word via MD5 → bucket index in a 1024-dim vector
3. Normalizing the result to unit length

**The problem:** Generic summary-intent words (`"summary"`, `"brief"`, `"uploaded"`, `"file"`) hash to completely different bucket indices than document body vocabulary (`"methodology"`, `"results"`, `"abstract"`, `"dataset"` etc.). Cosine similarity between the query vector and any stored chunk vector is near-zero.

The `retrieve_context` call used `score_threshold=0.05`. Because all scores were below this floor, the retriever returned an empty list, and `RAGTool.run()` short-circuited to `INSUFFICIENT_CONTEXT`.

### Iterations

**Iteration 1 (rejected):** Lower `score_threshold` globally to `0.0`.  
Problem: Would also surface irrelevant low-signal chunks for specific queries, degrading precision.

**Iteration 2 (adopted):** Detect summary intent via a compiled regex pattern `_SUMMARY_INTENT_PATTERNS` and apply `score_threshold=0.0` **only** for those queries.

### Fix Applied

**File:** [`src/rag/rag_pipeline.py`](./src/rag/rag_pipeline.py)

```python
# Phrases that signal the user wants a document overview
_SUMMARY_INTENT_PATTERNS = re.compile(
    r"\b(summar|brief|overview|abstract|outline|describe|what is this|what does this|"
    r"tell me about|explain this|give me a|what('s| is) in|content of|about this|this document|this paper|this file)",
    re.IGNORECASE,
)

def retrieve_context(self, query, top_k=5, score_threshold=0.05):
    is_summary_query = bool(_SUMMARY_INTENT_PATTERNS.search(query))
    effective_threshold = 0.0 if is_summary_query else score_threshold
    ...
    # For summary intent: sort by document order, not similarity score
    if is_summary_query and results:
        results.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0)))
```

**Why document order:** When similarity scores are all near-zero (hash collision artifact), ranking by score is meaningless. Sorting by `(page_number, chunk_index)` ensures the synthesizer receives content in the logical reading sequence of the document.

---

## RCA-002 · Evaluator Scores RAG at 0% for Overview Intent Queries

**Severity:** High  
**Discovered:** Same live test — even after RAG returned `OK` with content, evaluator would have excluded it.  
**Symptom:** Evaluator's term-overlap formula assigns 0% relevance to any source whose answer text doesn't contain the query words.

### Root Cause

The `EvaluatorEngine` computes relevance as:

```python
overlap_ratio = len(query_terms ∩ content_terms) / len(query_terms)
score = 0.5 * overlap_ratio + 0.5 * tool_confidence
```

For a query like `"summarize the document"`:
- `query_terms` = `{"summarize", "document"}` (after stop-word removal)
- `content_terms` = typical document body vocabulary (never contains `"summarize"`)
- `overlap = {}` → `overlap_ratio = 0.0`
- Even with `tool_confidence = 0.8` → `score = 0.4` — **below the 0.4 threshold**, excluded

The term-overlap heuristic is **categorically wrong** for intent-based queries where the query verb (`"summarize"`, `"describe"`, `"brief"`) is an instruction to the system, not a keyword to match against content.

### Iterations

**Iteration 1 (rejected):** Lower `relevance_threshold` globally to `0.2`.  
Problem: Would pass weak, tangentially-relevant sources for specific queries.

**Iteration 2 (adopted):** Mirror the same summary-intent detection in `evaluator.py`. When detected, skip the overlap penalty entirely and score `OK`-status sources by `tool_confidence` alone.

### Fix Applied

**File:** [`src/workflows/evaluator.py`](./src/workflows/evaluator.py)

```python
_SUMMARY_INTENT_RE = re.compile(...)   # Same pattern as rag_pipeline.py

# Inside evaluate_sources():
is_summary_query = bool(_SUMMARY_INTENT_RE.search(query))

if is_summary_query and tool_resp.status == "OK" and tool_resp.answer.strip():
    # Bypass term-overlap: score purely from tool confidence
    score = round(min(1.0, max(self.relevance_threshold, tool_resp.confidence)), 2)
else:
    # Standard path: weighted overlap + confidence
    overlap_ratio = len(overlap) / max(1, len(query_terms))
    score = round(min(1.0, 0.5 * overlap_ratio + 0.5 * tool_resp.confidence), 2)
```

**Design constraint preserved:** Error isolation (FR-605) is untouched — `ERROR` and `INSUFFICIENT_CONTEXT` statuses still short-circuit to 0 regardless of intent.

---

## RCA-003 · Insufficient Document Coverage for Summary Answers

**Severity:** Medium  
**Discovered:** Code review during RCA-001 fix.  
**Symptom:** Summary answers were generated from only 4 chunks — typically the first ~2 pages of a multi-section document.

### Root Cause

`RAGTool.run()` defaults to `top_k=4`. For a specific factual query, 4 highly-scored chunks are sufficient. But for a summary/overview query, 4 chunks represent a narrow slice of the document — the synthesizer cannot produce a representative summary from partial coverage.

Because the hash-based embedder assigns near-zero scores, the top-4 results were essentially random from the document, not the most informative sections.

### Fix Applied

**File:** [`src/tools/rag_tool.py`](./src/tools/rag_tool.py)

```python
from ..rag.rag_pipeline import RAGPipeline, _SUMMARY_INTENT_PATTERNS

def run(self, query, top_k=4):
    # Scale up chunk retrieval for summary/overview queries
    effective_top_k = 8 if _SUMMARY_INTENT_PATTERNS.search(query) else top_k
    chunks = self.rag_pipeline.retrieve_context(query, top_k=effective_top_k)
```

Combined with the document-order sort from RCA-001, the synthesizer now receives 8 chunks spanning the full document in reading order — enough to construct an accurate overview.

---

## RCA-004 · EvaluatorEngine Premature Exclusion on Short Queries

**Severity:** Medium  
**Discovered:** Earlier in development (pre-live-test), during automated test suite development.  
**Symptom:** Queries with only 1–2 meaningful terms (after stop-word removal) were being excluded from all sources, producing spurious `INSUFFICIENT_CONTEXT` refusals.

### Root Cause

The original stop-word list was overly broad and included common domain terms (`"show"`, `"how"`, `"about"`). A query like `"how does attention work"` was reduced to `{"attention", "work"}` — only 2 terms. Even a highly relevant source answer could fail to contain both exact words, producing a low overlap ratio.

Additionally, queries reduced to an **empty** `query_terms` set (all words were stop words) caused `overlap_ratio = 0/0` — a ZeroDivisionError risk.

### Fix Applied

**File:** [`src/workflows/evaluator.py`](./src/workflows/evaluator.py)

```python
# Fallback: if stop-word filtering empties the set, use all terms
all_terms = re.findall(r"\w+", query.lower())
query_terms = set(t for t in all_terms if t not in STOP_WORDS)
if not query_terms:
    query_terms = set(all_terms)   # prevents empty-set division
```

**Division guard:**
```python
overlap_ratio = len(overlap) / max(1, len(query_terms))  # max(1,...) prevents ZeroDivisionError
```

---

## Test Suite Verification

All fixes were verified against the full automated test suite with zero regressions:

```
platform win32 -- Python 3.13.14, pytest-9.1.1
collected 27 items

tests/test_config_loader.py        5 passed
tests/test_error_isolation.py      1 passed
tests/test_evaluator.py            4 passed
tests/test_external_clients.py     2 passed
tests/test_flow.py                 1 passed
tests/test_groundedness.py         1 passed
tests/test_memory.py               4 passed
tests/test_rag_pipeline.py         4 passed
tests/test_tools.py                5 passed

========================= 27 passed in 24.12s =========================
```

Intent pattern correctness was additionally verified with 8 hand-crafted cases covering true positives (summary intent) and true negatives (specific factual queries):

| Query | Expected | Result |
|---|---|---|
| `"a summary of uploaded file"` | summary | ✅ |
| `"give me a brief"` | summary | ✅ |
| `"summarize this document"` | summary | ✅ |
| `"overview of this paper"` | summary | ✅ |
| `"give me a list of findings"` | summary | ✅ |
| `"describe the methodology"` | summary | ✅ |
| `"what is transformer attention"` | specific | ✅ |
| `"how does BERT handle tokenization"` | specific | ✅ |

---

*Document generated: 2026-09-15 · Project: context-workflow · Status: All issues resolved*

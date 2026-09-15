# Sample Test Data & Demo Queries

This directory contains sample PDF documents and recommended evaluation queries designed to test each pathway of the Multi-Source Context Engineering Research Assistant.

## Available Sample Documents

- **`sample_research_paper.pdf`**: A 2-page sample technical research paper titled *"Multi-Agent Context Engineering Systems"* covering context gathering, parallel latency benchmarks (62% reduction), and evaluator filtering accuracy (94%).

## Recommended Test Queries

### 1. In-Document Grounded Query (Tests RAG Pathway)
- **Query:** `"What did the empirical results show about latency reduction and evaluator accuracy?"`
- **Expected Behavior:**
  - `status: OK`
  - Answer cites Page 2 of `sample_research_paper.pdf`.
  - Confidence > 0.80.

### 2. Temporal / Live Web Query (Tests Web Pathway)
- **Query:** `"What are the latest public developments in CrewAI and multi-agent workflows this year?"`
- **Expected Behavior:**
  - Evaluator scores `WEB` as relevant.
  - Returns web citations with links.

### 3. Academic Foundation Query (Tests ArXiv Pathway)
- **Query:** `"What foundational papers exist on attention mechanisms and transformer models?"`
- **Expected Behavior:**
  - Evaluator scores `ARXIV` as relevant.
  - Returns ArXiv citations and paper links.

### 4. Unanswerable / Hallucination-Defense Query (Tests Strict Grounding)
- **Query:** `"What were the third-quarter revenue figures for Acme Widget Corp in 1923?"`
- **Expected Behavior:**
  - `status: INSUFFICIENT_CONTEXT`
  - Evaluator excludes all sources due to lack of factual evidence.
  - Synthesizer explicitly refuses to answer from parametric memory.

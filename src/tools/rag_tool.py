"""RAG agent tool adhering to the universal response envelope (FR-106, FR-501 - FR-503)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..rag.rag_pipeline import RAGPipeline, _SUMMARY_INTENT_PATTERNS
from .schemas import Citation, ToolResponse


class RAGTool:
    """Agent tool for querying the indexed document vector database."""

    name: str = "rag_search_tool"
    description: str = "Search indexed PDF documents for passages, data points, and page citations."

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.rag_pipeline = rag_pipeline or RAGPipeline()

    def run(self, query: str, top_k: int = 4) -> str:
        """Executes similarity retrieval on indexed documents and returns serialized ToolResponse JSON."""
        try:
            # Use more chunks for summary/overview queries so the synthesizer has
            # full document coverage rather than just 4 arbitrarily-scored passages.
            effective_top_k = 8 if _SUMMARY_INTENT_PATTERNS.search(query) else top_k
            chunks = self.rag_pipeline.retrieve_context(query, top_k=effective_top_k)

            if not chunks:
                resp = ToolResponse(
                    status="INSUFFICIENT_CONTEXT",
                    source_used="RAG",
                    answer="No relevant passages found in the uploaded documents for this query.",
                    citations=[],
                    confidence=0.0,
                    details={"matched_chunks": 0},
                )
                return resp.to_json()

            # Compile citations and excerpt answer
            citations: List[Citation] = []
            answer_snippets: List[str] = []
            scores: List[float] = []

            for idx, c in enumerate(chunks):
                label = c.get("source_file", "Document")
                page = c.get("page_number", 1)
                score = c.get("score", 0.0)
                scores.append(score)
                citations.append(Citation(label=label, locator=f"Page {page}"))
                answer_snippets.append(f"[{label}, p.{page}]: {c.get('text', '').strip()}")

            top_score = max(scores) if scores else 0.0
            confidence = min(1.0, max(0.2, top_score))

            resp = ToolResponse(
                status="OK",
                source_used="RAG",
                answer="\n\n".join(answer_snippets[:3]),
                citations=citations,
                confidence=round(confidence, 2),
                details={"retrieved_chunks": len(chunks), "top_score": top_score},
            )
            return resp.to_json()

        except Exception as exc:
            resp = ToolResponse(
                status="ERROR",
                source_used="RAG",
                answer=f"RAG retrieval failure: {str(exc)}",
                citations=[],
                confidence=0.0,
                details={"error": str(exc)},
            )
            return resp.to_json()

    def __call__(self, query: str) -> str:
        return self.run(query)

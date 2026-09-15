"""Integration tests for groundedness and anti-hallucination defense (PRD Acceptance Criterion 3)."""

from unittest.mock import MagicMock
from src.workflows.flow import ResearchAssistantFlow


def test_unanswerable_query_returns_insufficient_context(sample_pdf_path):
    """Test that a query absent from all sources returns INSUFFICIENT_CONTEXT rather than fabricating facts."""
    flow = ResearchAssistantFlow()
    flow.rag_pipeline.process_documents([sample_pdf_path])

    # Mock external web and arxiv to return insufficient context
    mock_web = MagicMock()
    mock_web.search.return_value = {"status": "INSUFFICIENT_CONTEXT", "results": []}
    flow.tools["web_tool"].client = mock_web

    mock_arxiv = MagicMock()
    mock_arxiv.search.return_value = {"status": "INSUFFICIENT_CONTEXT", "papers": []}
    flow.tools["external_api_tool"].client = mock_arxiv

    # Query for completely unrelated and unmentioned fact
    unrelated_query = "What is the specific recipe for Grandma's chocolate walnut cookies in 1892?"
    result = flow.kickoff(unrelated_query)

    # Must return INSUFFICIENT_CONTEXT, zero confidence, and refusal answer
    assert result["status"] == "INSUFFICIENT_CONTEXT"
    assert result["confidence"] == 0.0
    assert "cannot answer" in result["answer"].lower() or "insufficient" in result["answer"].lower()
    # Confirm no citations fabricated
    assert len(result["citations"]) == 0

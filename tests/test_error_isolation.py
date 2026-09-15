"""Integration tests for error isolation and graceful degradation (PRD Acceptance Criterion 4)."""

from unittest.mock import MagicMock
from src.generation.schemas import ContextEvaluationResult
from src.tools.schemas import ToolResponse
from src.workflows.flow import ResearchAssistantFlow


def test_missing_web_key_does_not_crash_pipeline(sample_pdf_path, tmp_path):
    """Test that missing or invalid web search key does not prevent pipeline from completing.

    Acceptance criterion: the pipeline must complete without raising an exception,
    WEB must be isolated as ERROR, and the answer must never be sourced from WEB.
    The final status (OK vs INSUFFICIENT_CONTEXT) depends on whether other sources
    returned sufficient grounded content for the specific query.
    """
    # Setup flow with invalid web key
    flow = ResearchAssistantFlow()
    # Populate document in RAG
    flow.rag_pipeline.process_documents([sample_pdf_path])

    # Invalidate web client to simulate missing or broken key
    mock_web = MagicMock()
    mock_web.search.return_value = {
        "status": "ERROR",
        "error": "Simulated missing WEB_SEARCH_API_KEY",
        "results": [],
    }
    flow.tools["web_tool"].client = mock_web

    # Execute query — must complete without raising
    result = flow.kickoff("What are the empirical results of context engineering latency?")

    # Pipeline must return a structured result (not crash)
    assert "status" in result
    assert result["status"] in ("OK", "INSUFFICIENT_CONTEXT"), (
        f"Pipeline must return a known status, got: {result['status']}"
    )

    # WEB source must be isolated and marked ERROR
    assert result["raw_sources"]["WEB"]["status"] == "ERROR"

    # Evaluator must have excluded WEB from relevant sources
    eval_data = result["evaluation"]
    assert "WEB" not in eval_data["relevant_sources"]
    assert "error" in eval_data["reasoning"].lower() or "excluded 'web'" in eval_data["reasoning"].lower()

    # Answer must never claim WEB as a source
    assert "WEB" not in result.get("source_used", "")

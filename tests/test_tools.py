"""Tests for universal tool response contract and tool wrappers."""

import json
from unittest.mock import MagicMock
from src.tools.schemas import Citation, ToolResponse
from src.tools.rag_tool import RAGTool
from src.tools.memory_tool import MemoryTool
from src.tools.web_search_tool import WebSearchTool
from src.tools.external_api_tool import ExternalAPITool


def test_tool_response_contract_validation():
    """Verify ToolResponse validation rules (FR-501, FR-502, FR-503)."""
    # Non-OK status must force confidence to 0.0
    err_resp = ToolResponse(
        status="ERROR",
        source_used="WEB",
        answer="Connection failed",
        confidence=0.85,  # should be forced to 0.0
    )
    assert err_resp.confidence == 0.0
    assert err_resp.citations == []

    ok_resp = ToolResponse(
        status="OK",
        source_used="RAG",
        answer="Found passage",
        citations=[Citation(label="Doc A", locator="Page 1")],
        confidence=0.92,
    )
    assert ok_resp.confidence == 0.92
    assert len(ok_resp.citations) == 1

    # Verify JSON serialization
    serialized = ok_resp.to_json()
    data = json.loads(serialized)
    assert data["status"] == "OK"
    assert data["source_used"] == "RAG"


def test_rag_tool_insufficient_context():
    """Verify RAGTool returns INSUFFICIENT_CONTEXT when no chunks match."""
    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_context.return_value = []
    tool = RAGTool(rag_pipeline=mock_pipeline)

    output = tool.run("test query")
    resp = ToolResponse.model_validate_json(output)
    assert resp.status == "INSUFFICIENT_CONTEXT"
    assert resp.confidence == 0.0
    assert resp.citations == []


def test_memory_tool_returns_contract():
    """Verify MemoryTool adheres to response contract."""
    mock_mem = MagicMock()
    mock_mem.get_context.return_value = None
    tool = MemoryTool(memory_manager=mock_mem)

    output = tool.run("prior query")
    resp = ToolResponse.model_validate_json(output)
    assert resp.status == "INSUFFICIENT_CONTEXT"
    assert resp.source_used == "MEMORY"


def test_web_search_tool_returns_contract():
    """Verify WebSearchTool handles error and returns valid JSON envelope."""
    mock_client = MagicMock()
    mock_client.search.return_value = {"status": "ERROR", "error": "No API key", "results": []}
    tool = WebSearchTool(client=mock_client)

    output = tool.run("latest news")
    resp = ToolResponse.model_validate_json(output)
    assert resp.status == "ERROR"
    assert resp.confidence == 0.0
    assert "No API key" in resp.answer


def test_external_api_tool_returns_contract():
    """Verify ExternalAPITool handles successful query."""
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "status": "OK",
        "papers": [{
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "abstract": "Transformer architecture",
            "url": "https://arxiv.org/abs/1706.03762",
            "published_date": "2017",
            "category": "cs.CL",
        }],
    }
    tool = ExternalAPITool(client=mock_client)

    output = tool.run("transformers")
    resp = ToolResponse.model_validate_json(output)
    assert resp.status == "OK"
    assert resp.source_used == "ARXIV"
    assert len(resp.citations) == 1
    assert resp.citations[0].label == "Attention Is All You Need"

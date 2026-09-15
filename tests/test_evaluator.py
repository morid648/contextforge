"""Tests for EvaluatorEngine and GroundedSynthesizer."""

from src.generation.schemas import ContextEvaluationResult
from src.generation.generation import GroundedSynthesizer
from src.tools.schemas import Citation, ToolResponse
from src.workflows.evaluator import EvaluatorEngine


def test_evaluator_engine_error_isolation():
    """Verify that a source with ERROR status is NEVER included in relevant_sources (FR-605)."""
    evaluator = EvaluatorEngine(relevance_threshold=0.3)
    raw_sources = {
        "WEB": ToolResponse(
            status="ERROR",
            source_used="WEB",
            answer="Web search API key is invalid",
            confidence=0.0,
        ),
        "RAG": ToolResponse(
            status="OK",
            source_used="RAG",
            answer="Multi-agent context engineering achieves 62% latency reduction.",
            citations=[Citation(label="Paper", locator="Page 2")],
            confidence=0.9,
        ),
    }

    result = evaluator.evaluate_sources("multi-agent context engineering latency", raw_sources)

    assert "WEB" not in result.relevant_sources
    assert "WEB" in result.reasoning
    assert "error" in result.reasoning.lower()
    assert "RAG" in result.relevant_sources
    assert "RAG" in result.filtered_context


def test_evaluator_insufficient_context_exclusion():
    """Verify sources reporting INSUFFICIENT_CONTEXT are excluded from relevant_sources."""
    evaluator = EvaluatorEngine()
    raw_sources = {
        "MEMORY": ToolResponse(
            status="INSUFFICIENT_CONTEXT",
            source_used="MEMORY",
            answer="No prior conversation turns recorded.",
            confidence=0.0,
        )
    }

    result = evaluator.evaluate_sources("some query", raw_sources)
    assert "MEMORY" not in result.relevant_sources
    assert result.relevance_scores["MEMORY"] == 0.0


def test_grounded_synthesizer_empty_context_refusal():
    """Verify that synthesizer refuses to hallucinate when context is empty (G2)."""
    synthesizer = GroundedSynthesizer()
    empty_eval = ContextEvaluationResult(
        relevant_sources=[],
        filtered_context={},
        relevance_scores={"RAG": 0.0, "WEB": 0.0},
        reasoning="All sources lacked sufficient data.",
    )

    response = synthesizer.synthesize("What is the Q3 EBITDA for Company X?", empty_eval)
    assert response.status == "INSUFFICIENT_CONTEXT"
    assert response.confidence == 0.0
    assert "cannot answer" in response.answer.lower()
    assert len(response.missing) > 0


def test_grounded_synthesizer_with_verified_context():
    """Verify synthesizer produces answer and aggregates citations."""
    synthesizer = GroundedSynthesizer()
    eval_result = ContextEvaluationResult(
        relevant_sources=["RAG"],
        filtered_context={
            "RAG": {
                "answer": "Operating profit was $45M.",
                "citations": [{"label": "Annual Report", "locator": "Page 14"}],
                "confidence": 0.95,
            }
        },
        relevance_scores={"RAG": 0.95},
        reasoning="RAG source is highly relevant.",
    )

    response = synthesizer.synthesize("What was operating profit?", eval_result)
    assert response.status == "OK"
    assert "45M" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].label == "Annual Report"
    assert response.citations[0].locator == "Page 14"

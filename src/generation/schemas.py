"""Schemas for context evaluation and final response synthesis (PRD §7.2, §7.3)."""

from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field

from ..tools.schemas import Citation


class ContextEvaluationResult(BaseModel):
    """Schema for the evaluator agent's audit and filtering output (PRD §7.2)."""
    relevant_sources: List[str] = Field(
        default_factory=list,
        description="List of source keys (e.g. ['RAG', 'WEB']) that met relevance threshold and have valid status.",
    )
    filtered_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyed by source name, containing filtered, cleaned payload without errors.",
    )
    relevance_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevance score in [0.0, 1.0] for every inspected source, including excluded ones.",
    )
    reasoning: str = Field(
        ...,
        description="Detailed rationale for source inclusion/exclusion; must explicitly mention any ERROR sources.",
    )


class FinalSynthesizedResponse(BaseModel):
    """Schema for final structured synthesis response (PRD §7.3)."""
    status: Literal["OK", "INSUFFICIENT_CONTEXT"] = Field(
        ..., description="Whether a grounded answer could be formulated."
    )
    source_used: str = Field(
        default="MULTIPLE",
        description="Summary of sources used (e.g. 'RAG + WEB', 'NONE').",
    )
    answer: str = Field(..., description="Final synthesized, fully-grounded response text.")
    citations: List[Citation] = Field(
        default_factory=list, description="All verified source citations supporting the answer."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall answer confidence score."
    )
    missing: List[str] = Field(
        default_factory=list,
        description="Information gaps or missing context if status is INSUFFICIENT_CONTEXT.",
    )

"""Universal tool response envelope schemas (PRD §5.5, FR-501, FR-502, FR-503)."""

from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """Citation pointing to a specific source and locator (page, URL, or thread)."""
    label: str = Field(..., description="Human-readable title or source identifier.")
    locator: str = Field(..., description="Pinpoint locator, e.g. 'Page 2' or 'https://...'")


class ToolResponse(BaseModel):
    """Standardized response contract returned by every context gathering tool (PRD §5.5)."""
    status: Literal["OK", "INSUFFICIENT_CONTEXT", "ERROR"] = Field(
        ..., description="Execution status of the tool."
    )
    source_used: Literal["RAG", "MEMORY", "WEB", "ARXIV", "EXTERNAL_API"] = Field(
        ..., description="Source identifier."
    )
    answer: str = Field(..., description="Concise summary or human-readable extracted content.")
    citations: List[Citation] = Field(
        default_factory=list, description="List of pinpoint citations; never null."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0."
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Source-specific raw records or metadata."
    )

    @model_validator(mode="after")
    def enforce_contract_rules(self) -> "ToolResponse":
        """Ensures non-OK status zeroes out confidence and citations is never None."""
        if self.status != "OK":
            self.confidence = 0.0
        if self.citations is None:
            self.citations = []
        return self

    def to_json(self) -> str:
        """Serializes response to clean JSON string."""
        return self.model_dump_json(indent=2)

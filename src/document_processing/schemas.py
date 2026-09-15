"""Schemas for document processing, structured extraction, and chunking."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Section(BaseModel):
    """A section within a parsed document."""
    heading: str = Field(..., description="The heading or title of the section.")
    summary: str = Field(..., description="A concise summary of the section contents.")


class StructuredDocument(BaseModel):
    """Structured extraction schema for ingested documents (PRD §7.1)."""
    title: str = Field(..., description="Document title.")
    authors: List[str] = Field(default_factory=list, description="List of authors or issuing organization.")
    abstract: str = Field("", description="Executive summary or abstract.")
    keywords: List[str] = Field(default_factory=list, description="Top thematic keywords.")
    key_findings: List[str] = Field(default_factory=list, description="Key takeaways or findings.")
    sections: List[Section] = Field(default_factory=list, description="Structured section breakdowns.")


class DocumentChunk(BaseModel):
    """An indexed chunk of a document with citation metadata (PRD §5.1, §7.4)."""
    text: str = Field(..., description="Text content of the chunk.")
    page_number: int = Field(1, description="1-indexed page number where this chunk originates.")
    chunk_index: int = Field(0, description="Sequential chunk index across the document.")
    source_file: str = Field(..., description="Source filename for multi-document citations.")

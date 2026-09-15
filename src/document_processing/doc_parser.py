"""PDF document parser and structured chunk extractor."""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from pypdf import PdfReader

from .schemas import DocumentChunk, Section, StructuredDocument


class DocumentParser:
    """Parses PDF documents into structured metadata and indexed chunks."""

    def __init__(self, api_key: Optional[str] = None, chunk_size: int = 600, chunk_overlap: int = 100):
        self.api_key = api_key or os.getenv("DOC_PARSER_API_KEY")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, file_path: str | Path) -> Tuple[StructuredDocument, List[DocumentChunk]]:
        """Parses a PDF file into a StructuredDocument and List[DocumentChunk]."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        reader = PdfReader(str(path))
        num_pages = len(reader.pages)
        if num_pages == 0:
            raise ValueError(f"PDF file '{path.name}' contains 0 pages.")

        raw_pages: List[Tuple[int, str]] = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Clean common PDF whitespace artifacts
            cleaned_text = re.sub(r"[ \t]+", " ", text).strip()
            if cleaned_text:
                raw_pages.append((idx + 1, cleaned_text))

        if not raw_pages:
            raise ValueError(f"Could not extract any readable text from '{path.name}'.")

        # Chunk the text preserving page boundaries
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page_num, page_text in raw_pages:
            page_chunks = self._chunk_text(page_text, page_num, path.name, global_chunk_idx)
            chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        if not chunks:
            raise ValueError(f"Parsing produced zero valid chunks for '{path.name}'.")

        # Extract structured overview
        structured_doc = self._extract_structured_metadata(path.name, raw_pages, chunks)
        return structured_doc, chunks

    def _chunk_text(
        self, text: str, page_number: int, source_file: str, start_index: int
    ) -> List[DocumentChunk]:
        """Splits a single page's text into overlapping chunks."""
        chunks: List[DocumentChunk] = []
        words = text.split()
        if not words:
            return chunks

        # If text is small, treat page as single chunk
        if len(text) <= self.chunk_size:
            return [
                DocumentChunk(
                    text=text,
                    page_number=page_number,
                    chunk_index=start_index,
                    source_file=source_file,
                )
            ]

        # Word-based sliding window
        current_words: List[str] = []
        current_len = 0
        local_idx = 0

        for word in words:
            current_words.append(word)
            current_len += len(word) + 1
            if current_len >= self.chunk_size:
                chunk_str = " ".join(current_words)
                chunks.append(
                    DocumentChunk(
                        text=chunk_str,
                        page_number=page_number,
                        chunk_index=start_index + local_idx,
                        source_file=source_file,
                    )
                )
                local_idx += 1
                # Overlap step: keep the trailing words that fit in chunk_overlap
                overlap_words: List[str] = []
                overlap_len = 0
                for w in reversed(current_words):
                    if overlap_len + len(w) + 1 <= self.chunk_overlap:
                        overlap_words.insert(0, w)
                        overlap_len += len(w) + 1
                    else:
                        break
                current_words = overlap_words
                current_len = sum(len(w) + 1 for w in current_words)

        if current_words:
            chunk_str = " ".join(current_words)
            chunks.append(
                DocumentChunk(
                    text=chunk_str,
                    page_number=page_number,
                    chunk_index=start_index + local_idx,
                    source_file=source_file,
                )
            )

        return chunks

    def _extract_structured_metadata(
        self, filename: str, pages: List[Tuple[int, str]], chunks: List[DocumentChunk]
    ) -> StructuredDocument:
        """Extracts structured overview metadata from the document."""
        first_page_text = pages[0][1]
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]

        # Guess title from the first non-empty lines
        title = lines[0] if lines else Path(filename).stem
        if len(title) > 120 and lines:
            title = title[:120] + "..."

        # Guess abstract from first page
        abstract = ""
        abstract_match = re.search(r"(?i)abstract[:\s]*(.*?)(?:\n\n|introduction|1\.|1\s)", first_page_text, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip().replace("\n", " ")[:600]
        elif len(pages) > 0:
            abstract = first_page_text[:400] + "..."

        # Identify headings
        sections: List[Section] = []
        heading_pattern = re.compile(r"^(?:[0-9]+\.?\s+|[I|V|X]+\.?\s+)?([A-Z][A-Za-z0-9\s,-]{3,50})$")
        for _, page_text in pages:
            for line in page_text.split("\n"):
                line_str = line.strip()
                if heading_pattern.match(line_str) and not line_str.lower().startswith("page"):
                    sections.append(Section(heading=line_str, summary=f"Section covering {line_str}."))
                    if len(sections) >= 10:
                        break
            if len(sections) >= 10:
                break

        if not sections:
            sections = [Section(heading="Main Content", summary="Primary body of the document.")]

        return StructuredDocument(
            title=title,
            authors=["Extracted Document Source"],
            abstract=abstract,
            keywords=[Path(filename).stem],
            key_findings=[f"Document '{filename}' contains {len(pages)} pages and {len(chunks)} contextual chunks."],
            sections=sections,
        )

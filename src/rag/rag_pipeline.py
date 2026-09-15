"""End-to-end RAG pipeline orchestrating parsing, embeddings, vector indexing, and retrieval."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..document_processing.doc_parser import DocumentParser
from ..document_processing.schemas import DocumentChunk, StructuredDocument
from .embeddings import EmbeddingsClient
from .retriever import VectorStoreClient

# Phrases that signal the user wants a document overview — match regardless of similarity score
_SUMMARY_INTENT_PATTERNS = re.compile(
    r"\b(summar|brief|overview|abstract|outline|describe|what is this|what does this|"
    r"tell me about|explain this|give me a|what('s| is) in|content of|about this|this document|this paper|this file)",
    re.IGNORECASE,
)


class RAGPipeline:
    """Orchestrates document parsing, embedding generation, vector indexing, and retrieval."""

    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        embeddings: Optional[EmbeddingsClient] = None,
        retriever: Optional[VectorStoreClient] = None,
    ):
        self.parser = parser or DocumentParser()
        self.embeddings = embeddings or EmbeddingsClient()
        self.retriever = retriever or VectorStoreClient(dimension=self.embeddings.dimension)
        self.indexed_documents: Dict[str, StructuredDocument] = {}

    def process_documents(self, file_paths: List[str | Path]) -> int:
        """Parses, embeds, and indexes a list of PDF document paths."""
        total_chunks_indexed = 0

        for file_path in file_paths:
            path = Path(file_path)
            structured_doc, chunks = self.parser.parse_pdf(path)
            self.indexed_documents[path.name] = structured_doc

            # Extract chunk text list for batch contextualized embedding
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings_list = self.embeddings.embed_documents(chunk_texts)

            # Store in vector database
            inserted = self.retriever.insert_chunks(chunks, embeddings_list)
            total_chunks_indexed += inserted

        return total_chunks_indexed

    def retrieve_context(self, query: str, top_k: int = 5, score_threshold: float = 0.05) -> List[Dict[str, Any]]:
        """Given a search query, embeds it and retrieves the top-k matching chunks.

        For summary/overview intents the similarity threshold is bypassed so that
        the top-k chunks are always returned when documents are indexed, regardless
        of the cosine score produced by the local hash-based fallback embedder.
        """
        if self.retriever.count() == 0:
            return []

        # Detect summary intent — bypass threshold so generic queries always surface content
        is_summary_query = bool(_SUMMARY_INTENT_PATTERNS.search(query))
        effective_threshold = 0.0 if is_summary_query else score_threshold

        query_vector = self.embeddings.embed_query(query)
        results = self.retriever.search(query_vector, top_k=top_k, score_threshold=effective_threshold)

        # For summary intent, if similarity-ranked results are all near zero, sort by chunk_index
        # so the answer reads in document order rather than random hash order.
        if is_summary_query and results:
            results.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0)))

        return results

    def reset(self) -> None:
        """Clears indexed documents and resets vector collection."""
        self.retriever.reset_collection()
        self.indexed_documents.clear()

"""Tests for document parsing, embeddings, vector store, and RAG pipeline."""

from pathlib import Path
import pytest
from src.document_processing.doc_parser import DocumentParser
from src.document_processing.schemas import DocumentChunk, StructuredDocument
from src.rag.embeddings import EmbeddingsClient
from src.rag.retriever import VectorStoreClient
from src.rag.rag_pipeline import RAGPipeline


def test_document_parser_on_sample_pdf(sample_pdf_path):
    """Test parsing a valid PDF into structured doc and chunks."""
    parser = DocumentParser()
    doc, chunks = parser.parse_pdf(sample_pdf_path)

    assert isinstance(doc, StructuredDocument)
    assert len(chunks) > 0
    assert chunks[0].source_file == sample_pdf_path.name
    assert chunks[0].page_number == 1
    assert "Context Engineering" in chunks[0].text


def test_embeddings_client_deterministic():
    """Test hash-based deterministic fallback (no external dependencies)."""
    client = EmbeddingsClient(provider="local_fallback", dimension=128)
    chunks = ["Multi-agent context orchestration", "Vector similarity retrieval"]
    vectors = client.embed_documents(chunks)

    assert len(vectors) == 2
    assert len(vectors[0]) == 128
    assert len(vectors[1]) == 128

    query_vec = client.embed_query("Multi-agent systems")
    assert len(query_vec) == 128


def test_embeddings_client_sklearn_tfidf():
    """Test sklearn TF-IDF backend: correct dimensions and meaningful cosine ordering."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    client = EmbeddingsClient(provider="sklearn_tfidf", dimension=256)
    chunks = [
        "Quarterly earnings increased by 15 percent versus last year.",
        "Global inflation remained steady at 3.2 percent annually.",
    ]
    vectors = client.embed_documents(chunks)

    assert len(vectors) == 2
    assert len(vectors[0]) == 256, f"Expected 256 dims, got {len(vectors[0])}"

    # Query about earnings should score higher against chunk 0 than chunk 1
    query_vec = client.embed_query("quarterly earnings growth")
    assert len(query_vec) == 256

    scores = cosine_similarity([query_vec], vectors)[0]
    assert scores[0] > scores[1], (
        f"Expected earnings chunk (score={scores[0]:.4f}) > "
        f"inflation chunk (score={scores[1]:.4f})"
    )


def test_vector_store_client(tmp_path):
    """Test inserting and searching vector chunks."""
    db_file = tmp_path / "test_vec.db"
    store = VectorStoreClient(db_path=db_file, dimension=64)
    chunks = [
        DocumentChunk(text="Earnings increased by 15 percent", page_number=1, chunk_index=0, source_file="doc1.pdf"),
        DocumentChunk(text="Global inflation remained steady", page_number=2, chunk_index=1, source_file="doc1.pdf"),
    ]
    emb_client = EmbeddingsClient(provider="local_fallback", dimension=64)
    embeddings = emb_client.embed_documents([c.text for c in chunks])

    inserted = store.insert_chunks(chunks, embeddings)
    assert inserted == 2
    assert store.count() == 2

    # Query search
    query_vec = emb_client.embed_query("earnings growth")
    results = store.search(query_vec, top_k=1)
    assert len(results) == 1
    assert results[0]["page_number"] == 1
    assert "Earnings" in results[0]["text"]


def test_rag_pipeline_end_to_end(sample_pdf_path, tmp_path):
    """Test end-to-end RAG pipeline from PDF to retrieved context."""
    db_file = tmp_path / "rag_pipeline_test.db"
    store = VectorStoreClient(db_path=db_file, dimension=128)
    emb_client = EmbeddingsClient(provider="local_fallback", dimension=128)
    pipeline = RAGPipeline(embeddings=emb_client, retriever=store)

    # Empty store retrieval test (FR-106)
    empty_results = pipeline.retrieve_context("query")
    assert empty_results == []

    # Process document
    indexed_count = pipeline.process_documents([sample_pdf_path])
    assert indexed_count > 0

    # Retrieve context
    results = pipeline.retrieve_context("context engineering", top_k=2)
    assert len(results) > 0
    assert results[0]["source_file"] == sample_pdf_path.name

"""Tests for end-to-end ResearchAssistantFlow."""

from pathlib import Path
from src.memory.memory import MemoryManager
from src.rag.embeddings import EmbeddingsClient
from src.rag.retriever import VectorStoreClient
from src.rag.rag_pipeline import RAGPipeline
from src.workflows.flow import ResearchAssistantFlow


def test_research_assistant_flow_e2e(sample_pdf_path, tmp_path):
    """Test full 4-stage pipeline execution end-to-end."""
    # Setup test components
    db_file = tmp_path / "flow_test_vec.db"
    mem_file = tmp_path / "flow_test_mem.db"

    store = VectorStoreClient(db_path=db_file, dimension=128)
    emb_client = EmbeddingsClient(provider="local_fallback", dimension=128)
    rag = RAGPipeline(embeddings=emb_client, retriever=store)
    rag.process_documents([sample_pdf_path])

    mem = MemoryManager(user_id="test_user", thread_id="test_thread", db_path=mem_file)

    flow = ResearchAssistantFlow(rag_pipeline=rag, memory_manager=mem)

    # Execute query that exists in the document
    result = flow.kickoff("What did empirical results show about latency reduction?")

    assert result["status"] == "OK"
    assert "answer" in result
    assert len(result["citations"]) > 0
    assert "evaluation" in result
    assert "raw_sources" in result
    assert "RAG" in result["raw_sources"]

    # Verify memory recorded turns
    assert mem.count_turns() == 2  # 1 user + 1 assistant

"""Sidebar UI component for document ingestion and assistant status (FR-701, FR-704)."""

import os
import tempfile
from pathlib import Path
import streamlit as st

from .session import reset_chat


def render_sidebar() -> None:
    """Renders the document management sidebar and status monitors."""
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.markdown(
            "**Multi-Source Context Engineering**\n\n"
            "Parallel Context Gathering • Evaluator Filtering • Strict Grounding"
        )
        st.divider()

        # Status Badge
        has_docs = len(st.session_state.indexed_documents) > 0
        if has_docs:
            st.success("🟢 **Assistant Ready** (Document Indexed)")
        else:
            st.warning("🟡 **Awaiting Document** (Upload PDF to begin)")

        st.subheader("📄 Ingest Document")
        uploaded_file = st.file_uploader(
            "Upload a research paper or financial report (PDF only)",
            type=["pdf"],
            help="PDF document to be parsed, contextualized, and indexed into the vector database.",
        )

        if uploaded_file is not None:
            doc_name = uploaded_file.name
            if doc_name not in st.session_state.indexed_documents:
                progress_bar = st.progress(0, text="1/4: Uploading PDF...")
                
                # Save file to temporary storage for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = Path(tmp_file.name)

                try:
                    progress_bar.progress(25, text="2/4: Parsing document structure & sections...")
                    # Process document in pipeline
                    rag_pipe = st.session_state.rag_pipeline
                    progress_bar.progress(50, text="3/4: Computing contextualized embeddings...")
                    chunk_count = rag_pipe.process_documents([tmp_path])
                    
                    progress_bar.progress(85, text="4/4: Populating vector database index...")
                    st.session_state.indexed_documents.append(doc_name)
                    progress_bar.progress(100, text="Indexing complete!")
                    st.toast(f"Indexed {chunk_count} chunks from '{doc_name}'!", icon="✅")
                    st.rerun()

                except Exception as exc:
                    st.error(f"Ingestion failed: {str(exc)}")
                finally:
                    if tmp_path.exists():
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

        # Quick-load sample document for testing
        sample_path = Path("data/sample_research_paper.pdf")
        if sample_path.exists() and "sample_research_paper.pdf" not in st.session_state.indexed_documents:
            if st.button("📥 Load Sample Research Paper", use_container_width=True, help="Load and index the included sample research paper for quick testing"):
                with st.spinner("Indexing sample research paper..."):
                    try:
                        rag_pipe = st.session_state.rag_pipeline
                        chunk_count = rag_pipe.process_documents([sample_path])
                        st.session_state.indexed_documents.append("sample_research_paper.pdf")
                        st.toast(f"Indexed {chunk_count} chunks from sample paper!", icon="✅")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Sample ingestion failed: {str(exc)}")

        # Display indexed files
        if st.session_state.indexed_documents:
            st.markdown("### 📚 Indexed Files")
            for doc in st.session_state.indexed_documents:
                st.caption(f"• `{doc}`")

        st.divider()

        # Chat Reset Control (FR-704)
        if st.button("🔄 Reset Conversation History", use_container_width=True):
            reset_chat()
            st.toast("Conversation history cleared!", icon="🧹")
            st.rerun()

        st.caption(f"Session ID: `{st.session_state.session_id}`")

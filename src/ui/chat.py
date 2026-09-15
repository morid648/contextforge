"""Chat interface component for user queries and research dialogue (FR-702, FR-703)."""

from typing import Any, Dict
import streamlit as st

from .citations_drawer import render_citations_drawer


def render_chat_interface() -> None:
    """Renders the interactive chat history and query submission input."""
    st.header("💬 Multi-Source Context Research Assistant")
    st.caption("Ask questions grounded across Document RAG, Conversation Memory, Live Web, and Academic Literature.")

    has_docs = len(st.session_state.indexed_documents) > 0

    # Display prior conversation history
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            if role == "assistant" and "turn_data" in msg:
                render_citations_drawer(msg["turn_data"])

    # If no documents have been indexed yet, guide input (FR-702)
    if not has_docs:
        st.info("👈 Please upload and index a PDF document in the sidebar, or click below to load the included sample research paper.")
        from pathlib import Path
        sample_path = Path("data/sample_research_paper.pdf")
        if sample_path.exists():
            if st.button("📥 Load Sample Research Paper Now", type="primary", use_container_width=True):
                with st.spinner("Indexing sample research paper..."):
                    try:
                        rag_pipe = st.session_state.rag_pipeline
                        chunk_count = rag_pipe.process_documents([sample_path])
                        st.session_state.indexed_documents.append("sample_research_paper.pdf")
                        st.toast(f"Indexed {chunk_count} chunks from sample paper!", icon="✅")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Sample ingestion failed: {str(exc)}")
        st.chat_input("Chat disabled until a document is indexed...", disabled=True)
        return

    # Active chat input
    user_query = st.chat_input("Ask a research or factual question about the document...")
    if user_query:
        # Append user turn
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process through 4-stage multi-agent flow
        with st.chat_message("assistant"):
            with st.spinner("Gathering context in parallel across RAG, Memory, Web & ArXiv..."):
                try:
                    flow = st.session_state.flow
                    turn_data = flow.kickoff(user_query)
                    answer = turn_data.get("answer", "")
                    st.markdown(answer)
                    render_citations_drawer(turn_data)

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "turn_data": turn_data,
                    })

                except Exception as exc:
                    err_msg = f"An error occurred while executing the research flow: {str(exc)}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                    })

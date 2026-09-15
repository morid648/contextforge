"""Session state management for the Streamlit research assistant (FR-701, FR-704)."""

import uuid
from typing import Any, Dict
import streamlit as st

from ..rag.rag_pipeline import RAGPipeline
from ..workflows.flow import ResearchAssistantFlow


def init_session() -> None:
    """Initializes persistent session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()

    if "flow" not in st.session_state:
        st.session_state.flow = ResearchAssistantFlow(rag_pipeline=st.session_state.rag_pipeline)

    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = []


def reset_chat() -> None:
    """Resets the chat conversation history without dropping the indexed documents (FR-704)."""
    st.session_state.messages = []
    if "flow" in st.session_state and hasattr(st.session_state.flow, "memory_manager"):
        st.session_state.flow.memory_manager.reset()

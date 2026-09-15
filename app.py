"""Streamlit application entry point for Multi-Agent Context Engineering Research Assistant."""

import streamlit as st
from dotenv import load_dotenv

from src.ui.session import init_session
from src.ui.sidebar import render_sidebar
from src.ui.chat import render_chat_interface

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Context Engineering Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stMetric {
        background-color: rgba(128, 128, 128, 0.05);
        padding: 8px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
init_session()

# Render application UI
render_sidebar()
render_chat_interface()

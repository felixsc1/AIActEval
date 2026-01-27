"""
EU AI Act Compliance Testing POC - Streamlit Application

Main application entry point for multipage Streamlit app.
"""

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules for session state initialization
from dataset_handler import load_dataset

# Page configuration
st.set_page_config(
    page_title="EU AI Act Compliance Testing",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state (shared across all pages)
if 'dataset' not in st.session_state:
    st.session_state.dataset = load_dataset()

if 'last_evaluation' not in st.session_state:
    st.session_state.last_evaluation = None

if 'utility_bias_results' not in st.session_state:
    st.session_state.utility_bias_results = None

# Footer
st.markdown("---")
st.markdown("""
**EU AI Act Compliance Testing POC** | Built with Streamlit, DeepEval, and Ollama

*Focus: Bias detection using configurable GPT-5.x judge models*
""")


if __name__ == "__main__":
    pass  # Pages are handled automatically by Streamlit
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

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="ProbeAI - EU AI Act Compliance Testing",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define pages using programmatic navigation API
pages = [
    st.Page("pages/1_📝_Dataset_Management.py", title="Dataset Management", icon="📝"),
    st.Page("pages/2_🧪_Test_Runner.py", title="Test Runner", icon="🧪"),
    st.Page("pages/3_⚖️_Utility_Bias_Testing.py", title="Utility Bias Testing", icon="⚖️"),
    st.Page("pages/4_📊_Utility_Bias_Results.py", title="Utility Bias Results", icon="📊"),
]

# Initialize navigation with HIDDEN position so we can build a custom sidebar
pg = st.navigation(pages, position="hidden")

# --- Custom Sidebar ---
with st.sidebar:
    # 1. Logo at the very top (full width)
    st.image("assets/logo.jpg", width="stretch")
    
    # 2. Add some spacing or a separator
    st.markdown("---")
    
    # 3. Manual Navigation Links
    st.markdown("**Navigation**")
    for page in pages:
        st.page_link(page)

# --- End Custom Sidebar ---

# Initialize session state (shared across all pages)
if 'dataset' not in st.session_state:
    st.session_state.dataset = load_dataset()

if 'last_evaluation' not in st.session_state:
    st.session_state.last_evaluation = None

if 'utility_bias_results' not in st.session_state:
    st.session_state.utility_bias_results = None

# Run the selected page
pg.run()

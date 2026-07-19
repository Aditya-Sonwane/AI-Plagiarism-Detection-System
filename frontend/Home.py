"""Home Page - AI Plagiarism Detection System"""

from __future__ import annotations

import streamlit as st

from api_client import (
    apply_page_config,
    inject_global_css,
)

apply_page_config("Home")

st.session_state.setdefault("dark_mode", False)
inject_global_css(st.session_state.dark_mode)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp{
    font-family: "Inter", sans-serif;
}

h1, h2, h3, h4{
    font-weight:700;
}

div[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:12px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.divider()

st.sidebar.markdown("### Supported Files")

st.sidebar.write("• PDF")
st.sidebar.write("• DOCX")
st.sidebar.write("• TXT")

st.title("🛡️ AI Plagiarism Detection System")

st.markdown(
"""
Compare uploaded documents against your private reference library using
semantic search and AI-assisted plagiarism detection.
"""
)

st.divider()

st.subheader("🚀 Quick Start")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
"""
### 📚 Build Library

Upload reference documents that will be used for plagiarism comparison.
"""
    )

with col2:
    st.info(
"""
### 🔍 Scan Document

Compare a document against the indexed reference library.
"""
    )

with col3:
    st.info(
"""
### 📄 View Report

Review plagiarism results and download the PDF report.
"""
    )

st.divider()

st.subheader("✨ Key Features")

left, right = st.columns(2)

with left:
    st.markdown("✅ Semantic Search")
    st.markdown("✅ FAISS Vector Search")

with right:
    st.markdown("✅ Cross-Encoder Reranking")
    st.markdown("✅ LLM Verification")

st.divider()

st.subheader("ℹ️ About")

st.info(
"""
This application compares uploaded documents **only against your private reference library**.

It does **not** search the internet or external databases.

For accurate results, upload reference documents before scanning new documents.
"""
)

st.divider()

st.caption(
    "Built with FastAPI • Streamlit • FAISS • Sentence Transformers • OpenAI / Ollama"
)

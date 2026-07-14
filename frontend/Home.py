# """Home page for the AI Plagiarism Detection System."""
# from __future__ import annotations

# import streamlit as st

# from api_client import apply_page_config, get_client, inject_global_css

# apply_page_config("Home")

# st.session_state.setdefault("dark_mode", True)
# inject_global_css(st.session_state.dark_mode)

# client = get_client()

# st.title("Plagiarism Checker")
# st.caption(
#     "Upload documents, compare them against a reference library, and generate detailed plagiarism reports."
# )

# try:
#     health = client.health()
#     backend_ok = True
# except Exception:
#     backend_ok = False

# if not backend_ok:
#     st.error(
#         "Unable to connect to the backend service. "
#         "Please ensure the FastAPI server is running."
#     )

# st.divider()

# st.subheader("Getting Started")

# st.markdown(
#     """
# 1. **Build a Reference Library**

#    Open **Scan Document** and upload the documents you want to use as reference material.

# 2. **Scan a Document**

#    Upload a document and start the plagiarism scan.

# 3. **Review the Results**

#    Analyze the plagiarism percentage, sentence-level classifications, and download the PDF report.
# """
# )

# st.info(
#     "The system compares documents only with the reference library you have uploaded. "
#     "It does not search the internet or external databases. If the reference library is "
#     "empty, plagiarism results will always be 0% because there are no documents available "
#     "for comparison."
# )

# st.divider()

# with st.expander("System Architecture"):
#     st.code(
#         """
# Upload (PDF / DOCX / TXT)
#         ↓
# Text Extraction
#         ↓
# Sentence Tokenization
#         ↓
# Semantic Chunking
#         ↓
# Sentence Embeddings
#         ↓
# FAISS Vector Search + BM25 Retrieval
#         ↓
# Cross-Encoder Reranking
#         ↓
# LLM Verification
#         ↓
# Classification
#         ↓
# PDF Report Generation
# """,
#         language="text",
#     )

#     if backend_ok:
#         col1, col2, col3 = st.columns(3)

#         with col1:
#             st.metric(
#                 "Embedding Model",
#                 health["embedding_model"].split("/")[-1],
#             )

#         with col2:
#             st.metric(
#                 "LLM Provider",
#                 health["llm_provider"],
#             )

#         with col3:
#             st.metric(
#                 "Indexed Chunks",
#                 health["faiss_vectors_indexed"],
#             )

#     st.caption(
#         "Embedding models, retrieval settings, and LLM providers can be configured from the Settings page."
#     )

# st.divider()

# st.markdown(
#     "**Navigation:** Home → Scan Document → Results → History → Settings"
# )

"""Home page for the AI Plagiarism Detection System."""

from __future__ import annotations

import streamlit as st

from api_client import apply_page_config, get_client, inject_global_css

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
apply_page_config("Home")

st.session_state.setdefault("dark_mode", True)
inject_global_css(st.session_state.dark_mode)

client = get_client()

# -----------------------------------------------------
# Backend Health
# -----------------------------------------------------
try:
    health = client.health()
    backend_ok = True
except Exception:
    backend_ok = False
    health = {}

# -----------------------------------------------------
# Hero Section
# -----------------------------------------------------

st.title("🛡️ AI Plagiarism Detection System")

st.markdown(
    """
Detect **Exact Copy**, **Near Copy**, and **Paraphrased Content**
using **Semantic Search**, **FAISS Vector Database**,
**Cross-Encoder Reranking**, and **LLM Verification**.
"""
)

st.divider()

# # -----------------------------------------------------
# # Backend Status
# # -----------------------------------------------------

# if backend_ok:
#     st.success("🟢 Backend Connected")
# else:
#     st.error(
#         "🔴 Backend Offline\n\nPlease start the FastAPI server before using the application."
#     )

# # -----------------------------------------------------
# # Metrics
# # -----------------------------------------------------

# if backend_ok:

#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         st.metric(
#             "Indexed Chunks",
#             health["faiss_vectors_indexed"],
#         )

#     with col2:
#         st.metric(
#             "Embedding",
#             health["embedding_model"].split("/")[-1],
#         )

#     with col3:
#         st.metric(
#             "LLM",
#             health["llm_provider"].title(),
#         )

#     with col4:
#         st.metric(
#             "Status",
#             "Online",
#         )

# st.divider()

# -----------------------------------------------------
# Quick Start
# -----------------------------------------------------

st.subheader("🚀 Quick Start")

c1, c2, c3 = st.columns(3)

with c1:
    st.info(
        """
### 📚

### Reference Library

Upload documents that will be used as the plagiarism database.
"""
    )

with c2:
    st.info(
        """
### 🔍

### Scan Document

Upload a document to compare against the indexed library.
"""
    )

with c3:
    st.info(
        """
### 📄

### Download Report

Generate a detailed plagiarism analysis PDF.
"""
    )

st.divider()

# -----------------------------------------------------
# Features
# -----------------------------------------------------

st.subheader("✨ Key Features")

left, right = st.columns(2)

with left:
    st.success("Semantic Plagiarism Detection")
    st.success("Sentence Embeddings")
    st.success("FAISS Vector Search")
    st.success("Hybrid Retrieval")

with right:
    st.success("Cross-Encoder Reranking")
    st.success("LLM Verification")
    st.success("Sentence-Level Classification")
    st.success("PDF Report Generation")

st.divider()

# -----------------------------------------------------
# Workflow
# -----------------------------------------------------

st.subheader("⚙️ Workflow")

st.code(
    """
Reference Documents
        │
        ▼
Text Extraction
        │
        ▼
Sentence Tokenization
        │
        ▼
Semantic Embeddings
        │
        ▼
FAISS + BM25 Retrieval
        │
        ▼
Cross Encoder Reranking
        │
        ▼
LLM Verification
        │
        ▼
Plagiarism Classification
        │
        ▼
PDF Report
""",
    language="text",
)

st.divider()

# -----------------------------------------------------
# About
# -----------------------------------------------------

st.subheader("ℹ️ About")

st.info(
    """
This system compares uploaded documents **only against your private reference library**.

It performs semantic similarity search using transformer embeddings,
retrieves the most relevant passages through FAISS and BM25,
reranks them using a Cross-Encoder,
verifies plagiarism with an LLM,
and generates a detailed plagiarism report.

**Internet search is NOT performed.**
"""
)

st.divider()

# -----------------------------------------------------
# Architecture
# -----------------------------------------------------

with st.expander("🏗️ System Architecture"):

    st.code(
        """
Upload (PDF / DOCX / TXT)
        ↓
Text Extraction
        ↓
Sentence Tokenization
        ↓
Semantic Chunking
        ↓
Sentence Embeddings
        ↓
FAISS Vector Search + BM25 Retrieval
        ↓
Cross-Encoder Reranking
        ↓
LLM Verification
        ↓
Classification
        ↓
PDF Report Generation
""",
        language="text",
    )

st.divider()

st.caption(
    "AI Plagiarism Detection System • Semantic Search • FAISS • Cross Encoder • LLM Verification"
)
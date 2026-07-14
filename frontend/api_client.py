"""HTTP client and shared UI utilities for the Streamlit frontend."""
from __future__ import annotations

import os
from typing import Optional

import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.environ.get("PLAG_API_BASE_URL", "http://localhost:8000")


class ApiClient:
    def __init__(self, base_url: str = API_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def upload(self, filename: str, file_bytes: bytes, doc_type: str) -> dict:
        files = {"file": (filename, file_bytes)}
        data = {"doc_type": doc_type}
        r = requests.post(
            f"{self.base_url}/upload",
            files=files,
            data=data,
            timeout=180,
        )
        r.raise_for_status()
        return r.json()

    def scan(
        self,
        document_id: str,
        top_k: Optional[int] = None,
        llm_provider: Optional[str] = None,
        add_to_library: bool = False,
    ) -> dict:
        payload = {
            "document_id": document_id,
            "add_to_library": add_to_library,
        }

        if top_k:
            payload["top_k"] = top_k
        if llm_provider:
            payload["llm_provider"] = llm_provider

        r = requests.post(
            f"{self.base_url}/scan",
            json=payload,
            timeout=900,
        )
        r.raise_for_status()
        return r.json()

    def get_scan(self, scan_id: str) -> dict:
        r = requests.get(f"{self.base_url}/scan/{scan_id}", timeout=30)
        r.raise_for_status()
        return r.json()

    def history(self, limit: int = 50) -> list[dict]:
        r = requests.get(
            f"{self.base_url}/history",
            params={"limit": limit},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def documents(self, doc_type: Optional[str] = None) -> list[dict]:
        params = {"doc_type": doc_type} if doc_type else {}
        r = requests.get(
            f"{self.base_url}/documents",
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def delete_document(self, document_id: str) -> dict:
        r = requests.delete(
            f"{self.base_url}/documents/{document_id}",
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def report_url(self, scan_id: str) -> str:
        return f"{self.base_url}/report/{scan_id}"

    def download_report(self, scan_id: str) -> bytes:
        r = requests.get(self.report_url(scan_id), timeout=60)
        r.raise_for_status()
        return r.content


def get_client() -> ApiClient:
    if "api_client" not in st.session_state:
        st.session_state.api_client = ApiClient()
    return st.session_state.api_client


CLASSIFICATION_COLORS = {
    "Exact Copy": "#dc3545",
    "Near Copy": "#fd7e14",
    "Paraphrased": "#ffc107",
    "Original": "#28a745",
}


def apply_page_config(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} | Plagiarism Detector",
        layout="wide",
    )


def inject_global_css(dark_mode: bool) -> None:
    if dark_mode:
        bg = "#0E1117"
        sidebar = "#161B22"
        card = "#1F2937"
        text = "#F3F4F6"
        border = "#374151"
        input_bg = "#1F2937"
        accent = "#2563EB"
    else:
        bg = "#FFFFFF"
        sidebar = "#F8F9FA"
        card = "#FFFFFF"
        text = "#111827"
        border = "#D1D5DB"
        input_bg = "#FFFFFF"
        accent = "#2563EB"

    st.markdown(
        f"""
<style>

/* =========================
   Main App
========================= */

.stApp {{
    background-color: {bg};
    color: {text};
}}

/* Header */

[data-testid="stHeader"] {{
    background-color: {bg};
}}

/* Sidebar */

[data-testid="stSidebar"] {{
    background-color: {sidebar};
    border-right: 1px solid {border};
}}

[data-testid="stSidebar"] * {{
    color: {text} !important;
}}

/* Metric Cards */

div[data-testid="stMetric"] {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 15px;
}}

/* Custom Cards */

.plag-card {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 15px;
}}

/* Buttons */

.stButton > button {{
    border-radius: 8px;
    border: none;
    background-color: {accent};
    color: white;
}}

.stButton > button:hover {{
    opacity: 0.9;
}}

/* Inputs */

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {{
    background-color: {input_bg};
    color: {text};
    border: 1px solid {border};
}}

/* File uploader */

[data-testid="stFileUploader"] {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 10px;
}}

/* Expanders */

.streamlit-expanderHeader {{
    color: {text};
}}

[data-testid="stExpander"] {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 10px;
}}

/* Tables */

[data-testid="stTable"] {{
    background-color: {card};
}}

/* DataFrames */

[data-testid="stDataFrame"] {{
    background-color: {card};
}}

/* Code Blocks */

pre {{
    background-color: {card} !important;
}}

/* Divider */

hr {{
    border-color: {border};
}}

</style>
""",
        unsafe_allow_html=True,
    )
    
def make_gauge_chart(
    value: float,
    title: str = "Overall Plagiarism %",
) -> go.Figure:
    color = (
        "#28a745"
        if value < 20
        else "#ffc107"
        if value < 50
        else "#dc3545"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 20], "color": "rgba(40,167,69,0.25)"},
                    {"range": [20, 50], "color": "rgba(255,193,7,0.25)"},
                    {"range": [50, 100], "color": "rgba(220,53,69,0.25)"},
                ],
            },
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


def make_pie_chart(scan: dict) -> go.Figure:
    labels = [
        "Exact Copy",
        "Near Copy",
        "Paraphrased",
        "Original",
    ]

    values = [
        scan.get("exact_copy_pct", 0),
        scan.get("near_copy_pct", 0),
        scan.get("paraphrased_pct", 0),
        scan.get("original_pct", 0),
    ]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(
                colors=[CLASSIFICATION_COLORS[label] for label in labels]
            ),
            hole=0.45,
        )
    )

    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def make_history_bar_chart(history: list[dict]) -> go.Figure:
    filenames = [item["id"][:8] for item in history]
    values = [item["overall_plagiarism_pct"] for item in history]

    fig = go.Figure(
        go.Bar(
            x=filenames,
            y=values,
            marker_color="#4c78a8",
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Plagiarism %",
        xaxis_title="Scan ID",
    )

    return fig
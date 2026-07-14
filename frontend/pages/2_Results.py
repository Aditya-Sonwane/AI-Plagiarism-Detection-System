from __future__ import annotations

import streamlit as st

from api_client import (
    apply_page_config,
    get_client,
    inject_global_css,
    make_gauge_chart,
    make_pie_chart,
    CLASSIFICATION_COLORS,
)

apply_page_config("Results")
st.session_state.setdefault("dark_mode", True)
inject_global_css(st.session_state.dark_mode)

client = get_client()

st.title("Scan Results")

default_scan_id = st.session_state.get("last_scan_id", "")

scan_id = st.text_input(
    "Scan ID",
    value=default_scan_id,
    placeholder="Paste a scan ID or run a scan first",
)

if not scan_id:
    st.info("Run a scan on the Scan Document page, or paste an existing Scan ID.")
    st.stop()

try:
    scan = client.get_scan(scan_id)
except Exception as exc:
    st.error(f"Could not load scan '{scan_id}': {exc}")
    st.stop()

if scan.get("warning"):
    st.warning(scan["warning"])

col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(
        make_gauge_chart(scan["overall_plagiarism_pct"]),
        use_container_width=True,
    )

with col2:
    st.plotly_chart(
        make_pie_chart(scan),
        use_container_width=True,
    )

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Sentences", scan["total_sentences"])
m2.metric("Execution Time", f"{scan['execution_time_seconds']:.2f}s")
m3.metric("Embedding Model", scan["embedding_model"].split("/")[-1])
m4.metric("LLM Provider", scan["llm_provider"])

st.divider()

col_dl, _ = st.columns([1, 3])

with col_dl:
    try:
        pdf_bytes = client.download_report(scan_id)

        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"plagiarism_report_{scan_id}.pdf",
            mime="application/pdf",
            type="primary",
        )

    except Exception as exc:
        st.warning(f"Report not available yet: {exc}")

st.divider()

st.subheader("Sentence-Level Matches")

classification_filter = st.multiselect(
    "Filter by Classification",
    options=list(CLASSIFICATION_COLORS.keys()),
    default=["Exact Copy", "Near Copy", "Paraphrased"],
)

matches = [
    m
    for m in scan.get("matches", [])
    if m["classification"] in classification_filter
]

if not matches:
    st.info("No matches found for the selected filters.")

else:
    for m in matches:
        color = CLASSIFICATION_COLORS.get(m["classification"], "#888")

        with st.expander(
            f"[{m['classification']}] "
            f"Sentence #{m['query_sentence_index']} "
            f"(Similarity {m['similarity_score']:.2f}, "
            f"Confidence {m['confidence_score']:.2f})"
        ):
            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>"
                f"{m['classification']}</span>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Query Sentence:** {m['query_sentence']}")

            if m.get("matched_text"):
                st.markdown(f"**Matched Text:** {m['matched_text']}")

            st.markdown(
                f"**Source:** "
                f"{m.get('source_document_name') or 'N/A'} "
                f"(Page {m.get('source_page_number', 'N/A')})"
            )

            if m.get("reason"):
                st.caption(f"Reason: {m['reason']}")
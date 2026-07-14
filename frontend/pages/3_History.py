from __future__ import annotations

import streamlit as st

from api_client import (
    apply_page_config,
    get_client,
    inject_global_css,
    make_history_bar_chart,
)

apply_page_config("History")
st.session_state.setdefault("dark_mode", True)
inject_global_css(st.session_state.dark_mode)

client = get_client()

st.title("Scan History")

try:
    history = client.history(limit=100)
except Exception as exc:
    st.error(f"Could not load history: {exc}")
    st.stop()

if not history:
    st.info("No scans have been run yet.")
    st.stop()

st.plotly_chart(
    make_history_bar_chart(history),
    use_container_width=True,
)

st.divider()

for h in history:
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

    col1.markdown(f"**Scan** `{h['id'][:12]}...`")
    col2.metric("Plagiarism %", f"{h['overall_plagiarism_pct']:.1f}%")
    col3.metric("Sentences", h["total_sentences"])
    col4.metric("Time (s)", f"{h['execution_time_seconds']:.1f}")

    with col5:
        if st.button("View", key=f"view_{h['id']}"):
            st.session_state["last_scan_id"] = h["id"]
            st.switch_page("pages/2_Results.py")

    st.caption(
        f"Document ID: {h['document_id']} · "
        f"Scanned: {h['created_at']}"
    )

    st.divider()
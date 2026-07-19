from __future__ import annotations

import streamlit as st

from api_client import apply_page_config, get_client, inject_global_css

apply_page_config("Settings")
st.session_state.setdefault("dark_mode", False)

client = get_client()

st.title("Settings")

# Appearance

st.subheader("Appearance")

dark_mode = st.toggle(
    "Dark Mode",
    value=st.session_state.dark_mode,
)

st.session_state.dark_mode = dark_mode
inject_global_css(dark_mode)

st.divider()

# Corpus Management

st.subheader("Corpus Management")

try:
    docs = client.documents(doc_type="corpus")

    if docs:
        for d in docs:
            c1, c2 = st.columns([4, 1])

            c1.write(
                f"{d['filename']} — {d['num_pages']} page(s), "
                f"{d['char_count']} characters"
            )

            if c2.button("Delete", key=f"del_{d['id']}"):
                client.delete_document(d["id"])
                st.rerun()
    else:
        st.info("No corpus documents available.")

except Exception as exc:
    st.error(f"Could not load documents: {exc}")

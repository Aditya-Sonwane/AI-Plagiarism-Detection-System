from __future__ import annotations

import time

import streamlit as st

from api_client import apply_page_config, get_client, inject_global_css

apply_page_config("Scan Document")
st.session_state.setdefault("dark_mode", True)
inject_global_css(st.session_state.dark_mode)

client = get_client()

st.title("Check a Document")

tab_corpus, tab_scan = st.tabs(
    ["Step 1 · Reference Library", "Step 2 · Check for Plagiarism"]
)

with tab_corpus:
    st.markdown(
        "Upload the documents you want to check new submissions **against** — for example, "
        "previously submitted work, source articles, or reference material. The checker can "
        "only catch overlap with documents that are in this library."
    )

    corpus_files = st.file_uploader(
        "Drag and drop PDF / DOCX / TXT files here",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="corpus_uploader",
    )

    if st.button("Add to Reference Library", disabled=not corpus_files, type="primary"):
        progress = st.progress(0, text="Starting...")
        results = []

        for i, f in enumerate(corpus_files):
            progress.progress(i / len(corpus_files), text=f"Adding {f.name}...")
            try:
                resp = client.upload(f.name, f.getvalue(), doc_type="corpus")
                results.append((f.name, resp, None))
            except Exception as exc:
                results.append((f.name, None, str(exc)))

            progress.progress((i + 1) / len(corpus_files))

        progress.progress(1.0, text="Done")

        for name, resp, err in results:
            if err:
                st.error(f"{name}: {err}")
            elif resp["is_duplicate"]:
                st.warning(f"{name}: Already in the library.")
            else:
                st.success(f"{name}: Added successfully.")

    st.divider()
    st.subheader("Reference Library")

    try:
        corpus_docs = client.documents(doc_type="corpus")

        if corpus_docs:
            st.dataframe(
                [
                    {
                        "Filename": d["filename"],
                        "Type": d["file_type"],
                        "Pages": d["num_pages"],
                        "Added": d["created_at"],
                    }
                    for d in corpus_docs
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "The reference library is empty. Add at least one document before checking plagiarism."
            )

    except Exception as exc:
        st.error(f"Could not load the library: {exc}")

with tab_scan:
    try:
        library_docs = client.documents(doc_type="corpus")
    except Exception:
        library_docs = None

    if library_docs is not None and len(library_docs) == 0:
        st.warning(
            "The reference library is empty. Results will always be 0% until documents are added."
        )

    st.markdown("Upload the document you want to check.")

    scan_file = st.file_uploader(
        "Drag and drop a PDF / DOCX / TXT file here",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        key="scan_uploader",
    )

    add_to_library = st.checkbox(
        "Add this document to the reference library after scanning",
        value=False,
        help="Useful if future documents should also be compared against this one.",
    )

    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            top_k = st.slider(
                "Matches to consider per sentence",
                1,
                10,
                5,
            )

        with col2:
            llm_provider = st.selectbox(
                "Verification Model",
                ["ollama", "openai", "none"],
                index=0,
            )

    if st.button("Check for Plagiarism", disabled=not scan_file, type="primary"):
        with st.spinner(f"Uploading {scan_file.name}..."):
            upload_resp = client.upload(
                scan_file.name,
                scan_file.getvalue(),
                doc_type="scan",
            )

        document_id = upload_resp["document"]["id"]

        progress_placeholder = st.empty()
        progress_placeholder.progress(0, text="Checking document...")

        start = time.time()

        try:
            scan_result = client.scan(
                document_id,
                top_k=top_k,
                llm_provider=llm_provider,
                add_to_library=add_to_library,
            )

        except Exception as exc:
            progress_placeholder.empty()
            st.error(f"Check failed: {exc}")

        else:
            progress_placeholder.progress(
                1.0,
                text=f"Completed in {time.time() - start:.1f}s",
            )

            st.session_state["last_scan_id"] = scan_result["id"]

            if scan_result.get("warning"):
                st.warning(scan_result["warning"])

            st.success(
                f"Overall Plagiarism: {scan_result['overall_plagiarism_pct']:.1f}%"
            )

            st.page_link(
                "pages/2_Results.py",
                label="View Full Results",
            )
import pandas as pd
import requests
import streamlit as st

from shared.client import (
    delete_document,
    get_training_status,
    list_documents,
    update_document_corpus,
    upload_document,
)


st.set_page_config(page_title="LinguaComAI Documents", page_icon="AI", layout="wide")
st.title("Documents")
st.caption("Browse indexed documents, choose which ones belong in the training corpus, and upload new files.")

try:
    training_state = get_training_status()
    trainer_busy = training_state.get("busy", False)
except requests.RequestException as exc:
    trainer_busy = False
    st.warning(f"Could not load trainer status: {exc}")

left_col, right_col = st.columns([2.2, 1])

with left_col:
    st.subheader("Stored documents")
    try:
        documents = list_documents()
    except requests.RequestException as exc:
        st.error(f"Could not load documents: {exc}")
        if getattr(exc, "response", None) is not None:
            st.code(exc.response.text)
        documents = []

    if not documents:
        st.info("No documents have been uploaded yet.")
    else:
        documents_table = [
            {
                "ID": document["id"],
                "Filename": document["filename"],
                "Chunks": document["chunk_count"],
                "In corpus": "Yes" if document["include_in_corpus"] else "No",
                "Uploaded": pd.to_datetime(document["uploaded_at"]).strftime("%d-%m-%Y"),
            }
            for document in documents
        ]
        documents_df = pd.DataFrame(documents_table)
        selection_state = st.dataframe(
            documents_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode=["multi-row"],
        )

        selected_rows = selection_state.selection.rows if selection_state else []
        selected_documents = [documents[row_index] for row_index in selected_rows]

        if selected_documents:
            st.caption(f"{len(selected_documents)} document(s) selected.")
        else:
            st.caption("Select one or more rows to remove documents or change corpus inclusion.")

        action_col, include_col, exclude_col = st.columns([1, 1, 1])
        with action_col:
            if st.button(
                "Remove selected",
                use_container_width=True,
                disabled=trainer_busy or not selected_documents,
            ):
                try:
                    for document in selected_documents:
                        delete_document(document["id"])
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Delete failed: {exc}")
                    if getattr(exc, "response", None) is not None:
                        st.code(exc.response.text)
        with include_col:
            if st.button(
                "Include selected in corpus",
                use_container_width=True,
                disabled=trainer_busy or not selected_documents,
            ):
                try:
                    for document in selected_documents:
                        update_document_corpus(document["id"], True)
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Corpus update failed: {exc}")
                    if getattr(exc, "response", None) is not None:
                        st.code(exc.response.text)
        with exclude_col:
            if st.button(
                "Exclude selected from corpus",
                use_container_width=True,
                disabled=trainer_busy or not selected_documents,
            ):
                try:
                    for document in selected_documents:
                        update_document_corpus(document["id"], False)
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Corpus update failed: {exc}")
                    if getattr(exc, "response", None) is not None:
                        st.code(exc.response.text)

with right_col:
    st.subheader("Add document")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .docx files",
        type=["txt", "md", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Add documents", use_container_width=True, disabled=trainer_busy):
        try:
            results = [upload_document(uploaded_file) for uploaded_file in uploaded_files]
            total_chunks = sum(result["chunks_indexed"] for result in results)
            st.success(f"Indexed {len(results)} documents and {total_chunks} chunks.")
            st.rerun()
        except UnicodeDecodeError:
            st.error("Text files must be UTF-8 encoded.")
        except requests.RequestException as exc:
            st.error(f"Document upload failed: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

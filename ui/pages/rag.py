import requests
import streamlit as st

from shared.client import get_rag_settings, update_rag_settings


st.set_page_config(page_title="LinguaComAI RAG", page_icon="AI", layout="wide")
st.title("RAG Adjustment")
st.caption("Tune live retrieval behavior for chat. These settings affect query-time document fetching immediately.")

try:
    rag_settings = get_rag_settings()
except requests.RequestException as exc:
    st.error(f"Could not load RAG settings: {exc}")
    if getattr(exc, "response", None) is not None:
        st.code(exc.response.text)
    st.stop()

left_col, right_col = st.columns([1.4, 1])

with left_col:
    with st.form("rag_settings_form"):
        top_k = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=12,
            value=int(rag_settings.get("top_k", 4)),
            help="Higher values bring in more context, but also increase noise and prompt size.",
        )
        min_score = st.slider(
            "Minimum similarity",
            min_value=0.0,
            max_value=1.0,
            value=float(rag_settings.get("min_score", 0.2)),
            step=0.01,
            help="Higher values are stricter and drop weak matches. Lower values retrieve more aggressively.",
        )
        small_talk_bypass = st.toggle(
            "Skip retrieval for obvious small talk",
            value=bool(rag_settings.get("small_talk_bypass", True)),
            help="Greetings like 'hello' or 'thanks' can be answered without searching the document store.",
        )

        save_clicked = st.form_submit_button("Save RAG settings", use_container_width=True)

    if save_clicked:
        try:
            update_rag_settings(
                {
                    "top_k": top_k,
                    "min_score": min_score,
                    "small_talk_bypass": small_talk_bypass,
                }
            )
            st.success("RAG settings saved.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not save RAG settings: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

with right_col:
    st.subheader("What These Do")
    st.markdown(
        """
`Documents to retrieve` controls how many chunks the retriever asks for before score filtering.

`Minimum similarity` prevents weak matches from being included just because they were the nearest available chunks.

`Skip retrieval for obvious small talk` bypasses RAG for short greetings and similar chatter.
"""
    )
    st.info(
        "Chunk size and overlap are not on this page yet because changing them would require reindexing stored documents."
    )

import streamlit as st


st.set_page_config(page_title="LinguaComAI", page_icon="AI", layout="wide")
st.title("LinguaComAI")
st.caption("A local workspace for chatting with documents, tuning retrieval, compiling system prompts, and training models.")

st.markdown(
    """
Use the sidebar to move between pages. The app is organized so each area has a single job:
- `Chat`: ask questions against your uploaded document set.
- `Documents`: upload files, inspect indexed content, and choose what belongs in the corpus.
- `RAG Adjustment`: tune retrieval behavior like top-k, similarity threshold, and small-talk bypass.
- `System Prompt`: edit the standing system prompt or compile one from the selected corpus.
- `Models`: inspect installed Ollama and HF models, choose defaults, and manage visibility in chat.
- `Model Training`: run and monitor HF fine-tuning jobs.
"""
)

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Suggested flow")
    st.markdown(
        """
1. Add documents on the `Documents` page.
2. Tune retrieval on `RAG Adjustment` if chat feels too noisy or too sparse.
3. Try questions on `Chat`.
4. Refine behavior on `System Prompt`.
5. Use `Models` and `Model Training` when you want to change runtimes or experiment with fine-tuned models.
"""
    )

with right_col:
    st.subheader("Page guide")
    page_links = [
        ("Open Chat", "pages/chat.py", "Start a conversation with the current runtime and model."),
        ("Open Documents", "pages/documents.py", "Manage uploads and corpus inclusion."),
        ("Open RAG Adjustment", "pages/rag.py", "Tune live retrieval settings."),
        ("Open System Prompt", "pages/system_prompt.py", "Edit or compile the assistant instructions."),
        ("Open Models", "pages/models.py", "Inspect local models and choose defaults."),
        ("Open Model Training", "pages/model_training.py", "Launch and monitor training runs."),
    ]
    if hasattr(st, "page_link"):
        for label, path, help_text in page_links:
            st.page_link(path, label=label, help=help_text)
    else:
        for label, _, help_text in page_links:
            st.markdown(f"- **{label.replace('Open ', '')}**: {help_text}")

import requests
import streamlit as st

from shared.client import (
    compile_system_prompt,
    get_models_settings,
    get_system_prompt,
    get_system_prompt_compile_status,
    list_ollama_models,
    update_models_settings,
    update_system_prompt,
)


st.set_page_config(page_title="LinguaComAI System Prompt", page_icon="AI", layout="wide")
st.title("System Prompt")
st.caption("Define the assistant's standing instructions. This prompt is applied to every chat request.")

try:
    system_prompt_state = get_system_prompt()
    current_value = system_prompt_state.get("value", "")
    default_value = system_prompt_state.get("default", "")
except requests.RequestException as exc:
    current_value = ""
    default_value = ""
    st.warning(f"Could not load the current system prompt: {exc}")

try:
    compile_state = get_system_prompt_compile_status()
    compile_busy = compile_state.get("busy", False)
    latest_compile_run = compile_state.get("latest_run")
    current_compile_run = compile_state.get("current_run")
except requests.RequestException as exc:
    compile_busy = False
    latest_compile_run = None
    current_compile_run = None
    st.warning(f"Could not load compiler status: {exc}")

try:
    model_settings = get_models_settings()
    ollama_models_state = list_ollama_models()
    compiler_model_options = [model["name"] for model in ollama_models_state.get("models", [])]
except requests.RequestException as exc:
    model_settings = {}
    compiler_model_options = []
    st.warning(f"Could not load compiler model settings: {exc}")

if "system_prompt_editor" not in st.session_state:
    st.session_state["system_prompt_editor"] = current_value
if "compile_distillation_mode" not in st.session_state:
    st.session_state["compile_distillation_mode"] = "reuse"
if "compile_was_busy" not in st.session_state:
    st.session_state["compile_was_busy"] = compile_busy

status_run = current_compile_run or latest_compile_run


def render_compile_status_panel() -> None:
    try:
        live_compile_state = get_system_prompt_compile_status()
        live_compile_busy = live_compile_state.get("busy", False)
        live_latest_compile_run = live_compile_state.get("latest_run")
        live_current_compile_run = live_compile_state.get("current_run")
    except requests.RequestException as exc:
        st.warning(f"Could not load compiler status: {exc}")
        return

    live_status_run = live_current_compile_run or live_latest_compile_run
    if not live_status_run:
        st.info("No compiler run has been recorded yet.")
        st.session_state["compile_was_busy"] = live_compile_busy
        return

    st.subheader("Compiler status")
    st.info(
        f"Run #{live_status_run['id']} | Status: {live_status_run['status']} | "
        f"Model: {live_status_run.get('model_name') or 'n/a'}"
    )

    total_units = live_status_run.get("total_units") or 0
    completed_units = min(live_status_run.get("completed_units") or 0, total_units) if total_units else 0
    if total_units > 0:
        percentage = (completed_units / total_units) * 100
        st.progress(completed_units / total_units, text=f"Progress: {percentage:.1f}%")

    compile_mode = live_status_run.get("compile_mode")
    if compile_mode == "redistill":
        st.caption("Compilation mode: re-distill all selected documents")
    elif compile_mode == "reuse":
        st.caption("Compilation mode: reuse cached distillations when available")
    if live_status_run.get("message"):
        st.caption(live_status_run["message"])
    if live_status_run.get("error_text"):
        st.error(live_status_run["error_text"])

    stats_bits = []
    if live_status_run.get("documents_selected") is not None:
        stats_bits.append(f"Selected docs: {live_status_run['documents_selected']}")
    if live_status_run.get("documents_used") is not None:
        stats_bits.append(f"Used docs: {live_status_run['documents_used']}")
    if live_status_run.get("reused_distillations") is not None:
        stats_bits.append(f"Reused distillations: {live_status_run['reused_distillations']}")
    if live_status_run.get("new_distillations") is not None:
        stats_bits.append(f"New distillations: {live_status_run['new_distillations']}")
    if live_status_run.get("reduction_rounds") is not None:
        stats_bits.append(f"Reduction rounds: {live_status_run['reduction_rounds']}")
    if stats_bits:
        st.caption(" | ".join(stats_bits))

    if st.session_state.get("compile_was_busy") and not live_compile_busy:
        st.session_state["compile_was_busy"] = False
        st.rerun()

    st.session_state["compile_was_busy"] = live_compile_busy

prompt_editor_col_1, prompt_editor_col_2 = st.columns(2)
with prompt_editor_col_1:
    with st.form("current_system_prompt_form"):
        prompt_value = st.text_area(
            "Current system prompt",
            value=st.session_state["system_prompt_editor"],
            height=280,
            help="These instructions are sent separately from the retrieved context and user question.",
        )
        save_current_clicked = st.form_submit_button("Save")

    if save_current_clicked:
        try:
            update_system_prompt(prompt_value)
            st.session_state["system_prompt_editor"] = prompt_value
            st.success("System prompt saved.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not save the system prompt: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

with prompt_editor_col_2:
    with st.form("compiled_system_prompt_form"):
        prompt_value = st.text_area(
            "Compiled system prompt",
            value=(status_run or {}).get("prompt_draft", ""),
            height=280,
            help="These instructions are sent separately from the retrieved context and user question.",
        )
        compiled_current_clicked = st.form_submit_button(
            "Save",
            disabled=not bool((status_run or {}).get("prompt_draft")),
        )

    if compiled_current_clicked:
        try:
            update_system_prompt(prompt_value)
            st.session_state["system_prompt_editor"] = prompt_value
            st.success("System prompt saved.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not save the system prompt: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

status_container = st.container()
with status_container:
    if hasattr(st, "fragment"):
        @st.fragment(run_every="2s")
        def _compile_status_fragment() -> None:
            render_compile_status_panel()

        _compile_status_fragment()
    else:
        render_compile_status_panel()


action_col, _ = st.columns([1, 2])
with action_col:
    with st.form("compiler_model_settings_form"):
        system_prompt_model = st.selectbox(
            "System prompt compiler model",
            options=compiler_model_options or [model_settings.get("system_prompt_model", "")],
            index=(
                compiler_model_options.index(model_settings.get("system_prompt_model"))
                if model_settings.get("system_prompt_model") in compiler_model_options
                else 0
            )
            if (compiler_model_options or [model_settings.get("system_prompt_model", "")])
            else 0,
        )
        compiler_model_save_clicked = st.form_submit_button("Save compiler model", use_container_width=True)

    if compiler_model_save_clicked:
        try:
            update_models_settings({"system_prompt_model": system_prompt_model})
            st.success("Compiler model saved.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not save the compiler model: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

    compile_mode_label = st.radio(
        "Distillation mode",
        options=["Reuse existing distillations", "Re-distill all documents"],
        index=0 if st.session_state["compile_distillation_mode"] == "reuse" else 1,
        disabled=compile_busy,
    )
    st.session_state["compile_distillation_mode"] = (
        "redistill" if compile_mode_label == "Re-distill all documents" else "reuse"
    )
    if st.button("Compile system prompt from corpus", use_container_width=True, disabled=compile_busy):
        try:
            force_redistill = st.session_state["compile_distillation_mode"] == "redistill"
            result = compile_system_prompt(force_redistill=force_redistill)
            mode_text = "re-distill" if force_redistill else "reuse cached distillations"
            st.info(f"Compilation job #{result['run_id']} queued to {mode_text}.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not compile a system prompt from the corpus: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

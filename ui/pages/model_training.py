import json

import requests
import streamlit as st

from shared.client import abort_training, get_training_log_tail, get_training_status, start_training


st.set_page_config(page_title="LinguaComAI Model Training", page_icon="AI", layout="wide")
st.title("Model Training")
st.caption("Review the current training state, inspect the latest run, and launch a new OFT training job.")

try:
    training_state = get_training_status()
    trainer_busy = training_state.get("busy", False)
    active_model = training_state.get("active_model", "unknown")
    training_status = training_state.get("training_status", "ready")
    selected_count = training_state.get("selected_corpus_documents", 0)
    latest_run = training_state.get("latest_run")
except requests.RequestException as exc:
    training_state = None
    trainer_busy = False
    active_model = "unknown"
    training_status = "unavailable"
    selected_count = 0
    latest_run = None
    st.warning(f"Could not load trainer status: {exc}")

st.info(
    f"Active model: {active_model} | Training status: {training_status} | Corpus documents selected: {selected_count}"
)

action_col, _ = st.columns([1, 2])
with action_col:
    with st.form("training_form"):
        submitted = st.form_submit_button("Train model", use_container_width=True, disabled=trainer_busy)
    if submitted:
        try:
            result = start_training()
            st.success(
                f"OFT run #{result['run_id']} queued with {result['selected_documents']} selected documents."
            )
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not start training: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

    abort_disabled = not trainer_busy or training_status == "aborting"
    if st.button("Abort active run", use_container_width=True, disabled=abort_disabled):
        try:
            result = abort_training()
            stopped_count = len(result.get("stopped_containers", []))
            summary = f"Abort requested for run #{result['run_id']}."
            if stopped_count:
                summary += f" Stopped {stopped_count} active container(s)."
            if result.get("stop_errors"):
                summary += " Some containers could not be stopped immediately."
            st.warning(summary)
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not abort training: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

st.subheader("Latest run")
if not latest_run:
    st.info("No training run has been recorded yet.")
else:
    st.caption(
        f"Run #{latest_run['id']} | Status: {latest_run['status']} | Output model: {latest_run.get('output_model') or 'n/a'}"
    )
    if latest_run.get("message"):
        st.caption(f"Message: {latest_run['message']}")
    if latest_run.get("error_text"):
        st.error(latest_run["error_text"])
    if latest_run.get("log_path"):
        st.caption(f"Log path: {latest_run['log_path']}")

    if latest_run.get("status") in {"failed", "aborted"}:
        try:
            log_tail = get_training_log_tail(50)
        except requests.RequestException as exc:
            st.warning(f"Could not load run log tail: {exc}")
        else:
            lines = log_tail.get("lines", [])
            if lines:
                with st.expander("Last 50 log lines", expanded=True):
                    st.code("\n".join(lines))

    validation_results = latest_run.get("validation_results")
    if validation_results:
        try:
            parsed_validation = json.loads(validation_results)
        except (TypeError, json.JSONDecodeError):
            parsed_validation = None
        if parsed_validation:
            with st.expander("Validation answers", expanded=True):
                for label, value in parsed_validation.items():
                    st.markdown(f"**{label.title()}**")
                    st.write(value)

import json
import time

import requests
import shared.client as client
import streamlit as st

from shared.client import (
    build_prompt,
    generate_hf,
    generate_ollama,
    get_chat_runtime,
    get_system_prompt,
    get_training_status,
    retrieve_context,
    update_chat_model,
    update_chat_runtime,
)


st.set_page_config(page_title="LinguaComAI Chat", page_icon="AI", layout="wide")
st.title("Chat")


def render_status(container, label: str, color: str) -> None:
    container.markdown(
        (
            f"<span style='display:inline-block;padding:0.35rem 0.7rem;border-radius:999px;"
            f"background:{color};color:white;font-weight:600;font-size:0.9rem;'>{label}</span>"
        ),
        unsafe_allow_html=True,
    )


def format_duration_ns(value: int | None) -> str:
    if not value:
        return "0.0s"
    return f"{value / 1_000_000_000:.1f}s"


try:
    training_state = get_training_status()
    trainer_busy = training_state.get("busy", False)
    trainer_status = training_state.get("training_status", "ready")
    system_prompt = get_system_prompt().get("value", "")
    chat_runtime = get_chat_runtime()
except requests.RequestException as exc:
    training_state = None
    trainer_busy = False
    trainer_status = "unavailable"
    system_prompt = ""
    chat_runtime = {"backend": "ollama", "ollama_model": "unknown", "hf_model": "unknown"}
    st.warning(f"Could not load trainer status: {exc}")

backend_label_map = {"ollama": "Ollama", "hf": "Custom HF"}
backend_value_map = {label: value for value, label in backend_label_map.items()}
selected_backend = chat_runtime.get("backend", "ollama")
selected_label = st.radio(
    "Chat runtime",
    options=list(backend_value_map.keys()),
    index=list(backend_value_map.values()).index(selected_backend) if selected_backend in backend_value_map.values() else 0,
    horizontal=True,
)
chosen_backend = backend_value_map[selected_label]
if chosen_backend != selected_backend:
    try:
        chat_runtime = update_chat_runtime(chosen_backend)
        selected_backend = chat_runtime.get("backend", chosen_backend)
        st.rerun()
    except requests.RequestException as exc:
        st.warning(f"Could not update chat runtime: {exc}")

active_model = (
    chat_runtime.get("ollama_model", "unknown")
    if selected_backend == "ollama"
    else chat_runtime.get("hf_model", "unknown")
)
visible_models = (
    chat_runtime.get("visible_ollama_models") or []
    if selected_backend == "ollama"
    else chat_runtime.get("visible_hf_models") or []
)
model_options = visible_models or ([active_model] if active_model != "unknown" else [])
if model_options:
    selected_model = st.selectbox(
        "Chat model",
        options=model_options,
        index=model_options.index(active_model) if active_model in model_options else 0,
    )
    if selected_model != active_model:
        try:
            chat_runtime = update_chat_model(selected_backend, selected_model)
            active_model = selected_model
            st.rerun()
        except requests.RequestException as exc:
            st.warning(f"Could not update chat model: {exc}")
st.caption(f"Active {backend_label_map.get(selected_backend, selected_backend)} model: {active_model}")
if trainer_status == "ready":
    render_status(st.empty(), "Ready for chat", "#16a34a")
elif trainer_status == "failed":
    render_status(st.empty(), "Training failed", "#dc2626")
else:
    render_status(st.empty(), f"Training status: {trainer_status.replace('_', ' ')}", "#d97706")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("timings"):
            timing_parts = []
            if message["timings"].get("retrieval_ns") is not None:
                timing_parts.append(f"Retrieval {format_duration_ns(message['timings']['retrieval_ns'])}")
            if message["timings"].get("generation_ns") is not None:
                timing_parts.append(f"Generation {format_duration_ns(message['timings']['generation_ns'])}")
            if timing_parts:
                st.caption(" | ".join(timing_parts))
        if message.get("sources"):
            with st.expander(f"{len(message['sources'])} documents retrieved"):
                shown = set()
                for source in message["sources"]:
                    label = f"{source['filename']} | {source.get('heading', 'Introduction')}"
                    if label in shown:
                        continue
                    shown.add(label)
                    st.markdown(f"- {label}")


if user_question := st.chat_input(
    "Ask a question about your uploaded documents",
    disabled=trainer_busy,
):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    try:
        retrieval_payload = retrieve_context(user_question)
        if isinstance(retrieval_payload, list):
            matches = retrieval_payload
            retrieval_backend = "legacy"
            retrieval_duration_ns = None
        else:
            matches = retrieval_payload.get("matches", [])
            retrieval_backend = retrieval_payload.get("backend", "unknown")
            retrieval_duration_ns = retrieval_payload.get("request_duration_ns")
        prompt = build_prompt(user_question, matches, st.session_state.messages[:-1])
        stream_failed = False
    except requests.RequestException as exc:
        answer = f"Request failed: {exc}"
        matches = []
        retrieval_backend = "error"
        retrieval_duration_ns = None
        if getattr(exc, "response", None) is not None:
            answer = f"{answer}\n\n{exc.response.text}"
        stream_failed = True

    with st.chat_message("assistant"):
        if stream_failed:
            st.markdown(answer)
        else:
            status_placeholder = st.empty()
            placeholder = st.empty()
            timing_placeholder = st.empty()
            full_response = ""
            generation_duration_ns = None

            try:
                if retrieval_backend == "bypass":
                    render_status(status_placeholder, "Skipping retrieval...", "#6b7280")
                else:
                    render_status(status_placeholder, "Loading model...", "#d97706")
                render_status(status_placeholder, "Generating answer...", "#2563eb")
                generation_started_at = time.perf_counter_ns()
                if selected_backend == "ollama":
                    stream_ollama = getattr(client, "stream_ollama", None)
                    if callable(stream_ollama):
                        with placeholder.container():
                            streamed_response = st.write_stream(
                                stream_ollama(
                                    prompt,
                                    active_model,
                                    system_prompt=system_prompt,
                                    keep_alive=chat_runtime.get("ollama_keep_alive"),
                                )
                            )
                        full_response = (streamed_response or "").strip()
                    else:
                        response_payload = generate_ollama(
                            prompt,
                            active_model,
                            system_prompt=system_prompt,
                            keep_alive=chat_runtime.get("ollama_keep_alive"),
                        )
                        full_response = response_payload.get("response", "").strip()
                        placeholder.markdown(full_response or "No response")
                else:
                    response_payload = generate_hf(prompt, active_model, system_prompt=system_prompt)
                    full_response = response_payload.get("response", "").strip()
                    placeholder.markdown(full_response or "No response")
                generation_duration_ns = time.perf_counter_ns() - generation_started_at
                render_status(status_placeholder, "Answer complete", "#16a34a")
                timing_parts = []
                if retrieval_duration_ns is not None:
                    timing_parts.append(f"Retrieval {format_duration_ns(retrieval_duration_ns)}")
                if generation_duration_ns is not None:
                    timing_parts.append(f"Generation {format_duration_ns(generation_duration_ns)}")
                if timing_parts:
                    timing_placeholder.caption(" | ".join(timing_parts))
                answer = full_response or "No response"
            except requests.RequestException as exc:
                answer = f"Request failed: {exc}"
                if getattr(exc, "response", None) is not None:
                    answer = f"{answer}\n\n{exc.response.text}"
                render_status(status_placeholder, "Request failed", "#dc2626")
                placeholder.markdown(answer)
            except json.JSONDecodeError as exc:
                answer = f"Streaming response could not be decoded: {exc}"
                render_status(status_placeholder, "Streaming failed", "#dc2626")
                placeholder.markdown(answer)

        if matches:
            with st.expander(f"{len(matches)} documents retrieved"):
                shown = set()
                for source in matches:
                    label = f"{source['filename']} | {source.get('heading', 'Introduction')}"
                    if label in shown:
                        continue
                    shown.add(label)
                    st.markdown(f"- {label}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": matches,
            "timings": {
                "retrieval_ns": retrieval_duration_ns,
                "generation_ns": generation_duration_ns if not stream_failed else None,
            },
        }
    )

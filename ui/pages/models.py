import math

import pandas as pd
import requests
import streamlit as st

from shared.client import (
    download_hf_model,
    get_models_settings,
    list_hf_models,
    list_ollama_models,
    update_models_settings,
)


st.set_page_config(page_title="LinguaComAI Models", page_icon="AI", layout="wide")
st.title("Models")
st.caption("Inspect installed models, choose chat defaults, control Ollama keep-warm behavior, and download HF models.")


def format_size(value: int | None) -> str:
    if not value:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    unit_index = min(int(math.log(size, 1024)), len(units) - 1) if size > 0 else 0
    scaled = size / (1024 ** unit_index)
    return f"{scaled:.1f} {units[unit_index]}"


try:
    settings = get_models_settings()
    ollama_state = list_ollama_models()
    hf_state = list_hf_models()
except requests.RequestException as exc:
    st.error(f"Could not load model data: {exc}")
    if getattr(exc, "response", None) is not None:
        st.code(exc.response.text)
    st.stop()

ollama_models = ollama_state.get("models", [])
hf_models = hf_state.get("models", [])

ollama_names = [model["name"] for model in ollama_models]
hf_names = [model["name"] for model in hf_models]

settings_col, download_col = st.columns([1.5, 1])

with settings_col:
    st.subheader("Defaults and routing")
    with st.form("models_settings_form"):
        chat_backend = st.radio(
            "Default chat backend",
            options=["ollama", "hf"],
            index=0 if settings.get("chat_backend", "ollama") == "ollama" else 1,
            format_func=lambda value: "Ollama" if value == "ollama" else "Custom HF",
            horizontal=True,
        )

        active_ollama_model = st.selectbox(
            "Default Ollama chat model",
            options=ollama_names or [settings.get("active_ollama_model", "")],
            index=(
                ollama_names.index(settings.get("active_ollama_model"))
                if settings.get("active_ollama_model") in ollama_names
                else 0
            )
            if (ollama_names or [settings.get("active_ollama_model", "")])
            else 0,
        )

        active_hf_model = st.selectbox(
            "Default HF chat model",
            options=hf_names or [settings.get("active_hf_model", "")],
            index=(
                hf_names.index(settings.get("active_hf_model"))
                if settings.get("active_hf_model") in hf_names
                else 0
            )
            if (hf_names or [settings.get("active_hf_model", "")])
            else 0,
        )

        ollama_keep_alive = st.text_input(
            "Ollama keep-alive",
            value=settings.get("ollama_keep_alive", "30m"),
            help="Examples: `30m`, `5m`, `-1` to keep loaded indefinitely, or `0` to unload immediately after use.",
        )

        visible_ollama = st.multiselect(
            "Ollama models shown in chat",
            options=ollama_names,
            default=settings.get("chat_visible_ollama_models") or ollama_names,
        )
        visible_hf = st.multiselect(
            "HF models shown in chat",
            options=hf_names,
            default=settings.get("chat_visible_hf_models") or hf_names,
        )

        save_clicked = st.form_submit_button("Save model settings", use_container_width=True)

    if save_clicked:
        try:
            update_models_settings(
                {
                    "chat_backend": chat_backend,
                    "active_ollama_model": active_ollama_model,
                    "active_hf_model": active_hf_model,
                    "ollama_keep_alive": ollama_keep_alive,
                    "chat_visible_ollama_models": visible_ollama,
                    "chat_visible_hf_models": visible_hf,
                }
            )
            st.success("Model settings saved.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not save model settings: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

with download_col:
    st.subheader("Download HF model")
    with st.form("download_hf_model_form"):
        model_ref = st.text_input("HF model ID", placeholder="e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        hf_token = st.text_input(
            "HF token (optional)",
            type="password",
            help="Leave blank to use the existing hidden .env token if present. Enter a token here for a one-off download without changing saved settings.",
        )
        download_clicked = st.form_submit_button("Download model", use_container_width=True)

    if download_clicked:
        try:
            result = download_hf_model(model_ref, hf_token or None)
            st.success(f"Downloaded {result['model']} to {result['path']}.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not download model: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

st.subheader("Installed models")
combined_rows = [
    {
        "Backend": "Ollama",
        "Name": model["name"],
        "Size": format_size(model.get("size")),
        "Framework": model.get("framework") or "n/a",
        "Details": model.get("family") or model.get("parameter_size") or "n/a",
        "Path": "",
        "Requires token": "No",
        "Shown in chat": "Yes" if model.get("show_in_chat") else "No",
        "Default chat": "Yes" if model.get("is_active_chat_model") else "No",
        "Loaded now": "n/a",
    }
    for model in ollama_models
]
combined_rows.extend(
    [
        {
            "Backend": "HF",
            "Name": model["name"],
            "Size": format_size(model.get("size_bytes")),
            "Framework": model.get("framework") or "n/a",
            "Details": model.get("repo_id") or "local model",
            "Path": model.get("path") or "n/a",
            "Requires token": "Yes" if model.get("requires_hf_token") else "No",
            "Shown in chat": "Yes" if model["name"] in (settings.get("chat_visible_hf_models") or hf_names) else "No",
            "Default chat": "Yes" if model["name"] == settings.get("active_hf_model") else "No",
            "Loaded now": "Yes" if model.get("path") == hf_state.get("loaded_model") else "No",
        }
        for model in hf_models
    ]
)

if not combined_rows:
    st.info("No models were found.")
else:
    st.dataframe(pd.DataFrame(combined_rows), use_container_width=True, hide_index=True)

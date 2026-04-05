import json
import os
import time
from io import BytesIO
from pathlib import Path

import requests
from docx import Document
from dotenv import load_dotenv


load_dotenv()

FAISS_SERVICE_URL = os.getenv("FAISS_SERVICE_URL", "http://retrieval:8000").rstrip("/")
TRAINER_SERVICE_URL = os.getenv("TRAINER_SERVICE_URL", "http://trainer:8100").rstrip("/")
HF_SERVICE_URL = os.getenv("HF_SERVICE_URL", "http://hf:8200").rstrip("/")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
SYSTEM_PROMPTS_SERVICE_URL = os.getenv("SYSTEM_PROMPTS_SERVICE_URL", "http://system-prompts:8300").rstrip("/")
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def extract_text(file) -> str:
    if file.name.lower().endswith(".docx"):
        document = Document(BytesIO(file.getvalue()))
        blocks = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            style_name = (paragraph.style.name or "").lower()
            if style_name.startswith("heading"):
                blocks.append(f"# {text}")
            else:
                blocks.append(text)

        return "\n\n".join(blocks)

    return file.getvalue().decode("utf-8")


def upload_document(file) -> dict:
    content = extract_text(file)
    response = requests.post(
        f"{FAISS_SERVICE_URL}/documents",
        json={"filename": file.name, "content": content},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def list_documents() -> list[dict]:
    response = requests.get(f"{FAISS_SERVICE_URL}/documents", timeout=30)
    response.raise_for_status()
    return response.json().get("documents", [])


def delete_document(document_id: int) -> dict:
    response = requests.delete(f"{FAISS_SERVICE_URL}/documents/{document_id}", timeout=120)
    response.raise_for_status()
    return response.json()


def update_document_corpus(document_id: int, include_in_corpus: bool) -> dict:
    response = requests.patch(
        f"{FAISS_SERVICE_URL}/documents/{document_id}/corpus",
        json={"include_in_corpus": include_in_corpus},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_system_prompt() -> dict:
    response = requests.get(f"{SYSTEM_PROMPTS_SERVICE_URL}/system-prompt", timeout=30)
    response.raise_for_status()
    return response.json()


def update_system_prompt(value: str) -> dict:
    response = requests.put(
        f"{SYSTEM_PROMPTS_SERVICE_URL}/system-prompt",
        json={"value": value},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def compile_system_prompt(force_redistill: bool = False) -> dict:
    response = requests.post(
        f"{SYSTEM_PROMPTS_SERVICE_URL}/system-prompt/compile",
        json={"force_redistill": force_redistill},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_system_prompt_compile_status() -> dict:
    response = requests.get(f"{SYSTEM_PROMPTS_SERVICE_URL}/system-prompt/compile/status", timeout=30)
    response.raise_for_status()
    return response.json()


def get_chat_runtime() -> dict:
    response = requests.get(f"{FAISS_SERVICE_URL}/settings/chat-runtime", timeout=30)
    response.raise_for_status()
    return response.json()


def update_chat_runtime(backend: str) -> dict:
    response = requests.put(
        f"{FAISS_SERVICE_URL}/settings/chat-runtime",
        json={"backend": backend},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def update_chat_model(backend: str, model: str) -> dict:
    response = requests.put(
        f"{FAISS_SERVICE_URL}/settings/chat-model",
        json={"backend": backend, "model": model},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_models_settings() -> dict:
    response = requests.get(f"{FAISS_SERVICE_URL}/settings/models", timeout=30)
    response.raise_for_status()
    return response.json()


def update_models_settings(payload: dict) -> dict:
    response = requests.put(f"{FAISS_SERVICE_URL}/settings/models", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def get_rag_settings() -> dict:
    response = requests.get(f"{FAISS_SERVICE_URL}/settings/rag", timeout=30)
    response.raise_for_status()
    return response.json()


def update_rag_settings(payload: dict) -> dict:
    response = requests.put(f"{FAISS_SERVICE_URL}/settings/rag", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def list_ollama_models() -> dict:
    response = requests.get(f"{FAISS_SERVICE_URL}/models/ollama", timeout=30)
    response.raise_for_status()
    return response.json()


def list_hf_models() -> dict:
    response = requests.get(f"{HF_SERVICE_URL}/models", timeout=30)
    response.raise_for_status()
    return response.json()


def download_hf_model(model: str, hf_token: str | None = None) -> dict:
    response = requests.post(
        f"{HF_SERVICE_URL}/models/download",
        json={"model": model, "hf_token": hf_token},
        timeout=1800,
    )
    response.raise_for_status()
    return response.json()


def retrieve_context(query: str, top_k: int | None = None) -> dict:
    started_at = time.perf_counter_ns()
    payload = {"query": query}
    if top_k is not None:
        payload["top_k"] = top_k
    response = requests.post(
        f"{FAISS_SERVICE_URL}/query",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    response_payload = response.json()
    response_payload["request_duration_ns"] = time.perf_counter_ns() - started_at
    return response_payload


def get_training_status() -> dict:
    response = requests.get(f"{TRAINER_SERVICE_URL}/status", timeout=30)
    response.raise_for_status()
    return response.json()


def get_training_log_tail(lines: int = 50) -> dict:
    response = requests.get(f"{TRAINER_SERVICE_URL}/log-tail", params={"lines": lines}, timeout=30)
    response.raise_for_status()
    return response.json()


def start_training() -> dict:
    response = requests.post(f"{TRAINER_SERVICE_URL}/train/oft", timeout=30)
    response.raise_for_status()
    return response.json()


def abort_training() -> dict:
    response = requests.post(f"{TRAINER_SERVICE_URL}/train/abort", timeout=30)
    response.raise_for_status()
    return response.json()


def build_prompt(question: str, matches: list[dict], chat_history: list[dict]) -> str:
    context_blocks = []
    for idx, match in enumerate(matches, start=1):
        context_blocks.append(
            f"Source {idx} ({match['filename']} | {match.get('heading', 'Introduction')}):\n{match['content']}"
        )

    history_blocks = []
    for message in chat_history[-6:]:
        history_blocks.append(f"{message['role'].title()}: {message['content']}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant documents were found."
    history_text = "\n".join(history_blocks) if history_blocks else "No prior conversation."
    return load_prompt("chat_prompt_template.txt").format(
        history_text=history_text,
        context_text=context_text,
        question=question,
    )


def generate_hf(prompt: str, model_name: str, system_prompt: str | None = None) -> dict:
    response = requests.post(
        f"{HF_SERVICE_URL}/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": 160,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def generate_ollama(prompt: str, model_name: str, system_prompt: str | None = None, keep_alive: str | None = None) -> dict:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "keep_alive": keep_alive,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def stream_ollama(prompt: str, model_name: str, system_prompt: str | None = None, keep_alive: str | None = None):
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "keep_alive": keep_alive,
        },
        timeout=600,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            payload = json.loads(line)
            chunk = payload.get("response", "")
            if chunk:
                yield chunk

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
import psycopg
import requests
import tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg.rows import dict_row


DEFAULT_GENERATION_MODEL = os.getenv("TRAINING_BASE_MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3:8b")
DEFAULT_SYSTEM_PROMPT_MODEL = os.getenv("OLLAMA_SYSTEM_PROMPT_MODEL", DEFAULT_OLLAMA_MODEL)
DEFAULT_OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
DEFAULT_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
DEFAULT_RAG_SMALL_TALK_BYPASS = os.getenv("RAG_SMALL_TALK_BYPASS", "true").strip().lower() not in {"0", "false", "no"}
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
HF_SERVICE_URL = os.getenv("HF_SERVICE_URL", "http://hf:8200").rstrip("/")
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://linguacomai:linguacomai@postgres:5432/linguacomai",
)
DATA_DIR = Path(os.getenv("FAISS_DATA_DIR", "/data"))
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "/corpus"))
INDEX_PATH = DATA_DIR / "documents.index"
METADATA_PATH = DATA_DIR / "documents.json"
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE_TOKENS", os.getenv("RAG_CHUNK_SIZE", "450")))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", os.getenv("RAG_CHUNK_OVERLAP", "75")))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.2"))
DB_READY_TIMEOUT = int(os.getenv("POSTGRES_READY_TIMEOUT_SECONDS", "60"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LinguaComAI Retrieval Service")
lock = threading.Lock()
index = None
documents: List[dict] = []
encoding = tiktoken.get_encoding("cl100k_base")


class DocumentUpload(BaseModel):
    filename: str
    content: str


class DocumentCorpusUpdate(BaseModel):
    include_in_corpus: bool


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class ChatRuntimeUpdate(BaseModel):
    backend: str


class ChatModelSelectionUpdate(BaseModel):
    backend: str
    model: str


class ModelSettingsUpdate(BaseModel):
    chat_backend: str | None = None
    active_ollama_model: str | None = None
    active_hf_model: str | None = None
    system_prompt_model: str | None = None
    ollama_keep_alive: str | None = None
    chat_visible_ollama_models: list[str] | None = None
    chat_visible_hf_models: list[str] | None = None


class RagSettingsUpdate(BaseModel):
    top_k: int | None = None
    min_score: float | None = None
    small_talk_bypass: bool | None = None


def get_connection():
    return psycopg.connect(POSTGRES_DSN, row_factory=dict_row)


def wait_for_database() -> None:
    deadline = time.time() + DB_READY_TIMEOUT
    last_error = None
    while time.time() < deadline:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return
        except psycopg.Error as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"Postgres did not become ready in time: {last_error}")


def ensure_database_schema() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    corpus_file_path TEXT,
                    include_in_corpus BOOLEAN NOT NULL DEFAULT FALSE,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS corpus_file_path TEXT")
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS include_in_corpus BOOLEAN NOT NULL DEFAULT FALSE"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    heading TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding vector NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id BIGSERIAL PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT,
                    base_model TEXT,
                    output_model TEXT,
                    corpus_snapshot_dir TEXT,
                    output_dir TEXT,
                    log_path TEXT,
                    error_text TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks (document_id)"
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('active_model', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (DEFAULT_GENERATION_MODEL,),
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('training_status', 'ready')
                ON CONFLICT (key) DO NOTHING
                """
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('chat_backend', 'ollama')
                ON CONFLICT (key) DO NOTHING
                """
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('active_ollama_model', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (DEFAULT_OLLAMA_MODEL,),
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('active_hf_model', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (DEFAULT_GENERATION_MODEL,),
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('system_prompt_model', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (DEFAULT_SYSTEM_PROMPT_MODEL,),
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('ollama_keep_alive', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (DEFAULT_OLLAMA_KEEP_ALIVE,),
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('chat_visible_ollama_models', '[]')
                ON CONFLICT (key) DO NOTHING
                """
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('chat_visible_hf_models', '[]')
                ON CONFLICT (key) DO NOTHING
                """
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('rag_top_k', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (str(DEFAULT_RAG_TOP_K),),
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('rag_min_score', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (str(RAG_MIN_SCORE),),
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('rag_small_talk_bypass', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                ("true" if DEFAULT_RAG_SMALL_TALK_BYPASS else "false",),
            )


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def should_bypass_retrieval(query: str) -> bool:
    normalized = normalize_query_text(query)
    if not normalized:
        return True

    exact_small_talk = {
        "hi",
        "hello",
        "hey",
        "yo",
        "thanks",
        "thank you",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "bye",
        "goodbye",
    }
    if normalized in exact_small_talk:
        return True

    tokens = normalized.split()
    if len(tokens) <= 2 and all(token in {"hi", "hello", "hey", "thanks", "bye", "yo"} for token in tokens):
        return True

    return False


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if re.match(r"^\d+(\.\d+)*\s+\S+", stripped):
        return True
    if len(stripped) <= 80 and stripped == stripped.upper() and any(char.isalpha() for char in stripped):
        return True
    return False


def split_structured_sections(text: str) -> List[dict]:
    sections: List[dict] = []
    current_heading = "Introduction"
    current_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if is_heading(line):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({"heading": current_heading, "content": body})
                current_lines = []
            current_heading = normalize_line(line.lstrip("#")) or current_heading
            continue

        if not line.strip():
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"heading": current_heading, "content": body})

    return sections


def split_paragraphs(section_text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", section_text)
    return [normalize_line(paragraph) for paragraph in paragraphs if normalize_line(paragraph)]


def token_count(text: str) -> int:
    return len(encoding.encode(text))


def chunk_by_tokens(text: str, heading: str) -> List[dict]:
    tokens = encoding.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(len(tokens), start + CHUNK_SIZE)
        chunk_text = encoding.decode(tokens[start:end]).strip()
        if chunk_text:
            chunks.append(
                {
                    "heading": heading,
                    "content": chunk_text,
                    "token_count": end - start,
                }
            )
        if end == len(tokens):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def split_text(text: str) -> List[dict]:
    sections = split_structured_sections(text)
    chunks: List[dict] = []

    for section in sections:
        paragraph_buffer: List[str] = []
        paragraph_tokens = 0

        for paragraph in split_paragraphs(section["content"]):
            paragraph_token_count = token_count(paragraph)

            if paragraph_token_count > CHUNK_SIZE:
                if paragraph_buffer:
                    joined = "\n\n".join(paragraph_buffer)
                    chunks.append(
                        {
                            "heading": section["heading"],
                            "content": joined,
                            "token_count": token_count(joined),
                        }
                    )
                    paragraph_buffer = []
                    paragraph_tokens = 0
                chunks.extend(chunk_by_tokens(paragraph, section["heading"]))
                continue

            next_tokens = paragraph_tokens + paragraph_token_count
            if paragraph_buffer and next_tokens > CHUNK_SIZE:
                joined = "\n\n".join(paragraph_buffer)
                chunks.append(
                    {
                        "heading": section["heading"],
                        "content": joined,
                        "token_count": token_count(joined),
                    }
                )

                overlap_buffer: List[str] = []
                overlap_tokens = 0
                for buffered_paragraph in reversed(paragraph_buffer):
                    buffered_tokens = token_count(buffered_paragraph)
                    if overlap_tokens + buffered_tokens > CHUNK_OVERLAP:
                        break
                    overlap_buffer.insert(0, buffered_paragraph)
                    overlap_tokens += buffered_tokens

                paragraph_buffer = overlap_buffer
                paragraph_tokens = overlap_tokens

            paragraph_buffer.append(paragraph)
            paragraph_tokens += paragraph_token_count

        if paragraph_buffer:
            joined = "\n\n".join(paragraph_buffer)
            chunks.append(
                {
                    "heading": section["heading"],
                    "content": joined,
                    "token_count": token_count(joined),
                }
            )

    return [chunk for chunk in chunks if chunk["content"].strip()]


def embed_texts(texts: List[str]) -> np.ndarray:
    response = requests.post(
        f"{HF_SERVICE_URL}/embed",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings")
    if not embeddings:
        raise HTTPException(status_code=502, detail="Embedding model returned no embeddings.")

    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector.tolist()) + "]"


def corpus_file_for_document(document_id: int, filename: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._") or "document"
    return CORPUS_DIR / "documents" / f"{document_id}_{safe_name}.txt"


def write_corpus_file(document_id: int, filename: str, content: str) -> str:
    path = corpus_file_for_document(document_id, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def remove_corpus_file(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if path.exists():
        path.unlink()


def get_setting(key: str, default: str | None = None) -> str | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM app_settings WHERE key = %s", (key,))
            row = cur.fetchone()
    if row is None:
        return default
    return row["value"]


def set_setting(key: str, value: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (key, value),
            )


def get_json_setting(key: str) -> list[str]:
    raw = get_setting(key, "[]") or "[]"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def get_int_setting(key: str, default: int) -> int:
    raw = get_setting(key, str(default))
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default
    return value


def get_float_setting(key: str, default: float) -> float:
    raw = get_setting(key, str(default))
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default
    return value


def get_bool_setting(key: str, default: bool) -> bool:
    raw = get_setting(key, "true" if default else "false")
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def get_model_settings() -> dict:
    return {
        "chat_backend": get_setting("chat_backend", "ollama") or "ollama",
        "active_ollama_model": get_setting("active_ollama_model", DEFAULT_OLLAMA_MODEL) or DEFAULT_OLLAMA_MODEL,
        "active_hf_model": get_setting("active_hf_model", DEFAULT_GENERATION_MODEL) or DEFAULT_GENERATION_MODEL,
        "system_prompt_model": get_setting("system_prompt_model", DEFAULT_SYSTEM_PROMPT_MODEL) or DEFAULT_SYSTEM_PROMPT_MODEL,
        "ollama_keep_alive": get_setting("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE) or DEFAULT_OLLAMA_KEEP_ALIVE,
        "chat_visible_ollama_models": get_json_setting("chat_visible_ollama_models"),
        "chat_visible_hf_models": get_json_setting("chat_visible_hf_models"),
    }


def get_rag_settings() -> dict:
    return {
        "top_k": max(1, get_int_setting("rag_top_k", DEFAULT_RAG_TOP_K)),
        "min_score": max(0.0, min(1.0, get_float_setting("rag_min_score", RAG_MIN_SCORE))),
        "small_talk_bypass": get_bool_setting("rag_small_talk_bypass", DEFAULT_RAG_SMALL_TALK_BYPASS),
    }


def ensure_index(dimension: int) -> None:
    global index
    if index is None:
        index = faiss.IndexFlatIP(dimension)


def save_cache() -> None:
    if index is not None:
        faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(json.dumps(documents, indent=2), encoding="utf-8")


def rebuild_cache_from_database() -> None:
    global index, documents
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    d.filename,
                    c.heading,
                    c.content,
                    c.token_count,
                    c.embedding
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding_model = %s
                ORDER BY c.id
                """,
                (EMBEDDING_MODEL,),
            )
            rows = cur.fetchall()

    if not rows:
        index = None
        documents = []
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        METADATA_PATH.write_text("[]", encoding="utf-8")
        return

    vectors = []
    cache_documents = []
    for row in rows:
        embedding = row["embedding"]
        if isinstance(embedding, str):
            vector = np.array(json.loads(embedding.replace("{", "[").replace("}", "]")), dtype="float32")
        else:
            vector = np.array(embedding, dtype="float32")
        vectors.append(vector)
        cache_documents.append(
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "filename": row["filename"],
                "heading": row["heading"],
                "content": row["content"],
                "token_count": row["token_count"],
            }
        )

    matrix = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(matrix)
    ensure_index(matrix.shape[1])
    if index.d != matrix.shape[1]:
        index = faiss.IndexFlatIP(matrix.shape[1])
    else:
        index.reset()
    index.add(matrix)
    documents = cache_documents
    save_cache()


def search_postgres(query_vector: np.ndarray, top_k: int) -> List[dict]:
    query_embedding = vector_literal(query_vector)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    d.filename,
                    c.heading,
                    c.content,
                    c.token_count,
                    1 - (c.embedding <=> %s::vector) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding_model = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, EMBEDDING_MODEL, query_embedding, top_k),
            )
            rows = cur.fetchall()

    return [
        {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "filename": row["filename"],
            "heading": row["heading"],
            "content": row["content"],
            "token_count": row["token_count"],
            "score": float(row["score"]),
        }
        for row in rows
    ]


def filter_matches_by_score(matches: List[dict], min_score: float) -> List[dict]:
    return [match for match in matches if match.get("score", 0.0) >= min_score]


@app.on_event("startup")
def startup_event() -> None:
    wait_for_database()
    ensure_database_schema()
    rebuild_cache_from_database()


@app.get("/health")
def health() -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM documents")
            document_count = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) AS count FROM document_chunks WHERE embedding_model = %s", (EMBEDDING_MODEL,))
            chunk_count = cur.fetchone()["count"]

    return {
        "status": "ok",
        "documents": document_count,
        "chunks": chunk_count,
        "embedding_model": EMBEDDING_MODEL,
        "faiss_cached_chunks": len(documents),
        "active_model": get_setting("active_model", DEFAULT_GENERATION_MODEL),
        "training_status": get_setting("training_status", "ready"),
    }


@app.get("/documents")
def list_documents() -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.id,
                    d.filename,
                    d.uploaded_at,
                    d.include_in_corpus,
                    d.corpus_file_path,
                    COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.uploaded_at DESC, d.id DESC
                """
            )
            rows = cur.fetchall()

    return {"documents": rows}


@app.get("/settings/chat-runtime")
def get_chat_runtime() -> dict:
    settings = get_model_settings()
    return {
        "backend": settings["chat_backend"],
        "ollama_model": settings["active_ollama_model"],
        "hf_model": settings["active_hf_model"],
        "visible_ollama_models": settings["chat_visible_ollama_models"],
        "visible_hf_models": settings["chat_visible_hf_models"],
        "ollama_keep_alive": settings["ollama_keep_alive"],
    }


@app.put("/settings/chat-runtime")
def update_chat_runtime(update: ChatRuntimeUpdate) -> dict:
    backend = update.backend.strip().lower()
    if backend not in {"ollama", "hf"}:
        raise HTTPException(status_code=400, detail="Chat backend must be either 'ollama' or 'hf'.")

    set_setting("chat_backend", backend)
    return get_chat_runtime()


@app.put("/settings/chat-model")
def update_chat_model(update: ChatModelSelectionUpdate) -> dict:
    backend = update.backend.strip().lower()
    model = update.model.strip()
    if backend not in {"ollama", "hf"}:
        raise HTTPException(status_code=400, detail="Chat backend must be either 'ollama' or 'hf'.")
    if not model:
        raise HTTPException(status_code=400, detail="Model cannot be empty.")

    set_setting("chat_backend", backend)
    if backend == "ollama":
        set_setting("active_ollama_model", model)
    else:
        set_setting("active_hf_model", model)
        set_setting("active_model", model)

    return get_chat_runtime()


@app.get("/settings/models")
def get_models_settings() -> dict:
    return get_model_settings()


@app.put("/settings/models")
def update_models_settings(update: ModelSettingsUpdate) -> dict:
    if update.chat_backend is not None:
        backend = update.chat_backend.strip().lower()
        if backend not in {"ollama", "hf"}:
            raise HTTPException(status_code=400, detail="Chat backend must be either 'ollama' or 'hf'.")
        set_setting("chat_backend", backend)

    if update.active_ollama_model is not None:
        model = update.active_ollama_model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="Active Ollama model cannot be empty.")
        set_setting("active_ollama_model", model)

    if update.active_hf_model is not None:
        model = update.active_hf_model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="Active HF model cannot be empty.")
        set_setting("active_hf_model", model)
        set_setting("active_model", model)

    if update.system_prompt_model is not None:
        model = update.system_prompt_model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="System prompt model cannot be empty.")
        set_setting("system_prompt_model", model)

    if update.ollama_keep_alive is not None:
        value = update.ollama_keep_alive.strip()
        if not value:
            raise HTTPException(status_code=400, detail="Ollama keep-alive cannot be empty.")
        set_setting("ollama_keep_alive", value)

    if update.chat_visible_ollama_models is not None:
        set_setting("chat_visible_ollama_models", json.dumps(update.chat_visible_ollama_models))

    if update.chat_visible_hf_models is not None:
        set_setting("chat_visible_hf_models", json.dumps(update.chat_visible_hf_models))

    return get_model_settings()


@app.get("/settings/rag")
def read_rag_settings() -> dict:
    return get_rag_settings()


@app.put("/settings/rag")
def update_rag_settings(update: RagSettingsUpdate) -> dict:
    if update.top_k is not None:
        if update.top_k < 1:
            raise HTTPException(status_code=400, detail="RAG top-k must be at least 1.")
        set_setting("rag_top_k", str(update.top_k))

    if update.min_score is not None:
        if update.min_score < 0 or update.min_score > 1:
            raise HTTPException(status_code=400, detail="RAG minimum score must be between 0 and 1.")
        set_setting("rag_min_score", str(update.min_score))

    if update.small_talk_bypass is not None:
        set_setting("rag_small_talk_bypass", "true" if update.small_talk_bypass else "false")

    return get_rag_settings()


@app.get("/models/ollama")
def list_ollama_models() -> dict:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama models could not be loaded: {exc}") from exc

    settings = get_model_settings()
    visible_models = set(settings["chat_visible_ollama_models"])
    payload = response.json()
    models = []
    for model in payload.get("models", []):
        details = model.get("details") or {}
        name = model.get("name", "")
        models.append(
            {
                "name": name,
                "size": model.get("size"),
                "modified_at": model.get("modified_at"),
                "format": details.get("format"),
                "family": details.get("family"),
                "families": details.get("families") or [],
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
                "backend": "ollama",
                "framework": "gguf" if details.get("format") else "ollama",
                "requires_hf_token": False,
                "show_in_chat": name in visible_models if visible_models else True,
                "is_active_chat_model": name == settings["active_ollama_model"],
                "is_system_prompt_model": name == settings["system_prompt_model"],
            }
        )

    return {
        "models": models,
        "keep_alive": settings["ollama_keep_alive"],
        "active_model": settings["active_ollama_model"],
        "system_prompt_model": settings["system_prompt_model"],
    }


@app.post("/documents")
def upload_document(document: DocumentUpload) -> dict:
    text = document.content.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Document is empty.")

    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document does not contain usable text.")

    try:
        vectors = embed_texts([chunk["content"] for chunk in chunks])
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    with lock:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO documents (filename, content)
                        VALUES (%s, %s)
                        RETURNING id
                        """,
                        (document.filename, document.content),
                    )
                    document_id = cur.fetchone()["id"]
                    corpus_file_path = write_corpus_file(document_id, document.filename, document.content)
                    cur.execute(
                        "UPDATE documents SET corpus_file_path = %s WHERE id = %s",
                        (corpus_file_path, document_id),
                    )

                    chunk_rows = []
                    for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                        cur.execute(
                            """
                            INSERT INTO document_chunks (
                                document_id,
                                chunk_index,
                                heading,
                                content,
                                token_count,
                                embedding_model,
                                embedding
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                            RETURNING id
                            """,
                            (
                                document_id,
                                chunk_index,
                                chunk["heading"],
                                chunk["content"],
                                chunk["token_count"],
                                EMBEDDING_MODEL,
                                vector_literal(vector),
                            ),
                        )
                        chunk_id = cur.fetchone()["id"]
                        chunk_rows.append(
                            {
                                "chunk_id": chunk_id,
                                "document_id": document_id,
                                "filename": document.filename,
                                "heading": chunk["heading"],
                                "content": chunk["content"],
                                "token_count": chunk["token_count"],
                            }
                        )

            ensure_index(vectors.shape[1])
            if index is None or index.d != vectors.shape[1]:
                rebuild_cache_from_database()
            else:
                index.add(vectors)
                documents.extend(chunk_rows)
                save_cache()
        except psycopg.Error as exc:
            raise HTTPException(status_code=502, detail=f"Postgres write failed: {exc}") from exc

    return {
        "document_id": document_id,
        "chunks_indexed": len(chunks),
        "filename": document.filename,
        "embedding_model": EMBEDDING_MODEL,
        "corpus_file_path": corpus_file_path,
    }


@app.patch("/documents/{document_id}/corpus")
def update_document_corpus(document_id: int, update: DocumentCorpusUpdate) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET include_in_corpus = %s
                WHERE id = %s
                RETURNING id, filename, include_in_corpus
                """,
                (update.include_in_corpus, document_id),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Document not found.")

    return row


@app.delete("/documents/{document_id}")
def delete_document(document_id: int) -> dict:
    with lock:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT filename, corpus_file_path FROM documents WHERE id = %s",
                        (document_id,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise HTTPException(status_code=404, detail="Document not found.")

                    filename = row["filename"]
                    corpus_file_path = row["corpus_file_path"]
                    cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        except psycopg.Error as exc:
            raise HTTPException(status_code=502, detail=f"Postgres delete failed: {exc}") from exc

        remove_corpus_file(corpus_file_path)
        rebuild_cache_from_database()

    return {"deleted": True, "document_id": document_id, "filename": filename}


@app.post("/query")
def query_documents(request: QueryRequest) -> dict:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag_settings = get_rag_settings()

    if rag_settings["small_talk_bypass"] and should_bypass_retrieval(request.query):
        return {"matches": [], "backend": "bypass", "reason": "small_talk"}

    try:
        query_vector = embed_texts([request.query])[0]
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    with lock:
        top_k = max(1, request.top_k if request.top_k is not None else rag_settings["top_k"])
        if index is None or not documents:
            matches = filter_matches_by_score(
                search_postgres(query_vector, top_k),
                rag_settings["min_score"],
            )
            return {"matches": matches, "backend": "pgvector"}

        search_k = min(top_k, len(documents))
        scores, ids = index.search(np.expand_dims(query_vector, axis=0), search_k)

        matches = []
        for score, doc_index in zip(scores[0], ids[0]):
            if doc_index < 0:
                continue
            doc = documents[doc_index]
            matches.append(
                {
                    "chunk_id": doc["chunk_id"],
                    "document_id": doc["document_id"],
                    "filename": doc["filename"],
                    "heading": doc.get("heading", "Introduction"),
                    "content": doc["content"],
                    "token_count": doc.get("token_count", 0),
                    "score": float(score),
                }
            )

    return {"matches": filter_matches_by_score(matches, rag_settings["min_score"]), "backend": "faiss"}

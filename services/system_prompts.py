import json
import os
import threading
import time
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from psycopg.rows import dict_row


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://linguacomai:linguacomai@postgres:5432/linguacomai",
)
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3:8b")
SYSTEM_PROMPT_OLLAMA_MODEL = os.getenv("OLLAMA_SYSTEM_PROMPT_MODEL", DEFAULT_OLLAMA_MODEL)
SYSTEM_PROMPT_COMPILATION_CHAR_LIMIT = int(os.getenv("SYSTEM_PROMPT_COMPILATION_CHAR_LIMIT", "24000"))
SYSTEM_PROMPT_BATCH_CHAR_LIMIT = int(os.getenv("SYSTEM_PROMPT_BATCH_CHAR_LIMIT", "9000"))
SYSTEM_PROMPT_DOCUMENT_CHAR_LIMIT = int(os.getenv("SYSTEM_PROMPT_DOCUMENT_CHAR_LIMIT", "12000"))
SYSTEM_PROMPT_DISTILL_MAX_TOKENS = int(os.getenv("SYSTEM_PROMPT_DISTILL_MAX_TOKENS", "220"))
SYSTEM_PROMPT_FINAL_MAX_TOKENS = int(os.getenv("SYSTEM_PROMPT_FINAL_MAX_TOKENS", "320"))
DB_READY_TIMEOUT = int(os.getenv("POSTGRES_READY_TIMEOUT_SECONDS", "60"))
DISTILLATION_MAX_RETRIES = int(os.getenv("SYSTEM_PROMPT_DISTILL_MAX_RETRIES", "30"))
MAX_REDUCTION_ROUNDS = int(os.getenv("SYSTEM_PROMPT_MAX_REDUCTION_ROUNDS", "12"))

app = FastAPI(title="LinguaComAI System Prompts Service")
compile_lock = threading.Lock()
active_compile_thread = None
active_compile_run_id = None


class SystemPromptUpdate(BaseModel):
    value: str


class CompileRequest(BaseModel):
    force_redistill: bool = False


class Distillation(BaseModel):
    tone: str
    personality: str
    english_nuances: str
    methodology: str
    response_style: str
    constraints: list[str]
    summary: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


DEFAULT_SYSTEM_PROMPT = load_prompt("default_system_prompt.txt")
DOCUMENT_DISTILLATION_SYSTEM_PROMPT = load_prompt("document_distillation_system_prompt.txt")
DOCUMENT_DISTILLATION_USER_PROMPT = load_prompt("document_distillation_user_prompt.txt")
DISTILLATION_COMPRESSION_SYSTEM_PROMPT = load_prompt("distillation_compression_system_prompt.txt")
SYSTEM_PROMPT_SYNTHESIS_SYSTEM_PROMPT = load_prompt("system_prompt_synthesis_system_prompt.txt")


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
            cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content TEXT")
            cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS include_in_corpus BOOLEAN NOT NULL DEFAULT FALSE")
            cur.execute("UPDATE documents SET content = '' WHERE content IS NULL")
            cur.execute("ALTER TABLE documents ALTER COLUMN content SET NOT NULL")
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
                """
                CREATE TABLE IF NOT EXISTS document_distillations (
                    document_id BIGINT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    source_hash TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    distillation_text TEXT,
                    distillation_json TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS system_prompt_compile_runs (
                    id BIGSERIAL PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT,
                    compile_mode TEXT,
                    total_units INTEGER,
                    completed_units INTEGER,
                    prompt_draft TEXT,
                    documents_used INTEGER,
                    documents_selected INTEGER,
                    reused_distillations INTEGER,
                    new_distillations INTEGER,
                    reduction_rounds INTEGER,
                    model_name TEXT,
                    error_text TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ
                )
                """
            )
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS compile_mode TEXT")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS total_units INTEGER")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS completed_units INTEGER")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS prompt_draft TEXT")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS documents_used INTEGER")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS documents_selected INTEGER")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS reused_distillations INTEGER")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS new_distillations INTEGER")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS reduction_rounds INTEGER")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS model_name TEXT")
            cur.execute("ALTER TABLE system_prompt_compile_runs ADD COLUMN IF NOT EXISTS error_text TEXT")
            cur.execute("ALTER TABLE document_distillations ADD COLUMN IF NOT EXISTS distillation_text TEXT")
            cur.execute("ALTER TABLE document_distillations ADD COLUMN IF NOT EXISTS distillation_json TEXT")
            cur.execute("ALTER TABLE document_distillations ADD COLUMN IF NOT EXISTS source_hash TEXT")
            cur.execute("ALTER TABLE document_distillations ADD COLUMN IF NOT EXISTS model_name TEXT")
            cur.execute(
                """
                UPDATE document_distillations
                SET distillation_text = COALESCE(distillation_text, distillation_json, '')
                WHERE distillation_text IS NULL
                """
            )
            cur.execute(
                """
                UPDATE document_distillations
                SET distillation_json = distillation_text
                WHERE distillation_json IS NULL AND distillation_text IS NOT NULL
                """
            )
            cur.execute("ALTER TABLE document_distillations ALTER COLUMN distillation_text SET NOT NULL")
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('system_prompt', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (DEFAULT_SYSTEM_PROMPT,),
            )


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


def selected_system_prompt_model() -> str:
    return get_setting("system_prompt_model", SYSTEM_PROMPT_OLLAMA_MODEL) or SYSTEM_PROMPT_OLLAMA_MODEL


def create_compile_run(force_redistill: bool) -> dict:
    compile_mode = "redistill" if force_redistill else "reuse"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO system_prompt_compile_runs (status, message, compile_mode, total_units, completed_units)
                VALUES ('queued', 'System prompt compilation requested from UI.', %s, 0, 0)
                RETURNING *
                """,
                (compile_mode,),
            )
            return cur.fetchone()


def update_compile_run(run_id: int, *, status: str, message: str | None = None, **extra_fields) -> None:
    fields = {"status": status, "message": message}
    fields.update(extra_fields)
    assignments = ", ".join(f"{column} = %s" for column in fields)
    values = list(fields.values()) + [run_id]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE system_prompt_compile_runs SET {assignments} WHERE id = %s", values)


def latest_compile_run() -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM system_prompt_compile_runs ORDER BY id DESC LIMIT 1")
            return cur.fetchone()


def current_compile_run() -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM system_prompt_compile_runs
                WHERE status NOT IN ('completed', 'failed')
                ORDER BY id DESC
                LIMIT 1
                """
            )
            return cur.fetchone()


def recover_interrupted_compile_runs() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE system_prompt_compile_runs
                SET
                    status = 'failed',
                    message = 'System prompt compilation was interrupted by a service restart.',
                    error_text = COALESCE(error_text, 'Compiler worker was interrupted before completion.'),
                    completed_units = COALESCE(completed_units, 0),
                    completed_at = COALESCE(completed_at, NOW())
                WHERE status NOT IN ('completed', 'failed')
                """
            )


def selected_corpus_documents() -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, content
                FROM documents
                WHERE include_in_corpus = TRUE
                  AND content IS NOT NULL
                  AND BTRIM(content) <> ''
                ORDER BY id
                """
            )
            return cur.fetchall()


def ollama_generate(model: str, prompt: str, system_prompt: str, max_tokens: int) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
            },
        },
        timeout=900,
    )
    response.raise_for_status()
    return (response.json().get("response") or "").strip()


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model output was not a JSON object.")
    return payload


def validate_distillation_payload(payload: dict) -> Distillation:
    if isinstance(payload.get("constraints"), str):
        payload["constraints"] = [payload["constraints"]]
    return Distillation.model_validate(payload)


def split_text_by_char_limit(text: str, char_limit: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = [paragraph.strip() for paragraph in cleaned.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return [cleaned[:char_limit]] if char_limit > 0 else [cleaned]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if paragraph_length > char_limit:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_length = 0

            start = 0
            while start < paragraph_length:
                end = min(paragraph_length, start + char_limit)
                chunk = paragraph[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end
            continue

        projected_length = current_length + paragraph_length + (2 if current_parts else 0)
        if current_parts and projected_length > char_limit:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_length = 0

        current_parts.append(paragraph)
        current_length += paragraph_length + (2 if len(current_parts) > 1 else 0)

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def distill_text(text: str, model_name: str, error_label: str) -> Distillation:
    return generate_validated_distillation(
        model_name=model_name,
        prompt=text,
        system_prompt=DOCUMENT_DISTILLATION_SYSTEM_PROMPT,
        max_tokens=SYSTEM_PROMPT_DISTILL_MAX_TOKENS,
        error_label=error_label,
    )


def generate_validated_distillation(
    *,
    model_name: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    error_label: str,
) -> Distillation:
    last_error: Exception | None = None

    for attempt in range(1, DISTILLATION_MAX_RETRIES + 1):
        raw = ollama_generate(model_name, prompt, system_prompt, max_tokens)
        try:
            return validate_distillation_payload(extract_json_object(raw))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            if attempt == DISTILLATION_MAX_RETRIES:
                break

    raise HTTPException(
        status_code=502,
        detail=(
            f"{error_label} did not produce valid distillation JSON after "
            f"{DISTILLATION_MAX_RETRIES} attempts: {last_error}"
        ),
    ) from last_error


def distill_document(document: dict, model_name: str) -> Distillation:
    content = (document.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail=f"Document '{document['filename']}' has no text to distill.")

    content_chunks = split_text_by_char_limit(content, SYSTEM_PROMPT_DOCUMENT_CHAR_LIMIT)
    if not content_chunks:
        raise HTTPException(status_code=400, detail=f"Document '{document['filename']}' has no usable text to distill.")

    chunk_distillations = [
        distill_text(
            DOCUMENT_DISTILLATION_USER_PROMPT.format(filename=document["filename"], content=chunk),
            model_name,
            f"Document '{document['filename']}' chunk {index}",
        )
        for index, chunk in enumerate(content_chunks, start=1)
    ]

    if len(chunk_distillations) == 1:
        return chunk_distillations[0]

    reduced_distillations, _ = reduce_distillations_to_fit(
        chunk_distillations,
        model_name,
        char_limit=SYSTEM_PROMPT_DOCUMENT_CHAR_LIMIT,
        batch_char_limit=max(1, SYSTEM_PROMPT_DOCUMENT_CHAR_LIMIT // 2),
    )
    return compress_distillation_batch(reduced_distillations, model_name)


def compress_distillation_batch(batch: list[Distillation], model_name: str) -> Distillation:
    prompt = json.dumps([distillation.model_dump() for distillation in batch], ensure_ascii=False, indent=2)
    return generate_validated_distillation(
        model_name=model_name,
        prompt=prompt,
        system_prompt=DISTILLATION_COMPRESSION_SYSTEM_PROMPT,
        max_tokens=SYSTEM_PROMPT_DISTILL_MAX_TOKENS,
        error_label="Reduced distillation batch",
    )


def synthesize_system_prompt(distillations: list[Distillation], model_name: str) -> str:
    prompt = json.dumps([distillation.model_dump() for distillation in distillations], ensure_ascii=False, indent=2)
    print(prompt)
    return ollama_generate(
        model_name,
        prompt,
        SYSTEM_PROMPT_SYNTHESIS_SYSTEM_PROMPT,
        SYSTEM_PROMPT_FINAL_MAX_TOKENS,
    )


def batch_by_char_limit(items: list[Distillation], char_limit: int) -> list[list[Distillation]]:
    batches: list[list[Distillation]] = []
    current_batch: list[Distillation] = []
    current_length = 0

    for item in items:
        item_text = json.dumps(item.model_dump(), ensure_ascii=False)
        item_length = len(item_text)
        if current_batch and current_length + item_length > char_limit:
            batches.append(current_batch)
            current_batch = []
            current_length = 0
        current_batch.append(item)
        current_length += item_length

    if current_batch:
        batches.append(current_batch)

    return batches


def get_or_create_document_distillations(
    corpus_documents: list[dict], model_name: str, *, force_redistill: bool = False, progress_callback=None
) -> tuple[list[Distillation], dict]:
    distillations: list[Distillation] = []
    reused = 0
    created = 0

    processed_documents = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for document in corpus_documents:
                source_hash = sha256(document["content"].encode("utf-8")).hexdigest()
                cur.execute(
                    """
                    SELECT source_hash, model_name, COALESCE(distillation_json, distillation_text) AS distillation_payload
                    FROM document_distillations
                    WHERE document_id = %s
                    """,
                    (document["id"],),
                )
                row = cur.fetchone()
                if (
                    not force_redistill
                    and row
                    and row["source_hash"] == source_hash
                    and row["model_name"] == model_name
                    and row["distillation_payload"]
                ):
                    try:
                        distillation = validate_distillation_payload(json.loads(row["distillation_payload"]))
                        distillations.append(distillation)
                        reused += 1
                        processed_documents += 1
                        if progress_callback is not None:
                            progress_callback(processed_documents, reused, created)
                        continue
                    except (json.JSONDecodeError, ValidationError):
                        pass

                distillation = distill_document(document, model_name)
                distillation_payload = json.dumps(distillation.model_dump(), ensure_ascii=False)
                cur.execute(
                    """
                    INSERT INTO document_distillations (
                        document_id,
                        source_hash,
                        model_name,
                        distillation_text,
                        distillation_json,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (document_id)
                    DO UPDATE SET
                        source_hash = EXCLUDED.source_hash,
                        model_name = EXCLUDED.model_name,
                        distillation_text = EXCLUDED.distillation_text,
                        distillation_json = EXCLUDED.distillation_json,
                        updated_at = NOW()
                    """,
                    (
                        document["id"],
                        source_hash,
                        model_name,
                        distillation_payload,
                        distillation_payload,
                    ),
                )
                distillations.append(distillation)
                created += 1
                processed_documents += 1
                if progress_callback is not None:
                    progress_callback(processed_documents, reused, created)

    return distillations, {"reused": reused, "created": created, "processed_documents": processed_documents}


def reduce_distillations_to_fit(
    distillations: list[Distillation],
    model_name: str,
    *,
    char_limit: int = SYSTEM_PROMPT_COMPILATION_CHAR_LIMIT,
    batch_char_limit: int = SYSTEM_PROMPT_BATCH_CHAR_LIMIT,
) -> tuple[list[Distillation], int]:
    current = list(distillations)
    rounds = 0
    previous_length = len(json.dumps([item.model_dump() for item in current], ensure_ascii=False))

    while previous_length > char_limit:
        rounds += 1
        if rounds > MAX_REDUCTION_ROUNDS:
            raise HTTPException(
                status_code=502,
                detail=f"System prompt reduction did not converge after {MAX_REDUCTION_ROUNDS} rounds.",
            )
        batches = batch_by_char_limit(current, batch_char_limit)
        if len(batches) == 1 and len(batches[0]) == 1:
            compressed = compress_distillation_batch(batches[0], model_name)
            current_length = len(json.dumps([compressed.model_dump()], ensure_ascii=False))
            if current_length >= previous_length:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "System prompt reduction could not shrink an oversized distillation. "
                        "Try lowering the corpus size or document distillation limit."
                    ),
                )
            current = [compressed]
            previous_length = current_length
            continue

        current = [compress_distillation_batch(batch, model_name) for batch in batches]
        current_length = len(json.dumps([item.model_dump() for item in current], ensure_ascii=False))
        if current_length >= previous_length:
            raise HTTPException(
                status_code=502,
                detail="System prompt reduction stalled without reducing prompt size.",
            )
        previous_length = current_length

    return current, rounds


def compile_system_prompt_from_corpus_documents(*, force_redistill: bool = False, progress_callback=None) -> dict:
    corpus_documents = selected_corpus_documents()
    if not corpus_documents:
        raise HTTPException(status_code=400, detail="No corpus documents are selected.")

    compiler_model = selected_system_prompt_model()
    total_units = 10 + len(corpus_documents) + 3
    if progress_callback is not None:
        mode_text = "Re-distilling all selected corpus documents." if force_redistill else "Reusing cached distillations when available."
        progress_callback(
            status="distilling",
            message=f"Loading the model and starting distillation for {len(corpus_documents)} selected corpus documents. {mode_text}",
            total_units=total_units,
            completed_units=0,
            documents_selected=len(corpus_documents),
            model_name=compiler_model,
            compile_mode="redistill" if force_redistill else "reuse",
        )
    def on_document_progress(processed_documents: int, reused: int, created: int) -> None:
        if progress_callback is None:
            return
        progress_callback(
            status="distilling",
            message=f"Distilling {len(corpus_documents)} selected corpus documents. {mode_text}",
            total_units=total_units,
            completed_units=10 + processed_documents,
            documents_used=processed_documents,
            documents_selected=len(corpus_documents),
            reused_distillations=reused,
            new_distillations=created,
            model_name=compiler_model,
            compile_mode="redistill" if force_redistill else "reuse",
        )

    distillations, stats = get_or_create_document_distillations(
        corpus_documents,
        compiler_model,
        force_redistill=force_redistill,
        progress_callback=on_document_progress,
    )
    if progress_callback is not None:
        progress_callback(
            status="reducing",
            message="Reducing document distillations into a prompt-sized summary.",
            total_units=total_units,
            completed_units=10 + stats["processed_documents"],
            documents_used=len(distillations),
            documents_selected=len(corpus_documents),
            reused_distillations=stats["reused"],
            new_distillations=stats["created"],
            model_name=compiler_model,
            compile_mode="redistill" if force_redistill else "reuse",
        )
    reduced_distillations, reduction_rounds = reduce_distillations_to_fit(distillations, compiler_model)
    if progress_callback is not None:
        progress_callback(
            status="synthesizing",
            message="Synthesizing the final system prompt draft.",
            total_units=total_units,
            completed_units=10 + len(corpus_documents),
            documents_used=len(distillations),
            documents_selected=len(corpus_documents),
            reused_distillations=stats["reused"],
            new_distillations=stats["created"],
            reduction_rounds=reduction_rounds,
            model_name=compiler_model,
            compile_mode="redistill" if force_redistill else "reuse",
        )
    draft = synthesize_system_prompt(reduced_distillations, compiler_model)
    if not draft:
        raise HTTPException(status_code=502, detail="The model returned an empty system prompt draft.")

    return {
        "draft": draft,
        "documents_used": len(distillations),
        "documents_selected": len(corpus_documents),
        "reused_distillations": stats["reused"],
        "new_distillations": stats["created"],
        "reduction_rounds": reduction_rounds,
        "model": compiler_model,
        "compile_mode": "redistill" if force_redistill else "reuse",
        "total_units": total_units,
        "completed_units": total_units,
    }


def compile_worker(run_id: int, force_redistill: bool) -> None:
    global active_compile_thread, active_compile_run_id

    def progress_callback(*, status: str, message: str, **extra_fields) -> None:
        update_compile_run(run_id, status=status, message=message, **extra_fields)

    try:
        update_compile_run(
            run_id,
            status="running",
            message="Loading selected corpus documents.",
            started_at=utc_now(),
            compile_mode="redistill" if force_redistill else "reuse",
            total_units=0,
            completed_units=0,
        )
        result = compile_system_prompt_from_corpus_documents(
            force_redistill=force_redistill,
            progress_callback=progress_callback,
        )
        update_compile_run(
            run_id,
            status="completed",
            message="System prompt compilation finished.",
            compile_mode=result["compile_mode"],
            total_units=result["total_units"],
            completed_units=result["completed_units"],
            prompt_draft=result["draft"],
            documents_used=result["documents_used"],
            documents_selected=result["documents_selected"],
            reused_distillations=result["reused_distillations"],
            new_distillations=result["new_distillations"],
            reduction_rounds=result["reduction_rounds"],
            model_name=result["model"],
            error_text=None,
            completed_at=utc_now(),
        )
    except Exception as exc:
        update_compile_run(
            run_id,
            status="failed",
            message="System prompt compilation failed.",
            error_text=str(exc),
            completed_at=utc_now(),
        )
    finally:
        with compile_lock:
            if active_compile_run_id == run_id:
                active_compile_run_id = None
            active_compile_thread = None


@app.on_event("startup")
def startup_event() -> None:
    wait_for_database()
    ensure_database_schema()
    recover_interrupted_compile_runs()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/system-prompt")
def get_system_prompt() -> dict:
    return {
        "value": get_setting("system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT,
        "default": DEFAULT_SYSTEM_PROMPT,
    }


@app.put("/system-prompt")
def update_system_prompt(update: SystemPromptUpdate) -> dict:
    value = update.value.strip()
    if not value:
        raise HTTPException(status_code=400, detail="System prompt cannot be empty.")

    set_setting("system_prompt", value)
    return {"value": value}


@app.get("/system-prompt/compile/status")
def get_compile_status() -> dict:
    with compile_lock:
        compile_busy = active_compile_thread is not None and active_compile_thread.is_alive()
        active_run = active_compile_run_id
    latest_run = latest_compile_run()
    current_run = current_compile_run()
    return {
        "busy": compile_busy or current_run is not None,
        "active_run_id": active_run,
        "latest_run": latest_run,
        "current_run": current_run,
    }


@app.post("/system-prompt/compile")
def compile_system_prompt(request: CompileRequest) -> dict:
    global active_compile_thread, active_compile_run_id
    with compile_lock:
        if active_compile_thread is not None and active_compile_thread.is_alive():
            raise HTTPException(status_code=409, detail="A system prompt compilation job is already running.")

        run = create_compile_run(request.force_redistill)
        active_compile_run_id = run["id"]
        active_compile_thread = threading.Thread(
            target=compile_worker,
            args=(run["id"], request.force_redistill),
            daemon=True,
        )
        active_compile_thread.start()

    return {
        "run_id": run["id"],
        "status": "queued",
        "compile_mode": run.get("compile_mode"),
    }

import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import docker
import psycopg
from docker.errors import DockerException, NotFound
from docker.types import DeviceRequest
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from psycopg.rows import dict_row


POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://linguacomai:linguacomai@postgres:5432/linguacomai",
)
TRAINING_BASE_MODEL_PATH = os.getenv("TRAINING_BASE_MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
HF_SERVICE_URL = os.getenv("HF_SERVICE_URL", "http://hf:8200").rstrip("/")
TRAINER_IMAGE = os.getenv("LLAMA_FACTORY_IMAGE", "llamafactory:latest")
OFT_COMMAND_TEMPLATE = os.getenv(
    "OFT_COMMAND_TEMPLATE",
    "echo 'Set OFT_COMMAND_TEMPLATE in .env before training.' >&2; exit 1",
)
TRAINER_CONTAINER_NAME_PREFIX = os.getenv("TRAINER_CONTAINER_NAME_PREFIX", "linguacomai_llamafactory_run")
TRAINER_USE_GPU = os.getenv("TRAINER_USE_GPU", "true").lower() == "true"
TRAINING_OUTPUT_DIR = Path(os.getenv("TRAINING_OUTPUT_DIR", "/training_output"))
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "/corpus"))
CORPUS_VOLUME_NAME = os.getenv("CORPUS_VOLUME_NAME", "linguacomai_corpus_data")
TRAINING_OUTPUT_VOLUME_NAME = os.getenv("TRAINING_OUTPUT_VOLUME_NAME", "linguacomai_training_output")
HF_CACHE_VOLUME_NAME = os.getenv("HF_CACHE_VOLUME_NAME", "linguacomai_hf_cache")
HF_TOKEN = os.getenv("HF_TOKEN")
DOCKER_READY_TIMEOUT = int(os.getenv("DOCKER_READY_TIMEOUT_SECONDS", "60"))
DOCKER_HOST_VALUE = os.getenv("DOCKER_HOST")
DOCKER_TLS_VERIFY = os.getenv("DOCKER_TLS_VERIFY")
BASE_MODEL_SUBDIR = os.getenv("BASE_MODEL_SUBDIR", "base_model")
TRAINING_CHUNK_CHAR_LENGTH = int(os.getenv("TRAINING_CHUNK_CHAR_LENGTH", "1200"))
TRAINING_CHUNK_CHAR_OVERLAP = int(os.getenv("TRAINING_CHUNK_CHAR_OVERLAP", "200"))
DB_READY_TIMEOUT = int(os.getenv("POSTGRES_READY_TIMEOUT_SECONDS", "60"))
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
OFT_MODE = "oft"

TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LinguaComAI Trainer Service")
run_lock = threading.Lock()
active_thread = None
active_run_id = None
abort_requested_run_id = None
active_container_names: set[str] = set()
logger = logging.getLogger("trainer_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
MAX_ERROR_TEXT_LENGTH = 500


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


VALIDATION_PROMPT = load_prompt("training_validation_prompt.txt")


class TrainingAborted(Exception):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
            cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS corpus_file_path TEXT")
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS include_in_corpus BOOLEAN NOT NULL DEFAULT FALSE"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id BIGSERIAL PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT,
                    training_mode TEXT NOT NULL DEFAULT 'oft',
                    base_model TEXT,
                    output_model TEXT,
                    validation_results TEXT,
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
                "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS validation_results TEXT"
            )
            cur.execute(
                "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS training_mode TEXT NOT NULL DEFAULT 'oft'"
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
                """
                INSERT INTO app_settings (key, value)
                VALUES ('active_model', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (TRAINING_BASE_MODEL_PATH,),
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('active_hf_model', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (TRAINING_BASE_MODEL_PATH,),
            )
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES ('training_status', 'ready')
                ON CONFLICT (key) DO NOTHING
                """
            )


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


def get_setting(key: str, default: str | None = None) -> str | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM app_settings WHERE key = %s", (key,))
            row = cur.fetchone()
    if row is None:
        return default
    return row["value"]


def create_training_run(base_model: str, training_mode: str) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_runs (status, message, training_mode, base_model)
                VALUES ('queued', 'Training requested from UI.', %s, %s)
                RETURNING *
                """,
                (training_mode, base_model),
            )
            return cur.fetchone()


def update_training_run(run_id: int, *, status: str, message: str | None = None, **extra_fields) -> None:
    fields = {"status": status, "message": message}
    fields.update(extra_fields)
    assignments = ", ".join(f"{column} = %s" for column in fields)
    values = list(fields.values()) + [run_id]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE training_runs SET {assignments} WHERE id = %s", values)


def current_training_run() -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM training_runs
                WHERE status NOT IN ('completed', 'failed', 'aborted')
                ORDER BY id DESC
                LIMIT 1
                """
            )
            return cur.fetchone()


def latest_training_run() -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1")
            return cur.fetchone()


def failed_runs() -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM training_runs
                WHERE status = 'failed'
                ORDER BY id DESC
                """
            )
            return cur.fetchall()


def selected_corpus_documents() -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, corpus_file_path
                FROM documents
                WHERE include_in_corpus = TRUE AND corpus_file_path IS NOT NULL
                ORDER BY id
                """
            )
            return cur.fetchall()


def normalize_training_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def split_training_chunks(text: str) -> list[str]:
    cleaned = normalize_training_text(text)
    if not cleaned:
        return []

    paragraphs = [paragraph.strip() for paragraph in cleaned.split("\n\n") if paragraph.strip()]
    chunks = []
    current_parts = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if paragraph_length > TRAINING_CHUNK_CHAR_LENGTH:
            if current_parts:
                chunk_text = "\n\n".join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_parts = []
                current_length = 0

            start = 0
            while start < paragraph_length:
                end = min(paragraph_length, start + TRAINING_CHUNK_CHAR_LENGTH)
                chunk_text = paragraph[start:end].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                if end == paragraph_length:
                    break
                start = max(end - TRAINING_CHUNK_CHAR_OVERLAP, start + 1)
            continue

        projected = current_length + paragraph_length + (2 if current_parts else 0)
        if current_parts and projected > TRAINING_CHUNK_CHAR_LENGTH:
            chunk_text = "\n\n".join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

            overlap_parts = []
            overlap_length = 0
            for part in reversed(current_parts):
                part_length = len(part)
                if overlap_length + part_length > TRAINING_CHUNK_CHAR_OVERLAP:
                    break
                overlap_parts.insert(0, part)
                overlap_length += part_length

            current_parts = overlap_parts
            current_length = sum(len(part) for part in current_parts)

        current_parts.append(paragraph)
        current_length += paragraph_length + (2 if len(current_parts) > 1 else 0)

    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def cleanup_failed_run_artifacts(keep_run_id: int) -> None:
    for run in failed_runs():
        if run["id"] == keep_run_id:
            continue

        for field in ("output_dir", "corpus_snapshot_dir"):
            path_value = run.get(field)
            if not path_value:
                continue
            path = Path(path_value)
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)


def resolve_model_ref(model_ref: str) -> str:
    local_path = Path(model_ref)
    if local_path.exists():
        return str(local_path)

    try:
        return snapshot_download(
            repo_id=model_ref,
            token=HF_TOKEN or None,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, FileNotFoundError):
        logger.info("Model '%s' not found locally. Downloading into the shared Hugging Face cache.", model_ref)
        return snapshot_download(
            repo_id=model_ref,
            token=HF_TOKEN or None,
        )


def is_abort_requested(run_id: int) -> bool:
    with run_lock:
        return abort_requested_run_id == run_id


def check_abort_requested(run_id: int) -> None:
    if is_abort_requested(run_id):
        raise TrainingAborted("Training was aborted by the user.")


def mark_container_active(container_name: str) -> None:
    with run_lock:
        active_container_names.add(container_name)


def mark_container_inactive(container_name: str) -> None:
    with run_lock:
        active_container_names.discard(container_name)


def expected_container_names(run_id: int) -> list[str]:
    return [
        f"{TRAINER_CONTAINER_NAME_PREFIX}_{run_id}",
        f"{TRAINER_CONTAINER_NAME_PREFIX}_base_{run_id}",
    ]


def finalize_aborted_run(run_id: int, message: str) -> None:
    global active_run_id, abort_requested_run_id, active_thread
    set_setting("training_status", "ready")
    update_training_run(
        run_id,
        status="aborted",
        message=message,
        error_text=None,
        completed_at=utc_now(),
    )
    with run_lock:
        if active_run_id == run_id:
            active_run_id = None
        if abort_requested_run_id == run_id:
            abort_requested_run_id = None
        active_thread = None


def get_docker_client():
    deadline = time.time() + DOCKER_READY_TIMEOUT
    last_error = None
    while time.time() < deadline:
        try:
            if DOCKER_HOST_VALUE:
                tls_enabled = str(DOCKER_TLS_VERIFY).lower() not in {"", "0", "false", "none"}
                client = docker.DockerClient(base_url=DOCKER_HOST_VALUE, tls=tls_enabled or False)
            else:
                client = docker.from_env()
            client.ping()
            return client
        except DockerException as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"Docker daemon did not become ready in time: {last_error}")


def write_snapshot(run_id: int, docs: list[dict]) -> tuple[Path, Path]:
    snapshot_dir = CORPUS_DIR / "snapshots" / f"run_{run_id}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_dir / "manifest.json"
    dataset_path = snapshot_dir / "corpus.jsonl"

    manifest = []
    example_count = 0
    with dataset_path.open("w", encoding="utf-8") as dataset_file:
        for doc in docs:
            source_path = Path(doc["corpus_file_path"])
            if not source_path.exists():
                continue

            text = source_path.read_text(encoding="utf-8")
            target_name = source_path.name
            snapshot_file = snapshot_dir / target_name
            snapshot_file.write_text(text, encoding="utf-8")
            chunks = split_training_chunks(text)
            manifest.append(
                {
                    "document_id": doc["id"],
                    "filename": doc["filename"],
                    "snapshot_file": str(snapshot_file),
                    "training_examples": len(chunks),
                }
            )
            for chunk in chunks:
                dataset_file.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                example_count += 1

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Snapshot for run %s created with %s training examples.", run_id, example_count)
    return snapshot_dir, dataset_path


def render_training_command(
    run_id: int,
    dataset_path: Path,
    output_dir: Path,
    training_mode: str,
    base_model_ref: str,
) -> str:
    command = OFT_COMMAND_TEMPLATE
    replacements = {
        "{run_id}": str(run_id),
        "{dataset_path}": str(dataset_path),
        "{output_dir}": str(output_dir),
        "{base_model}": base_model_ref,
    }
    for placeholder, value in replacements.items():
        command = command.replace(placeholder, value)
    return command


def run_llama_factory_job(
    client, run_id: int, dataset_path: Path, output_dir: Path, log_path: Path, training_mode: str, base_model_ref: str
) -> None:
    check_abort_requested(run_id)
    command = render_training_command(run_id, dataset_path, output_dir, training_mode, base_model_ref)
    container_name = f"{TRAINER_CONTAINER_NAME_PREFIX}_{run_id}"
    device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])] if TRAINER_USE_GPU else None
    logger.info("Launching training container '%s' with image '%s'.", container_name, TRAINER_IMAGE)
    logger.info("Training dataset path: %s", dataset_path)
    logger.info("Training output path: %s", output_dir)

    try:
        existing = client.containers.get(container_name)
        existing.remove(force=True)
    except NotFound:
        pass

    container = client.containers.run(
        TRAINER_IMAGE,
        command=["bash", "-lc", command],
        detach=True,
        name=container_name,
        remove=False,
        environment={
            "RUN_ID": str(run_id),
            "DATASET_PATH": str(dataset_path),
            "OUTPUT_DIR": str(output_dir),
            "BASE_MODEL": base_model_ref,
            "HF_TOKEN": HF_TOKEN or "",
        },
        volumes={
            CORPUS_VOLUME_NAME: {"bind": "/corpus", "mode": "rw"},
            TRAINING_OUTPUT_VOLUME_NAME: {"bind": "/training_output", "mode": "rw"},
            HF_CACHE_VOLUME_NAME: {"bind": "/root/.cache/huggingface", "mode": "rw"},
        },
        device_requests=device_requests,
    )
    mark_container_active(container_name)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            for chunk in container.logs(stream=True, follow=True):
                check_abort_requested(run_id)
                log_file.write(chunk.decode("utf-8", errors="replace"))
                log_file.flush()

        check_abort_requested(run_id)
        result = container.wait()
        status_code = result.get("StatusCode", 1)
        logger.info("Training container '%s' exited with status %s.", container_name, status_code)
        if status_code != 0:
            raise RuntimeError(f"LLaMA-Factory container exited with status {status_code}.")
    except DockerException as exc:
        if is_abort_requested(run_id):
            raise TrainingAborted("Training was aborted while stopping the active training container.") from exc
        raise
    finally:
        mark_container_inactive(container_name)
        try:
            container.remove(force=True)
        except DockerException:
            pass


def find_adapter_directory(output_dir: Path) -> Path:
    candidate_dirs = []
    for path in output_dir.rglob("adapter_config.json"):
        candidate_dir = path.parent
        if (
            (candidate_dir / "adapter_model.safetensors").exists()
            or (candidate_dir / "adapter_model.bin").exists()
        ):
            candidate_dirs.append(candidate_dir)

    if candidate_dirs:
        candidate_dirs.sort(key=lambda path: (len(path.parts), str(path)))
        return candidate_dirs[0]

    visible_files = [str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()]
    raise RuntimeError(
        "No complete adapter directory was found in the training output. "
        f"Looked under {output_dir}. Files found: {visible_files}"
    )


def export_base_model(client, run_id: int, output_dir: Path, log_path: Path, base_model_ref: str) -> Path:
    check_abort_requested(run_id)
    base_model_dir = output_dir / BASE_MODEL_SUBDIR
    base_model_dir.mkdir(parents=True, exist_ok=True)
    container_name = f"{TRAINER_CONTAINER_NAME_PREFIX}_base_{run_id}"
    export_script = (
        "python - <<'PY'\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        f"base_model_name = r'{base_model_ref}'\n"
        f"base_model_dir = r'{base_model_dir}'\n"
        "model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype='auto', device_map='cpu', local_files_only=True)\n"
        "model.save_pretrained(base_model_dir, safe_serialization=True)\n"
        "tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)\n"
        "tokenizer.save_pretrained(base_model_dir)\n"
        "print(f'Base model written to {base_model_dir}')\n"
        "PY"
    )

    logger.info(
        "Exporting exact training base model '%s' under '%s'.",
        base_model_ref,
        base_model_dir,
    )

    try:
        existing = client.containers.get(container_name)
        existing.remove(force=True)
    except NotFound:
        pass

    container = client.containers.run(
        TRAINER_IMAGE,
        command=["bash", "-lc", export_script],
        detach=True,
        name=container_name,
        remove=False,
        environment={
            "HF_TOKEN": HF_TOKEN or "",
        },
        volumes={
            TRAINING_OUTPUT_VOLUME_NAME: {"bind": "/training_output", "mode": "rw"},
            HF_CACHE_VOLUME_NAME: {"bind": "/root/.cache/huggingface", "mode": "rw"},
        },
    )
    mark_container_active(container_name)

    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n=== BASE EXPORT STEP ===\n")
            for chunk in container.logs(stream=True, follow=True):
                check_abort_requested(run_id)
                log_file.write(chunk.decode("utf-8", errors="replace"))
                log_file.flush()

        check_abort_requested(run_id)
        result = container.wait()
        status_code = result.get("StatusCode", 1)
        logger.info("Base export container '%s' exited with status %s.", container_name, status_code)
        if status_code != 0:
            raise RuntimeError(f"Base export container exited with status {status_code}.")
    except DockerException as exc:
        if is_abort_requested(run_id):
            raise TrainingAborted("Training was aborted while stopping the base export container.") from exc
        raise
    finally:
        mark_container_inactive(container_name)
        try:
            container.remove(force=True)
        except DockerException:
            pass

    if not any(base_model_dir.glob("*.safetensors")):
        visible_files = [str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()]
        raise RuntimeError(
            "Base model directory was created but no safetensors model files were found. "
            f"Files found: {visible_files}"
        )

    return base_model_dir


def hf_generate(model_ref: str, prompt: str) -> str:
    payload = json.dumps(
        {
            "model": model_ref,
            "prompt": prompt,
            "max_new_tokens": 160,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{HF_SERVICE_URL}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc

    response_payload = json.loads(body)
    return (response_payload.get("response") or "").strip()


def validate_models(
    client,
    run_id: int,
    output_dir: Path,
    base_model_dir: Path,
    candidate_model_dir: Path,
    log_path: Path,
    has_training_data: bool,
    training_mode: str,
) -> dict:
    adapter_dir = find_adapter_directory(output_dir) if has_training_data else None
    logger.info("Validating base, adapter, and merged models through the HF service for run %s.", run_id)

    cases = [
        {
            "label": "BASE",
            "model_ref": str(base_model_dir),
        },
        {
            "label": "CANDIDATE",
            "model_ref": str(candidate_model_dir),
        },
    ]
    if adapter_dir is not None:
        cases.insert(
            1,
            {
                "label": "ADAPTER",
                "model_ref": str(adapter_dir),
            },
        )

    results = {}
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n=== VALIDATION STEP ===\n")
        log_file.write(f"Prompt: {VALIDATION_PROMPT}\n")
        for case in cases:
            try:
                response_text = hf_generate(case["model_ref"], VALIDATION_PROMPT) or "<empty response>"
                results[case["label"].lower()] = response_text
                log_file.write(f"[{case['label']}] {response_text}\n")
            except Exception as exc:
                error_text = f"ERROR: {exc}"
                results[case["label"].lower()] = error_text
                log_file.write(f"[{case['label']}] {error_text}\n")
            log_file.flush()
        if adapter_dir is None:
            results["adapter"] = "SKIPPED: no training corpus selected."
            log_file.write("[ADAPTER] SKIPPED: no training corpus selected.\n")
            log_file.flush()

    return results


def run_training_pipeline(run_id: int, training_mode: str) -> None:
    logger.info("Starting %s training pipeline for run %s.", training_mode, run_id)
    check_abort_requested(run_id)
    set_setting("training_status", "snapshotting")
    update_training_run(run_id, status="snapshotting", message="Collecting selected corpus documents.", started_at=utc_now())

    base_model_ref = get_setting("active_hf_model", TRAINING_BASE_MODEL_PATH) or TRAINING_BASE_MODEL_PATH
    resolved_base_model_ref = resolve_model_ref(base_model_ref)
    check_abort_requested(run_id)

    docs = selected_corpus_documents()
    logger.info("Run %s will train on %s selected documents.", run_id, len(docs))

    output_dir = TRAINING_OUTPUT_DIR / "runs" / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    if docs:
        snapshot_dir, dataset_path = write_snapshot(run_id, docs)
    else:
        snapshot_dir = CORPUS_DIR / "snapshots" / f"run_{run_id}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = snapshot_dir / "corpus.jsonl"
        dataset_path.write_text("", encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("No documents selected. Running control pipeline with the unchanged base model.\n")
    check_abort_requested(run_id)
    update_training_run(
        run_id,
        status="preparing_training",
        message="Preparing training run." if docs else "Preparing control validation run.",
        corpus_snapshot_dir=str(snapshot_dir),
        output_dir=str(output_dir),
        log_path=str(log_path),
    )
    set_setting("training_status", "preparing_training")

    client = get_docker_client()

    try:
        base_model_dir = export_base_model(client, run_id, output_dir, log_path, resolved_base_model_ref)
        check_abort_requested(run_id)
        if docs:
            update_training_run(run_id, status="training", message=f"Running {training_mode} LLaMA-Factory training job.")
            set_setting("training_status", "training")
            run_llama_factory_job(
                client,
                run_id,
                dataset_path,
                output_dir,
                log_path,
                training_mode,
                resolved_base_model_ref,
            )
            check_abort_requested(run_id)
            candidate_model_dir = find_adapter_directory(output_dir)
        else:
            candidate_model_dir = base_model_dir
            update_training_run(run_id, status="training", message="No documents selected; using the exported base model as control output.")
            set_setting("training_status", "training")
        check_abort_requested(run_id)
        update_training_run(run_id, status="activating_model", message="Preparing model validation and activation.")
        set_setting("training_status", "activating_model")

        update_training_run(run_id, status="validating_model", message="Validating base, adapter, and merged model outputs through HF.")
        set_setting("training_status", "validating_model")
        validation_results = validate_models(
            client, run_id, output_dir, base_model_dir, candidate_model_dir, log_path, bool(docs), training_mode
        )
        check_abort_requested(run_id)
        update_training_run(
            run_id,
            status="activating_model",
            message="Activating the trained HF model.",
            validation_results=json.dumps(validation_results),
        )
        set_setting("training_status", "activating_model")

        set_setting("active_hf_model", str(candidate_model_dir))
        set_setting("active_model", str(candidate_model_dir))
        set_setting("training_status", "ready")
        update_training_run(
            run_id,
            status="completed",
            message="Training finished successfully.",
            output_model=str(candidate_model_dir),
            validation_results=json.dumps(validation_results),
            completed_at=utc_now(),
        )
        logger.info("Training pipeline for run %s completed. Active HF model is now '%s'.", run_id, candidate_model_dir)
    except TrainingAborted as exc:
        logger.info("Training pipeline for run %s aborted: %s", run_id, exc)
        set_setting("training_status", "ready")
        update_training_run(
            run_id,
            status="aborted",
            message="Training aborted by the user.",
            error_text=None,
            completed_at=utc_now(),
        )
    except Exception as exc:
        logger.exception("Training pipeline for run %s failed: %s", run_id, exc)
        short_error = str(exc).strip().replace("\n", " ")
        if len(short_error) > MAX_ERROR_TEXT_LENGTH:
            short_error = short_error[: MAX_ERROR_TEXT_LENGTH - 3] + "..."
        set_setting("training_status", "failed")
        update_training_run(
            run_id,
            status="failed",
            message="Training failed.",
            error_text=short_error,
            completed_at=utc_now(),
        )
        cleanup_failed_run_artifacts(run_id)
        raise


def training_worker(run_id: int, training_mode: str) -> None:
    global active_thread, active_run_id, abort_requested_run_id
    try:
        run_training_pipeline(run_id, training_mode)
    except Exception:
        pass
    finally:
        with run_lock:
            if active_run_id == run_id:
                active_run_id = None
            if abort_requested_run_id == run_id:
                abort_requested_run_id = None
            active_thread = None


@app.on_event("startup")
def startup_event() -> None:
    wait_for_database()
    ensure_database_schema()


@app.get("/status")
def training_status() -> dict:
    active_model = get_setting("active_hf_model", TRAINING_BASE_MODEL_PATH)
    status = get_setting("training_status", "ready")
    latest_run = latest_training_run()
    selected_docs = selected_corpus_documents()
    with run_lock:
        active_run = active_run_id
    return {
        "active_model": active_model,
        "training_status": status,
        "selected_corpus_documents": len(selected_docs),
        "latest_run": latest_run,
        "busy": status not in {"ready", "failed", "aborted"},
        "active_run_id": active_run,
    }


@app.get("/log-tail")
def training_log_tail(lines: int = 50) -> dict:
    run = latest_training_run()
    if run is None or not run.get("log_path"):
        return {"lines": [], "run_id": None}

    log_path = Path(run["log_path"])
    if not log_path.exists():
        return {"lines": [], "run_id": run["id"]}

    tail_lines = deque(maxlen=max(lines, 1))
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            tail_lines.append(line.rstrip("\n"))

    return {
        "run_id": run["id"],
        "lines": list(tail_lines),
    }


def queue_training(training_mode: str) -> dict:
    global active_thread, active_run_id, abort_requested_run_id
    with run_lock:
        if active_thread is not None and active_thread.is_alive():
            raise HTTPException(status_code=409, detail="A training job is already running.")

        docs = selected_corpus_documents()
        base_model = get_setting("active_hf_model", TRAINING_BASE_MODEL_PATH) or TRAINING_BASE_MODEL_PATH
        run = create_training_run(base_model, training_mode)
        set_setting("training_status", "queued")
        logger.info("Queued %s run %s using base model '%s'.", training_mode, run["id"], base_model)
        active_run_id = run["id"]
        abort_requested_run_id = None
        active_thread = threading.Thread(target=training_worker, args=(run["id"], training_mode), daemon=True)
        active_thread.start()

    return {
        "run_id": run["id"],
        "status": "queued",
        "training_mode": training_mode,
        "selected_documents": len(docs),
    }


@app.post("/train/abort")
def abort_training() -> dict:
    global abort_requested_run_id
    with run_lock:
        run = current_training_run()
        local_thread_alive = active_thread is not None and active_thread.is_alive()
        if run is None:
            raise HTTPException(status_code=409, detail="No active training job is running.")

        run_id = run["id"]
        abort_requested_run_id = run_id
        active_names = sorted(set(active_container_names).union(expected_container_names(run_id)))

    set_setting("training_status", "aborting")
    update_training_run(run_id, status="aborting", message="Abort requested. Stopping active training containers.")

    stopped_containers = []
    stop_errors = []
    found_container = False
    response_status = "aborting"
    if active_names:
        try:
            client = get_docker_client()
            for container_name in active_names:
                try:
                    container = client.containers.get(container_name)
                    found_container = True
                    container.remove(force=True)
                    stopped_containers.append(container_name)
                except NotFound:
                    continue
                except DockerException as exc:
                    stop_errors.append(f"{container_name}: {exc}")
        except Exception as exc:
            stop_errors.append(str(exc))

    if not local_thread_alive:
        if stop_errors:
            set_setting("training_status", "failed")
            update_training_run(
                run_id,
                status="failed",
                message="Abort was requested, but one or more training containers could not be stopped.",
                error_text=" | ".join(stop_errors)[:MAX_ERROR_TEXT_LENGTH],
                completed_at=utc_now(),
            )
        else:
            message = "Training aborted by the user."
            if not found_container:
                message = "Training marked as aborted after the trainer lost track of the active run."
            finalize_aborted_run(run_id, message)
            response_status = "aborted"

    return {
        "run_id": run_id,
        "status": response_status,
        "stopped_containers": stopped_containers,
        "stop_errors": stop_errors,
    }


@app.post("/train")
def start_training() -> dict:
    return queue_training(OFT_MODE)


@app.post("/train/oft")
def start_oft_tuning() -> dict:
    return queue_training(OFT_MODE)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

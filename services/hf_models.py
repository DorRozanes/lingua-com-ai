import os
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import scan_cache_dir, snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from pydantic import BaseModel
from peft import AutoPeftModelForCausalLM
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


app = FastAPI(title="LinguaComAI HF Inference Service")
model_lock = threading.Lock()
loaded_model_ref = None
loaded_model = None
loaded_tokenizer = None
loaded_embedding_model_ref = None
loaded_embedding_model = None
loaded_embedding_tokenizer = None
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HOME = Path(os.getenv("HF_HOME", "/root/.cache/huggingface"))
TRAINING_OUTPUT_DIR = Path(os.getenv("TRAINING_OUTPUT_DIR", "/training_output"))


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system_prompt: str | None = None
    max_new_tokens: int = 160


class EmbedRequest(BaseModel):
    model: str
    input: list[str]


class ModelDownloadRequest(BaseModel):
    model: str
    hf_token: str | None = None


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
        return snapshot_download(
            repo_id=model_ref,
            token=HF_TOKEN or None,
        )


def resolve_model_ref_with_token(model_ref: str, hf_token: str | None = None) -> str:
    local_path = Path(model_ref)
    if local_path.exists():
        return str(local_path)

    try:
        return snapshot_download(
            repo_id=model_ref,
            token=hf_token or HF_TOKEN or None,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, FileNotFoundError):
        return snapshot_download(
            repo_id=model_ref,
            token=hf_token or HF_TOKEN or None,
        )


def infer_framework_from_path(path: Path) -> str:
    if (path / "adapter_config.json").exists():
        return "peft-adapter"
    if list(path.glob("*.gguf")):
        return "gguf"
    if (path / "config.json").exists():
        return "transformers"
    if list(path.glob("*.safetensors")):
        return "safetensors"
    return "local-files"


def summarize_local_model_path(path: Path, model_id: str | None = None, backend: str = "hf") -> dict:
    size_bytes = sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) if path.exists() else 0
    framework = infer_framework_from_path(path)
    return {
        "id": model_id or str(path),
        "name": model_id or path.name,
        "backend": backend,
        "framework": framework,
        "path": str(path),
        "size_bytes": size_bytes,
        "requires_hf_token": backend == "hf",
    }


def list_cached_hf_models() -> list[dict]:
    models: list[dict] = []
    try:
        cache_info = scan_cache_dir(HF_HOME / "hub")
    except Exception:
        cache_info = None

    if cache_info is not None:
        for repo in cache_info.repos:
            if repo.repo_type != "model":
                continue
            revision = next(iter(repo.revisions), None)
            snapshot_path = Path(revision.snapshot_path) if revision is not None else None
            model_path = snapshot_path or Path(repo.repo_path)
            summary = summarize_local_model_path(model_path, model_id=repo.repo_id, backend="hf")
            summary["size_bytes"] = repo.size_on_disk
            summary["repo_id"] = repo.repo_id
            summary["last_accessed"] = revision.last_modified if revision is not None else None
            models.append(summary)

    return models


def list_training_output_models() -> list[dict]:
    models: list[dict] = []
    if not TRAINING_OUTPUT_DIR.exists():
        return models

    for path in TRAINING_OUTPUT_DIR.rglob("*"):
        if not path.is_dir():
            continue
        if not ((path / "adapter_config.json").exists() or (path / "config.json").exists()):
            continue
        models.append(summarize_local_model_path(path, backend="hf"))

    unique_models = {}
    for model in models:
        unique_models[model["path"]] = model
    return list(unique_models.values())


def load_model(model_ref: str):
    global loaded_model_ref, loaded_model, loaded_tokenizer

    resolved_ref = resolve_model_ref(model_ref)
    local_path = Path(resolved_ref)
    is_adapter = local_path.exists() and (local_path / "adapter_config.json").exists()

    if loaded_model_ref == resolved_ref and loaded_model is not None and loaded_tokenizer is not None:
        return loaded_model, loaded_tokenizer

    loaded_model = None
    loaded_tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_adapter:
        tokenizer = AutoTokenizer.from_pretrained(resolved_ref, local_files_only=True)
        model = AutoPeftModelForCausalLM.from_pretrained(
            resolved_ref,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(resolved_ref, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_ref,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    loaded_model_ref = resolved_ref
    loaded_model = model
    loaded_tokenizer = tokenizer
    return model, tokenizer


def load_embedding_model(model_ref: str):
    global loaded_embedding_model_ref, loaded_embedding_model, loaded_embedding_tokenizer

    resolved_ref = resolve_model_ref(model_ref)

    if (
        loaded_embedding_model_ref == resolved_ref
        and loaded_embedding_model is not None
        and loaded_embedding_tokenizer is not None
    ):
        return loaded_embedding_model, loaded_embedding_tokenizer

    loaded_embedding_model = None
    loaded_embedding_tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(resolved_ref, local_files_only=True)
    model = AutoModel.from_pretrained(
        resolved_ref,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    loaded_embedding_model_ref = resolved_ref
    loaded_embedding_model = model
    loaded_embedding_tokenizer = tokenizer
    return model, tokenizer


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "loaded_model": loaded_model_ref,
        "loaded_embedding_model": loaded_embedding_model_ref,
    }


@app.get("/models")
def list_models() -> dict:
    cache_models = list_cached_hf_models()
    output_models = list_training_output_models()
    combined = {}
    for model in cache_models + output_models:
        combined[model["path"]] = model
    return {
        "models": sorted(combined.values(), key=lambda item: (item["backend"], item["name"].lower())),
        "loaded_model": loaded_model_ref,
        "loaded_embedding_model": loaded_embedding_model_ref,
    }


@app.post("/models/download")
def download_model(request: ModelDownloadRequest) -> dict:
    model_ref = request.model.strip()
    if not model_ref:
        raise HTTPException(status_code=400, detail="Model cannot be empty.")

    resolved_path = resolve_model_ref_with_token(model_ref, request.hf_token.strip() if request.hf_token else None)
    summary = summarize_local_model_path(Path(resolved_path), model_id=model_ref, backend="hf")
    return {
        "downloaded": True,
        "model": model_ref,
        "path": resolved_path,
        "size_bytes": summary["size_bytes"],
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    with model_lock:
        model, tokenizer = load_model(request.model)
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        if getattr(tokenizer, "chat_template", None):
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = tokenizer(request.prompt, return_tensors="pt")

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return {
            "model": request.model,
            "response": text,
        }


@app.post("/embed")
def embed(request: EmbedRequest) -> dict:
    with model_lock:
        model, tokenizer = load_embedding_model(request.model)
        inputs = tokenizer(
            request.input,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return {
            "model": request.model,
            "embeddings": embeddings.detach().cpu().tolist(),
        }

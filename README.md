# LinguaComAI

LinguaComAI is a local-first workspace for working with LLMs on your own machine. The project is meant for mostly offline use after setup and model download: you can upload documents, chat against them with RAG, compile a reusable system prompt from your corpus, inspect and switch models, and run local HF fine-tuning jobs.

The general idea is to keep the workflow adjustable instead of hiding everything behind one fixed pipeline. You can tune retrieval behavior, choose which models appear in chat, control Ollama keep-warm behavior, edit or compile the system prompt, and manage training runs from the UI.

## What It Does

- Chat with your uploaded documents through a Streamlit UI
- Index `.txt`, `.md`, and `.docx` files for retrieval
- Run local RAG with adjustable retrieval settings
- Generate a system prompt from selected corpus documents
- Use Ollama for chat and prompt synthesis
- Use Hugging Face models for inference and fine-tuning
- Track training and prompt-compilation jobs in the UI

## Installation

### Prerequisites

- `git`
- Docker Engine / Docker Desktop
- Docker Compose

### Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd LinguaComAI
```

2. Create your environment file from the sample:

```bash
cp .env.sample .env
```

If you are on Windows PowerShell:

```powershell
Copy-Item .env.sample .env
```

3. Edit `.env` as needed.

Important notes:
- An HF token may be useful for downloading gated or private HF models.
- After models are downloaded and cached locally, most normal usage can stay offline.

4. Start the app:

```bash
docker compose up -d --build
```

5. Open the UI:

`http://localhost:8501`

## Offline Use

This project is designed to support offline LLM work as much as practical.

You will still usually need internet for:
- pulling Docker images the first time
- downloading Ollama models the first time
- downloading Hugging Face models the first time

After that, normal workflows can stay local:
- document upload and indexing
- chat
- retrieval
- system prompt compilation
- local training workflows, assuming the required base model is already cached

## Pages

- `Intro`: overview of the app and where to go next
- `Chat`: ask questions against uploaded documents
- `Documents`: upload files and choose which ones belong in the corpus
- `RAG Adjustment`: tune top-k, similarity threshold, and small-talk bypass
- `System Prompt`: edit the active system prompt or compile one from the corpus
- `Models`: inspect installed Ollama and HF models and set defaults
- `Model Training`: start, monitor, and abort fine-tuning runs

## Performance Notes

Offline LLMs are usually slower than hosted APIs, especially on consumer hardware.

Things that can make responses feel slow:
- large local models
- model cold starts after unload
- limited GPU VRAM
- CPU-only inference
- retrieval overhead before generation
- first-time model loading from disk

This means a local setup may feel excellent for privacy and control, but not always fast. Short greetings can still feel delayed if a model has to load into memory first.

## GPU / VRAM Detection

The current code does some GPU-awareness, but it does not yet do a deep, user-facing VRAM inspection pass.

Right now:
- [services/hf_models.py](/C:/Users/Dor/Desktop/Projects/LinguaComAI/services/hf_models.py) checks `torch.cuda.is_available()`
- it clears CUDA cache with `torch.cuda.empty_cache()`
- it loads models with `device_map="auto"` so Transformers can place weights automatically

So the app can detect whether CUDA is available and will try to use the GPU when possible, but it does not currently show a detailed VRAM report in the UI or make explicit scheduling decisions based on measured free VRAM.

If we want to improve that later, the next step would be to add code that reads things like:
- `torch.cuda.get_device_properties(...)`
- `torch.cuda.mem_get_info()`

That would let the app estimate how much VRAM is present and how much is free before loading a model.

## Current Architecture

The app runs as several Docker services:
- Streamlit UI
- retrieval service
- HF inference service
- training service
- system-prompt service
- Ollama
- Postgres / pgvector

Only the UI is exposed to the host by default; the rest communicate internally over Docker networking.

## Notes

- This project is optimized for local control and experimentation, not for maximum throughput.
- If you want the smoothest experience, use smaller models first and scale up only when needed.
- If chat feels noisy or irrelevant, adjust the RAG settings before assuming the model is the problem.

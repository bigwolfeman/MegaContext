# MegaContext

Extend any LLM to **million-token contexts** with zero training. Drop-in OpenAI-compatible server that adds embedding retrieval transparently.

## How It Works

MegaContext uses the model's own embedding table to compress overflow context into a retrieval index. When a query comes in, it retrieves the most relevant chunks via cosine similarity and inserts them into the prompt — all behind a standard `/v1/chat/completions` endpoint.

```
[HEAD tokens] + [RETRIEVED top-k chunks] + [TAIL tokens]
```

- **No training required** — uses raw embedding table from any HuggingFace model
- **No external vector DB** — GPU-resident slab-based index, append-only
- **Scales linearly** — ingest 1M tokens in ~2.4s, query time flat from 128K to 1M

## Quick Start with Qwen3.6-27B + vLLM

### 1. Start vLLM backend

```bash
docker run --gpus all -p 8421:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.6-27B \
  --gpu-memory-utilization 0.70 \
  --max-model-len 131072 \
  --dtype bfloat16
```

### 2. Install MegaContext

```bash
pip install -e .
```

### 3. Start the MegaContext server

```bash
MEGA_MODEL=Qwen/Qwen3.6-27B \
MEGA_BACKEND_URL=http://localhost:8421 \
MEGA_BACKEND_TYPE=vllm \
MEGA_PORT=8422 \
python server.py
```

### 4. Use it like any OpenAI API

```python
import httpx

BASE = "http://localhost:8422/v1"

# Ingest a large document
with open("my_book.txt") as f:
    httpx.post(f"{BASE}/mega-context/add-document", json={"text": f.read()})

# Chat with the full context
resp = httpx.post(f"{BASE}/chat/completions", json={
    "messages": [{"role": "user", "content": "What happened in chapter 3?"}],
    "max_tokens": 2048,
    "temperature": 0.6,
})
print(resp.json()["choices"][0]["message"]["content"])
```

## API Endpoints

### Standard OpenAI-compatible
- `GET /v1/models` — list available models
- `POST /v1/chat/completions` — chat with retrieval (supports streaming)

### MegaContext extensions
- `POST /v1/mega-context/add-document` — bulk ingest text `{"text": "..."}`
- `POST /v1/mega-context/add` — add to conversation history
- `POST /v1/mega-context/clear` — reset context
- `GET /v1/mega-context/stats` — index statistics
- `POST /v1/mega-context/save` — persist index to disk `{"path": "/tmp/idx.pt"}`
- `POST /v1/mega-context/load` — restore saved index

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `MEGA_MODEL` | `Qwen/Qwen3.6-27B` | HuggingFace model ID (for tokenizer + embeddings) |
| `MEGA_BACKEND_URL` | `http://localhost:8421` | Backend inference server URL |
| `MEGA_BACKEND_TYPE` | `llamacpp` | Backend type: `vllm` or `llamacpp` |
| `MEGA_BACKEND_MODEL` | same as MEGA_MODEL | Model name for vLLM API calls |
| `MEGA_PORT` | `8422` | Port for MegaContext server |
| `MEGA_DEVICE` | `cuda` | Device for embedding index |

## Using with llama.cpp

```bash
# Start llama.cpp server
./llama-server -m model.gguf -c 131072 --port 8421

# Start MegaContext (llamacpp is the default backend)
MEGA_MODEL=Qwen/Qwen3.6-27B \
MEGA_BACKEND_TYPE=llamacpp \
python server.py
```

## Architecture

```
Client (OpenAI SDK / curl)
    │
    ▼
MegaContext Server (:8422)
    ├── Tokenizer (from HuggingFace)
    ├── Embedding Table (~2GB, bf16)
    ├── Slab-based Index (GPU, fp32 normalized)
    │     └── Mean-pooled 64-token chunks
    └── Prompt Builder
          └── [HEAD] + [top-k retrieved] + [TAIL]
    │
    ▼
Backend (vLLM / llama.cpp :8421)
    └── Token IDs → Generation
```

## BABILong Benchmark Results (Qwen3.6-35B-A3B)

| Length | qa1 (baseline → mega) | qa2 (baseline → mega) | qa3 (baseline → mega) |
|--------|---|---|---|
| 0k | 100% → 100% | 96% → 100% | 85% → 99% |
| 1k | 97% → 100% | 86% → 97% | 75% → 99% |
| 4k | 99% → 100% | 75% → 93% | 47% → 96% |
| 8k | 97% → 99% | 67% → 89% | 45% → 90% |
| 16k | 95% → 99% | 55% → 86% | 37% → 91% |
| 32k | 88% → 99% | 43% → 71% | 35% → TBD |
| 64k | 72% → 93% | 37% → 67% | 30% → TBD |
| 128k | 70% → 83% | 33% → 55% | 28% → TBD |

## License

Apache 2.0

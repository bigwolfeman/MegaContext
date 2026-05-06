# MegaContext

Extend any LLM to **million-token contexts** with zero training. Drop-in OpenAI-compatible proxy — point your coding tools at it and context overflow is handled automatically.

## How It Works

MegaContext sits between your client and any inference backend (vLLM, llama.cpp). When a conversation fits in the backend's context window, requests pass through unchanged. When it overflows, MegaContext automatically:

1. Indexes the overflow into a GPU-resident embedding index
2. Retrieves the most relevant chunks for the current query
3. Builds a compressed prompt that fits the backend's context window

```
Conversation fits?  → Pass through directly (zero overhead)
Conversation overflows? → Index → Retrieve → Compressed prompt
```

- **Truly drop-in** — standard `/v1/chat/completions`, no special API calls needed
- **No training required** — uses the model's own embedding table
- **No external vector DB** — GPU-resident slab-based index
- **Works with any tool** — OpenCode, Cursor, aider, Continue, SWE-agent, or plain curl

## Quick Start

### 1. Start your backend

**vLLM:**
```bash
docker run --gpus all -p 8421:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.6-27B \
  --gpu-memory-utilization 0.70 \
  --max-model-len 131072 \
  --dtype bfloat16
```

**llama.cpp:**
```bash
llama-server -m model.gguf -c 131072 --port 8421
```

### 2. Start MegaContext

```bash
pip install -e .

MEGA_MODEL=Qwen/Qwen3.6-27B \
MEGA_BACKEND_URL=http://localhost:8421 \
MEGA_BACKEND_TYPE=vllm \
python server.py
```

### 3. Point your tools at it

```bash
# OpenCode
OPENAI_BASE_URL=http://localhost:8422/v1 opencode

# aider
aider --openai-api-base http://localhost:8422/v1

# Any OpenAI SDK client
export OPENAI_BASE_URL=http://localhost:8422/v1
```

That's it. Your tools work exactly as before, but now they can handle conversations that exceed the model's native context window.

### Python example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8422/v1", api_key="not-needed")

# Works like normal — if the conversation is small, it passes through.
# If it's huge (100K+ tokens of code context), MegaContext handles it.
resp = client.chat.completions.create(
    model="qwen3.6-27b-mega",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": huge_codebase_context + "\n\nFix the bug in auth.py"},
    ],
    max_tokens=4096,
)
print(resp.choices[0].message.content)
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `MEGA_MODEL` | `Qwen/Qwen3.6-27B` | HuggingFace model ID (tokenizer + embeddings) |
| `MEGA_BACKEND_URL` | `http://localhost:8421` | Backend inference server URL |
| `MEGA_BACKEND_TYPE` | `llamacpp` | Backend type: `vllm` or `llamacpp` |
| `MEGA_BACKEND_MODEL` | same as MEGA_MODEL | Model name for vLLM API calls |
| `MEGA_PORT` | `8422` | Port for MegaContext server |
| `MEGA_DEVICE` | `cuda` | Device for embedding index |
| `MEGA_MAX_CONTEXT` | `131072` | Backend's max context length |
| `MEGA_CONTEXT_MARGIN` | `8192` | Reserved tokens for generation headroom |

## Architecture

```
Client (OpenCode / Cursor / aider / curl)
    │
    ▼
MegaContext Server (:8422)
    ├── Fits in context? → Pass through to backend (zero overhead)
    └── Overflow?
          ├── Tokenizer (from HuggingFace)
          ├── Embedding Table (~2GB, bf16)
          ├── Slab Index (GPU, fp32 normalized)
          │     └── 64-token chunks, mean-pooled embeddings
          └── Prompt: [system] + [recent history tail] + [retrieved chunks] + [query]
    │
    ▼
Backend (vLLM / llama.cpp :8421)
```

## Power-user endpoints

For advanced use cases (bulk document ingestion, persistent sessions):

- `POST /v1/mega-context/add-document` — bulk ingest text `{"text": "..."}`
- `POST /v1/mega-context/add` — add to conversation context
- `POST /v1/mega-context/clear` — reset context
- `GET /v1/mega-context/stats` — index statistics
- `POST /v1/mega-context/save` — persist index to disk
- `POST /v1/mega-context/load` — restore saved index

## BABILong Benchmark Results (Qwen3.6-35B-A3B)

| Length | qa1 (baseline / mega) | qa2 (baseline / mega) | qa3 (baseline / mega) |
|--------|---|---|---|
| 0k | 100% / 100% | 96% / 100% | 85% / 99% |
| 1k | 97% / 100% | 86% / 97% | 75% / 99% |
| 4k | 99% / 100% | 75% / 93% | 47% / 96% |
| 8k | 97% / 99% | 67% / 89% | 45% / 90% |
| 16k | 95% / 99% | 55% / 86% | 37% / 91% |
| 32k | 88% / 99% | 43% / 71% | 35% / 87% |
| 64k | 72% / 93% | 37% / 67% | 30% / 83% |
| 128k | 70% / 83% | 33% / 55% | 28% / TBD |

## License

Apache 2.0

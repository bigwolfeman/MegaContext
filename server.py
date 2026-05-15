"""MegaContext Server: OpenAI-compatible proxy with million-token context.

Drop-in replacement for any OpenAI-compatible endpoint. Point your coding
tool (OpenCode, Cursor, aider, etc.) at this server instead of the backend
directly. If the conversation fits in the backend's context window, requests
pass through unchanged. When it overflows, MegaContext automatically compresses
and retrieves relevant context.

No special API calls needed — just use /v1/chat/completions as normal.
"""

import asyncio
import os
import sys
import json
import time
import re

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

MODEL_ID = os.environ.get('MEGA_MODEL', 'Qwen/Qwen3.6-27B')
MODEL_NAME = os.environ.get('MEGA_MODEL_NAME', 'qwen3.6-27b-mega')
BACKEND_URL = os.environ.get('MEGA_BACKEND_URL', 'http://localhost:8421')
BACKEND_TYPE = os.environ.get('MEGA_BACKEND_TYPE', 'llamacpp')
BACKEND_MODEL = os.environ.get('MEGA_BACKEND_MODEL', MODEL_ID)
PORT = int(os.environ.get('MEGA_PORT', '8422'))
DEVICE = os.environ.get('MEGA_DEVICE', 'cuda')
MAX_CONTEXT = int(os.environ.get('MEGA_MAX_CONTEXT', '131072'))
CONTEXT_MARGIN = int(os.environ.get('MEGA_CONTEXT_MARGIN', '8192'))

app = FastAPI(title='MegaContext Server')
mega_ctx = None


def _encode(text):
    return mega_ctx.tokenizer.encode(text, add_special_tokens=False)


def _count_tokens(text):
    return len(_encode(text))


def _messages_to_text(messages):
    """Serialize messages to chat-template text for tokenization/ingestion."""
    parts = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if isinstance(content, list):
            content = ' '.join(
                c.get('text', '') for c in content if c.get('type') == 'text')
        parts.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
    return ''.join(parts)


QUERY_TAIL_CHARS = 4000

def _split_user_message(text):
    """Split a large user message into (query, context_to_index).

    Tries semantic boundaries first (common prompt markers), then falls back
    to keeping the last QUERY_TAIL_CHARS as the query.
    """
    markers = [
        '\nPlease first localize',
        '\nIMPORTANT:',
        '\nRULES:',
        '\n--- END FILE ---',
        '\n</code>',
        '\n```\n\n',
    ]
    best_split = -1
    for marker in markers:
        idx = text.rfind(marker)
        if idx > 0:
            split_at = idx + len(marker)
            if split_at < len(text) - 100:
                best_split = split_at
                break

    if best_split > 0:
        context = text[:best_split]
        query = text[best_split:].strip()
        if not query:
            query = text[-QUERY_TAIL_CHARS:]
            context = text[:-QUERY_TAIL_CHARS]
    else:
        query = text[-QUERY_TAIL_CHARS:]
        context = text[:-QUERY_TAIL_CHARS]

    return query, context


def _build_backend_request_passthrough(messages, max_tokens, temperature, top_p, stream, extra=None):
    """Build a passthrough request — send messages directly to backend's chat endpoint."""
    if BACKEND_TYPE == 'vllm':
        body = {
            'model': BACKEND_MODEL,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stream': stream,
        }
        if extra:
            for k in ('n', 'stop', 'frequency_penalty', 'presence_penalty',
                       'logprobs', 'top_logprobs', 'response_format'):
                if k in extra:
                    body[k] = extra[k]
        url = f'{BACKEND_URL}/v1/chat/completions'
    else:
        text = _messages_to_text(messages)
        tokens = _encode(text + '<|im_start|>assistant\n')
        body = {
            'prompt': tokens,
            'n_predict': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stream': stream,
        }
        url = f'{BACKEND_URL}/completion'
    return url, body


def _build_backend_request_tokens(prompt_tokens, max_tokens, temperature, top_p, stream):
    """Build a token-level request for MegaContext prompts."""
    if BACKEND_TYPE == 'vllm':
        body = {
            'model': BACKEND_MODEL,
            'prompt': prompt_tokens,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stream': stream,
        }
        url = f'{BACKEND_URL}/v1/completions'
    else:
        body = {
            'prompt': prompt_tokens,
            'n_predict': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stream': stream,
        }
        url = f'{BACKEND_URL}/completion'
    return url, body


def _build_mega_prompt(retrieved_tokens, last_user, system_msg='', history_tokens=None):
    """Build prompt with retrieved context injected.

    Structure:
      [system message]
      [history tail — recent conversation turns]
      <|im_start|>user
      <context>
      {retrieved chunks}
      </context>

      {last user message}
      <|im_end|>
      <|im_start|>assistant
      <think>
    """
    parts = []

    if system_msg:
        parts.extend(_encode(f'<|im_start|>system\n{system_msg}<|im_end|>\n'))

    if history_tokens:
        parts.extend(history_tokens)

    if retrieved_tokens:
        parts.extend(_encode('<|im_start|>user\n<context>\n'))
        parts.extend(retrieved_tokens)
        parts.extend(_encode(f'\n</context>\n\n{last_user}<|im_end|>\n'))
    else:
        parts.extend(_encode(f'<|im_start|>user\n{last_user}<|im_end|>\n'))

    parts.extend(_encode('<|im_start|>assistant\n<think>\n'))
    return parts


def init():
    global mega_ctx
    from mega_context import MegaContext

    print(f"Loading MegaContext for {MODEL_ID}...")
    mega_ctx = MegaContext(
        MODEL_ID,
        device=DEVICE,
        head_size=0,
        tail_size=65536,
        retrieve_k=1024,
        chunk_batch=131072,
        embed_batch=2048,
    )
    print(f"Ready. Backend: {BACKEND_URL} ({BACKEND_TYPE})")
    print(f"Max context: {MAX_CONTEXT} (margin: {CONTEXT_MARGIN})")


@app.get('/v1/models')
async def list_models():
    return {'object': 'list', 'data': [{
        'id': MODEL_NAME,
        'object': 'model',
        'created': int(time.time()),
        'owned_by': 'mega-context',
    }]}


@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get('messages', [])
    stream = body.get('stream', False)
    max_tokens = body.get('max_tokens', 4096)
    temperature = body.get('temperature', 0.7)
    top_p = body.get('top_p', 0.9)

    full_text = _messages_to_text(messages)
    total_tokens = await asyncio.to_thread(_count_tokens, full_text)
    budget = MAX_CONTEXT - CONTEXT_MARGIN - max_tokens

    if total_tokens <= budget:
        extra = {k: body[k] for k in ('n', 'stop', 'frequency_penalty', 'presence_penalty',
                                       'logprobs', 'top_logprobs', 'response_format')
                 if k in body}
        url, backend_body = _build_backend_request_passthrough(
            messages, max_tokens, temperature, top_p, stream, extra)
        async with httpx.AsyncClient(timeout=900.0) as client:
            if stream:
                return await _handle_stream_passthrough(client, url, backend_body)
            else:
                return await _handle_sync_passthrough(client, url, backend_body)

    system_msg = ''
    history_msgs = []
    last_user = ''
    for msg in messages:
        if msg.get('role') == 'system':
            system_msg = msg.get('content', '')
        elif msg.get('role') == 'user':
            if last_user:
                history_msgs.append({'role': 'user', 'content': last_user})
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(
                    c.get('text', '') for c in content if c.get('type') == 'text')
            last_user = content
        elif msg.get('role') == 'assistant':
            if last_user:
                history_msgs.append({'role': 'user', 'content': last_user})
                last_user = ''
            history_msgs.append(msg)

    mega_ctx.clear()

    if history_msgs:
        history_text = _messages_to_text(history_msgs)
        await asyncio.to_thread(mega_ctx.add_document, history_text)
    elif last_user:
        last_user_tokens = await asyncio.to_thread(_count_tokens, last_user)
        if last_user_tokens > budget:
            query_text, context_text = _split_user_message(last_user)
            if context_text:
                await asyncio.to_thread(mega_ctx.add_document, context_text)
            last_user = query_text

    retrieved_tokens = await asyncio.to_thread(mega_ctx.build_prompt, last_user)

    history_tail = mega_ctx._tail_tokens[:] if mega_ctx._tail_tokens else None

    full_prompt = _build_mega_prompt(
        retrieved_tokens, last_user, system_msg,
        history_tokens=history_tail)

    prompt_len = len(full_prompt)
    if prompt_len > budget:
        overshoot = prompt_len - budget
        if retrieved_tokens and len(retrieved_tokens) > overshoot:
            retrieved_tokens = retrieved_tokens[:-overshoot]
            full_prompt = _build_mega_prompt(
                retrieved_tokens, last_user, system_msg,
                history_tokens=history_tail)

    stats = mega_ctx.stats
    print(f"[MegaContext] {total_tokens:,} tok → {len(full_prompt):,} tok prompt "
          f"({stats['index_chunks']} chunks indexed, "
          f"{len(retrieved_tokens) if retrieved_tokens else 0:,} retrieved)")

    url, backend_body = _build_backend_request_tokens(
        full_prompt, max_tokens, temperature, top_p, stream)

    async with httpx.AsyncClient(timeout=900.0) as client:
        if stream:
            return await _handle_stream_mega(client, url, backend_body, stats)
        else:
            return await _handle_sync_mega(client, url, backend_body, stats)


# --- Passthrough handlers (conversation fits in context) ---

async def _handle_stream_passthrough(client, url, backend_body):
    async def proxy():
        async with client.stream('POST', url, json=backend_body) as resp:
            async for line in resp.aiter_lines():
                yield line + '\n'
    return StreamingResponse(proxy(), media_type='text/event-stream')


async def _handle_sync_passthrough(client, url, backend_body):
    resp = await client.post(url, json=backend_body)
    result = resp.json()
    if BACKEND_TYPE == 'vllm':
        for choice in result.get('choices', []):
            msg = choice.get('message', {})
            if msg.get('content') and '</think>' in msg['content']:
                msg['content'] = re.sub(
                    r'^.*?</think>\s*', '', msg['content'], flags=re.DOTALL).strip()
        result['model'] = MODEL_NAME
        return result
    else:
        text = result.get('content', '')
        text = re.sub(r'^.*?</think>\s*', '', text, flags=re.DOTALL).strip() if '</think>' in text else text
        usage = {
            'prompt_tokens': result.get('tokens_evaluated', 0),
            'completion_tokens': result.get('tokens_predicted', 0),
            'total_tokens': result.get('tokens_evaluated', 0) + result.get('tokens_predicted', 0),
        }
        return {
            'id': f'chatcmpl-{int(time.time())}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': MODEL_NAME,
            'choices': [{'index': 0,
                        'message': {'role': 'assistant', 'content': text},
                        'finish_reason': 'stop'}],
            'usage': usage,
        }


# --- MegaContext handlers (overflow path) ---

async def _handle_stream_mega(client, url, backend_body, stats):
    async def stream_proxy():
        full_text = ""
        cid = f'chatcmpl-{int(time.time())}'
        async with client.stream('POST', url, json=backend_body) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith('data: '):
                    continue
                payload = line[6:]
                if payload.strip() == '[DONE]':
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue

                if BACKEND_TYPE == 'vllm':
                    choices = chunk.get('choices', [{}])
                    token_text = choices[0].get('text', '') if choices else ''
                    stop = choices[0].get('finish_reason') is not None if choices else False
                else:
                    token_text = chunk.get('content', '')
                    stop = chunk.get('stop', False)

                full_text += token_text
                oa_chunk = {
                    'id': cid,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': MODEL_NAME,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': token_text},
                        'finish_reason': 'stop' if stop else None,
                    }],
                }
                yield f'data: {json.dumps(oa_chunk)}\n\n'
                if stop:
                    break

        yield 'data: [DONE]\n\n'

    return StreamingResponse(stream_proxy(), media_type='text/event-stream')


async def _handle_sync_mega(client, url, backend_body, stats):
    resp = await client.post(url, json=backend_body)
    result = resp.json()

    if 'error' in result:
        return JSONResponse(result, status_code=500)

    if BACKEND_TYPE == 'vllm':
        choices = result.get('choices', [{}])
        text = choices[0].get('text', '') if choices else ''
        usage = result.get('usage', {})
    else:
        text = result.get('content', '')
        usage = {
            'prompt_tokens': result.get('tokens_evaluated', 0),
            'completion_tokens': result.get('tokens_predicted', 0),
            'total_tokens': result.get('tokens_evaluated', 0) + result.get('tokens_predicted', 0),
        }

    text = re.sub(r'^.*?</think>\s*', '', text, flags=re.DOTALL).strip() if '</think>' in text else text

    return {
        'id': f'chatcmpl-{int(time.time())}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': MODEL_NAME,
        'choices': [{'index': 0,
                    'message': {'role': 'assistant', 'content': text},
                    'finish_reason': 'stop'}],
        'usage': usage,
    }


# --- Power-user endpoints (optional, not needed for drop-in use) ---

@app.post('/v1/mega-context/add')
async def add_context(request: Request):
    body = await request.json()
    text = body.get('text', '')
    if text:
        await asyncio.to_thread(mega_ctx.add, text)
    return mega_ctx.stats


@app.post('/v1/mega-context/add-document')
async def add_document(request: Request):
    body = await request.json()
    text = body.get('text', '')
    if text:
        await asyncio.to_thread(mega_ctx.add_document, text)
    return mega_ctx.stats


@app.post('/v1/mega-context/clear')
async def clear_context():
    mega_ctx.clear()
    return {'status': 'cleared'}


@app.get('/v1/mega-context/stats')
async def get_stats():
    return mega_ctx.stats if mega_ctx else {'error': 'not initialized'}


@app.post('/v1/mega-context/save')
async def save_index(request: Request):
    body = await request.json()
    path = body.get('path', '/tmp/mega_index.pt')
    await asyncio.to_thread(mega_ctx.save_index, path)
    return {'saved': path, **mega_ctx.stats}


@app.post('/v1/mega-context/load')
async def load_index(request: Request):
    body = await request.json()
    path = body.get('path', '/tmp/mega_index.pt')
    await asyncio.to_thread(mega_ctx.load_index, path)
    return {'loaded': path, **mega_ctx.stats}


if __name__ == '__main__':
    init()
    print(f"\nMegaContext Server: http://0.0.0.0:{PORT}/v1")
    print(f"Backend: {BACKEND_TYPE} @ {BACKEND_URL}")
    uvicorn.run(app, host='0.0.0.0', port=PORT)

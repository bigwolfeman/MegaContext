"""MegaContext Server: OpenAI-compatible API with million-token context.

Sits between any OpenAI-compatible client and a backend (llama.cpp or vLLM).
Handles context compression + retrieval transparently.
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

app = FastAPI(title='MegaContext Server')
mega_ctx = None


def _build_backend_request(prompt_tokens, max_tokens, temperature, top_p, stream):
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


def _encode(text):
    return mega_ctx.tokenizer.encode(text, add_special_tokens=False)


def _build_chat_prompt(context_tokens, last_user, system_msg=''):
    """Build a prompt that matches the model's chat template structure.

    Target structure:
      [<|im_start|>system\n{sys}<|im_end|>\n]
      <|im_start|>user\n{preamble}\n\n<context>\n{context}\n</context>\n\n{question}
      <|im_end|>\n
      <|im_start|>assistant\n<think>\n
    """
    parts = []

    if system_msg:
        parts.extend(_encode(f'<|im_start|>system\n{system_msg}<|im_end|>\n'))

    split_marker = '\nQuestion:'
    idx = last_user.rfind(split_marker)
    if idx >= 0 and context_tokens:
        preamble = last_user[:idx]
        question_part = last_user[idx+1:]
        parts.extend(_encode(f'<|im_start|>user\n{preamble}\n\n<context>\n'))
        parts.extend(context_tokens)
        parts.extend(_encode(f'\n</context>\n\n{question_part}<|im_end|>\n'))
    elif context_tokens:
        parts.extend(_encode('<|im_start|>user\n<context>\n'))
        parts.extend(context_tokens)
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
        head_size=65536,
        tail_size=65536,
        retrieve_k=1024,
        chunk_batch=131072,
        embed_batch=2048,
    )
    print(f"Ready. Backend: {BACKEND_URL} ({BACKEND_TYPE})")
    print(f"Stats: {mega_ctx.stats}")


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

    system_msg = next(
        (m['content'] for m in messages if m.get('role') == 'system'), '')
    last_user = next(
        (m.get('content', '') for m in reversed(messages)
         if m.get('role') == 'user'), '')

    stats = mega_ctx.stats

    context_tokens = await asyncio.to_thread(mega_ctx.build_prompt, last_user)

    full_prompt = _build_chat_prompt(context_tokens, last_user, system_msg)

    url, backend_body = _build_backend_request(
        full_prompt, max_tokens, temperature, body.get('top_p', 0.9), stream)

    async with httpx.AsyncClient(timeout=900.0) as client:
        if stream:
            return await _handle_stream(client, url, backend_body, stats, last_user)
        else:
            return await _handle_sync(client, url, backend_body, stats, last_user)


async def _handle_stream(client, url, backend_body, stats, last_user):
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

        if stats['index_tokens'] > 0:
            footer = _stats_footer(stats)
            footer_chunk = {
                'id': cid,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': MODEL_NAME,
                'choices': [{'index': 0,
                            'delta': {'content': footer},
                            'finish_reason': None}],
            }
            yield f'data: {json.dumps(footer_chunk)}\n\n'
        yield 'data: [DONE]\n\n'

        clean = re.sub(
            r'<think>.*?</think>\s*', '', full_text, flags=re.DOTALL).strip()
        if clean:
            await asyncio.to_thread(
                mega_ctx.add,
                f'<|im_start|>user\n{last_user}<|im_end|>\n'
                f'<|im_start|>assistant\n{clean}<|im_end|>\n')

    return StreamingResponse(stream_proxy(), media_type='text/event-stream')


async def _handle_sync(client, url, backend_body, stats, last_user):
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
        tokens_predicted = result.get('tokens_predicted', 0)
        tokens_evaluated = result.get('tokens_evaluated', 0)
        usage = {
            'prompt_tokens': tokens_evaluated,
            'completion_tokens': tokens_predicted,
            'total_tokens': tokens_evaluated + tokens_predicted,
        }

    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    await asyncio.to_thread(
        mega_ctx.add,
        f'<|im_start|>user\n{last_user}<|im_end|>\n'
        f'<|im_start|>assistant\n{text}<|im_end|>\n')

    if stats['index_tokens'] > 0:
        text += _stats_footer(stats)

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


def _stats_footer(stats: dict) -> str:
    return (f"\n\n---\n*📚 {stats['index_tokens']:,} tokens indexed "
            f"| {stats['index_chunks']} chunks "
            f"| tail: {stats['tail_tokens']:,}*")


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

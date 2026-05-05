"""MegaContext — extend any LLM to million-token contexts.

Uses mean-embedding cosine retrieval over an incremental slab-based index.
No learned compressor needed — raw embeddings with 65%+ score margins.

Usage:
    ctx = MegaContext("Qwen/Qwen3.6-27B", device="cuda")
    ctx.add_document(book_text)
    ctx.add("User: What happened in chapter 3?\nAssistant:")
    prompt_ids = ctx.build_prompt()
"""

import numpy as np
import torch

from mega_context.index import IncrementalIndex


class MegaContext:
    """Extend any frozen LLM to million-token contexts.

    Three-part prompt: [HEAD] + [RETRIEVED] + [TAIL]
    Overflow from tail auto-compresses into the retrieval index.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        head_size: int = 65536,
        tail_size: int = 65536,
        retrieve_k: int = 1024,
        chunk_size: int = 64,
        chunk_batch: int = 131072,
        embed_batch: int = 2048,
    ):
        from transformers import AutoTokenizer
        from mega_context.embeddings import load_embed_table

        self.device = torch.device(device)
        self.head_size = head_size
        self.tail_size = tail_size
        self.retrieve_k = retrieve_k
        self.chunk_size = chunk_size
        self.chunk_batch = (chunk_batch // chunk_size) * chunk_size
        self.embed_batch = embed_batch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.embed_fn, d_model = load_embed_table(model_id, device=device)
        self.d_model = d_model

        self.index = IncrementalIndex(
            d_model=d_model, chunk_size=chunk_size, device=device)

        self._head_tokens: list[int] = []
        self._tail_tokens: list[int] = []

    def set_head(self, text_or_tokens):
        if isinstance(text_or_tokens, str):
            self._head_tokens = self.tokenizer.encode(
                text_or_tokens, add_special_tokens=False)[:self.head_size]
        else:
            self._head_tokens = list(text_or_tokens)[:self.head_size]

    def add(self, text_or_tokens):
        """Add content to conversation. Overflow auto-compresses to index."""
        if isinstance(text_or_tokens, str):
            new_tokens = self.tokenizer.encode(text_or_tokens, add_special_tokens=False)
        else:
            new_tokens = list(text_or_tokens)
        self._tail_tokens.extend(new_tokens)
        self._maybe_compress()

    def add_document(self, text_or_tokens):
        """Bulk-ingest a large document directly into the index."""
        if isinstance(text_or_tokens, str):
            tokens = self.tokenizer.encode(text_or_tokens, add_special_tokens=False)
        else:
            tokens = list(text_or_tokens)

        all_tokens = self._tail_tokens + tokens
        self._tail_tokens = []

        if len(all_tokens) <= self.tail_size:
            self._tail_tokens = all_tokens
            return

        compress_tokens = all_tokens[:-self.tail_size]
        self._tail_tokens = all_tokens[-self.tail_size:]
        self._compress_and_append(compress_tokens)

    def _maybe_compress(self):
        while len(self._tail_tokens) > self.tail_size + self.chunk_batch:
            overflow = self._tail_tokens[:self.chunk_batch]
            self._tail_tokens = self._tail_tokens[self.chunk_batch:]
            self._compress_and_append(overflow)

    def _compress_and_append(self, tokens: list[int]):
        ids = np.array(tokens, dtype=np.int32)
        n_chunks = len(ids) // self.chunk_size
        if n_chunks == 0:
            return

        usable = n_chunks * self.chunk_size
        chunk_ids = ids[:usable].reshape(n_chunks, self.chunk_size)

        all_embs = []
        with torch.no_grad():
            for i in range(0, n_chunks, self.embed_batch):
                batch = torch.from_numpy(chunk_ids[i:i+self.embed_batch]).to(
                    device=self.device, dtype=torch.long)
                embs = self.embed_fn(batch).float().mean(dim=1)
                all_embs.append(embs)

        mean_embs = torch.cat(all_embs, dim=0)
        mean_embs = torch.nn.functional.normalize(mean_embs, dim=-1)
        self.index.append(mean_embs, ids[:usable])
        del mean_embs, all_embs

    def build_prompt(self, query: str | list[int] | None = None) -> list[int]:
        """Build prompt: head + retrieved chunks + tail."""
        if query is None:
            q_tokens = self._tail_tokens[-256:] if self._tail_tokens else []
        elif isinstance(query, str):
            q_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        else:
            q_tokens = list(query)

        retrieved_tokens = []
        if self.index.size > 0 and q_tokens:
            with torch.no_grad():
                qt = torch.tensor([q_tokens], device=self.device, dtype=torch.long)
                qe = self.embed_fn(qt).float().mean(dim=1)
                qe = torch.nn.functional.normalize(qe, dim=-1).squeeze(0)
            retrieved_tokens = self.index.retrieve(qe, self.retrieve_k)

        return self._head_tokens + retrieved_tokens + self._tail_tokens

    def clear(self):
        self._tail_tokens = []
        self.index = IncrementalIndex(
            d_model=self.d_model, chunk_size=self.chunk_size, device=self.device)

    def save_index(self, path: str):
        self.index.save(path)

    def load_index(self, path: str):
        self.index.load(path)

    @property
    def stats(self) -> dict:
        return {
            'head_tokens': len(self._head_tokens),
            'tail_tokens': len(self._tail_tokens),
            'index_chunks': self.index.size,
            'index_tokens': self.index.total_tokens,
            'index_memory_mb': self.index.memory_bytes / 1e6,
            'total_tokens': (len(self._head_tokens) + len(self._tail_tokens)
                            + self.index.total_tokens),
            'prompt_tokens': (len(self._head_tokens)
                             + min(self.retrieve_k * self.chunk_size,
                                   self.index.total_tokens)
                             + len(self._tail_tokens)),
        }

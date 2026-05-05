"""Append-only incremental index for chunk retrieval.

Stores mean embedding per chunk for cosine similarity retrieval.
Uses chunked storage to avoid CUDA allocator bloat from repeated torch.cat.
"""

import numpy as np
import torch
import torch.nn.functional as F

SLAB_SIZE = 4096


class IncrementalIndex:
    """Append-only index of chunk embeddings. Never rebuilds."""

    def __init__(self, d_model=3584, chunk_size=64, device='cuda', **kwargs):
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.device = device

        self._slabs = []
        self._slab_idx = -1
        self._slab_offset = 0
        self._token_ids = []
        self._n_chunks = 0

    def pre_allocate(self, n_chunks: int):
        """Pre-allocate slabs for expected number of chunks."""
        n_slabs_needed = (n_chunks + SLAB_SIZE - 1) // SLAB_SIZE
        for _ in range(len(self._slabs), n_slabs_needed):
            self._slabs.append(torch.zeros(
                SLAB_SIZE, self.d_model, dtype=torch.float32, device=self.device))

    def _ensure_slab(self):
        if self._slab_idx < 0 or self._slab_offset >= SLAB_SIZE:
            next_idx = self._slab_idx + 1
            if next_idx < len(self._slabs):
                self._slab_idx = next_idx
            else:
                self._slabs.append(torch.zeros(
                    SLAB_SIZE, self.d_model, dtype=torch.float32, device=self.device))
                self._slab_idx = len(self._slabs) - 1
            self._slab_offset = 0

    def append(self, embeddings: torch.Tensor, token_ids: np.ndarray):
        """Append normalized mean chunk embeddings and source token IDs."""
        n_new = embeddings.shape[0]
        embs = embeddings.to(device=self.device, dtype=torch.float32).detach()
        ids = token_ids.reshape(n_new, self.chunk_size)

        offset = 0
        while offset < n_new:
            self._ensure_slab()
            slab = self._slabs[self._slab_idx]
            space = SLAB_SIZE - self._slab_offset
            batch = min(space, n_new - offset)
            slab[self._slab_offset:self._slab_offset + batch] = embs[offset:offset + batch]
            self._slab_offset += batch
            offset += batch

        for i in range(n_new):
            self._token_ids.append(ids[i].copy())
        self._n_chunks += n_new

    def _get_all_embeddings(self) -> torch.Tensor:
        """Materialize all embeddings as a single contiguous tensor for scoring."""
        if self._n_chunks == 0:
            return torch.empty(0, self.d_model, device=self.device)
        parts = []
        for i, slab in enumerate(self._slabs):
            if i < self._slab_idx:
                parts.append(slab)
            elif i == self._slab_idx:
                parts.append(slab[:self._slab_offset])
        return torch.cat(parts, dim=0)

    def retrieve(self, query_emb: torch.Tensor, top_k: int) -> list[int]:
        """Cosine similarity retrieval."""
        if self._n_chunks == 0:
            return []

        k = min(top_k, self._n_chunks)
        q = query_emb.to(dtype=torch.float32, device=self.device)
        if q.dim() == 1:
            q = q.unsqueeze(0)

        all_embs = self._get_all_embeddings()
        scores = (q @ all_embs.T).squeeze(0)
        top_indices = scores.topk(k).indices.tolist()

        selected = sorted(top_indices)
        token_ids = []
        for idx in selected:
            token_ids.extend(self._token_ids[idx].tolist())
        return token_ids

    def save(self, path: str):
        data = {
            'embeddings': self._get_all_embeddings().cpu(),
            'token_ids': np.stack(self._token_ids) if self._token_ids else np.array([]),
            'n_chunks': self._n_chunks,
            'd_model': self.d_model,
            'chunk_size': self.chunk_size,
        }
        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.d_model = data['d_model']
        self.chunk_size = data['chunk_size']
        self._n_chunks = 0
        self._slabs = []
        self._slab_idx = -1
        self._slab_offset = 0
        self._token_ids = []
        if data['embeddings'] is not None and len(data['embeddings']) > 0:
            embs = data['embeddings'].to(device=self.device, dtype=torch.float32)
            self.append(embs, data['token_ids'].reshape(-1))
        ids = data['token_ids']
        self._token_ids = [ids[i] for i in range(len(ids))] if len(ids) > 0 else []

    @property
    def size(self) -> int:
        return self._n_chunks

    @property
    def total_tokens(self) -> int:
        return self._n_chunks * self.chunk_size

    @property
    def memory_bytes(self) -> int:
        return self._n_chunks * self.d_model * 4

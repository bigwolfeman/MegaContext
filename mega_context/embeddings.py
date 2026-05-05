"""Load only the embedding table from a model — ~2-3GB instead of the full model.

Supports loading from:
  - HuggingFace model ID (downloads/finds cached safetensors)
  - Local directory with safetensors files
"""

import os
import torch
import torch.nn as nn


def load_embed_table(model_id_or_path: str, device='cuda') -> tuple[nn.Embedding, int]:
    """Load just the embedding table from safetensors shards.

    Returns (embedding_layer, d_model).
    """
    import safetensors.torch as st

    model_path = _resolve_path(model_id_or_path)
    shard_files = sorted([
        os.path.join(model_path, f) for f in os.listdir(model_path)
        if f.endswith('.safetensors')
    ])
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files in {model_path}")

    embed_weight = None
    for shard in shard_files:
        tensors = st.load_file(shard)
        for key in tensors:
            if 'embed_tokens' in key:
                embed_weight = tensors[key]
                break
        if embed_weight is not None:
            break
        del tensors

    if embed_weight is None:
        raise ValueError(f"Could not find embed_tokens weight in {model_path}")

    vocab_size, d_model = embed_weight.shape
    embed_layer = nn.Embedding(
        vocab_size, d_model,
        _weight=embed_weight.to(device=device, dtype=torch.bfloat16))
    embed_layer.weight.requires_grad_(False)

    size_gb = embed_weight.numel() * 2 / 1e9
    print(f"Embed table: vocab={vocab_size:,} d_model={d_model} ({size_gb:.1f}GB)")
    return embed_layer, d_model


def _resolve_path(model_id: str) -> str:
    if os.path.isdir(model_id):
        return model_id

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = f"models--{model_id.replace('/', '--')}"
    model_cache = os.path.join(cache_dir, model_dir_name)
    if os.path.isdir(model_cache):
        snapshots = os.path.join(model_cache, "snapshots")
        if os.path.isdir(snapshots):
            versions = os.listdir(snapshots)
            if versions:
                return os.path.join(snapshots, versions[0])

    from huggingface_hub import snapshot_download
    return snapshot_download(model_id)

"""Dense embeddings.

Thin, explicit wrapper around one OpenAI call. This is the only piece that
isn't "from scratch" -- computing a text embedding model is out of scope for
an application -- but nothing about the retrieval *logic* (cosine, ranking,
fusion) lives in here or is hidden by it; that's all in retrieval.py.

Batches requests because embedding one chunk at a time is slow and burns
rate-limit headroom for no reason.
"""

import os
from typing import List

from openai import OpenAI

_BATCH_SIZE = 100


def make_embed_fn(model: str = None, client: OpenAI = None):
    """Returns a function List[str] -> List[List[float]] suitable for
    `retrieval.HybridRetriever(embed_fn=...)`."""
    model = model or os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = client or OpenAI()

    def embed_fn(texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i:i + _BATCH_SIZE]
            resp = client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    return embed_fn

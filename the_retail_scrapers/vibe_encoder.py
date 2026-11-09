# vibe_encoder.py

from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Load once per process
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1024)
def _encode_cached(text: str) -> np.ndarray:
    if not text:
        text = ""
    emb = _MODEL.encode(text, normalize_embeddings=True)
    # ensure numpy array
    return np.asarray(emb, dtype="float32")


def vibe_similarity(user_text: Optional[str], product_text: Optional[str]) -> float:
    """
    Return cosine similarity between user vibe text and product text in [0, 1] ish.
    If either is empty, return 0.0.
    """
    if not user_text or not product_text:
        return 0.0

    u = _encode_cached(user_text)
    p = _encode_cached(product_text)

    # since embeddings are normalized, dot product is cosine similarity
    return float(np.dot(u, p))

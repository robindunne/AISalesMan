#!/usr/bin/env python3
"""
vibe_search_core.py

Provides:
  - load_vibe_model()
  - get_vibe_embedding(model, text)
  - rank_products_by_vibe(products, vibe_emb, top_k)

Used by planner_astar.py to inject transformer "vibe" similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_vibe_model() -> SentenceTransformer:
    """Load transformer model once."""
    return SentenceTransformer(MODEL_NAME)


def get_vibe_embedding(model: SentenceTransformer, text: str) -> np.ndarray:
    """Return L2-normalized embedding for user vibe text."""
    emb = model.encode([text], normalize_embeddings=True)
    return emb[0]


def rank_products_by_vibe(
    products: list,
    vibe_emb: np.ndarray,
    top_k: int = 50,
) -> list:
    """
    Returns top-k products sorted by cosine similarity to vibe_emb.
    Each product must contain "raw_description".
    """
    descs = [(p.get("raw_description") or "") for p in products]
    if not descs:
        return []

    # Encode product descriptions
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    prod_embs = model.encode(descs, normalize_embeddings=True)

    sims = prod_embs @ vibe_emb  # cosine since normalized

    # Top-K indices
    top_k = min(top_k, len(products))
    idxs = np.argsort(-sims)[:top_k]

    ranked = []
    for i in idxs:
        p = dict(products[i])
        p["_vibe_sim"] = float(sims[i])
        ranked.append(p)

    return ranked

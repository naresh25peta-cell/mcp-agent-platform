"""
embedders.py — embedding models the retrieval eval can run with.

- ``HashingEmbedder``: deterministic bag-of-words hashing "embedding". No
  download, no network, no cost, identical results on every machine. It is
  what CI gates on, so a failing build means the *pipeline* regressed
  (chunking, fusion, the MCP tool layer), never that a model download
  hiccuped. It is NOT a measure of production embedding quality.
- ``load_minilm``: the real ``sentence-transformers/all-MiniLM-L6-v2`` model
  the server uses in production. Needs a one-time download from Hugging Face.
"""
from __future__ import annotations

import hashlib

import numpy as np

_DIM = 32


class HashingEmbedder:
    """Same algorithm as the fake model in tests/test_document_qa_server.py."""

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            vec = np.zeros(_DIM, dtype="float32")
            for word in text.lower().split():
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.asarray(vectors, dtype="float32")


def load_minilm():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embedder(name: str):
    if name == "hashing":
        return HashingEmbedder()
    if name == "minilm":
        return load_minilm()
    raise ValueError(f"Unknown embedder '{name}'. Use 'hashing' or 'minilm'.")

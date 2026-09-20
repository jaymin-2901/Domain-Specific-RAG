"""
Phase 2 -- Embeddings & Vector Store.

Two embedders are implemented so you can compare them (Phase 2's explicit
requirement to benchmark at least two on cost/latency/accuracy):

- GeminiEmbedder: calls the Gemini API (gemini-embedding-001). Free tier,
  but has network latency and a rate limit.
- LocalEmbedder: runs a small sentence-transformers model on your own CPU.
  Zero cost, zero rate limit, slightly lower quality on domain-specific text.

VectorStore wraps FAISS (local, free, no server) for the actual similarity
search, using a flat inner-product index (cosine similarity on normalized
vectors). Flat is exact and plenty fast below ~1M vectors, which is why it
was chosen over HNSW/IVF here -- see the README for the trade-off writeup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss
from google import genai
from google.genai import types

import config


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return mat / norms


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class GeminiEmbedder:
    """Embeds text using Gemini's gemini-embedding-001 model (free tier)."""

    name = "gemini-embedding-001"

    def __init__(self, api_key: str | None = None, output_dim: int = config.GEMINI_EMBEDDING_DIM):
        self.client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)
        self.output_dim = output_dim

    def encode(self, texts: list[str], batch_size: int = 32, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # simple retry with backoff for transient 429/5xx errors
            for attempt in range(4):
                try:
                    resp = self.client.models.embed_content(
                        model=config.GEMINI_EMBEDDING_MODEL,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.output_dim,
                        ),
                    )
                    all_vecs.extend([e.values for e in resp.embeddings])
                    break
                except Exception as exc:
                    if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                        raise RuntimeError(
                            "Gemini embedding quota is exhausted. Use the local embedding backend "
                            "for indexing, or wait for the quota window to reset."
                        ) from exc
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        return np.array(all_vecs, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], task_type="RETRIEVAL_QUERY")[0]


class LocalEmbedder:
    """Embeds text locally with sentence-transformers. No API cost, no key."""

    def __init__(self, model_name: str = config.LOCAL_EMBEDDING_MODEL):
        # imported lazily so that using ONLY the Gemini embedder never requires
        # installing/loading the (much heavier) sentence-transformers + torch stack
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts: list[str], batch_size: int = 32, **_) -> np.ndarray:
        return np.array(self.model.encode(texts, batch_size=batch_size, show_progress_bar=False), dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]


# ---------------------------------------------------------------------------
# Vector store (FAISS)
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # inner product on normalized vecs == cosine sim
        self.chunk_ids: list[str] = []

    def build(self, embeddings: np.ndarray, chunk_ids: list[str]):
        embeddings = _normalize(embeddings.astype("float32"))
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.chunk_ids = list(chunk_ids)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = _normalize(query_vec.reshape(1, -1).astype("float32"))
        scores, idxs = self.index.search(q, min(k, len(self.chunk_ids)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def save(self, path: str | Path):
        path = Path(path)
        faiss.write_index(self.index, str(path))

    def load(self, path: str | Path, chunk_ids: list[str]):
        self.index = faiss.read_index(str(path))
        self.chunk_ids = list(chunk_ids)


# ---------------------------------------------------------------------------
# Phase 2 benchmark: compare embedders on latency + a tiny retrieval-accuracy proxy
# ---------------------------------------------------------------------------

@dataclass
class EmbedderBenchmark:
    name: str
    build_time_seconds: float
    avg_query_latency_ms: float
    dim: int


def benchmark_embedder(embedder, chunk_texts: list[str], sample_queries: list[str]) -> EmbedderBenchmark:
    t0 = time.time()
    vecs = embedder.encode(chunk_texts)
    build_time = time.time() - t0

    latencies = []
    for q in sample_queries:
        t0 = time.time()
        embedder.encode_query(q)
        latencies.append((time.time() - t0) * 1000)

    return EmbedderBenchmark(
        name=embedder.name,
        build_time_seconds=round(build_time, 3),
        avg_query_latency_ms=round(sum(latencies) / max(len(latencies), 1), 1),
        dim=vecs.shape[1],
    )

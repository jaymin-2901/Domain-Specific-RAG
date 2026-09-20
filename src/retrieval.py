"""
Phase 3 -- Retrieval + Hybrid Search.

Pure vector search misses exact-match terms (product IDs, error codes,
specific names) because embeddings blur precise tokens into "meaning".
BM25 is the opposite: great at exact terms, blind to paraphrasing.

This module runs both and fuses them with Reciprocal Rank Fusion (RRF),
which just needs each method's *rank* rather than trying to compare
BM25 scores and cosine scores on the same scale (they aren't comparable).

The fused candidates then go through a cross-encoder reranker
(sentence-transformers CrossEncoder, local, free) which reads the
query and each candidate chunk *together* -- far more accurate than
comparing two embeddings independently, at the cost of being too slow
to run over the whole corpus. That's why it only reranks the top ~12
candidates instead of the full index.
"""

from __future__ import annotations

from dataclasses import dataclass
from rank_bm25 import BM25Okapi

import config
from src.ingestion import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    vector_rank: int | None = None
    bm25_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._corpus_tokens = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


class CrossEncoderReranker:
    def __init__(self, model_name: str = config.RERANKER_MODEL):
        # lazy import: keeps sentence-transformers optional if someone strips
        # the reranker out of the pipeline entirely
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_k]]


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    rrf_k: int = config.RRF_K,
) -> dict[str, float]:
    """RRF score = sum over each ranking of 1 / (rrf_k + rank)."""
    scores: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(vector_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (chunk_id, _) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


class HybridRetriever:
    """Vector search + BM25 -> RRF fusion -> cross-encoder rerank."""

    def __init__(self, chunks_by_id: dict[str, Chunk], vector_store, embedder, bm25: BM25Retriever, reranker: CrossEncoderReranker):
        self.chunks_by_id = chunks_by_id
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k_vector: int = config.TOP_K_VECTOR,
        top_k_bm25: int = config.TOP_K_BM25,
        top_k_fused: int = config.TOP_K_RERANK_CANDIDATES,
        top_k_final: int = config.TOP_K_FINAL,
    ) -> list[ScoredChunk]:
        query_vec = self.embedder.encode_query(query)
        vector_results = self.vector_store.search(query_vec, top_k_vector)
        bm25_results = self.bm25.search(query, top_k_bm25)

        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        top_fused_ids = sorted(fused, key=fused.get, reverse=True)[:top_k_fused]
        candidates = [self.chunks_by_id[cid] for cid in top_fused_ids if cid in self.chunks_by_id]

        reranked = self.reranker.rerank(query, candidates, top_k_final)

        vector_rank = {cid: i for i, (cid, _) in enumerate(vector_results)}
        bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25_results)}

        return [
            ScoredChunk(
                chunk=chunk,
                vector_rank=vector_rank.get(chunk.chunk_id),
                bm25_rank=bm25_rank.get(chunk.chunk_id),
                fused_score=fused.get(chunk.chunk_id),
                rerank_score=score,
            )
            for chunk, score in reranked
        ]

    def retrieve_vector_only(self, query: str, top_k: int = config.TOP_K_FINAL) -> list[ScoredChunk]:
        """Used by the eval script to show the hybrid-search improvement concretely."""
        query_vec = self.embedder.encode_query(query)
        results = self.vector_store.search(query_vec, top_k)
        return [
            ScoredChunk(chunk=self.chunks_by_id[cid], vector_rank=i, fused_score=score)
            for i, (cid, score) in enumerate(results) if cid in self.chunks_by_id
        ]

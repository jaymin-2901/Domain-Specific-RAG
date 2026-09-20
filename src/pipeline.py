"""
RAGPipeline -- ties together every phase into one object:

    ingest()  -> Phase 1 (chunk) + Phase 2 (embed + index)
    load()    -> reload a previously built index from disk
    query()   -> Phase 3 (hybrid retrieve + rerank) + Phase 6 (guardrails)
                 + Phase 4 (generate)

This is what app.py (Streamlit, Phase 7) and scripts/run_eval.py (Phase 5)
both drive.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from google import genai

import config
from src.ingestion import load_documents, chunk_documents, Chunk
from src.embeddings import GeminiEmbedder, LocalEmbedder, VectorStore
from src.retrieval import BM25Retriever, CrossEncoderReranker, HybridRetriever, ScoredChunk
from src.generation import generate_answer
from src.guardrails import RateLimiter, TTLCache, CostTracker, sanitize_chunks_for_prompt


class RAGPipeline:
    def __init__(self, api_key: str | None = None, embedder_backend: str = "gemini"):
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY in your .env file "
                "or pass api_key= explicitly. Get a free key at https://aistudio.google.com/apikey"
            )

        self.client = genai.Client(api_key=self.api_key)
        self.embedder_backend = embedder_backend
        self.embedder = GeminiEmbedder(api_key=self.api_key) if embedder_backend == "gemini" else LocalEmbedder()

        self.chunks_by_id: dict[str, Chunk] = {}
        self.vector_store: VectorStore | None = None
        self.bm25: BM25Retriever | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.hybrid: HybridRetriever | None = None

        self.rate_limiter = RateLimiter()
        self.cache = TTLCache()
        self.cost_tracker = CostTracker()

    # ------------------------------------------------------------------
    # Phase 1 + 2: build the index
    # ------------------------------------------------------------------
    def ingest(self, data_dir: str | Path = config.DATA_DIR, chunk_strategy: str = "recursive") -> int:
        docs = load_documents(data_dir)
        if not docs:
            raise ValueError(f"No .md/.txt/.pdf files found in {data_dir}")

        local_embedder_for_chunking = LocalEmbedder() if chunk_strategy == "semantic" else None
        chunks = chunk_documents(
            docs,
            strategy=chunk_strategy,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            local_embedder=local_embedder_for_chunking,
            semantic_threshold=config.SEMANTIC_SIMILARITY_THRESHOLD,
            semantic_max_size=config.SEMANTIC_MAX_CHUNK_SIZE,
        )
        self.chunks_by_id = {c.chunk_id: c for c in chunks}

        texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        embeddings = self.embedder.encode(texts)
        self.cost_tracker.record_embedding(sum(len(t) for t in texts) // 4)  # rough token estimate

        dim = embeddings.shape[1]
        self.vector_store = VectorStore(dim=dim)
        self.vector_store.build(embeddings, chunk_ids)

        self.bm25 = BM25Retriever(chunks)
        self.reranker = CrossEncoderReranker()
        self.hybrid = HybridRetriever(self.chunks_by_id, self.vector_store, self.embedder, self.bm25, self.reranker)

        return len(chunks)

    def save(self, index_dir: str | Path = config.INDEX_DIR) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(index_dir / "vectors.faiss")
        with open(index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks_by_id, f)
        with open(index_dir / "meta.json", "w") as f:
            json.dump({"embedder_backend": self.embedder_backend, "dim": self.vector_store.dim}, f)

    def load(self, index_dir: str | Path = config.INDEX_DIR) -> None:
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())
        with open(index_dir / "chunks.pkl", "rb") as f:
            self.chunks_by_id = pickle.load(f)

        chunks = list(self.chunks_by_id.values())
        chunk_ids = [c.chunk_id for c in chunks]

        self.vector_store = VectorStore(dim=meta["dim"])
        self.vector_store.load(index_dir / "vectors.faiss", chunk_ids)

        self.bm25 = BM25Retriever(chunks)
        self.reranker = CrossEncoderReranker()
        self.hybrid = HybridRetriever(self.chunks_by_id, self.vector_store, self.embedder, self.bm25, self.reranker)

    def index_exists(self, index_dir: str | Path = config.INDEX_DIR) -> bool:
        index_dir = Path(index_dir)
        return (index_dir / "vectors.faiss").exists() and (index_dir / "chunks.pkl").exists()

    def index_backend(self, index_dir: str | Path = config.INDEX_DIR) -> str | None:
        """Return the backend used to create the saved index, if available."""
        meta_path = Path(index_dir) / "meta.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text()).get("embedder_backend")

    # ------------------------------------------------------------------
    # Phase 3 + 4 + 6: answer a question
    # ------------------------------------------------------------------
    def query(self, question: str, session_id: str = "default", use_cache: bool = True) -> dict:
        if self.hybrid is None:
            raise RuntimeError("Pipeline has no index loaded. Call ingest() or load() first.")

        allowed, retry_after = self.rate_limiter.allow(session_id)
        if not allowed:
            return {
                "answer": f"Rate limit exceeded. Try again in {retry_after:.0f} seconds.",
                "used_fallback": True,
                "retrieved_chunks": [],
                "warnings": ["rate_limited"],
                "cache_hit": False,
            }

        if use_cache:
            cached = self.cache.get(question)
            if cached is not None:
                self.cost_tracker.record_query(cache_hit=True)
                return {**cached, "cache_hit": True}

        scored_chunks = self.hybrid.retrieve(question)
        scored_chunks, warnings = sanitize_chunks_for_prompt(scored_chunks)

        result = generate_answer(self.client, question, scored_chunks)

        response = {
            "answer": result.answer,
            "used_fallback": result.used_fallback,
            "retrieved_chunks": scored_chunks,
            "cited_sources": result.cited_sources,
            "warnings": warnings,
            "cache_hit": False,
        }

        approx_in_tokens = len(question) // 4 + sum(len(sc.chunk.text) for sc in scored_chunks) // 4
        approx_out_tokens = len(result.answer) // 4
        self.cost_tracker.record_query(cache_hit=False, gen_input_tokens=approx_in_tokens, gen_output_tokens=approx_out_tokens)

        if use_cache:
            self.cache.set(question, response)

        return response

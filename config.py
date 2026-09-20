"""
Central configuration for the RAG system.

Everything here is free to run:
- Generation + embeddings: Gemini API (free tier, just needs GEMINI_API_KEY)
- Reranker + local embedding baseline: sentence-transformers, runs on your CPU, no API key
- Vector store: FAISS, local, no server needed
- Keyword search: rank_bm25, local, pure Python

Only one paid-capable API is used (Gemini), and only its free tier is required.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "sample_docs"
INDEX_DIR = ROOT_DIR / "index_store"
EVAL_DIR = ROOT_DIR / "eval"
CACHE_DIR = ROOT_DIR / ".cache"

INDEX_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Generation model. Override with GENERATION_MODEL when needed.
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-3.6-flash")

# Embedding model. text-embedding-004 was deprecated Jan 2026 -- do not use it.
# gemini-embedding-001 is the current free-tier embedding model.
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# gemini-embedding-001 natively outputs 3072-dim vectors but supports Matryoshka
# truncation. 768 keeps FAISS fast/small while losing very little quality --
# this trade-off is a good interview talking point for Phase 2.
GEMINI_EMBEDDING_DIM = int(os.environ.get("GEMINI_EMBEDDING_DIM", "768"))

# Model used as the LLM judge in the eval harness (Phase 5).
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.6-flash")

# ---------------------------------------------------------------------------
# Local (free, no API key) models -- used for the embedding comparison in
# Phase 2 and for the cross-encoder reranker in Phase 3.
# ---------------------------------------------------------------------------
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# Chunking (Phase 1)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800          # characters, not tokens -- see ingestion.py docstring
CHUNK_OVERLAP = 150
SEMANTIC_SIMILARITY_THRESHOLD = 0.55
SEMANTIC_MAX_CHUNK_SIZE = 1200

# ---------------------------------------------------------------------------
# Retrieval (Phase 3)
# ---------------------------------------------------------------------------
TOP_K_VECTOR = 8
TOP_K_BM25 = 8
TOP_K_RERANK_CANDIDATES = 12   # how many fused candidates go into the reranker
TOP_K_FINAL = 4                # how many chunks actually go into the prompt
RRF_K = 60                     # reciprocal rank fusion constant

# Below this reranker score, we treat retrieval as "nothing relevant found"
# and skip generation entirely (Phase 4's "I don't know" guardrail).
# Cross-encoder logits are model-relative and may be negative. The installed
# MS MARCO model commonly scores relevant handbook chunks around -9 to -3.
MIN_RERANK_SCORE = -10.0

# ---------------------------------------------------------------------------
# Guardrails (Phase 6)
# ---------------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60
CACHE_TTL_SECONDS = 60 * 60  # 1 hour

# ---------------------------------------------------------------------------
# Cost tracking (rough, informational only -- Gemini free tier costs $0,
# these numbers are what it WOULD cost past the free tier, per Phase 6/7)
# ---------------------------------------------------------------------------
COST_PER_1M_INPUT_TOKENS_USD = 0.15   # gemini-embedding-001 paid-tier rate
COST_PER_1M_GEN_INPUT_TOKENS_USD = 0.075   # gemini-3.6-flash paid-tier input (approx)
COST_PER_1M_GEN_OUTPUT_TOKENS_USD = 0.30   # gemini-3.6-flash paid-tier output (approx)

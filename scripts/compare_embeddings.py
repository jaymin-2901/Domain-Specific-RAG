"""
Phase 2 deliverable: compare the Gemini embedding model against a free local
sentence-transformers model on latency and a simple top-1 retrieval-accuracy
proxy (does the correct source document come back as the #1 vector hit?).

This produces the concrete numbers Phase 2 asks you to have ready in an
interview: "why did you pick this embedding model, what's the trade-off?"

Usage:
    python scripts/compare_embeddings.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.ingestion import load_documents, chunk_documents
from src.embeddings import GeminiEmbedder, LocalEmbedder, VectorStore, benchmark_embedder
from src.evaluation import load_test_questions


def top1_accuracy(embedder, chunks, questions) -> float:
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    embeddings = embedder.encode(texts)

    store = VectorStore(dim=embeddings.shape[1])
    store.build(embeddings, chunk_ids)

    chunks_by_id = {c.chunk_id: c for c in chunks}
    correct = 0
    scored = [q for q in questions if q.expected_source]
    for q in scored:
        qvec = embedder.encode_query(q.question)
        top1_id, _ = store.search(qvec, 1)[0]
        if q.expected_source.lower() in chunks_by_id[top1_id].source.lower():
            correct += 1
    return correct / len(scored) if scored else 0.0


def main():
    docs = load_documents(config.DATA_DIR)
    chunks = chunk_documents(docs, strategy="recursive", chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    questions = load_test_questions(config.EVAL_DIR / "test_questions.json")
    sample_queries = [q.question for q in questions[:10]]

    print(f"Comparing embedders on {len(chunks)} chunks, {len(sample_queries)} sample queries...\n")

    results = []
    for label, embedder in [("gemini-embedding-001", GeminiEmbedder()), ("local (bge-small-en-v1.5)", LocalEmbedder())]:
        print(f"--- {label} ---")
        t0 = time.time()
        bench = benchmark_embedder(embedder, [c.text for c in chunks], sample_queries)
        acc = top1_accuracy(embedder, chunks, questions)
        total_time = time.time() - t0

        print(f"  dim: {bench.dim}")
        print(f"  index build time ({len(chunks)} chunks): {bench.build_time_seconds}s")
        print(f"  avg query latency: {bench.avg_query_latency_ms}ms")
        print(f"  top-1 retrieval accuracy: {acc:.1%}")
        print(f"  total benchmark time: {total_time:.1f}s\n")

        results.append({"embedder": label, **bench.__dict__, "top1_accuracy": acc})

    print("=== Summary ===")
    for r in results:
        print(r)
    print("\nGemini is a hosted API call (network latency, free-tier rate limits, $0.15/1M "
          "tokens on the paid tier). Local sentence-transformers has zero cost and zero rate "
          "limit but uses your own CPU and a smaller, more general-purpose model. For a small "
          "or bursty workload, Gemini's free tier is simplest; for very high query volume, "
          "the local model removes rate-limit risk entirely.")


if __name__ == "__main__":
    main()

"""
Build the retrieval index from the documents in data/sample_docs (or any
other folder). Run this once before querying the pipeline or launching the
Streamlit app.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --data-dir path/to/your/docs --chunk-strategy semantic
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(config.DATA_DIR))
    parser.add_argument("--chunk-strategy", default="recursive", choices=["fixed_size", "recursive", "semantic"])
    parser.add_argument("--embedder", default="local", choices=["gemini", "local"])
    args = parser.parse_args()

    print(f"Loading documents from {args.data_dir} ...")
    pipeline = RAGPipeline(embedder_backend=args.embedder)

    t0 = time.time()
    num_chunks = pipeline.ingest(data_dir=args.data_dir, chunk_strategy=args.chunk_strategy)
    elapsed = time.time() - t0

    pipeline.save()

    print(f"Indexed {num_chunks} chunks in {elapsed:.1f}s using '{args.chunk_strategy}' chunking "
          f"and the '{args.embedder}' embedder.")
    print(f"Index saved to {config.INDEX_DIR}/")


if __name__ == "__main__":
    main()

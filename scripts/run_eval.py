"""
Run the evaluation harness against the built index and print/save results.
Re-run this after every pipeline change (new chunking strategy, added
reranker, tuned prompt, etc.) and diff the numbers against eval/history.json
-- that before/after delta is the evidence Phase 5 asks for.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --tag "after-hybrid-search" --no-judge
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.pipeline import RAGPipeline
from src.evaluation import load_test_questions, run_evaluation, aggregate_metrics, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(config.EVAL_DIR / "test_questions.json"))
    parser.add_argument("--output", default=str(config.EVAL_DIR / "results" / "latest.json"))
    parser.add_argument("--tag", default="", help="Label for this run, e.g. 'baseline' or 'after-reranker'")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-judge faithfulness/relevance scoring (faster, retrieval metrics only)")
    args = parser.parse_args()

    if not Path(config.INDEX_DIR / "vectors.faiss").exists():
        print("No index found. Run scripts/build_index.py first.")
        sys.exit(1)

    print("Loading pipeline and index...")
    pipeline = RAGPipeline()
    pipeline.load()

    questions = load_test_questions(args.questions)
    print(f"Running {len(questions)} test questions{' (retrieval metrics only)' if args.no_judge else ''} ...")

    results = run_evaluation(pipeline, questions, judge_client=pipeline.client, use_judge=not args.no_judge)
    metrics = aggregate_metrics(results)

    save_report(results, metrics, args.output, tag=args.tag)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nFull report saved to {args.output}")
    print(f"Cost/usage this run: {pipeline.cost_tracker.summary()}")


if __name__ == "__main__":
    main()

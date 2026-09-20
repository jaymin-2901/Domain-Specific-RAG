"""
Phase 5 -- Evaluation Framework.

Proves the system works with numbers, not vibes, using two kinds of metrics:

Retrieval metrics (no LLM call needed, deterministic):
- hit_rate: did the expected source document show up in the top-k retrieved
  chunks at all?
- mrr (mean reciprocal rank): how high up was it?

Generation metrics (LLM-as-judge, using Gemini itself in a cheap/flash
config as the judge -- this mirrors RAGAS's approach without pulling in the
extra dependency, and gives you full control over the judge prompt):
- faithfulness: is every claim in the answer actually supported by the
  retrieved context (i.e. did the model avoid hallucinating / using outside
  knowledge)?
- answer_relevance: does the answer actually address the question asked?

Unanswerable questions (is_unanswerable: true in the test set) are scored
separately: did the system correctly say "I don't know" instead of
fabricating an answer? This is graded as a simple correctness check, not
faithfulness/relevance.

Run this after every pipeline change and diff the numbers -- that before/after
delta is the single most convincing thing to bring to an interview.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import config


@dataclass
class TestQuestion:
    id: str
    question: str
    expected_source: str | None = None       # filename substring expected in retrieved chunks
    expected_keywords: list[str] | None = None  # keywords expected to appear in a correct answer
    is_unanswerable: bool = False


@dataclass
class QuestionResult:
    id: str
    question: str
    answer: str
    retrieved_sources: list[str]
    hit: bool | None
    reciprocal_rank: float | None
    faithfulness: float | None
    answer_relevance: float | None
    correct_refusal: bool | None
    latency_ms: float


def load_test_questions(path: str | Path) -> list[TestQuestion]:
    data = json.loads(Path(path).read_text())
    return [TestQuestion(**q) for q in data]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def retrieval_hit_and_rank(expected_source: str | None, retrieved_sources: list[str]) -> tuple[bool | None, float | None]:
    if not expected_source:
        return None, None
    for i, src in enumerate(retrieved_sources):
        if expected_source.lower() in src.lower():
            return True, 1.0 / (i + 1)
    return False, 0.0


# ---------------------------------------------------------------------------
# LLM-judge metrics
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are grading a RAG (retrieval-augmented generation) system's answer. \
Respond ONLY with a JSON object, no markdown fences, no extra text.

Context the system was given:
{context}

Question: {question}

System's answer: {answer}

Score two things from 0.0 to 1.0:
- "faithfulness": 1.0 if every claim in the answer is directly supported by the context above, \
0.0 if the answer contains claims not found in the context (hallucination / outside knowledge). \
Partial credit for partially-supported answers.
- "answer_relevance": 1.0 if the answer directly and completely addresses the question asked, \
0.0 if it is off-topic or non-responsive. Partial credit for partially-relevant answers.

Return exactly: {{"faithfulness": <float>, "answer_relevance": <float>}}"""


def judge_answer(client, question: str, answer: str, context: str, model: str = config.JUDGE_MODEL) -> tuple[float, float]:
    prompt = _JUDGE_PROMPT.format(context=context[:6000], question=question, answer=answer)
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = (resp.text or "").strip()
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(text)
        return float(parsed.get("faithfulness", 0.0)), float(parsed.get("answer_relevance", 0.0))
    except Exception:
        return 0.0, 0.0


_REFUSAL_MARKERS = ["i don't know", "cannot find", "not enough information", "no information"]


def is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(m in lower for m in _REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(pipeline, test_questions: list[TestQuestion], judge_client=None, use_judge: bool = True) -> list[QuestionResult]:
    results = []
    for tq in test_questions:
        t0 = time.time()
        response = pipeline.query(tq.question, session_id="eval", use_cache=False)
        latency_ms = (time.time() - t0) * 1000

        retrieved_sources = [sc.chunk.source for sc in response["retrieved_chunks"]]
        hit, rr = retrieval_hit_and_rank(tq.expected_source, retrieved_sources)

        correct_refusal = None
        faithfulness = None
        relevance = None

        if tq.is_unanswerable:
            correct_refusal = is_refusal(response["answer"])
        elif use_judge and judge_client is not None and not response["used_fallback"]:
            context = "\n\n".join(sc.chunk.text for sc in response["retrieved_chunks"])
            faithfulness, relevance = judge_answer(judge_client, tq.question, response["answer"], context)
        elif response["used_fallback"]:
            # system refused on an answerable question -- that's a miss, not "no score"
            faithfulness, relevance = 0.0, 0.0

        results.append(QuestionResult(
            id=tq.id,
            question=tq.question,
            answer=response["answer"],
            retrieved_sources=retrieved_sources,
            hit=hit,
            reciprocal_rank=rr,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            correct_refusal=correct_refusal,
            latency_ms=round(latency_ms, 1),
        ))
    return results


def aggregate_metrics(results: list[QuestionResult]) -> dict:
    answerable = [r for r in results if r.correct_refusal is None]
    unanswerable = [r for r in results if r.correct_refusal is not None]

    hits = [r.hit for r in answerable if r.hit is not None]
    rrs = [r.reciprocal_rank for r in answerable if r.reciprocal_rank is not None]
    faiths = [r.faithfulness for r in answerable if r.faithfulness is not None]
    rels = [r.answer_relevance for r in answerable if r.answer_relevance is not None]
    refusal_correct = [r.correct_refusal for r in unanswerable]

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "num_questions": len(results),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
        "retrieval_hit_rate": avg(hits),
        "retrieval_mrr": avg(rrs),
        "avg_faithfulness": avg(faiths),
        "avg_answer_relevance": avg(rels),
        "unanswerable_correct_refusal_rate": avg(refusal_correct),
        "avg_latency_ms": avg([r.latency_ms for r in results]),
    }


def save_report(results: list[QuestionResult], metrics: dict, path: str | Path, tag: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(report, indent=2))

    # append a one-line summary to a running history file for before/after tracking
    history_path = path.parent / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({"tag": tag, "timestamp": report["timestamp"], "metrics": metrics})
    history_path.write_text(json.dumps(history, indent=2))

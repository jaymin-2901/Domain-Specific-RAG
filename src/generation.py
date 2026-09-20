"""
Phase 4 -- Generation & Prompt Engineering.

The #1 thing interviewers probe on RAG projects: how do you stop the model
from answering out of its own parametric knowledge instead of the retrieved
context? Three things do that work here, stacked:

1. A system instruction that explicitly forbids using outside knowledge and
   requires every claim to be traceable to a numbered source chunk.
2. Chunks are injected as clearly delimited, numbered blocks (Source 1,
   Source 2, ...) and the model is told to cite them inline like [1], [2].
3. A pre-generation guardrail (see MIN_RERANK_SCORE in retrieval) that skips
   the LLM call entirely and returns "I don't know" when nothing retrieved
   clears a relevance bar -- so the model is never even given a chance to
   improvise from a weak or empty context.

Multi-hop questions (answer spans multiple chunks) are handled by simply
giving the model several distinct sources and explicitly asking it to
synthesize across them, citing each piece separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from google.genai import types

import config
from src.retrieval import ScoredChunk

SYSTEM_INSTRUCTION = """You are a precise question-answering assistant that ONLY uses the \
provided source excerpts to answer questions. Follow these rules exactly:

1. Use ONLY information found in the numbered sources below. Never use outside \
knowledge, even if you are confident it is correct.
2. Every factual sentence in your answer must end with a citation to the source \
number(s) it came from, like this: [1] or [2][3].
3. If the sources do not contain enough information to answer the question, \
respond with exactly: "I don't know based on the provided documents." Do not \
guess or partially answer from memory.
4. If the question requires combining facts from more than one source \
(a multi-hop question), explicitly connect them in your reasoning and cite \
every source you used.
5. Be concise. Do not repeat the sources verbatim; synthesize them in your own words.
"""

USER_TEMPLATE = """Sources:
{sources_block}

Question: {question}

Answer the question using only the sources above, following all the rules in \
your instructions."""


@dataclass
class GenerationResult:
    answer: str
    used_fallback: bool          # True if we never called the LLM (no relevant context)
    cited_sources: list[int]     # which [n] markers actually appear in the answer
    source_chunks: list[ScoredChunk]


def _build_sources_block(scored_chunks: list[ScoredChunk]) -> str:
    lines = []
    for i, sc in enumerate(scored_chunks, start=1):
        header = f" (section: {sc.chunk.section_header})" if sc.chunk.section_header else ""
        page = f", page {sc.chunk.page_number}" if sc.chunk.page_number else ""
        lines.append(f"[{i}] From {sc.chunk.source}{header}{page}:\n{sc.chunk.text}")
    return "\n\n".join(lines)


def _extract_cited_sources(answer: str) -> list[int]:
    nums = set()
    for match in re.findall(r"\[(\d+)\]", answer):
        nums.add(int(match))
    return sorted(nums)


def generate_answer(
    client,
    question: str,
    scored_chunks: list[ScoredChunk],
    model: str = config.GENERATION_MODEL,
    min_rerank_score: float = config.MIN_RERANK_SCORE,
) -> GenerationResult:
    if not scored_chunks:
        return GenerationResult("I don't know based on the provided documents.", True, [], [])

    best_score = max((sc.rerank_score for sc in scored_chunks if sc.rerank_score is not None), default=None)
    if best_score is not None and best_score < min_rerank_score:
        return GenerationResult("I don't know based on the provided documents.", True, [], scored_chunks)

    sources_block = _build_sources_block(scored_chunks)
    prompt = USER_TEMPLATE.format(sources_block=sources_block, question=question)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_output_tokens=1024,
        ),
    )
    answer = (resp.text or "").strip()
    return GenerationResult(
        answer=answer,
        used_fallback=False,
        cited_sources=_extract_cited_sources(answer),
        source_chunks=scored_chunks,
    )

"""
Phase 1 -- Ingestion & Chunking Strategy.

Turns raw files (.md, .txt, .pdf) into a list of Chunk objects with metadata
attached (source file, section header, page number, chunk strategy used).

Three chunking strategies are implemented so you can compare them directly
(this is the concrete "why chunk size matters" evidence an interviewer will
ask for):

1. fixed_size      -- naive sliding window over raw characters
2. recursive       -- splits on paragraph -> line -> sentence -> word
                       boundaries, only falling back to a harder split when
                       a piece is still too big. Keeps chunks semantically
                       cleaner than fixed_size for the same target size.
3. semantic        -- embeds each sentence with the LOCAL embedder (free,
                       no API cost) and starts a new chunk wherever the
                       similarity between consecutive sentences drops below
                       a threshold, i.e. wherever the topic shifts.

Note on "tokens": Gemini's tokenizer isn't installed locally, so chunk sizes
here are measured in characters. As a rule of thumb, 800 characters is
roughly 150-220 English tokens -- close enough for chunking purposes, and
you avoid pulling in an unrelated tokenizer library just to approximate size.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


@dataclass
class RawDocument:
    doc_id: str
    source: str          # filename
    text: str
    pages: list[str] = field(default_factory=list)  # non-empty only for PDFs


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    section_header: str | None
    page_number: int | None
    strategy: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: str | Path) -> list[RawDocument]:
    """Load every .md/.txt/.pdf file in data_dir into a RawDocument."""
    data_dir = Path(data_dir)
    docs: list[RawDocument] = []

    for path in sorted(data_dir.glob("**/*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()

        if suffix in (".md", ".txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(RawDocument(doc_id=str(uuid.uuid4())[:8], source=path.name, text=text))

        elif suffix == ".pdf":
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n\n".join(pages)
            docs.append(RawDocument(doc_id=str(uuid.uuid4())[:8], source=path.name, text=full_text, pages=pages))

        # unsupported types are silently skipped -- extend here for
        # .docx/.html etc. using unstructured.io if your corpus needs it

    return docs


# ---------------------------------------------------------------------------
# Helper: nearest markdown header above a character offset
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _nearest_header(text: str, offset: int) -> str | None:
    header = None
    for m in _HEADER_RE.finditer(text):
        if m.start() > offset:
            break
        header = m.group(2).strip()
    return header


def _page_for_offset(doc: RawDocument, offset: int) -> int | None:
    if not doc.pages:
        return None
    cursor = 0
    for i, page_text in enumerate(doc.pages):
        cursor += len(page_text) + 2  # account for the "\n\n" join
        if offset < cursor:
            return i + 1
    return len(doc.pages)


# ---------------------------------------------------------------------------
# Strategy 1: fixed-size with overlap
# ---------------------------------------------------------------------------

def fixed_size_chunks(doc: RawDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    text = doc.text
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}-{len(chunks)}",
                doc_id=doc.doc_id,
                source=doc.source,
                text=piece,
                section_header=_nearest_header(text, start),
                page_number=_page_for_offset(doc, start),
                strategy="fixed_size",
            ))
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: recursive character splitting
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # last resort: hard character cut
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    buf = ""
    for part in parts:
        candidate = (buf + sep + part) if buf else part
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                pieces.append(buf)
            if len(part) > chunk_size:
                pieces.extend(_recursive_split(part, chunk_size, rest))
                buf = ""
            else:
                buf = part
    if buf:
        pieces.append(buf)
    return pieces


def recursive_chunks(doc: RawDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    raw_pieces = _recursive_split(doc.text, chunk_size, _SEPARATORS)

    # stitch overlap back in by carrying the tail of the previous piece
    chunks = []
    cursor = 0
    prev_tail = ""
    for piece in raw_pieces:
        if prev_tail:
            joiner = "\n" if prev_tail.lstrip().startswith("#") else " "
            piece_with_overlap = prev_tail + joiner + piece
        else:
            piece_with_overlap = piece
        offset = doc.text.find(piece[:30], cursor) if piece else cursor
        offset = max(offset, 0)
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}-{len(chunks)}",
            doc_id=doc.doc_id,
            source=doc.source,
            text=piece_with_overlap.strip(),
            section_header=_nearest_header(doc.text, offset),
            page_number=_page_for_offset(doc, offset),
            strategy="recursive",
        ))
        if overlap:
            tail = piece[-overlap:]
            if tail.rstrip().endswith((".", "!", "?")):
                prev_tail = ""
            else:
                boundaries = [match.end() for match in re.finditer(r"\n\n|\n|(?<=[.!?])\s+", tail)]
                prev_tail = tail[max(boundaries):] if boundaries else tail
        else:
            prev_tail = ""
        cursor = offset + len(piece)
    return [c for c in chunks if c.text]


# ---------------------------------------------------------------------------
# Strategy 3: semantic chunking (topic-shift detection)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def semantic_chunks(
    doc: RawDocument,
    local_embedder,               # src.embeddings.LocalEmbedder instance
    similarity_threshold: float,
    max_chunk_size: int,
) -> list[Chunk]:
    import numpy as np

    sentences = [s.strip() for s in _SENTENCE_RE.split(doc.text) if s.strip()]
    if len(sentences) <= 1:
        return fixed_size_chunks(doc, max_chunk_size, 0)

    embeddings = local_embedder.encode(sentences)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    chunks = []
    current_sentences = [sentences[0]]
    current_len = len(sentences[0])
    cursor = 0

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        sent = sentences[i]
        topic_shift = sim < similarity_threshold
        too_big = current_len + len(sent) > max_chunk_size

        if topic_shift or too_big:
            text = " ".join(current_sentences)
            offset = doc.text.find(current_sentences[0][:30], cursor)
            offset = max(offset, 0)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}-{len(chunks)}",
                doc_id=doc.doc_id,
                source=doc.source,
                text=text,
                section_header=_nearest_header(doc.text, offset),
                page_number=_page_for_offset(doc, offset),
                strategy="semantic",
            ))
            cursor = offset + len(text)
            current_sentences = [sent]
            current_len = len(sent)
        else:
            current_sentences.append(sent)
            current_len += len(sent)

    if current_sentences:
        text = " ".join(current_sentences)
        offset = doc.text.find(current_sentences[0][:30], cursor)
        offset = max(offset, 0)
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}-{len(chunks)}",
            doc_id=doc.doc_id,
            source=doc.source,
            text=text,
            section_header=_nearest_header(doc.text, offset),
            page_number=_page_for_offset(doc, offset),
            strategy="semantic",
        ))
    return chunks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def chunk_documents(
    docs: Iterable[RawDocument],
    strategy: str = "recursive",
    chunk_size: int = 800,
    overlap: int = 150,
    local_embedder=None,
    semantic_threshold: float = 0.55,
    semantic_max_size: int = 1200,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        if strategy == "fixed_size":
            all_chunks.extend(fixed_size_chunks(doc, chunk_size, overlap))
        elif strategy == "recursive":
            all_chunks.extend(recursive_chunks(doc, chunk_size, overlap))
        elif strategy == "semantic":
            if local_embedder is None:
                raise ValueError("semantic chunking requires local_embedder")
            all_chunks.extend(semantic_chunks(doc, local_embedder, semantic_threshold, semantic_max_size))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    return all_chunks

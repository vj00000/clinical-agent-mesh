"""Overlapping word-window chunking with content-addressed ids.

Ids are a hash of source plus text rather than a counter, so rebuilding the
corpus does not invalidate citations already stored in checkpoints or eval
results.
"""

import hashlib

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str
    ordinal: int


def _chunk_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\x00{text}".encode())
    return digest.hexdigest()[:16]


def chunk_document(
    text: str,
    *,
    source: str,
    target_words: int = 300,
    overlap_words: int = 50,
) -> list[Chunk]:
    """Split `text` into overlapping windows of at most `target_words` words."""
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words")

    words = text.split()
    if not words:
        return []

    stride = target_words - overlap_words
    chunks: list[Chunk] = []
    for ordinal, start in enumerate(range(0, len(words), stride)):
        window = words[start : start + target_words]
        if not window:
            break
        body = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(source, body),
                source=source,
                text=body,
                ordinal=ordinal,
            )
        )
        if start + target_words >= len(words):
            break

    return chunks

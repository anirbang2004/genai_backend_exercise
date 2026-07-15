"""
Chunking.

Splits a Document into smaller, overlapping "chunks" that are small
enough to embed meaningfully and retrieve precisely, while retaining
enough surrounding context to be useful on their own.

A simple word-count sliding window is used rather than a sentence- or
token-aware splitter. It's dependency-free, fast, deterministic, and more
than adequate for the short policy-style documents in this exercise. See
NOTES.md for a discussion of the tradeoffs.
"""

from dataclasses import dataclass
from typing import List

from src.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
from src.loader import Document


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document, ready to be embedded/indexed."""

    chunk_id: str   # globally unique id, e.g. "refund-policy.md::0"
    source: str     # originating filename, carried through for citations
    text: str       # the chunk's text content


def chunk_document(
    document: Document,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[Chunk]:
    """
    Split a single document into overlapping word-window chunks.

    Args:
        document: the source Document to split.
        chunk_size_words: target number of words per chunk.
        overlap_words: number of words repeated between consecutive
            chunks, so a fact that happens to fall on a chunk boundary is
            still retrievable in full from at least one chunk.

    Returns:
        A list of Chunk objects in document order. A document shorter
        than `chunk_size_words` yields exactly one chunk.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be in [0, chunk_size_words)")

    words = document.text.split()
    if not words:
        return []

    stride = chunk_size_words - overlap_words
    chunks: List[Chunk] = []

    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + chunk_size_words]
        text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=f"{document.source}::{index}",
                source=document.source,
                text=text,
            )
        )
        index += 1
        start += stride

    return chunks


def chunk_documents(documents: List[Document], **kwargs) -> List[Chunk]:
    """Chunk a list of documents, preserving document order."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, **kwargs))
    return all_chunks

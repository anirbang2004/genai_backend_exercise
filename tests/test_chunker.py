import pytest

from src.chunker import chunk_document, chunk_documents
from src.loader import Document


def make_doc(word_count: int, source: str = "doc.txt") -> Document:
    text = " ".join(f"word{i}" for i in range(word_count))
    return Document(source=source, path=f"/tmp/{source}", text=text)


def test_short_document_yields_single_chunk():
    doc = make_doc(50)

    chunks = chunk_document(doc, chunk_size_words=200, overlap_words=40)

    assert len(chunks) == 1
    assert chunks[0].text == doc.text
    assert chunks[0].chunk_id == "doc.txt::0"
    assert chunks[0].source == "doc.txt"


def test_long_document_splits_into_multiple_overlapping_chunks():
    doc = make_doc(500)

    chunks = chunk_document(doc, chunk_size_words=200, overlap_words=40)

    assert len(chunks) > 1
    # Chunk ids are sequential and namespaced by source file.
    assert chunks[0].chunk_id == "doc.txt::0"
    assert chunks[1].chunk_id == "doc.txt::1"

    # Verify overlap: the tail of chunk 0 should reappear at the head of chunk 1.
    chunk0_words = chunks[0].text.split()
    chunk1_words = chunks[1].text.split()
    assert chunk0_words[-40:] == chunk1_words[:40]


def test_chunk_size_is_never_exceeded():
    doc = make_doc(500)

    chunks = chunk_document(doc, chunk_size_words=200, overlap_words=40)

    # Every chunk (including short trailing windows near the end of the
    # document) must stay within the requested chunk size.
    for chunk in chunks:
        assert len(chunk.text.split()) <= 200

    # The very first chunk should always be a full window.
    assert len(chunks[0].text.split()) == 200


def test_empty_document_yields_no_chunks():
    doc = Document(source="empty.txt", path="/tmp/empty.txt", text="")

    chunks = chunk_document(doc)

    assert chunks == []


def test_invalid_overlap_raises():
    doc = make_doc(50)

    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size_words=100, overlap_words=100)

    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size_words=100, overlap_words=-1)


def test_invalid_chunk_size_raises():
    doc = make_doc(50)

    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size_words=0)


def test_chunk_documents_preserves_order_across_files():
    doc_a = make_doc(30, source="a.txt")
    doc_b = make_doc(30, source="b.txt")

    chunks = chunk_documents([doc_a, doc_b])

    sources_in_order = [c.source for c in chunks]
    assert sources_in_order == ["a.txt", "b.txt"]

import pytest

from src.chunker import Chunk
from src.vector_store import IndexNotFoundError, VectorStore


def make_store(tmp_path, fake_embedding_function, name="test-collection"):
    return VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name=name,
        embedding_function=fake_embedding_function,
    )


def test_build_index_and_query_returns_relevant_chunk(tmp_path, fake_embedding_function):
    store = make_store(tmp_path, fake_embedding_function)
    chunks = [
        Chunk(chunk_id="refund.md::0", source="refund.md", text="refund policy 30 days"),
        Chunk(chunk_id="shipping.md::0", source="shipping.md", text="shipping takes 5 days"),
    ]

    indexed = store.build_index(chunks)
    assert indexed == 2

    results = store.query("refund policy", top_k=1)

    assert len(results) == 1
    assert results[0].source == "refund.md"


def test_query_before_ingest_raises_index_not_found(tmp_path, fake_embedding_function):
    store = make_store(tmp_path, fake_embedding_function)

    with pytest.raises(IndexNotFoundError):
        store.query("anything", top_k=3)


def test_query_returns_empty_list_when_index_has_no_chunks(tmp_path, fake_embedding_function):
    store = make_store(tmp_path, fake_embedding_function)
    store.build_index([])  # e.g. a docs dir that only had empty files

    results = store.query("anything", top_k=3)

    assert results == []


def test_rebuilding_index_replaces_previous_chunks(tmp_path, fake_embedding_function):
    store = make_store(tmp_path, fake_embedding_function)
    store.build_index(
        [Chunk(chunk_id="old.md::0", source="old.md", text="stale content")]
    )

    # Re-ingest with a completely different document set.
    store.build_index(
        [Chunk(chunk_id="new.md::0", source="new.md", text="fresh content")]
    )

    results = store.query("content", top_k=5)
    sources = {r.source for r in results}

    assert sources == {"new.md"}


def test_top_k_is_capped_at_available_chunk_count(tmp_path, fake_embedding_function):
    store = make_store(tmp_path, fake_embedding_function)
    store.build_index(
        [Chunk(chunk_id="only.md::0", source="only.md", text="the only chunk")]
    )

    # Requesting more results than exist should not raise.
    results = store.query("chunk", top_k=10)

    assert len(results) == 1

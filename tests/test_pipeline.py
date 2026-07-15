import pytest

from src.answer_composer import NOT_FOUND_MESSAGE
from src.loader import DocumentDirectoryNotFoundError, NoSupportedDocumentsError
from src.pipeline import run_ask, run_ingest
from src.vector_store import IndexNotFoundError, VectorStore


def make_store(tmp_path, fake_embedding_function):
    return VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="pipeline-test",
        embedding_function=fake_embedding_function,
    )


def test_end_to_end_ingest_then_ask_returns_grounded_answer(
    tmp_path, fake_embedding_function, fake_llm
):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "refund-policy.md").write_text(
        "Customers may request a refund within 30 days of purchase."
    )
    (docs_dir / "shipping.md").write_text(
        "Orders ship within 5 business days of confirmation."
    )

    store = make_store(tmp_path, fake_embedding_function)
    result = run_ingest(str(docs_dir), vector_store=store)

    assert result.documents_loaded == 2
    assert result.chunks_indexed == 2

    answer = run_ask(
        "refund within 30 days of purchase",
        vector_store=store,
        llm=fake_llm,
        top_k=2,
    )

    assert answer.text == fake_llm.response
    assert "refund-policy.md" in answer.sources


def test_ask_unrelated_question_returns_not_found(
    tmp_path, fake_embedding_function, fake_llm
):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "refund-policy.md").write_text(
        "Customers may request a refund within 30 days of purchase."
    )

    store = make_store(tmp_path, fake_embedding_function)
    run_ingest(str(docs_dir), vector_store=store)

    # A question sharing no vocabulary with any indexed chunk should be
    # rejected as unsupported by the FakeEmbeddingFunction's bag-of-words
    # similarity, exercising the grounding threshold end to end.
    answer = run_ask(
        "zzz qqq xyzzy plugh", vector_store=store, llm=fake_llm, top_k=2
    )

    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.sources == []


def test_ask_before_ingest_raises(tmp_path, fake_embedding_function, fake_llm):
    store = make_store(tmp_path, fake_embedding_function)

    with pytest.raises(IndexNotFoundError):
        run_ask("anything", vector_store=store, llm=fake_llm)


def test_ingest_missing_directory_raises(tmp_path, fake_embedding_function):
    store = make_store(tmp_path, fake_embedding_function)
    missing_dir = tmp_path / "nope"

    with pytest.raises(DocumentDirectoryNotFoundError):
        run_ingest(str(missing_dir), vector_store=store)


def test_ingest_directory_with_no_supported_files_raises(
    tmp_path, fake_embedding_function
):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "image.png").write_bytes(b"\x89PNG\r\n")
    store = make_store(tmp_path, fake_embedding_function)

    with pytest.raises(NoSupportedDocumentsError):
        run_ingest(str(docs_dir), vector_store=store)

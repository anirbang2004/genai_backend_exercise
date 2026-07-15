import pytest

import app
from src.answer_composer import Answer
from src.pipeline import IngestResult


def test_no_command_is_invalid_usage(capsys):
    with pytest.raises(SystemExit) as exc_info:
        app.main([])
    assert exc_info.value.code == 2


def test_unknown_command_is_invalid_usage(capsys):
    with pytest.raises(SystemExit) as exc_info:
        app.main(["frobnicate", "stuff"])
    assert exc_info.value.code == 2


def test_ask_missing_question_argument_is_invalid_usage():
    with pytest.raises(SystemExit) as exc_info:
        app.main(["ask"])
    assert exc_info.value.code == 2


def test_ingest_missing_directory_reports_error(capsys):
    exit_code = app.main(["ingest", "/path/does/not/exist"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err
    assert "not found" in captured.err.lower()


def test_ask_before_ingest_reports_error(tmp_path, monkeypatch, capsys):
    # Point the default vector store at an empty, isolated location so
    # this test never touches (or depends on) a previously built index.
    monkeypatch.setattr(app, "run_ask", _raise_index_not_found)

    exit_code = app.main(["ask", "What is the refund policy?"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err


def _raise_index_not_found(*args, **kwargs):
    from src.vector_store import IndexNotFoundError

    raise IndexNotFoundError("No index found. Run 'ingest' before asking questions.")


def test_ingest_happy_path_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        app,
        "run_ingest",
        lambda directory: IngestResult(
            documents_loaded=3, chunks_indexed=7, source_directory=directory
        ),
    )

    exit_code = app.main(["ingest", "./docs"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "3 document(s)" in captured.out
    assert "7 chunk(s)" in captured.out
    assert "./docs" in captured.out


def test_ask_happy_path_prints_answer_and_sources(monkeypatch, capsys):
    monkeypatch.setattr(
        app,
        "run_ask",
        lambda question, llm=None: Answer(
            text="Customers may request a refund within 30 days.",
            sources=["refund-policy.md"],
        ),
    )

    exit_code = app.main(["ask", "What is the refund policy?"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Answer: Customers may request a refund within 30 days." in captured.out
    assert "Sources: refund-policy.md" in captured.out


def test_ask_with_no_grounded_answer_prints_none_for_sources(monkeypatch, capsys):
    monkeypatch.setattr(
        app,
        "run_ask",
        lambda question, llm=None: Answer(
            text="I could not find enough support for that answer in the provided documents.",
            sources=[],
        ),
    )

    exit_code = app.main(["ask", "What is the meaning of life?"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Sources: None" in captured.out

import pytest

from src.loader import (
    DocumentDirectoryNotFoundError,
    NoSupportedDocumentsError,
    load_documents,
)


def test_load_documents_reads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("hello from a")
    (tmp_path / "b.md").write_text("# hello from b")

    docs = load_documents(str(tmp_path))

    sources = sorted(d.source for d in docs)
    assert sources == ["a.txt", "b.md"]


def test_load_documents_skips_unsupported_extensions(tmp_path):
    (tmp_path / "a.txt").write_text("supported")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].source == "a.txt"


def test_load_documents_skips_empty_files(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "real.txt").write_text("has content")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].source == "real.txt"


def test_load_documents_missing_directory_raises(tmp_path):
    missing = tmp_path / "does-not-exist"

    with pytest.raises(DocumentDirectoryNotFoundError):
        load_documents(str(missing))


def test_load_documents_path_is_a_file_raises(tmp_path):
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("I'm a file, not a directory")

    with pytest.raises(DocumentDirectoryNotFoundError):
        load_documents(str(file_path))


def test_load_documents_no_supported_files_raises(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    with pytest.raises(NoSupportedDocumentsError):
        load_documents(str(tmp_path))


def test_load_documents_preserves_text_content(tmp_path):
    content = "Line one.\nLine two.\n"
    (tmp_path / "doc.txt").write_text(content)

    docs = load_documents(str(tmp_path))

    assert docs[0].text == content.strip()

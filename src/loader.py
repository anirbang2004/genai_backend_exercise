"""
Document loading.

Responsible for turning a directory on disk into a list of in-memory
Document objects. Deliberately kept separate from chunking/indexing so
each concern can be tested and reasoned about independently.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.config import SUPPORTED_EXTENSIONS


class DocumentDirectoryNotFoundError(FileNotFoundError):
    """Raised when the path passed to `ingest` does not exist or isn't a directory."""


class NoSupportedDocumentsError(ValueError):
    """Raised when a valid directory contains no .txt/.md files to ingest."""


@dataclass(frozen=True)
class Document:
    """A single source document loaded from disk."""

    source: str    # filename, e.g. "refund-policy.md" - used for citations
    path: str      # full path on disk, kept for debugging/traceability
    text: str      # raw file contents


def load_documents(directory: str) -> List[Document]:
    """
    Read all supported (.txt, .md) files from `directory`.

    Unsupported file types are silently skipped (they are not an error -
    a docs folder may legitimately contain images, READMEs in other
    formats, etc.). Empty files are skipped too, since they contribute no
    retrievable content.

    Raises:
        DocumentDirectoryNotFoundError: if `directory` doesn't exist or
            isn't a directory.
        NoSupportedDocumentsError: if the directory exists but contains no
            usable .txt/.md files.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise DocumentDirectoryNotFoundError(
            f"Documents directory not found: '{directory}'"
        )
    if not dir_path.is_dir():
        raise DocumentDirectoryNotFoundError(
            f"Path exists but is not a directory: '{directory}'"
        )

    documents: List[Document] = []
    # sorted() keeps ingestion order deterministic, which makes debugging
    # and tests reproducible regardless of filesystem iteration order.
    for file_path in sorted(dir_path.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue

        documents.append(
            Document(source=file_path.name, path=str(file_path), text=text)
        )

    if not documents:
        raise NoSupportedDocumentsError(
            f"No supported .txt/.md documents found in '{directory}'"
        )

    return documents

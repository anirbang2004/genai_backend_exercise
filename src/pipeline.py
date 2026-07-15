"""
Pipeline orchestration.

Wires together the individual components (loader -> chunker -> vector
store -> answer composer) into the two operations the CLI exposes:
`run_ingest` and `run_ask`. Kept separate from `app.py` so the
orchestration logic can be unit tested without going through argparse or
stdout.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.answer_composer import Answer, compose_answer
from src.chunker import chunk_documents
from src.config import TOP_K
from src.loader import load_documents
from src.vector_store import VectorStore


@dataclass(frozen=True)
class IngestResult:
    documents_loaded: int
    chunks_indexed: int
    source_directory: str


def run_ingest(
    directory: str,
    vector_store: Optional[VectorStore] = None,
) -> IngestResult:
    """
    Load documents from `directory`, chunk them, and (re)build the index.

    Raises:
        DocumentDirectoryNotFoundError: bad path.
        NoSupportedDocumentsError: path exists but has no usable files.
    """
    documents = load_documents(directory)  # raises on bad path/empty dir
    chunks = chunk_documents(documents)

    store = vector_store or VectorStore()
    num_indexed = store.build_index(chunks)

    _write_metadata(
        store,
        {
            "source_directory": str(directory),
            "documents_loaded": len(documents),
            "chunks_indexed": num_indexed,
            "ingested_at": time.time(),
        },
    )

    return IngestResult(
        documents_loaded=len(documents),
        chunks_indexed=num_indexed,
        source_directory=str(directory),
    )


def run_ask(
    question: str,
    vector_store: Optional[VectorStore] = None,
    llm=None,
    top_k: int = TOP_K,
) -> Answer:
    """
    Answer `question` using the previously built index.

    Raises:
        IndexNotFoundError: if `ingest` has not been run yet.
    """
    store = vector_store or VectorStore()
    retrieved = store.query(question, top_k=top_k)
    return compose_answer(question, retrieved, llm=llm)


def _write_metadata(store: VectorStore, data: dict) -> None:
    """
    Best-effort write of small bookkeeping metadata about the last ingest.

    Stored next to the vector store's own persist directory (rather than
    a fixed global path) so each VectorStore instance - including the
    isolated, tmp_path-backed ones used in tests - owns its own metadata
    and never leaks writes into the real project directory.
    """
    try:
        metadata_path = Path(store.persist_directory).parent / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(data, indent=2))
    except OSError:
        # Metadata is informational only; never let it block ingestion.
        pass

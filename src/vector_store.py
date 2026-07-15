"""
Vector store.

Thin wrapper around a persistent Chroma collection. Isolating chromadb
usage behind this module means the rest of the app (retriever, CLI,
tests) never has to know it's Chroma specifically - it could be swapped
for another local vector store without touching callers.
"""

from dataclasses import dataclass
from typing import List

import chromadb

from src import config
from src.chunker import Chunk


class IndexNotFoundError(FileNotFoundError):
    """Raised when `ask` is invoked before any successful `ingest`."""


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from a similarity search, with its distance score."""

    chunk_id: str
    source: str
    text: str
    distance: float  # lower is more similar (Chroma's default: cosine distance)


class VectorStore:
    """
    Persistent, local vector store for document chunks.

    Backed by Chroma's DuckDB+Parquet persistence, so the index survives
    across CLI invocations without requiring a running server or any
    paid/hosted service.
    """

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embedding_function=None,
    ):
        # Resolved at call time (not baked into the signature at import
        # time) so tests can point isolated instances at a tmp_path
        # without needing to touch the real project's .index directory.
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or config.COLLECTION_NAME
        # Falls back to the real sentence-transformers embedding function
        # only when one isn't supplied, so tests can inject a fast fake.
        if embedding_function is None:
            from src.embeddings import get_default_embedding_function

            embedding_function = get_default_embedding_function()
        self.embedding_function = embedding_function
        self._client = chromadb.PersistentClient(path=self.persist_directory)

    def build_index(self, chunks: List[Chunk]) -> int:
        """
        (Re)build the collection from scratch using `chunks`.

        Rebuilding from scratch (rather than incrementally upserting)
        keeps `ingest` idempotent and avoids stale chunks lingering after
        a document is edited or removed from the docs folder.

        Returns the number of chunks indexed.
        """
        # Drop any previous collection so re-running `ingest` never leaves
        # orphaned chunks from a prior version of the docs folder.
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # collection didn't exist yet - nothing to drop

        collection = self._client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            # Explicit cosine space (rather than relying on whatever the
            # embedding function's default happens to be) so the
            # grounding distance threshold in answer_composer means the
            # same thing regardless of which embedding function is used.
            metadata={"hnsw:space": "cosine"},
        )

        if not chunks:
            return 0

        collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source} for c in chunks],
        )
        return len(chunks)

    def _load_collection(self):
        try:
            return self._client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception as exc:
            raise IndexNotFoundError(
                "No index found. Run 'ingest' before asking questions."
            ) from exc

    def query(self, question: str, top_k: int) -> List[RetrievedChunk]:
        """
        Retrieve the `top_k` chunks most similar to `question`.

        Raises:
            IndexNotFoundError: if `ingest` hasn't been run yet (or the
                persisted index is missing/corrupted).
        """
        collection = self._load_collection()

        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[question],
            n_results=min(top_k, collection.count()),
        )

        retrieved: List[RetrievedChunk] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source=meta.get("source", "unknown"),
                    text=text,
                    distance=distance,
                )
            )
        return retrieved

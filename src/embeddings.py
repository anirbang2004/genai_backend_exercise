"""
Embedding function.

Wraps a local sentence-transformers model behind Chroma's
`EmbeddingFunction` protocol (a callable: List[str] -> List[List[float]]).

The model is loaded lazily on first use rather than at import time. This
keeps `import src.embeddings` cheap (important for CLI startup and for
unit tests that inject a fake embedding function and never need the real
model at all).
"""

from typing import List, Optional

from chromadb.api.types import EmbeddingFunction

from src.config import EMBEDDING_MODEL_NAME


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Chroma-compatible embedding function backed by sentence-transformers.

    sentence-transformers models run fully on-device (CPU is fine for a
    model this small) and require no paid API - only a one-time download
    of model weights from the public Hugging Face hub the first time
    they're used.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None  # loaded on first call

    def _ensure_loaded(self):
        if self._model is None:
            # Imported lazily so environments that only run unit tests
            # (with a fake embedding function) never need torch installed.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        embeddings = self._model.encode(list(input), convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, input: List[str]) -> List[List[float]]:
        # Chroma calls this (rather than __call__) for the query side of
        # a search. We use the same embedding space for documents and
        # queries, which is standard for a general-purpose sentence
        # embedding model like MiniLM.
        return self.__call__(input)

    def name(self) -> str:
        # Required by chromadb's EmbeddingFunction protocol so it can
        # detect when a persisted collection's embedding function changes.
        return f"sentence-transformers:{self.model_name}"


def get_default_embedding_function(model_name: Optional[str] = None):
    """Factory so callers don't need to import the class directly."""
    return SentenceTransformerEmbeddingFunction(model_name or EMBEDDING_MODEL_NAME)

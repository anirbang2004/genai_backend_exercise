"""
Shared test fixtures.

Unit tests never load the real sentence-transformers or transformers
models: doing so would make the suite slow, network-dependent (first
run downloads weights), and non-deterministic. Instead we inject small,
fast fakes that satisfy the same interfaces:

- FakeEmbeddingFunction: deterministic bag-of-words vectors, good enough
  for cosine similarity to meaningfully separate unrelated texts in
  tests.
- FakeLLM: returns a canned/inspectable response instead of running
  real text generation.

This keeps the test suite fast, hermetic, and reflective of production
control flow (real code paths, fake heavy dependencies) rather than
mocking away the logic under test.
"""

import hashlib
from collections import Counter
from typing import List

import pytest
from chromadb.api.types import EmbeddingFunction


class FakeEmbeddingFunction(EmbeddingFunction):
    """
    Deterministic, dependency-free stand-in for a real embedding model.

    Uses the classic "hashing trick": each word is hashed into one of a
    fixed number of buckets, and the vector is the resulting word-count
    histogram. Unlike a growing vocabulary index, the output dimension
    never changes, which is required by Chroma (a collection is created
    with a fixed embedding dimension). Texts that share more words end up
    with smaller cosine distance, which is enough to exercise real
    retrieval/ranking logic without downloading an actual model.
    """

    def __init__(self, dimensions: int = 4096):
        self.dimensions = dimensions

    def _vectorize(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        for word, count in Counter(text.lower().split()).items():
            # Python's built-in hash() is randomized per-process (PYTHONHASHSEED),
            # which would make bucket assignment - and therefore test
            # outcomes - nondeterministic across runs. hashlib is stable.
            digest = hashlib.md5(word.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % self.dimensions
            vec[bucket] += float(count)
        return vec

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [self._vectorize(text) for text in input]

    def embed_query(self, input: List[str]) -> List[List[float]]:
        # Chroma routes query-side embedding through this method; use the
        # same vectorization as documents (consistent with a real
        # single-space sentence embedding model).
        return self.__call__(input)

    def name(self) -> str:
        return "fake-embedding-function"


class FakeLLM:
    """Fake LLM that returns a canned answer and records the last prompt."""

    def __init__(self, response: str = "This is a synthesized answer."):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


class RaisingLLM:
    """Fake LLM that always fails, to test the extractive fallback path."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("model unavailable")


@pytest.fixture
def fake_embedding_function():
    return FakeEmbeddingFunction()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def raising_llm():
    return RaisingLLM()

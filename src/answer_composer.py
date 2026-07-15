"""
Answer composition.

Takes the chunks returned by retrieval and turns them into a final,
citable answer. Two responsibilities are handled here:

1. Deciding whether the retrieved evidence is strong enough to answer
   at all (grounding check) - if not, we say so explicitly rather than
   letting the LLM hallucinate a confident-sounding answer.
2. Synthesizing a natural-language answer from the surviving context
   using a local LLM, with a safe extractive fallback if the LLM is
   unavailable (e.g. offline, first run before model weights are
   cached).
"""

from dataclasses import dataclass
from typing import List

from src.config import MAX_RELEVANT_DISTANCE
from src.vector_store import RetrievedChunk

NOT_FOUND_MESSAGE = (
    "I could not find enough support for that answer in the provided documents."
)

_PROMPT_TEMPLATE = (
    "Answer the question using ONLY the information in the context below. "
    "If the context does not contain the answer, respond exactly with: "
    "\"{not_found}\"\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


@dataclass(frozen=True)
class Answer:
    text: str
    sources: List[str]  # empty list means "no grounded answer"


def _select_grounded_chunks(
    retrieved: List[RetrievedChunk], max_distance: float
) -> List[RetrievedChunk]:
    """Keep only chunks similar enough to plausibly support an answer."""
    return [c for c in retrieved if c.distance <= max_distance]


def _unique_sources_by_relevance(chunks: List[RetrievedChunk]) -> List[str]:
    """De-duplicate source filenames while preserving relevance order."""
    seen = []
    for c in chunks:
        if c.source not in seen:
            seen.append(c.source)
    return seen


def _build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{c.source}]: {c.text}" for c in chunks)
    return _PROMPT_TEMPLATE.format(
        not_found=NOT_FOUND_MESSAGE, context=context, question=question
    )


def _extractive_fallback(chunks: List[RetrievedChunk]) -> str:
    """
    Best-effort answer when no LLM is available: surface the single most
    relevant chunk verbatim. Strictly worse than synthesis, but still
    grounded and never fabricated - a safe degradation path.
    """
    best = chunks[0]
    return best.text


def compose_answer(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    llm=None,
    max_distance: float = MAX_RELEVANT_DISTANCE,
) -> Answer:
    """
    Compose a final answer from retrieved chunks.

    Args:
        question: the user's original question.
        retrieved_chunks: chunks returned by VectorStore.query, ordered
            by relevance (most relevant first).
        llm: object exposing `.generate(prompt) -> str`. If None, an
            extractive fallback is used (useful for tests/offline runs).
        max_distance: grounding threshold; chunks less similar than this
            are discarded before synthesis.
    """
    grounded = _select_grounded_chunks(retrieved_chunks, max_distance)
    if not grounded:
        return Answer(text=NOT_FOUND_MESSAGE, sources=[])

    sources = _unique_sources_by_relevance(grounded)

    if llm is None:
        return Answer(text=_extractive_fallback(grounded), sources=sources)

    prompt = _build_prompt(question, grounded)
    try:
        text = llm.generate(prompt)
    except Exception:
        # Local model unavailable (e.g. not yet downloaded, no network
        # for the one-time fetch) - degrade gracefully instead of
        # crashing the CLI. The answer is still grounded in real content.
        text = _extractive_fallback(grounded)

    if not text:
        text = _extractive_fallback(grounded)

    # If the model itself decided the context didn't support an answer,
    # honor that and report no sources - it's not "not invented", it's
    # explicitly "unsupported".
    if text.strip().lower() == NOT_FOUND_MESSAGE.lower():
        return Answer(text=NOT_FOUND_MESSAGE, sources=[])

    return Answer(text=text, sources=sources)

from src.answer_composer import NOT_FOUND_MESSAGE, compose_answer
from src.vector_store import RetrievedChunk


def relevant_chunk(source="refund.md", distance=0.1):
    return RetrievedChunk(
        chunk_id=f"{source}::0",
        source=source,
        text="Customers may request a refund within 30 days of purchase.",
        distance=distance,
    )


def irrelevant_chunk(source="unrelated.md", distance=1.5):
    return RetrievedChunk(
        chunk_id=f"{source}::0",
        source=source,
        text="This document is about something completely unrelated.",
        distance=distance,
    )


def test_no_retrieved_chunks_returns_not_found():
    answer = compose_answer("What is the refund policy?", [], llm=None)

    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.sources == []


def test_chunks_beyond_distance_threshold_are_not_grounded():
    answer = compose_answer(
        "What is the refund policy?", [irrelevant_chunk()], llm=None, max_distance=0.8
    )

    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.sources == []


def test_grounded_chunk_uses_llm_synthesis(fake_llm):
    chunk = relevant_chunk()

    answer = compose_answer("What is the refund policy?", [chunk], llm=fake_llm)

    assert answer.text == fake_llm.response
    assert answer.sources == ["refund.md"]
    # The prompt actually given to the LLM should include the grounding context.
    assert "refund" in fake_llm.last_prompt.lower()


def test_llm_failure_falls_back_to_extractive_answer(raising_llm):
    chunk = relevant_chunk()

    answer = compose_answer("What is the refund policy?", [chunk], llm=raising_llm)

    assert answer.text == chunk.text  # extractive fallback: chunk text verbatim
    assert answer.sources == ["refund.md"]


def test_no_llm_provided_uses_extractive_fallback():
    chunk = relevant_chunk()

    answer = compose_answer("What is the refund policy?", [chunk], llm=None)

    assert answer.text == chunk.text
    assert answer.sources == ["refund.md"]


def test_sources_are_deduplicated_and_ordered_by_relevance(fake_llm):
    chunks = [
        relevant_chunk(source="refund.md", distance=0.1),
        relevant_chunk(source="refund.md", distance=0.2),  # same source again
        relevant_chunk(source="faq.md", distance=0.3),
    ]

    answer = compose_answer("What is the refund policy?", chunks, llm=fake_llm)

    assert answer.sources == ["refund.md", "faq.md"]


def test_llm_explicitly_signaling_not_found_clears_sources():
    class NotFoundLLM:
        def generate(self, prompt):
            return NOT_FOUND_MESSAGE

    answer = compose_answer(
        "What is the refund policy?", [relevant_chunk()], llm=NotFoundLLM()
    )

    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.sources == []

"""
Local LLM for answer synthesis.

Uses a small, free, locally-runnable sequence-to-sequence model
(google/flan-t5-small, ~80M params) via Hugging Face `transformers` to
compose a natural-language answer from retrieved context. No paid API
and no network calls at question-answering time beyond the one-time
model download.

The model is intentionally hidden behind a small interface
(`generate(prompt) -> str`) so `answer_composer` doesn't depend on
transformers directly, and tests can inject a trivial fake generator
instead of loading real model weights.
"""

from src.config import LLM_MAX_NEW_TOKENS, LLM_MODEL_NAME


class LocalSeq2SeqLLM:
    """Lazy-loaded wrapper around a local transformers text2text pipeline."""

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.model_name = model_name
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is None:
            # Imported lazily: unit tests that inject a fake LLM never
            # need `transformers`/`torch` installed just to run.
            from transformers import pipeline

            self._pipeline = pipeline("text2text-generation", model=self.model_name)

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        output = self._pipeline(
            prompt,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=False,  # deterministic output for a support/QA tool
        )
        return output[0]["generated_text"].strip()


def get_default_llm(model_name: str = None) -> LocalSeq2SeqLLM:
    """Factory so callers don't need to import the class directly."""
    return LocalSeq2SeqLLM(model_name or LLM_MODEL_NAME)

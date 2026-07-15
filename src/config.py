"""
Centralized configuration for the GenAI Backend Engineering exercise.

Keeping all tunables in one place makes it easy to reason about the
system's behavior and to override settings in tests without monkeypatching
scattered constants across modules.
"""

from pathlib import Path

# --- Storage locations -------------------------------------------------
# Everything is persisted under a single hidden project directory so the
# CLI can be run from any working directory and still find its index.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / ".index"
CHROMA_PERSIST_DIR = str(INDEX_DIR / "chroma")

# --- Ingestion / chunking ------------------------------------------------
SUPPORTED_EXTENSIONS = {".txt", ".md"}
CHUNK_SIZE_WORDS = 200          # target words per chunk
CHUNK_OVERLAP_WORDS = 40        # words shared between consecutive chunks

# --- Retrieval ------------------------------------------------------------
COLLECTION_NAME = "documents"
TOP_K = 4
# Chroma's default distance metric (cosine) yields values roughly in
# [0, 2], where 0 is a perfect match. Chunks farther than this threshold
# are treated as "not relevant enough" to support an answer.
MAX_RELEVANT_DISTANCE = 0.8

# --- Models ----------------------------------------------------------------
# Both models are small, free, and run fully locally/offline once
# downloaded once from Hugging Face's public model hub - no paid API
# keys or network calls are required at query time.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"
LLM_MAX_NEW_TOKENS = 128

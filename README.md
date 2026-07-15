# genai_backend_exercise
A small command-line RAG (retrieval-augmented generation) app that answers questions from a local collection of `.txt`/`.md` docs.

- **Vector store:** [ChromaDB](https://www.trychroma.com/) (local, persistent, no server needed)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, no charge)
- **Answer generation:** `google/flan-t5-small` using Hugging Face `transformers` (local, free)
- There are no paid APIs or hosted services in the pipeline anywhere.

Prerequisite:

- Python 3.9+ 
- ~500MB disk space for model weights (downloaded automatically, once, at first use, from the public Hugging Face model hub)
- Internet access is only required the *first* time you run `ingest` or `ask`, to download the embedding/LLM model weights. Then everything is completely offline.

Setup:

On linux distro run

```
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```

Usage: 

1. Ingesting Documents

Create a local persistent index from a directory of .txt/.md files:
python app.py ingest ./docs

Reads every `.txt`/`.md` file in the specified directory, splits it into overlapping chunks, embeds those chunks, and persists the resulting index under `.index/` in the project root (re-running `ingest` rebuilds the index from scratch, so it's always safe to re-run after editing the docs folder).

2. Ask a question

python app.py ask "What is the refund policy?" 

Sample output:
```Answer: Customers have the right to a refund within 30 days of purchase.```
Sources: refund-policy.md

If the indexed documents do not contain enough relevant content to support an answer, the app explicitly says so rather than guessing.
Answer: I did not find the necessary support in the supplied documents for that answer.
Sources: None

Error Handling:

The CLI handles common failure modes with clear messages and a non-zero exit code, e.g.
```python app.py # invalid usage -> prints usage/help python app.py ingest./nope # missing directory -> clear error python app.py ask "..." # before any ingest -> clear error ```

Running tests:

```
pip install -r requirements.txt   # includes pytest
pytest
```

Project layout:

```
app.py                  # CLI entry point 
src/
  config.py             # Tunable central configuration
  loader.py             # reads .txt/.md files from a directory
  chunker.py             # splits documents into overlapping chunks
  embeddings.py          # sentence-transformers embedding function
  vector_store.py        # ChromaDB persistence + retrieval
  llm.py                 # local flan-t5-small 
  answer_composer.py     # grounding check + answer/source composition
  pipeline.py             # hooks the above into ingest/ask operations
tests/                   # pytest suite (see conftest.py for fakes)
docs/                    # sample document collection used for manual testing
```






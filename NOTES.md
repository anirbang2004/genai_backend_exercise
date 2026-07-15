

Known limitations: 

- We’re splitting text strictly by word count right now, which means we occasionally slice a 
  sentence right down the middle. Switching to a sentence- or paragraph-aware splitter 
  (like using a lightweight NLP library) would give us much cleaner chunks, though it adds a bit of processing overhead.

- Raw Vector Search, We're trusting Chroma’s raw top-$k$ results based purely on cosine distance. 
  Adding a cross-encoder re-ranking step would definitely sharpen our precision for vague queries, 
  but it also introduces more latency and another model dependency we have to manage.

- The grounding threshold is currently a fixed constant. While it works perfectly 
  for our sample docs/ folder,   it hasn't been calibrated against a diverse, labeled dataset, 
  so it might need tweaking if we throw drastically different document styles or lengths at it.

- Every time the ingest i sinvoked, the tool does a full re-index. It’s safe and simple 
  because we never have to worry about stale chunks from deleted files, but it’s not incremental. It’s fine 
  for a small side  project, but it won't scale if we're dealing with a massive, rapidly changing corpus.

- This is built for a single machine and a single user. There’s no concurrency control around the persisted index, 
  meaning if we try to run an ingest and an ask at the exact same millisecond, they’ll hit a race condition. 
  For an interactive CLI tool, though, it’s a non-issue.

- First run requires internet access to download the embedding/LLM
  model weights, a few hundred MB, one-time, from the public Hugging
  Face hub. No paid API key or account is required.

- English-only tokenization/models, no multilingual support.


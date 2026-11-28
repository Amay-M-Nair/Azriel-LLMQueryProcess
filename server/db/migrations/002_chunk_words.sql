-- ROADMAP 2.2: each chunk's word count, so the store can report total_words
-- (what retrieve() compares with FULL_CONTEXT_WORDS) with a SUM instead of
-- pulling every chunk's text. Counted at ingest exactly as the FAISS store
-- counts it: len(text.split()). Filings text is full of non-breaking spaces,
-- which a Postgres regex would split differently.

ALTER TABLE chunks ADD COLUMN words integer NOT NULL CHECK (words >= 0);

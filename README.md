# Grounded QA for SEC Filings

Question answering over SEC 10-K (annual) and 10-Q (quarterly) filings. It
returns cited answers that are checked against the filing, or an explicit
"the filing does not say".

> **Work in progress.** The plan, its progress and every measured result live
> in **[ROADMAP.md](ROADMAP.md)**. Decisions made along the way, and anywhere
> the build differs from the plan, are in **[NEO.md](NEO.md)**. This README
> covers what exists today; Phase 9 writes the final one.

**Stack:** FastAPI (React frontend later), Postgres + pgvector on Neon, Groq
gpt-oss-120b / 20b for answers, local Hugging Face models for embeddings,
RAGAS for evaluation with DeepSeek on AWS Bedrock as the judge.

## How it works today

```
 SEC EDGAR  ->  [edgar]       fetch each company's 10-K and 10-Q filings
                    |
               [document_loader]  filing HTML -> plain text, tables as rows
                    |
               [chunker]      ~160-word overlapping chunks, with character
                    |         offsets and filing metadata (ticker, form,
                    |         fiscal year and period)
               [embedder]     all-MiniLM-L6-v2, 384 numbers per chunk, locally
                    |
               [store]        FAISS today; Postgres + pgvector from Phase 2

 a question -> classify -> route -> rewrite follow-ups -> hybrid search
            (meaning + BM25 keywords) -> top 5 chunks -> gpt-oss answers
            from those chunks only, with [1][2] citations
```

The evaluation runs on a **pinned snapshot**: 6 companies (AAPL, MSFT, JPM,
JNJ, WMT, CVX) x 3 fiscal years x one 10-K and three 10-Qs = 72 filings,
fixed by accession number and SHA-256 in `server/evaluation/snapshot.json`.
A golden set of 101 hand-verified questions (`golden.jsonl`) is scored on
retrieval (recall, MRR, nDCG) and with RAGAS. The baseline and every later
improvement are in ROADMAP.md's Results table.

## Setup

1. Create the environment and install the dependencies:

   ```bash
   python -m venv .venv
   .venv/Scripts/python.exe -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in what you use. Each variable is
   explained in the file:
   - `GROQ_API_KEY` for answers
   - `SEC_USER_AGENT` for EDGAR (a name and a contact email)
   - `DATABASE_URL` / `TEST_DATABASE_URL` for Neon
   - `AWS_BEARER_TOKEN_BEDROCK` for the evaluation judge

3. Check what is set up:

   ```bash
   cd server
   ../.venv/Scripts/python.exe -m utils.check
   ```

The first run downloads the embedding model (~90 MB), once. Everything
generated (filings, indexes, caches, logs, your `.env`) lives under `data/`
and is not tracked.

## Commands

Run these from `server/`, with `../.venv/Scripts/python.exe -m ...`:

| Command | What it does |
|---|---|
| `ingestion.edgar AAPL` | List a company's recent 10-K and 10-Q filings |
| `evaluation.snapshot` | Fetch the pinned filings and check every hash |
| `ingestion.filings` | Index the snapshot into `data/snapshot_index/` (FAISS) |
| `evaluation.evaluate` | Answer and grade the golden set, write a Results row |
| `evaluation.regression` | Check the last run against `thresholds.json` |
| `evaluation.golden check` | Validate the golden set against the filings |
| `evaluation.judge --ping` | Check the RAGAS judge is reachable |
| `db.migrate` | Apply schema migrations to Neon (`--test` for the test branch, `--status` to list) |

The golden-set review tool is a local Streamlit page, never deployed:

```bash
../.venv/Scripts/streamlit.exe run evaluation/review_app.py
```

Tests run from the repository root: `.venv/Scripts/python.exe -m pytest`.
Each run also ticks or unticks the subphases in ROADMAP.md. Tests that need
the database use only `TEST_DATABASE_URL`, and skip without it.

## Project layout

Everything Python is under `server/`:

| Package | Job |
|---|---|
| `backend/` | Classify a question, route it, rewrite follow-ups, build the prompt; `providers/` holds Groq and Ollama |
| `ingestion/` | EDGAR fetching, filing text, chunking, embedding, indexing |
| `vectorstore/` | The FAISS index and BM25, retired after the parity check (2.6) |
| `db/` | Neon: connections, migrations, the pgvector store |
| `evaluation/` | Snapshot, golden set, metrics, RAGAS judge, the evaluation run, regression floors |
| `tools/` | The ROADMAP.md generator, ONNX export, the memory measurement |
| `utils/` | Settings (`config.py`), logging with secret redaction, prompts |
| `tests/` | The test suite; `@pytest.mark.roadmap("x.y")` marks each subphase's "done when" check |

`web/` will hold the React app (Phase 9).

## Logs and secrets

Logs go to stderr and `data/groundedqa.log`. API keys, database URLs and the
EDGAR contact email are redacted before anything is logged, and tests plant
secrets to prove it.

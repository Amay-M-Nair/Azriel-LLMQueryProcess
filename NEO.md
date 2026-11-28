# NEO — notes beside the roadmap

**ROADMAP.md is the plan and is not edited.** This file holds what has come up since:
- decisions the roadmap left open,
- facts found along the way,
- proposals waiting for a yes.

**Rules:**
- Nothing here overrides ROADMAP.md on its own.
- Anything that differs from ROADMAP.md is listed under **Departures**, with who decided it and when.
- Proposals stay marked as proposals until agreed.

Last checked against ROADMAP.md: 2026-11-28.

---

## Departures from ROADMAP.md

### D1. The RAGAS judge runs on AWS Bedrock, not Gemini
- **Status:** decided by the owner on 2026-11-27 by setting up Bedrock ($120 of credits).
- **ROADMAP.md says:** "RAGAS (Gemini as judge)" (Stack); "RAGAS judge: Gemini (a different family, avoiding self-preference)" (Model settings); "The judge is Gemini" and "It runs on Gemini's free tier" (1.6).
- **What stays the same:** the roadmap's reason for its choice. The judge is still from a different family than the gpt-oss models that write the answers. The 1.6 metrics, caches and *done when* are unchanged.
- **Why:** 1.6 found that the Gemini free tier gives about 10–15 grades a day per model, so a golden-set run would take over a month. This also settles 1.6's open decision ("how 1.8 is judged"): a paid judge, paid from the credits.
- **The one-model-per-metric rule still holds:** each metric has one fixed model, never a fall-through list. On Bedrock it is the same model for all five metrics, so `JUDGE_MODELS` (a dict) became `JUDGE_MODEL` (one string) in `utils/config.py`.
- **The Gemini judge code was removed (owner, 2026-11-27):** the Gemini path, its five models and the daily-quota handling existed only for the free tier. The judge is Bedrock-only. A throttled call (429/503) is waited out, and any other failure is recorded and retried on the next run. `--candidates` went too: `golden.jsonl` is approved, so the judge always reads it. The Gemini runs stay in the history of `ragas_check.json` as a record; the 1.6 check only counts runs made by the current judge.

### D2. Gemini is removed from the project
- **Status:** decided by the owner on 2026-11-27.
- **ROADMAP.md says:** "Gemini stays as judge and fallback, Ollama for offline dev" (0.6); google-genai "held at newest 2.x" (Versions checked, Phase 0).
- **What changed:** the Gemini answer provider (`backend/providers/gemini.py`), its settings in `utils/config.py`, `utils.check --models`, the `google-genai` pin and `GOOGLE_API_KEY` are gone. The judge is Bedrock (D1). Answers come from Groq, with Ollama for offline work.
- **What stays:** `tests/test_phase0.py` still fails on a Google-key-shaped string anywhere in the repo, because 0.1 revoked an exposed one. Log redaction now covers `AWS_BEARER_TOKEN_BEDROCK` in place of the Google keys.
- **Why:** nothing in the plan uses Gemini any more, and the code existed only for its free-tier quota.

### D3. The old Azriel app is removed now, not in Phase 9
- **Status:** decided by the owner on 2026-11-28.
- **ROADMAP.md says:** the interim Streamlit page and `.streamlit/config.toml` stay until Phase 9 (0.2, 9.4). README describes the pipeline as it stands.
- **What went:**
  - The old document-chat app: `server/app.py`, `server/.streamlit/`, and `server/api/` (its in-process client and upload collections).
  - The app-only helpers: `utils/examples.py` and `utils/env_file.py` (typing API keys into the page).
  - The synthetic evaluation: `evaluation/run_eval.py`, `evaluation/corpus/` and `evaluation/dataset.jsonl`, replaced by `evaluate.py` and the golden set.
  - Their tests, and the settings only the app used (`INDEX_DIR`, `UPLOAD_DIR`, `DEFAULT_COLLECTION`).
- **Why:** the owner found the old build confusing next to the SEC system. Nothing in the SEC pipeline or the evaluation used these files, and none of their tests were roadmap checks, so no subphase lost its tick.
- **What changed with it:**
  - README.md now describes only the SEC build.
  - The stale-results hint in `tests/test_roadmap.py` and the roadmap tool now point to `evaluation.evaluate`. ROADMAP.md itself still names `run_eval` in "How this file stays current"; read it as `evaluate`.
  - The 0.3 layout test checks `db/` instead of `api/`.
- **What stays until its own phase** (changing `ingestion/` or `vectorstore/` marks the baseline stale and forces a new evaluation run):
  - The PDF, Word and OCR readers in `ingestion/document_loader.py`, the upload path in `ingestion/pipeline.py`, and their packages (pypdf, python-docx, pymupdf, rapidocr). These go in 4.1.
  - The FAISS store: 2.6.
  - `streamlit`, still needed by the local golden-set review tool.
- **Until Phase 9 there is no web page to try the system in.** The API (2.5) is the next way to ask it questions.

---

## Decisions within the roadmap

### Phase 2 — Versions checked (2026-11-28)
Logged here, not in ROADMAP.md's *Versions checked* table, since ROADMAP.md is not edited.

| Library / service | Version | Note |
|---|---|---|
| psycopg[binary] | 3.3.9 | Newest. The binary extra bundles libpq, so no Postgres install is needed here or on Render. |
| pgvector (Python) | 0.5.1 | Newest. Reads a `vector` column back as a `pgvector.Vector`, not a numpy array: call `.to_numpy()`. |
| Neon Postgres | 18.7 | What the free plan gave. |
| pgvector extension | 0.8.7 | Installed by migration 001 (`CREATE EXTENSION vector`). |
| psycopg-pool | not added | One connection is enough for now; revisit at 2.5 if the API needs a pool. |

`pip check` is clean.

### 2.1 — Neon setup (2026-11-28, owner)
- **Free plan** (Neon's FAQ, Nov 2026): 1 GB storage per project, 10 branches, 100 CU-hours a month, scale-to-zero after 5 idle minutes. If storage passes 1 GB, writes fail; nothing is billed. The 31k chunks with vectors should take about 120–150 MB.
- **Project** `grounded-qa` on AWS US East (Ohio), where Render also has a region (9.4).
- **Two branches:** `production` is Neon's default name; ROADMAP.md names none. It holds the live index (`DATABASE_URL`). `test` is for tests only (`TEST_DATABASE_URL`, a new variable).
  - `db.connection` refuses `TEST_DATABASE_URL` if it equals `DATABASE_URL`, so no test can reach the live data.
  - Both variables are redacted in logs.
- **Direct connections, no pooler.** Migrations need a direct connection, and nothing needs a pooler yet. `.env.example` says so now; it used to say "pooled".
- **Latency from this machine:** about 0.28 s per round trip and 1.7 s per new connection. For 2.4 this means ingest must write in batches (COPY or pipelined `executemany`). Row-by-row, 31k chunks would take about 2.5 hours in round trips alone. The 2.1 tests share one connection and one migrated schema for the same reason, which cut them from 105 s to 47 s.

### 2.1 — The schema (2026-11-28)
- **Migration runner: our own** (`server/db/migrate.py`), not Alembic, so no new package.
  - Files are named `NNN_name.sql` and applied in order, each in its own transaction. A failure keeps nothing.
  - `schema_migrations` stores a SHA-256 of each applied file. A file edited after it was applied is refused; changes go in a new migration.
  - An advisory lock keeps two runners apart.
  - CLI: `python -m db.migrate [--test] [--status]`, run from `server/`.
- **Tables** (`db/migrations/001_schema.sql`): `filings`, `chunks`, `tables`, `eval_runs`.
  - The roadmap's "company" on `chunks` is stored as `ticker` (plus `cik`), which is what the filters use. The company name sits on `filings`.
  - **The copies on `chunks` cannot drift from their filing:** a second foreign key on (accession, ticker, cik, form, fiscal_year, fiscal_period) enforces it.
  - `form` is free text. `fiscal_year` and `fiscal_period` may be null for a future form with no fiscal period (8-K). Checks require them for 10-K and 10-Q, `FY` for a 10-K, and Q1–Q3 for a 10-Q.
  - **The span** is `start_char` / `end_char`. `page`, `start_word` and `end_word` are kept too, so a `Chunk` read back from Postgres equals the one FAISS holds (parity, 2.6).
  - **No vector index:** an exact scan, as 2.2 asks. `1 - (embedding <=> q)` matches FAISS's inner product to 1e-6, and float32 vectors round-trip exactly (tested).
  - The commit column is `eval_runs.git_commit`, because `commit` is an SQL keyword.
- **Live index vs the snapshot:** both live in the same tables. The evaluation will filter to the accessions in `evaluation/snapshot.json` (2.2 / 2.6), so filings added by 2.7 never move a score.
- **Done when met:** 14 tests in `tests/test_db_schema.py` apply the migrations to a new empty schema on the `test` branch, then check constraints, cascades and cosine ranking. The CLI applied 001 to the `test` branch itself; a second run applied nothing. `pytest` ticked 2.1.
- **Not done yet:** the `production` branch is not migrated. That happens when 2.4 first ingests, or earlier with `python -m db.migrate`.

### 2.2 — The pgvector store (2026-11-28)
- **Where it lives:** `server/db/store.py`, not `server/vectorstore/`. `tests/test_roadmap.py` marks the baseline stale when anything under `backend/`, `ingestion/` or `vectorstore/` is committed after the last evaluation run. The pipeline doesn't use this store until 2.6, and 2.6 is itself an evaluation run, so it re-measures then.
- **Interface:** `PgVectorStore` answers what the pipeline asks of the FAISS store: `len(store)`, `sources`, `total_words` and `search(query_vector, k, query, min_similarity)`. `search` also takes `filters`. Writing chunks is 2.4's job.
- **Filters** (`Filters`): lists of tickers, forms, fiscal years, fiscal periods, sections and accessions. Each list keeps chunks that match any of its values, and the lists combine with AND.
  - They are lists because a Q4 question needs two filings: `fiscal_years=[2025], fiscal_periods=["FY", "Q3"]`.
  - A single string is refused, so `tickers="AAPL"` can't be read as five one-letter tickers.
  - A section filter leaves out chunks that have no section yet; sections are filled from 4.1.
- **Scope vs filters:** `scope` is fixed for the store's lifetime and limits everything it reports. `Filters.snapshot(snapshot)` scopes to the pinned filings, which keeps filings added by 2.7 out of the evaluation. `filters` narrows a single search further; 4.3 fills it from the question.
- **Ranking is by inner product (`<#>`), not cosine distance (`<=>`):** it is the metric FAISS's `IndexFlatIP` uses, and the vectors are unit length, so it equals cosine without the division. Ties are broken by row id, which is ingest order: the order FAISS holds the chunks in.
- **Keyword search:** `KeywordIndex` is BM25 over a fixed set of chunks, keyed by database id.
  - The blend is copied line for line from the FAISS store: the same pool sizes, the same normalisation, and the same rescue for strong keyword matches.
  - Without a keyword index, search ranks by meaning alone.
  - A chunk added after the keyword index was built raises an error, rather than quietly scoring 0.
  - 2.3 loads the index from Postgres at startup and measures its memory.
- **Migration 002** adds `chunks.words`, counted at ingest as `len(text.split())`, so `total_words` is a SUM. A Postgres regex splits non-breaking spaces differently, and filings are full of them. 001 was not edited, because it was already applied.
- **Shared test fixtures** live in `server/tests/conftest.py`: one admin connection per session, plus a fresh or migrated throwaway schema per test module. The 2.1 tests use them too.
- **Done when met:** 19 tests in `tests/test_db_store.py`.
  - 240 synthetic chunks across 4 filings go into a throwaway schema on the `test` branch.
  - The FAISS store and `PgVectorStore` return the same chunk ids in the same order, with equal `Chunk`s and scores within 1e-5. This holds in every branch: meaning only, hybrid over every chunk, hybrid over a pool, the similarity floor with keyword rescue, and a store scoped to 3 of the 4 filings.
  - The filters are checked separately: the Q4 pair, form, section, and no match.
  - The CLI applied 002 to the `test` branch, and `pytest` ticked 2.2.

### 1.6 — How the Bedrock judge is wired (2026-11-27)
- **Route:** Bedrock's OpenAI-compatible endpoint, `bedrock-mantle.us-east-1.api.aws/v1`, with a Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK` in `.env`). The existing RAGAS `llm_factory(provider="openai")` path works unchanged, with no new packages.
  - The model cards for Qwen, Kimi and GLM list Chat Completions on bedrock-mantle. bedrock-runtime's OpenAI route is mainly for OpenAI's own models.
  - mantle model ids drop the `:0` version suffix.
- **Not Amazon Nova:** AWS's compatibility table lists every Nova model as Invoke and Converse only, with no Chat Completions. So the judge is a third-party model sold on Bedrock.
- **Judge model:** DeepSeek V3.2 (`config.JUDGE_MODEL`). The exact model id is confirmed with `python -m evaluation.judge --ping` before the first run.
  - It is not an OpenAI model, so it is a different family from gpt-oss.
  - Price at 2026-11: $0.62 in / $1.85 out per 1M tokens (Standard tier, US East; rechecked on the pricing page 2026-11-27).
  - **Cost of a full run, measured (2026-11-27):** RAGAS 0.4.4's real prompts, rendered with the cached answers and passages plus the JSON schema instructor adds, come to about 16,700 input tokens per question (14 calls). For all 101 cases: 1,414 calls, 1.69M input tokens (o200k tokenizer; DeepSeek's may differ by ~10–15%). The cached Gemini replies average 74 tokens, so ~0.1M output. That is about **$1.25 a run**, or about $2 with 3× longer replies plus retries. The earlier estimate (3.5M in, 0.6M out, $3.30) is a safe upper bound.
  - Not yet known: whether DeepSeek on this endpoint thinks before it answers (output would grow). The ping prints the token counts.
  - Cheaper fallbacks: Mistral Large 3 ($0.50 / $1.50, about $2.65 a run) and Gemma 4 31B ($0.14 / $0.40, about $0.75 a run).
- **Models compared (2026-11-27, Standard tier, US East, per 1M tokens in / out; cost of one full run at ~3.5M in, ~0.6M out):**

  | Model | Price | ≈ $ per run | Fit as the judge |
  |---|---|---|---|
  | DeepSeek V3.2 | 0.62 / 1.85 | 3.30 | **Chosen.** Strong open model, not OpenAI. |
  | Mistral Large 3 | 0.50 / 1.50 | 2.65 | Good backup. |
  | Qwen3 Next 80B A3B | 0.15 / 1.20 | 1.25 | Cheap; only 3B active, a weaker judge. |
  | MiniMax M2.5 | 0.30 / 1.20 | 1.75 | A reasoning model, so it writes more tokens than listed. |
  | Kimi K2.5 | 0.60 / 3.00 | 3.90 | Strong; costlier output. |
  | GLM 4.7 / GLM 5 | 0.60 / 2.20; 1.00 / 3.20 | 3.40; 5.40 | GLM 4.7 is "Legacy" (EOL 2027-03-30). |
  | Gemma 4 31B | 0.14 / 0.40 | 0.75 | Cheap; mid-strength. |
  | Nemotron 3 Super 120B A12B | 0.15 / 0.65 | 0.90 | Cheap; mid-strength. |
  | Kimi K3 | 3.00 / 15.00 (global) | 19.50 | Too expensive for repeated runs. |
  | OpenAI gpt-oss on Bedrock | — | — | **Ruled out:** the same family that writes the answers. |
  | Amazon Nova (all) | not read | — | **Ruled out:** no Chat Completions; would need new code. |
  | Anthropic Claude | not listed as a table | — | **Ruled out:** billed through AWS Marketplace (Haiku 4.5's card), which credits do not cover. |

  Qwen3 235B and Qwen3 32B have no US East price on the page; xAI Grok has no price.
- **Embeddings** (answer relevancy, answer correctness) run locally with `sentence-transformers/all-MiniLM-L6-v2`, the same model as retrieval: free and the same every run. Bedrock's embedding models are not on the OpenAI endpoint.
- **Changing the judge re-grades everything:** grades are keyed by judge, so the 13 Gemini grades in the cache are not reused. Regrading the 5 check cases is about 70 calls, a few cents.
- **Credits check:** AWS credits do not cover AWS Marketplace charges. After the first small run, the Billing console must show the charge under "Amazon Bedrock", covered by credits, before a full run. Billing can lag up to a day.

### 1.6 — Done on Bedrock (2026-11-27)
- **Done when met:** the 5 check cases were graded on all 5 metrics (25 grades, 60 judge calls, 348 s), then a rerun gave 25 from cache, 0 calls. `pytest` ticked 1.6.
- **The ping:** `deepseek.v3.2` answered with 2 output tokens, so it does not think first on this endpoint. The default `AmazonBedrockLimitedAccess` key policy was enough for bedrock-mantle.
- **Bug found and fixed: RAGAS keys its call cache by prompt alone, not by model.** The first DeepSeek run shared the call cache with the Gemini runs. Wherever a prompt matched, it got Gemini's old reply: 34 calls instead of 60, and grades that were part Gemini. Those 25 grades and that run were thrown away. The call cache is now one folder per judge model (`judge_cache/calls/{model}/`), with a test.
- **Answer relevancy runs at `strictness=1`:** RAGAS's default of 3 sends the same prompt three times. The call cache answers the second and third from the first, so 3 would only look like an average of three. 1 says what happens. This leaves 12 calls per question, not 14: about 1,212 calls and $1.15 for a full run.
- **Scores (a check, not an evaluation):**

  | Case | Faithfulness | Answer relevancy | Context precision | Context recall | Answer correctness |
  |---|---|---|---|---|---|
  | net sales FY2026 | 1.00 | 1.00 | 0.20 | 1.00 | 0.88 |
  | employees FY2026 | 0.50 | 1.00 | 0.00 | 0.00 | 0.41 |
  | services growth FY2026 | 0.29 | 0.97 | 0.00 | 0.00 | 0.45 |
  | wearables definition | 1.00 | 0.92 | 0.58 | 1.00 | 0.74 |
  | sales drivers FY2026 | 1.00 | 0.75 | 0.00 | 0.25 | 0.38 |

  Where retrieval missed (context recall 0), correctness is low. This is the same weakness 1.5 found: the right company, the wrong filing.
- **Still to check before 1.8:** that the Billing console shows these calls under "Amazon Bedrock", paid by credits (it can lag a day).

### 1.6 — Made ready for 1.8 (2026-11-27, owner agreed)
- **Answer correctness is scored on facts alone** (`weights=[1.0, 0.0]`, `METRIC_SETTINGS` in `evaluation/judge.py`). RAGAS's default gives 25% to embedding similarity, which pays a wrong figure on the right topic.
- **Metric settings are part of every grade's key**, so changing one re-grades that metric. The regrade cost 0 judge calls, because the prompts were unchanged and came from the call cache.
- **Unanswerable cases stay out of RAGAS** (`judge.answerable`). Their right answer is a refusal, which RAGAS marks down; 5.4's refusal accuracy scores them. 1.8 grades the 89 answerable cases with RAGAS.
- **Grading runs 8 at a time** (`config.JUDGE_CONCURRENCY`); a throttled call backs off. A full run should take about 15 minutes, not 2 hours. The cost is unchanged.
- **The judge is lenient on hedged answers (found by hand, 2026-11-27).** The services-growth answer said the passages lacked the figure, then invented $35B → $38B; the filing says $109B → $123B. DeepSeek counted the three hedge sentences ("the excerpts do not include…") as true positives against the reference, so facts-only correctness still came to 0.46 (3 / (3 + ½·7)). The wrong figures were all caught as false positives. This is the case for P3 and P4.
- **Money:** about $0.10 spent so far (the ping and two 5-case runs, ~120K input tokens). A full 1.8 run is about $1.10 (89 answerable cases × 12 calls).

### 1.9 — The regression harness (2026-11-27)
- **The rule** (`evaluation/regression.py`, thresholds in `evaluation/thresholds.json`, set from `baseline.json`):
  - **Floors:** each quality metric's floor is the lower end of the baseline's 95% interval, rounded down. A drop below it is more than resampling the questions would explain. Recall@5 ≥ 0.095, recall@10 ≥ 0.149, MRR ≥ 0.078, nDCG@10 ≥ 0.090, faithfulness ≥ 0.678, answer relevancy ≥ 0.593, context precision ≥ 0.120, context recall ≥ 0.213, answer correctness ≥ 0.380.
  - **Ceilings:** $/query may grow by half (≤ $0.000714) and p95 latency may double (≤ 32.4 s), since latency moves with the network and the machine. An unknown cost fails.
  - **Same set:** a run on another snapshot, or with ungraded cases, fails: its numbers are not comparable.
- **How it works with the rest:** `tests/test_roadmap.py` already fails when the pipeline changes after the last run, so a change has to be re-measured (`python -m evaluation.evaluate`), and `tests/test_regression.py` then fails if the new numbers fall below the floors.
- **Raising the floors** is done on purpose, with `python -m evaluation.regression --set <result>`, once an improvement is the new current pipeline.
- **Phase 1 is complete (9/9).**

### 1.8 — The baseline (2026-11-27)
- **Run by** `python -m evaluation.evaluate` (from `server/`), not `run_eval`: `run_eval` still scores the old synthetic corpus. Results in `evaluation/results/baseline.json`, at commit 05d823f on snapshot v1.
- **Baseline (101 cases; RAGAS and retrieval on the 89 answerable; 95% bootstrap intervals):**

  | Metric | Value | 95% CI |
  |---|---|---|
  | Recall@5 (what the answer model saw) | 0.169 | 0.096–0.247 |
  | Recall@10 | 0.231 | 0.149–0.319 |
  | MRR | 0.137 | 0.079–0.204 |
  | nDCG@10 | 0.148 | 0.091–0.211 |
  | Faithfulness (n = 86) | 0.749 | 0.679–0.814 |
  | Answer relevancy | 0.679 | 0.593–0.758 |
  | Context precision | 0.186 | 0.121–0.259 |
  | Context recall | 0.302 | 0.213–0.399 |
  | Answer correctness (facts only) | 0.435 | 0.380–0.490 |
  | $/query (Groq) | $0.00048 | — |
  | p50 / p95 latency | 10.2 s / 16.2 s | — |

- **By question type** (recall@5 · recall@10 · faithfulness · correctness · context recall):
  - cross-period (15): 0.27 · 0.27 · 0.87 · 0.47 · 0.27
  - definition (10): 0.20 · 0.40 · 0.80 · 0.64 · 0.68
  - follow-up (12): 0.25 · 0.33 · 0.81 · 0.39 · 0.29
  - summary (10): 0.00 · 0.06 · 0.55 · 0.30 · 0.05
  - table lookup (22): 0.09 · 0.14 · 0.78 · 0.41 · 0.23
  - text lookup (20): 0.20 · 0.25 · 0.66 · 0.43 · 0.36
- **What it says:** retrieval is the bottleneck, as 1.5 found. Tables (0.09) and summaries (0.00) are worst: the minimal loader's tables (4.2) and multi-passage summaries. Definitions do best, being worded like the filing. Faithfulness is far higher than correctness: the answers mostly stay with their passages, but the passages are often the wrong ones.
- **Routes:** 96 retrieval, 5 clarify (follow-ups the classifier could not place; see 1.7). 9 follow-ups were rewritten by the model.
- **Cost:** answers $0.048 on Groq (2,287 input tokens a question, 702 of them cached; 409 output, 311 of them reasoning). The judge made about 1,000 calls, about $1 of the Bedrock credits by the 1.6 estimate; Billing still to confirm.
- **How the run was kept honest** (each a fix made during this run):
  - Warm-up before timing: the first retrieval takes about 16 s.
  - No model fallback: a rate-limited 120b quietly answered one case on 20b; that answer was deleted and remade.
  - Groq's rate limit (4 waits) is waited out, and an answer whose classifier or rewriter call quietly failed is remade (none were).
  - No passages retrieved (3 answerable follow-ups routed to clarify): context precision and recall are 0, and faithfulness does not apply (hence n = 86), instead of judge errors.
  - A case counts as judged only if no metric errored; 2 passing judge errors were retried.
- **Dates are UTC** in results files, so this run reads 2026-11-26.

### 1.7 — Cost and latency accounting (2026-11-27)
- **What is recorded:** every model call's stage, model and tokens (prompt, cached, completion, reasoning) go into `QueryTrace.usage`, through a per-question metered copy of the provider (`Provider.metered`), so a shared provider never mixes two questions' costs. `trace.cost` prices them from `config.PRICES`. `trace.total_seconds` now includes the streamed answer (the `answer` stage), which was not timed before.
- **Prices** (`config.PRICES`, checked 2026-11-27 on the console.groq.com model pages, USD per 1M tokens): gpt-oss-120b $0.15 in / $0.075 cached / $0.60 out; gpt-oss-20b $0.075 / $0.037 / $0.30. Reasoning tokens are part of completion_tokens and billed as output.
- **Unknown is never $0:** a call whose model has no price, or a question with fewer usage records than calls, makes the cost `None`, and a run with any unknown cost reports no $/query.
- **Eval results carry them:** each answer record has `usage`, `tokens`, `cost` and `seconds`. `answers.summarize` gives `cost_per_query`, `p50_seconds` and `p95_seconds`, the keys the Results table reads, for 1.8 to write. `run_eval` prints tokens, $ and p50/p95 too.
- **Bug found and fixed: the eval answered with no provider.** `answers.answer` passed none, so `process()` skipped the classifier and the rewriter: every cached answer shows `classify: 0.0, rewrite: 0.0` and 1 call. It now uses the configured provider. `RECORD_VERSION = 2` in the fingerprint starts a new answer set, so the 1.6 check's answers are not mixed with these.
- **Live check (2 questions, Groq):** net sales took 2 calls (classify on 20b, answer on 120b), 2,547 input and 148 output tokens (85 reasoning), $0.00039. Answering the whole golden set costs about $0.04 on Groq, which is separate from the Bedrock credits.
- **Found in the live check, left for the baseline to measure:**
  - **Cold start.** The first question's retrieval took 16.5 s (loading models); a warm one takes about 0.5 s. 1.8 warms up before timing, or p95 measures the start-up rather than the pipeline.
  - **Follow-ups fail at classification.** The classifier does not see the conversation, so "And how did that compare with the year before?" comes back `unknown`, routed to clarify ("I could not tell what that is asking"), before the rewriter can help. The app behaves the same way, so the baseline records it honestly. It is a strong first improvement row after 1.8: classify the rewritten question, or give the classifier the history.

### 1.6 — Facts that changed since ROADMAP.md was written
- The 5 check cases now come from the approved `golden.jsonl` (the default), not from drafts with `--candidates`. They are the same 5 Apple cases, one per answerable type.

### Judge options checked 2026-11-26 (kept for the record)
- **Gemini paid:** about $1–2 a run, no code change, $5 minimum prepay. Google Cloud's $300 trial credit can't pay for the Gemini API, nor for Vertex Gemini for sign-ups from 2026-11-16.
- **OpenRouter:** free models 50 requests a day, or 1,000 a day after $10 of credit (a run then spans about 2 days). No embeddings.
- **Groq Qwen:** preview only. 200K tokens a day (about 2.5 weeks a run), or $0.80 / $4.00 per 1M tokens paid.
- **Local Ollama** on the RTX 3050 (4 GB): free, 4–15 hours a run, a weaker judge.

---

## Proposals (not agreed)

- **P1. RAGAS only on key runs.** Run RAGAS on the baseline, on the best configuration of each phase, and on the final system: about 6–8 runs, not one per ablation. Retrieval metrics still run on every experiment, since they cost nothing. On Bedrock this is about $20–30 in all; RAGAS on every ablation (~20 runs) would be about $65.
- **P3. Check the judge early, not only in 7.5.** After the 1.8 run, the owner hand-labels about 25 grades (faithfulness and answer correctness). Agreement is reported as Cohen's kappa with a CI (`metrics.py` already has it), before the baseline numbers are used to decide anything. 7.5 keeps the full ~50-label calibration. Cost: the owner's time, $0.
- **P4. An exact figure check beside answer correctness.** Free and deterministic: does the answer state the reference's figures (allowing rounding and millions written as billions), and does it state figures the reference does not? `golden check` already matches numbers this way. For financial questions this is a firmer signal than an LLM's statement matching, and it catches the case above, where the judge's leniency did not.
- **P2. Somewhere to see online metrics.** The roadmap records traces (8.3 MLflow tracing), but nothing aggregates live metrics into a view, feedback signal or alert. Could join Phase 8 or 9.

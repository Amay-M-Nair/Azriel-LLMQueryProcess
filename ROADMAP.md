# Grounded QA for SEC Filings — Roadmap

A question-answering system for SEC 10-K (annual) and 10-Q (quarterly) filings. It returns
cited answers that are checked against the filing, or an explicit "the filing does not say".
The live index picks up new filings as they are filed; the evaluation runs on a pinned
snapshot. Each upgrade is measured against a baseline before it counts.

**Stack:**
- **Frontend and API:** React + FastAPI on Render.
- **Database:** Postgres + pgvector on Neon.
- **LLM:** Groq (gpt-oss-120b strong, gpt-oss-20b cheap).
- **Local models:** Hugging Face embeddings, reranker and NLI. PyTorch offline, ONNX in production.
- **Supporting:** Pydantic, RAGAS (Gemini as judge), MLflow.

**How this file stays current:**
1. **Generated blocks.** The **Status** and **Results** blocks below are generated, never typed:
   - `python -m evaluation.run_eval` rewrites Results (from `server/`).
   - Every `pytest` run ticks or unticks each subphase from its `@pytest.mark.roadmap("x.y")` tests (`server/tools/roadmap.py`, hooked in `server/conftest.py`). A skipped check counts as not done.
   - `server/tests/test_roadmap.py` fails when Results are older than the last pipeline change.
   - `python -m tools.roadmap --check` (from `server/`) reports whether this file is current.
2. **Live filings, pinned evaluation.** Filings come from SEC EDGAR.
   - The live index is kept current: a refresh job (2.7) adds each new 10-K and 10-Q from the tracked companies.
   - The evaluation always runs on the pinned snapshot in `evaluation/snapshot.json`, so a score changes only when the code does.
   - The snapshot is refreshed on purpose, as a new version (for example when a new fiscal year closes), and the golden set is extended with it. Every Results row names the snapshot it ran on.
3. **Version checks.** Every phase opens with a version check, logged under *Versions checked*.

---

## Status
<!-- status:start -->
| Phase | Done | Next |
|---|---|---|
| 0 Foundations | 7/7 | complete |
| 1 Data and baseline | 2/9 | 1.3 Baseline ingestion |
| 2 Database and API | 0/7 | 2.1 Neon schema |
| 3 Structured outputs | 0/3 | 3.1 Schemas |
| 4 Parsing and hybrid retrieval | 0/6 | 4.1 Layout-aware 10-K and 10-Q parsing |
| 5 Reranking, expansion, refusal | 0/5 | 5.1 Cross-encoder reranker |
| 6 Routing and cost | 0/3 | 6.1 Decompose route |
| 7 Verification | 0/5 | 7.1 Claim decomposition |
| 8 Safety and tracing | 0/3 | 8.1 Adversarial set |
| 9 React frontend and deployment | 0/6 | 9.1 React app |
<!-- status:end -->

## Results
<!-- results:start -->
No runs yet. The baseline (1.8) comes first; every later subphase adds a row, including rows where nothing improved.

| Configuration | Recall@5 | MRR | nDCG@10 | Faithfulness | Unsupported claims | Correct refusals | $/query | p95 latency | Run |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (current pipeline) | | | | | | | | | |
<!-- results:end -->

## Model settings per stage
These are starting points. Each is confirmed at its phase start and tuned by ablation.

| Stage | Model | `reasoning_effort` |
|---|---|---|
| Intent, rewrite, claim splitting | gpt-oss-20b | low |
| Answer generation | gpt-oss-120b | medium |
| Decompose / multi-hop | gpt-oss-120b | high |
| RAGAS judge | Gemini (a different family, avoiding self-preference) | — |

Temperature starts at OpenAI's recommended 1.0 for gpt-oss. 1.0 vs 0.3 is run as an ablation rather than assumed.

## Versions checked
| Date | Phase | Library / API | Version pinned | Note |
|---|---|---|---|---|
| 2026-11-24 | 0 | Python | 3.11.9 | |
| 2026-11-24 | 0 | groq | 1.7.0 | `reasoning_effort` (none/default/low/medium/high), `include_reasoning`, `usage.completion_tokens_details.reasoning_tokens` confirmed in the SDK |
| 2026-11-24 | 0 | torch | 2.15.0 (+cpu) | `torch.onnx.export` now defaults to dynamo; legacy `dynamo=False` still works but needs `onnx` |
| 2026-11-24 | 0 | sentence-transformers / transformers | 6.3.0 / 5.17.0 | |
| 2026-11-24 | 0 | onnxruntime / tokenizers / onnx | 1.32.0 / 0.23.2 / 1.23.2 | onnx is export-time only |
| 2026-11-24 | 0 | optimum / optimum-onnx | not used | every optimum-onnx release needs transformers<4.56; sentence-transformers 6 needs transformers 5 — so `sentence-transformers[onnx]` cannot install. Exported with torch directly instead |
| 2026-11-24 | 0 | fastapi / uvicorn | 0.165.0 / 0.54.0 | |
| 2026-11-24 | 0 | streamlit, langchain-*, faiss-cpu, httpx, pytest | newest in current major | see `requirements.txt` |
| 2026-11-24 | 0 | pypdf, pymupdf, google-genai, anthropic | held at newest 6.x / 1.x / 2.x / 1.x | new majors (7 / 2 / 3 / 2) exist; taken in the phase that touches that code |
| 2026-11-25 | 1 | httpx | 0.28.1 | still the newest release; `httpx.MockTransport` replays recorded EDGAR responses, so no mocking library is added |
| 2026-11-25 | 1 | pydantic | 2.14.4 newest (2.13.5 installed via fastapi) | pinned at 1.4, where the golden-set schema first needs it |
| 2026-11-25 | 1 | ragas | 0.4.4 newest | pinned at 1.6, after checking its Gemini judge support |

---

## Phase 0 — Foundations
*Goal: a clean base that can measure itself.*

- [x] **0.1 Secret hygiene**
  - Blank `GOOGLE_API_KEY` in `.env.example` and revoke the exposed key.
  - Add `GROQ_API_KEY`, `SEC_USER_AGENT`, `DATABASE_URL` placeholders.
  - *Done when:* `tests/test_env_file.py` passes and `git grep` finds no key-shaped strings.
- [x] **0.2 Remove what the new design replaces**
  - Delete `packages.txt`, visitor collections, `AZRIEL_PUBLIC`, `tests/test_deployment.py`, `HttpClient`.
  - `.streamlit/config.toml` stays (moved to `server/`) while the interim Streamlit page exists; it goes with the page in Phase 9.
  - *Done when:* the suite passes with them gone.
- [x] **0.3 Repo layout and rename**
  - Python code moves under `server/`; an empty `web/` is created.
  - Rename to *Grounded QA for SEC Filings*.
  - *Done when:* `pytest` passes from the new layout.
- [x] **0.4 Living roadmap**
  - `tools/roadmap.py` renders Results from `evaluation/results/*.json` and Status from `@pytest.mark.roadmap("x.y")`.
  - `tests/test_roadmap.py` checks freshness.
  - *Done when:* this file regenerates, with an empty ablation table.
- [x] **0.5 Version pinning**
  - Check every dependency against current docs and pin exact versions.
  - *Done when:* a fresh venv installs and the tests pass.
- [x] **0.6 Groq provider**
  - `backend/providers/groq.py` with:
    - streaming
    - `reasoning_effort` and temperature per call
    - `retry-after` handling
    - reasoning tokens counted but not streamed as answer text
  - Becomes the default `PROVIDER`; Gemini stays as judge and fallback, Ollama for offline dev.
  - *Done when:* mocked provider tests pass and `python -m utils.check` reports Groq ready.
- [x] **0.7 Memory spike for Render**
  - A bare FastAPI process loading MiniLM + a reranker as ONNX; measure RSS.
  - Decide between Render free and Starter, and whether NLI runs locally or as an LLM call.
  - *Done when:* the measurement and the decision are recorded here.

## Phase 1 — Data and baseline (doc step 1)
*Goal: real filings (10-K and 10-Q), a golden set, and "before" numbers for everything.*

- [x] **1.1 EDGAR fetcher**
  - `ingestion/edgar.py`: ticker → CIK → submissions JSON → the primary document of each 10-K and 10-Q.
  - Every filing carries `form` (10-K, 10-Q) and `fiscal_period` (FY, Q1, Q2, Q3) next to `fiscal_year`.
    - Year and period come from the filing's own cover data (`dei:DocumentFiscalYearFocus`, `dei:DocumentFiscalPeriodFocus`), not from dates: many fiscal years do not end in December.
    - Amendments (10-K/A, 10-Q/A) are listed but not fetched for now.
  - A `latest` mode lists the filings newer than what is cached, by accession number. The refresh job (2.7) builds on it.
  - Rules:
    - User-Agent from `SEC_USER_AGENT`.
    - At most 10 requests per second, with backoff.
    - Cache to `data/filings/{ticker}/{fy}/{period}/`.
  - *Done when:* recorded-response tests pass for both forms and for `latest`, and real filings download.
- [x] **1.2 Pinned snapshot**
  - `evaluation/snapshot.json`: 5–8 companies × 3 fiscal years, by accession number and SHA-256.
  - Companies: AAPL, MSFT, JPM, JNJ, WMT, CVX. Six industries, and fiscal years ending in September, June, December, January and on 52/53-week calendars.
  - Each fiscal year holds its 10-K and its three 10-Qs: 12 filings per company. There is no Q4 10-Q; the fourth quarter lives in the 10-K.
  - Only complete fiscal years are pinned: a year counts once its 10-K is filed, so a year still in progress is left out rather than half-included.
  - The snapshot has a `version`, recorded in every evaluation result. A rebuild bumps it only if the pinned filings changed.
  - `python -m evaluation.snapshot --build` picks the filings; `python -m evaluation.snapshot --fresh` re-downloads them into an empty folder and checks every hash.
  - *Done when:* `--fresh` reproduces identical files (hash check).
- [ ] **1.3 Baseline ingestion**
  - A minimal `.htm` loader, run through the unchanged chunker, embedder and FAISS pipeline.
  - `form`, `fiscal_year` and `fiscal_period` ride along as chunk metadata.
  - *Done when:* the snapshot is indexed.
- [ ] **1.4 Golden set**
  - 60–100 hand-verified questions with a Pydantic schema, covering table lookups, text lookups, follow-ups, multi-hop/cross-period, definitions, summaries and unanswerable questions.
  - Both forms are covered: at least 20 cases are answered from 10-Qs.
    - Cross-period means year over year and quarter over quarter.
    - Some Q4 questions must be derived. For flow figures (revenue, income, cash flow), Q4 is the 10-K's annual figure minus the Q3 10-Q's nine-month figure. Balance-sheet figures are a point in time, so Q4 is read straight from the 10-K.
  - Relevance is labelled by **text span** (accession + char offsets), so labels survive re-chunking and the database move.
  - *Done when:* every case validates, with at least 8 of each type and at least 20 on 10-Qs.
- [ ] **1.5 Exact metrics, from scratch**
  - NumPy recall@k, MRR, nDCG@10, bootstrap CI, Cohen's kappa.
  - *Done when:* unit tests against hand-worked examples pass.
- [ ] **1.6 RAGAS**
  - Faithfulness, answer relevancy, context precision and recall, answer correctness.
  - The judge is Gemini; results are cached.
  - *Done when:* it runs on 5 cases and the cache hits on rerun.
- [ ] **1.7 Cost and latency accounting**
  - Tokens, $ (from a price table), and p50/p95 latency recorded in `QueryTrace`.
  - *Done when:* every eval result carries them.
- [ ] **1.8 Baseline run**
  - The current pipeline over the snapshot.
  - *Done when:* the Baseline row is filled, with confidence intervals.
- [ ] **1.9 Regression harness**
  - `tests/test_regression.py` fails below `evaluation/thresholds.json`.
  - *Done when:* it passes at baseline thresholds.

## Phase 2 — Database and API
*Goal: the pipeline served by FastAPI from Neon, with numbers identical to the FAISS baseline.*

- [ ] **2.1 Neon schema**
  - Tables:
    - `filings` (accession, cik, form, fiscal_year, fiscal_period, period end, filed date).
    - `chunks` (`vector(384)` + company, cik, form, fiscal_year, fiscal_period, section, kind, span).
    - `tables` (JSONB) and `eval_runs`.
  - `form` is free text rather than an enum, so a new form type (8-K, later) needs no migration.
  - psycopg 3 + pgvector, with a migration runner.
  - *Done when:* migrations apply to a fresh Neon branch.
- [ ] **2.2 pgvector store**
  - Same `search` interface as `VectorStore`, plus company/form/year/period/section filters. Exact scan first.
  - *Done when:* store tests pass against a test database.
- [ ] **2.3 BM25 from Postgres**
  - Chunks loaded into `vectorstore/keyword.BM25` at startup; `pg_search` checked as an alternative.
  - With 10-Qs the index is roughly four times the filings of a 10-K-only design, so BM25's memory is measured at full size against the 0.7 headroom (~75 MB). If it does not fit, the keyword search moves into Postgres.
  - *Done when:* BM25 scores match the FAISS-era ones and its memory is recorded.
- [ ] **2.4 Ingest CLI**
  - `python -m ingestion.ingest --snapshot`: fetch → parse → chunk → embed → upsert, deduplicated by hash.
  - `--ticker X --latest` ingests one company's newest filings the same way.
  - *Done when:* a rerun inserts nothing.
- [ ] **2.5 FastAPI service**
  - Endpoints: `POST /ask` (NDJSON stream), `GET /filings`, `GET /health`.
  - CORS, per-IP rate limit, request size cap.
  - *Done when:* API tests pass and `/ask` streams locally.
- [ ] **2.6 Parity check**
  - pgvector + BM25 vs FAISS on the golden set; FAISS is retired afterwards.
  - *Done when:* the parity row equals the baseline.
- [ ] **2.7 Filing refresh**
  - `python -m ingestion.refresh`: for each tracked company, the 1.1 `latest` mode finds 10-K and 10-Q accession numbers not yet in `filings`, and 2.4 ingests them.
  - It writes to the live index only. The pinned snapshot and the evaluation are never touched.
  - Each run logs what it found and added; a failure on one company does not stop the others.
  - *Done when:* a recorded-response test with one new filing inserts exactly that filing, and a second run inserts nothing.

## Phase 3 — Structured outputs (doc step 2)
*Goal: every model reply is validated data.*

- [ ] **3.1 Schemas**
  - `IntentResult`, `RewriteResult`, `Answer` (claims with citation ids, `declined`, `decline_reason`). These double as API models.
- [ ] **3.2 Structured completion**
  - `Provider.complete_structured(schema)` using each provider's native JSON-schema mode (Groq `response_format`, Gemini `response_schema`, Ollama `format`).
- [ ] **3.3 Validate and retry**
  - Feed the validation error back to the model; count invalid outputs. Replaces `parse_reply`.
  - *Done when:* invalid-JSON rate ≈0 with no quality drop.

## Phase 4 — Parsing and hybrid retrieval (doc step 3)
*Goal: retrieval that understands 10-K structure.*

- [ ] **4.1 Layout-aware 10-K and 10-Q parsing**
  - BeautifulSoup + lxml.
  - 10-K: split by Item (1, 1A, 7, 7A, 8).
  - 10-Q: split by Part and Item (Part I Items 1–4, Part II Items 1–6, including 1A). The two forms number their Items differently, so sections are stored as names (for example "MD&A"), not bare Item numbers.
  - *Done when:* the section of every golden span is detected correctly, on both forms.
- [ ] **4.2 Table extraction**
  - Tables become `kind=table` chunks and structured JSONB.
  - *Done when:* the golden table figures are found in the extracted tables.
- [ ] **4.3 Metadata filters**
  - Company, year and quarter extracted from the question ("Q3 2025", "last quarter"); "latest" resolves to the newest filing. The UI filter overrides it.
  - *Done when:* the filter ablation row is filled.
- [ ] **4.4 Chunking comparison**
  - Recursive vs section-aware vs parent-child.
  - *Done when:* three rows exist and a winner is chosen.
- [ ] **4.5 RRF fusion**
  - Replaces the weighted blend (both reported); drop `FULL_CONTEXT_WORDS`.
  - *Done when:* the hybrid row is filled.
- [ ] **4.6 Embedding comparison**
  - MiniLM vs `bge-small-en-v1.5`.
  - *Done when:* the row is filled and the winner is re-ingested.

## Phase 5 — Reranking, expansion, refusal (doc step 4)
*Goal: every retrieval failure can be explained.*

- [ ] **5.1 Cross-encoder reranker**
  - Top-20 → top-5; ONNX in production.
  - *Done when:* the row shows recall vs added latency.
- [ ] **5.2 Multi-query**
  - Paraphrases fused with RRF.
  - *Done when:* it has its own row.
- [ ] **5.3 HyDE**
  - Embed a hypothetical answer passage.
  - *Done when:* it has its own row.
- [ ] **5.4 Refusal path**
  - "The filing does not say" replaces "Beyond your documents".
  - *Done when:* refusal accuracy is reported.
- [ ] **5.5 Failure taxonomy**
  - Retrieval / generation / parsing / wrong refusal, counted.
  - *Done when:* every failure is tagged.

## Phase 6 — Routing and cost (doc step 5)
*Goal: a measured cost vs quality trade-off.*

- [ ] **6.1 Decompose route**
  - Sub-questions with per-period filters, under a step limit.
  - `calculator.py` for growth, margins, and Q4 flow figures (annual minus nine months).
  - *Done when:* the multi-hop questions pass.
- [ ] **6.2 Model choice**
  - 20b vs 120b per route, each with its own effort.
  - Ablations: all-120b vs routed, effort levels, temperature 1.0 vs 0.3.
  - *Done when:* the rows show $/query, p95 and quality.
- [ ] **6.3 Router accuracy**
  - Confusion matrix, plus the cost of each wrong route.
  - *Done when:* both are reported.

## Phase 7 — Verification (doc step 6)
*Goal: the "Grounded" in the name is earned.*

- [ ] **7.1 Claim decomposition**
  - Split each answer into atomic claims, each with its citations.
- [ ] **7.2 Claim checking**
  - In production, gpt-oss-20b on Groq checks each claim against its cited chunks (decided in 0.7: local NLI does not fit). The fp32 NLI cross-encoder still runs offline in the evaluation, as a second opinion to compare against.
- [ ] **7.3 Numeric verifier**
  - Units, %, negatives in parentheses and fiscal periods normalised, then matched against the extracted tables.
  - A 10-Q table shows three-month and nine-month (year-to-date) columns side by side; a figure from the wrong column counts as unsupported.
- [ ] **7.4 Verdict**
  - A `verdict` event: supported / partly unsupported / declined.
  - *Done when:* the unsupported-claim rate is reported before and after.
- [ ] **7.5 Judge calibration**
  - ~50 hand labels vs RAGAS faithfulness.
  - *Done when:* kappa is reported with a CI.

## Phase 8 — Safety and tracing (doc steps 7, 8)
*Goal: attacks measured, and any bad answer debuggable.*

- [ ] **8.1 Adversarial set**
  - 20+ cases: direct injection, PII requests, instructions planted inside filings.
- [ ] **8.2 Guards**
  - Input guard, spotlighting of retrieved text, output guard.
  - *Done when:* attack success rate is reported before and after.
- [ ] **8.3 MLflow tracing**
  - A span per stage; ablation runs logged as MLflow runs.
  - *Done when:* a bad answer can be traced to its failing stage.

## Phase 9 — React frontend and deployment
*Goal: a live, public demo.*

- [ ] **9.1 React app**
  - Vite + React + TypeScript; an NDJSON stream reader; a company/year/quarter filter.
- [ ] **9.2 Answer view**
  - Expandable citations (passage or table) and the verdict badge.
- [ ] **9.3 Trace panel and sample questions**
  - Intent, route, rewrites, chunks, tokens, $, timings.
  - One sample-question button per question type.
- [ ] **9.4 Render deployment**
  - `render.yaml` with a static site and a FastAPI web service.
  - Env vars: Neon `DATABASE_URL`, `GROQ_API_KEY`, `SEC_USER_AGENT`, allowed origin.
  - The 2.7 refresh runs on a schedule: a Render cron job, or a GitHub Actions scheduled workflow if the free tier rules that out (checked at this step). Daily is enough: filings arrive a few times a year per company.
- [ ] **9.5 Public hardening**
  - Per-IP and daily LLM caps; an optional bring-your-own-key header that is never logged.
  - *Done when:* the live URL answers all 7 doc question types within its memory limit.
- [ ] **9.6 Write-up**
  - README with generated numbers, 5–10 failure cases, and resume bullets limited to what is built.

## Later (not scheduled)
- **8-K event reports.** Left out of Phases 1–9 on purpose (decided 2026-11-25):
  - Each 8-K reports a different kind of event (earnings, an executive change, a contract), and the substance is usually in an exhibit such as the EX-99.1 press release, which needs its own fetch and parse path.
  - Companies file many small 8-Ks, which adds retrieval noise and makes refusal and claim checking harder.
  - Taken up once verification (Phase 7) works. `form` is already free text (2.1), so the schema is ready for it.

---

## Findings
Notes on what each phase taught, including what did not help. Added as phases land.

### 0.7 — Memory on Render (2026-11-24)
Peak memory of a fresh process that imports the service (FastAPI + onnxruntime), loads the models and runs one request (embed a question, rerank 20 passages, check 5 claims). Raw data: `server/evaluation/results/spike_memory.json` (`python -m tools.spike_memory`).

| Configuration | Peak |
|---|---|
| Service only, no models | 77 MB |
| Embedder (fp32) | 228 MB |
| **Embedder (fp32) + reranker (int8)** | **435 MB** |
| Embedder + reranker (both int8) | 368 MB |
| Embedder + reranker (both fp32) | 513 MB |
| PyTorch path (sentence-transformers embedder + reranker) | 799 MB |
| + NLI verifier (int8) | 804 MB |
| + NLI verifier (fp32) | 1528 MB |

What it showed:
- **ONNX export is exact.** ONNX vs PyTorch differ by at most 1.6e-7 (embedder), 4.8e-6 (reranker), 2.4e-6 (NLI).
- **Without PyTorch the same two models take 46% less memory** (435 vs 799 MB).
- **int8 is safe for the reranker, not for the embedder.** The int8 reranker kept every top-1 choice in the check (4 queries; the real test is 5.1 on the golden set). The int8 embedder moved vectors (cosine down to 0.96). Filings are embedded offline at fp32, so the query embedder must stay fp32 or the two live in different spaces, and parity (2.6) could not hold.
- **int8 destroys the NLI model:** 0/8 labels agree with fp32. fp32 NLI alone (739 MB on disk) is three times the free instance.

**Decision:**
- Target **Render free (512 MB)** with an fp32 embedder and an int8 reranker. That leaves ~75 MB of headroom on a Windows measurement, so check it against Render's own metrics at 9.4.
- **Claim checking runs on gpt-oss-20b** (Groq) in production. Local fp32 NLI is kept for offline evaluation only.
- **If Render shows less headroom**, there are two fallbacks:
  - Embed both filings and queries with the int8 embedder (368 MB). This is consistent, but the 4.6 ablation must show it costs no recall.
  - Move to a 2 GB instance.

### 1.1–1.2 — Filings from EDGAR (2026-11-25)
- **Every one of the 72 pinned filings declares its fiscal year and period on its cover**, so none had to be placed by dates. The covers agree with how each company names its years. Walmart's year ending 31 January 2026 is fiscal 2026. Johnson & Johnson's 52/53-week years end on 31 December 2023, 29 December 2024 and 28 December 2025.
- **A ticker can move to a new company record.** In 2026 ExxonMobil reorganised under a new parent, ExxonMobil Holdings Corp. The XOM ticker now maps to the new CIK, whose record holds only the two newest 10-Qs; the years before sit under the old Exxon Mobil Corp CIK. The snapshot uses Chevron (CVX) instead. Following a predecessor CIK is left for the refresh job (2.7) to decide, if a tracked company does this.
- **Filings are large and uneven.** The 72 pinned filings come to 321 MB, from 12.5 MB for Apple's twelve to 141 MB for JPMorgan's. JPMorgan's largest single filing is 13 MB of HTML. 1.3 indexes them, and 2.3 measures what that costs in memory.

## Failure cases
5–10 concrete wrong answers, each with its tag and the trace that explained it. Added from Phase 5 on.

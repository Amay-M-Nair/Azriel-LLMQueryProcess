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
| 1 Data and baseline | 9/9 | complete |
| 2 Database and API | 2/7 | 2.3 BM25 from Postgres |
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
| Configuration | Recall@5 | MRR | nDCG@10 | Faithfulness | Unsupported claims | Correct refusals | $/query | p95 latency | Run |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (current pipeline) | 0.169 [0.096–0.247] | 0.137 [0.079–0.204] | 0.148 [0.091–0.211] | 0.749 [0.679–0.814] |  |  | $0.0005 | 16.2s | 2026-11-26 · 05d823f · v1 |
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
| 2026-11-26 | 1 | pydantic | 2.14.4 | pinned at 1.4 for the golden-set schema; upgraded from 2.13.5, which fastapi had pulled in. `pip check` clean |
| 2026-11-26 | 1 | ragas / instructor / openai | 0.4.4 / 1.17.0 / 3.3.0 | ragas adds ~20 packages (langgraph, langchain-openai, datasets, ...) and imports langchain-community, which prints a sunset warning; chosen over a hand-written judge so the scores are RAGAS's own. Its google-genai route sends safety settings Gemini rejects (instructor#1658), so the judge uses Gemini's OpenAI-compatible endpoint. instructor pinned: pip otherwise picks 1.3, which predates openai 2 |

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
- [x] **1.3 Baseline ingestion**
  - A minimal `.htm` loader (`document_loader.html_text`), run through the unchanged chunker, embedder and FAISS pipeline.
    - It drops scripts, styles and hidden content, including inline XBRL's hidden facts.
    - Each block is a line, and each table row is one line with its cells joined by ` | `.
    - Nothing else is attempted: no Items, no table structure. That is 4.1 and 4.2.
  - `ticker`, `accession`, `form`, `fiscal_year` and `fiscal_period` ride along as chunk metadata. So do character offsets into the filing's text, which the 1.4 span labels are matched against.
  - `python -m ingestion.filings` indexes into `data/snapshot_index/v{version}/`. It saves after each company and skips filings already indexed, so an interrupted run resumes.
  - *Done when:* the snapshot is indexed.
- [x] **1.4 Golden set**
  - 60–101 hand-verified questions (the cap was 100; all 101 reviewed drafts were approved, so it was raised by one) with a Pydantic schema, covering table lookups, text lookups, follow-ups, multi-hop/cross-period, definitions, summaries and unanswerable questions.
  - Both forms are covered: at least 20 cases are answered from 10-Qs.
    - Cross-period means year over year and quarter over quarter.
    - Some Q4 questions must be derived. For flow figures (revenue, income, cash flow), Q4 is the 10-K's annual figure minus the Q3 10-Q's nine-month figure. Balance-sheet figures are a point in time, so Q4 is read straight from the 10-K.
  - Relevance is labelled by **text span** (accession + char offsets), so labels survive re-chunking and the database move.
    - Offsets point into `html_text` of the pinned filing, the text 1.3 indexes. Each span also keeps its quote, so it can be found again if 4.1 changes the text.
  - How the set is made (`evaluation/golden.py`):
    - Cases are drafted into `golden_candidates.jsonl`. `python -m evaluation.golden locate` turns each quoted passage into offsets, and refuses a quote that is not unique in its filing.
    - A person reviews each one in `evaluation/review_app.py`, which shows the highlighted passage inside its filing. An unanswerable case comes with a search box over the filings.
    - The review app is a local Streamlit tool for building the golden set, not part of the product: it is never deployed. The product UI is React + FastAPI (Phase 9). When the interim Streamlit page goes in Phase 9, Streamlit stays only for this tool, kept out of the deployed service.
    - Approved cases go to `golden.jsonl`, stamped with the reviewer and date. Rejected ones go to `golden_rejected.jsonl` with the reason.
    - **A reference answer says only what its cited passages say.** Nothing from the surrounding text, and no interpretation ("led by Oncology"). A claim the answer needs is cited, or it is dropped. `check` enforces this for figures: every number in an answer must appear in a cited quote or in the calculation, allowing for rounding and millions written as billions. A derived figure (a growth %, a Q4) needs its calculation. Prose claims are checked by the reviewer, and the review app shows any uncited figure as a warning.
  - *Done when:* every case validates, with at least 8 of each type and at least 20 on 10-Qs.
  - *Done 2026-11-26:* all 101 drafts reviewed by hand and approved into `golden.jsonl` (22 table, 20 text, 15 cross-period, 12 follow-up, 12 unanswerable, 10 definition, 10 summary; 27 on 10-Qs). 15 were corrected during the review (see *Findings, 1.3–1.4*).
- [x] **1.5 Exact metrics, from scratch**
  - NumPy recall@k, MRR, nDCG@10, bootstrap CI, Cohen's kappa (`evaluation/metrics.py`).
  - Retrieval is scored against **evidence**, not chunks:
    - A chunk *covers* a golden span when their overlap is at least half of the shorter of the two. That counts a chunk inside a long passage, and a chunk holding most of a table row, but not a neighbouring window that only grazes the span.
    - Each span counts once however many chunks cover it, so a smaller chunk size cannot score better by cutting one answer into more pieces, and nDCG cannot exceed 1.
  - Unanswerable questions have no retrieval score; declining is measured in 5.4.
  - Each metric is a mean over questions with a 95% percentile-bootstrap interval. Two configurations are compared with a paired bootstrap of their per-question difference.
  - *Done when:* unit tests against hand-worked examples pass.
- [x] **1.6 RAGAS**
  - Faithfulness, answer relevancy, context precision and recall, answer correctness (`evaluation/judge.py`).
  - The judge is Gemini; results are cached.
    - Each metric has its own fixed model (`JUDGE_MODELS` in `utils/config.py`), never a fall-through list: a metric graded by two models would mix two scales. Five models means five free-tier quotas. Changing one metric's model re-grades that metric only.
    - It grades the top 5 retrieved passages, which is what the answer model saw. Its token limit is 8,192, because Gemini's thinking counts against it.
    - Two caches: every grade (keyed by answer, metric and judge) in `data/judge_cache/scores.jsonl`, and every judge call in RAGAS's disk cache. A rerun costs nothing, and a metric cut off halfway resumes without repeating calls.
  - It runs on Gemini's free tier. When one model's daily quota runs out, its metric stops and the others carry on; the next run continues from the cache. A per-minute limit or a 503 "high demand" is waited out.
  - The 5 check cases are drafts from `golden_candidates.jsonl` (`--candidates`), since `golden.jsonl` does not exist until 1.4's review. This is a check that the judge works, not an evaluation; the golden-set evaluation is 1.8.
  - **Open decision:** how 1.8 is judged. The free tier is too slow for the full set (see *Findings, 1.6*). Options: paid tier (about $1–2 a run), or free tier with fewer calls per question plus RAGAS on a 25-question subset for ablations.
  - Answers are generated once by the pipeline and cached under a fingerprint of everything that produced them (`evaluation/answers.py`). At temperature 1.0 a regenerated answer would be a different answer.
  - *Done when:* it runs on 5 cases and the cache hits on rerun.
- [x] **1.7 Cost and latency accounting**
  - Tokens, $ (from a price table), and p50/p95 latency recorded in `QueryTrace`.
  - *Done when:* every eval result carries them.
- [x] **1.8 Baseline run**
  - The current pipeline over the snapshot.
  - *Done when:* the Baseline row is filled, with confidence intervals.
- [x] **1.9 Regression harness**
  - `tests/test_regression.py` fails below `evaluation/thresholds.json`.
  - *Done when:* it passes at baseline thresholds.

## Phase 2 — Database and API
*Goal: the pipeline served by FastAPI from Neon, with numbers identical to the FAISS baseline.*

- [x] **2.1 Neon schema**
  - Tables:
    - `filings` (accession, cik, form, fiscal_year, fiscal_period, period end, filed date).
    - `chunks` (`vector(384)` + company, cik, form, fiscal_year, fiscal_period, section, kind, span).
    - `tables` (JSONB) and `eval_runs`.
  - `form` is free text rather than an enum, so a new form type (8-K, later) needs no migration.
  - psycopg 3 + pgvector, with a migration runner.
  - *Done when:* migrations apply to a fresh Neon branch.
- [x] **2.2 pgvector store**
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
  - Periods are the company's **fiscal** periods (Apple's year ends in September, Microsoft's in June), as stored on each chunk from the cover data.
  - No filing covers Q4 on its own, so a Q4 question filters to two filings: that fiscal year's 10-K and its Q3 10-Q. 6.1 then subtracts (flow figures) or reads the 10-K directly (balance-sheet figures).
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
  - The deployed service installs only what it serves: no Streamlit (the golden-set review tool stays local) and no evaluation-only packages (RAGAS and its dependencies).
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

### 1.3–1.4 — The baseline loader and the golden set (2026-11-26)
- **The 72 filings make 31,119 chunks**, indexed in 15.4 minutes on CPU. Size varies by a factor of 13: an Apple 10-Q is about 10,000 words (about 90 chunks), a JPMorgan 10-Q about 130,000 (about 1,100 chunks).
- **The minimal loader reads tables unevenly.** Most filings' tables come out one row per line ("Total net sales | 478,834 | 416,161 | 391,035"). Microsoft's come out one cell per line, with the row label, the "$" signs and each year's figure on separate lines. A chunk can then hold figures whose label sits in the previous chunk. This is what 4.2 is for, and the golden set's table questions will show how much it costs.
- **The same figure often appears several times in one filing** (the income statement, the MD&A table, the segment note). A golden span has to say which occurrence it means, so quotes are lengthened until they are unique: often by taking in the row above.
- **The drafts' answers often said more than their highlights** (found in review, 2026-11-26). 15 of 101 drafts:
  - The highlight stopped a sentence early, so the answer's figure or reason sat in the next sentence (Azure's 41%, JPMorgan's expense drivers, J&J's 53-week years).
  - A total came from a table row that was not cited (Walmart's segment net sales, JPMorgan's NII and NIR, Chevron's employee split).
  - A derived figure had no calculation (Apple's Q1 2025 "up 4%").
  - A claim came from no passage at all: Chevron's "lower prices cut revenues", which the filing explains differently (realizations, DD&A, operating costs, currency); and "led by Oncology".
  - One was a misreading: CARVYKTI grew on share gains and capacity expansion, not market growth.
  - All were fixed by citing the passage or dropping the claim. One already-approved case went back to the review. The figure check above now catches the mechanical kind.

### 1.5 — A first look at retrieval (2026-11-26, preview, not the baseline)
Run on the 101 drafted candidates before review, so it does not count as a Results row; 1.8 is the baseline. It shows the metrics working on real data, and where the baseline will be weak.
- **Every one of the 128 golden spans is covered by some chunk in the index** at the 50% threshold. So a miss is always a ranking failure, never an unfindable span.
- **Retrieval finds the company but not the filing.** On the 89 answerable questions, today's hybrid search (top 10, question text only) scored:

  | Metric | Value | 95% CI |
  |---|---|---|
  | recall@10 | 0.199 | 0.122–0.282 |
  | MRR | 0.109 | 0.057–0.169 |
  | nDCG@10 | 0.120 | 0.068–0.178 |

  94% of the top-10 chunks came from the right company, but only 16% from a filing that holds the answer. Each company has twelve filings that read alike, and nothing in the search knows which year or quarter was asked about. The metadata filters of 4.3 address exactly this.
- **By type, recall@10:** definitions 0.40, cross-period 0.27, text 0.25, tables 0.14, follow-ups 0.08, summaries 0.07. Follow-ups are low because the question alone ("And how did that compare with the year before?") names no company. The pipeline's rewriter is meant to fix that, and 1.8 runs it.

### 1.6 — The judge on the free tier (2026-11-26, in progress)
- **The free tier allows 20 requests a day per model**, and RAGAS needs about 14 judge calls per question: faithfulness 2, answer relevancy 3, context precision 5 (one per passage), context recall 1, answer correctness 3.
- **Real capacity is lower than 20.** Retries after 503 "high demand" appear to count against the daily quota, so a model gives roughly 10–15 grades a day. A run of 24 calls took 41 minutes.
- **What that means:** with one model per metric, context precision is the bottleneck. A full golden-set run (1.8) would take over a month on the free tier.
- **Progress on the 5 check cases:** 13 of 25 grades done; context recall is done for all five. Results in `server/evaluation/results/ragas_check.json`.
- **The scores make sense so far.** The employees answer had no supporting passage retrieved (context recall 0.00) and scored 0.17 on correctness: the same retrieval weakness 1.5 found.
- **Left to do:** rerun after the quotas reset (about 08:00 UTC) until all 25 grades exist, then one rerun served entirely from the cache. The judge decision (paid or free + subset) is still open.

## Failure cases
5–10 concrete wrong answers, each with its tag and the trace that explained it. Added from Phase 5 on.

# NEO — notes beside the roadmap

**ROADMAP.md is the plan and is not edited.** This file holds what has come up since:
- decisions the roadmap left open,
- facts found along the way,
- proposals waiting for a yes.

**Rules:**
- Nothing here overrides ROADMAP.md on its own.
- Anything that differs from ROADMAP.md is listed under **Departures**, with who decided it and when.
- Proposals stay marked as proposals until agreed.

Last checked against ROADMAP.md: 2026-11-27.

---

## Departures from ROADMAP.md

### D1. The RAGAS judge runs on AWS Bedrock, not Gemini
- **Status:** decided by the owner on 2026-11-27 by setting up Bedrock ($120 of credits).
- **ROADMAP.md says:** "RAGAS (Gemini as judge)" (Stack); "RAGAS judge: Gemini (a different family, avoiding self-preference)" (Model settings); "The judge is Gemini" and "It runs on Gemini's free tier" (1.6).
- **What stays the same:** the roadmap's reason for its choice. The judge is still from a different family than the gpt-oss models that write the answers. The 1.6 metrics, caches and *done when* are unchanged.
- **Why:** 1.6 found that the Gemini free tier gives about 10–15 grades a day per model, so a golden-set run would take over a month. This also settles 1.6's open decision ("how 1.8 is judged"): a paid judge, paid from the credits.
- **The one-model-per-metric rule still holds:** each metric has one fixed model, never a fall-through list. On Bedrock it is the same model for all five metrics, so `JUDGE_MODELS` (a dict) became `JUDGE_MODEL` (one string) in `utils/config.py`.
- **The Gemini judge code was removed (owner, 2026-11-27):** the Gemini path, its five models and the daily-quota handling existed only for the free tier. The judge is Bedrock-only. A throttled call (429/503) is waited out, and any other failure is recorded and retried on the next run. `--candidates` went too: `golden.jsonl` is approved, so the judge always reads it. The Gemini runs stay in the history of `ragas_check.json` as a record; the 1.6 check only counts runs made by the current judge.

---

## Decisions within the roadmap

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
- **P2. Somewhere to see online metrics.** The roadmap records traces (8.3 MLflow tracing), but nothing aggregates live metrics into a view, feedback signal or alert. Could join Phase 8 or 9.

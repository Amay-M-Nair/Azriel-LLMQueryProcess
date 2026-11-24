"""Every tunable number in the project, in one place."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # server/
REPO_ROOT = PROJECT_ROOT.parent                          # holds .env and data/

load_dotenv(REPO_ROOT / ".env")  # API keys land in the environment

DATA_DIR = REPO_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
UPLOAD_DIR = DATA_DIR / "uploads"

# --- Retrieval (runs locally, costs nothing) -------------------------------
# A small, fast embedding model. 90 MB, downloads once on first use.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # output size of the model above

# MUST stay under what the embedding model can read in one go - all-MiniLM
# truncates at 256 tokens, about 175 words. A larger chunk is not an error and
# not a warning: the overflow is simply never seen by the search, so those
# words can never be found. build_index() checks this and tells you.
CHUNK_WORDS = 160
CHUNK_OVERLAP_WORDS = 40   # overlap so a sentence split across chunks isn't lost
TOP_K = 5                  # how many chunks to hand the model

# Cosine similarity below this counts as "not actually related". Without a
# floor the store always returns its k closest chunks even when nothing in
# the index is relevant, which invites the model to answer from noise.
MIN_SIMILARITY = 0.12

# ... but a chunk can be a decisive keyword match and still score badly on
# cosine - "who is Ingrid Halvorsen?" against a page that names her once.
# A chunk whose keyword score reaches this fraction of the best keyword score
# in the candidate set survives MIN_SIMILARITY on that evidence alone.
# Lower it to rescue more, raise it towards 1.0 to rescue only exact hits.
KEYWORD_RESCUE = 0.6

# If the whole index is smaller than this, skip retrieval and send everything.
# A short document does not need compressing, and retrieving over one can only
# lose information. Raise it if your model has a large context window.
FULL_CONTEXT_WORDS = 4000

# How much keyword (BM25) matching counts against vector similarity when
# ranking. 0.0 = pure meaning, 1.0 = pure keyword. Keywords matter for lookups
# like a name or an error code; meaning matters for everything else.
KEYWORD_WEIGHT = 0.35

# Blending BM25 into the ranking needs a candidate set that already contains
# the chunk which wins on keywords. Up to this many chunks the store scores
# every one of them, so the blend is exact. Past it, it takes a pool of the
# best vector matches instead - fast, but a chunk that only a keyword would
# have found can fall outside the pool and be missed.
HYBRID_EXACT_LIMIT = 20000
HYBRID_POOL_MULTIPLIER = 8   # pool = TOP_K * this ...
HYBRID_POOL_MIN = 50         # ... but never smaller than this

# --- Which model writes the answer -----------------------------------------
# "groq"      gpt-oss on Groq, needs GROQ_API_KEY -> https://console.groq.com/keys
# "gemini"    free tier, needs GOOGLE_API_KEY  -> https://aistudio.google.com/apikey
#             (also the evaluation judge: a different model family from gpt-oss)
# "ollama"    free and offline, needs Ollama running locally
PROVIDER = "groq"

TEMPERATURE = 0.2  # gemini / ollama. Groq has its own, below.

# Groq serves OpenAI's open-weight gpt-oss models. Two sizes, one API: the
# strong model writes answers, the cheap one does the small structured jobs.
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_CHEAP_MODEL = "openai/gpt-oss-20b"
# Tried in order when the model above is rate limited or overloaded.
GROQ_FALLBACK_MODELS = [GROQ_CHEAP_MODEL]

# gpt-oss is a reasoning model. How hard it thinks is set by reasoning_effort
# ("low" | "medium" | "high"); OpenAI's guidance for gpt-oss is to sample at
# temperature 1.0 rather than to turn temperature down for accuracy. Both
# are starting points: ROADMAP 6.2 measures effort levels and 1.0 vs 0.3.
GROQ_TEMPERATURE = 1.0
GROQ_REASONING_EFFORT = "medium"

# Model and effort per pipeline stage (ROADMAP "Model settings per stage").
# A stage not listed uses GROQ_MODEL at GROQ_REASONING_EFFORT.
GROQ_STAGES: dict[str, tuple[str, str]] = {
    "classify":  (GROQ_CHEAP_MODEL, "low"),
    "rewrite":   (GROQ_CHEAP_MODEL, "low"),
    "claims":    (GROQ_CHEAP_MODEL, "low"),
    "answer":    (GROQ_MODEL, "medium"),
    "decompose": (GROQ_MODEL, "high"),
}

# Reasoning tokens count toward this, so it has to leave room for thinking
# as well as for a Thorough answer.
GROQ_MAX_TOKENS = 8192
GROQ_RETRY_ROUNDS = 2        # passes over the model list before giving up
GROQ_RETRY_BACKOFF = 2.0     # seconds, used only if the API suggests nothing
GROQ_MAX_WAIT = 30.0         # never block longer than this between tries

# Verified working on 2026-09-15. Free-tier models get retired and go busy,
# so if the first is unavailable the provider falls through this list.
GEMINI_MODEL = "gemini-3.6-flash"
# The free tier allows only ~20 requests PER DAY PER MODEL, so each extra
# model in this list adds roughly another 20 questions a day. The provider
# falls through them in order when one is exhausted or overloaded.
GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]
# Room for a Thorough answer. The models themselves allow 65536, so this is
# a guard against a runaway reply rather than a real ceiling - raise it if an
# answer ever stops mid-sentence.
GEMINI_MAX_TOKENS = 8192
GEMINI_RETRY_ROUNDS = 3      # passes over the model list before giving up
GEMINI_RETRY_BACKOFF = 2.0   # seconds, used only if the API suggests nothing
GEMINI_MAX_WAIT = 45.0       # never block the UI longer than this between tries

OLLAMA_MODEL = "llama3.2:3b"     # fits comfortably in 4 GB of VRAM
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_NUM_PREDICT = 2048
OLLAMA_TIMEOUT = 300             # seconds; local models can be slow

USE_REFUSAL_FALLBACKS = True     # retry on another model if a request is declined

# --- Collections -----------------------------------------------------------
# The namespace documents are kept under. Each collection has its own index,
# and no collection can retrieve another's documents.
DEFAULT_COLLECTION = "default"


# --- SEC EDGAR (ingestion/edgar.py) ----------------------------------------
# Filings are cached here as data/filings/{ticker}/{fiscal_year}/{period}/.
FILINGS_DIR = DATA_DIR / "filings"

# The SEC's fair-access policy allows at most 10 requests per second per
# client, and answers a client that goes faster with a 403 lockout rather
# than a slower reply. So the limit is kept here, not discovered there.
EDGAR_MAX_REQUESTS_PER_SECOND = 10
EDGAR_TIMEOUT = 30.0          # seconds per request; a 10-K can be 10 MB
EDGAR_RETRIES = 5             # tries per request on 429, 5xx or a dropped connection
EDGAR_RETRY_BACKOFF = 1.0     # seconds, doubled per try, when the SEC suggests nothing
EDGAR_MAX_WAIT = 60.0         # never wait longer than this between tries

# The ticker -> CIK map changes when companies list, delist or rename, which
# is rare. Re-downloaded when the cached copy is older than this.
EDGAR_TICKER_MAP_MAX_AGE_DAYS = 7


# --- Logging ---------------------------------------------------------------
# Where the detail goes when the user gets a sentence. Set LOG_FILE to None to
# keep everything on stderr, which is what you want if the terminal is right
# there in front of you.
LOG_LEVEL = "INFO"
LOG_FILE = DATA_DIR / "groundedqa.log"


# --- Reading documents -----------------------------------------------------
# A scanned PDF has no text layer, so its pages are rasterised and read back
# with OCR. It costs a few seconds per page, so it only runs on pages that
# yielded nothing on their own - never on a document that is already text.
OCR_ENABLED = True
OCR_DPI = 200          # lower is faster and loses small print; 200 reads 9pt fine


# --- Query processing ------------------------------------------------------
# How many past exchanges the rewriter and the direct-answer path can see.
# Enough to resolve "it" without burying the current question.
HISTORY_TURNS = 4

# A follow-up this short is usually a fragment leaning on the previous turn,
# so it is worth resolving before searching.
SHORT_FOLLOWUP_WORDS = 4

# The prompts themselves live in utils/prompts.py.

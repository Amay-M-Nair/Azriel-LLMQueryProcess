-- ROADMAP 2.1: the first schema. Filings, their chunks with vectors, extracted
-- tables, and evaluation runs.
--
-- `form` is free text, not an enum, so a new form type (8-K, later) needs no
-- migration. The checks below only pin down what is known about the forms we
-- have: a 10-K covers a fiscal year (FY), a 10-Q one of Q1-Q3 (there is no Q4
-- 10-Q; the fourth quarter lives in the 10-K).

-- Always into public: the extension is database-wide, and must not land in
-- whatever schema happens to be first on the search path (tests use their own).
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE TABLE filings (
    accession        text        PRIMARY KEY CHECK (accession ~ '^\d{10}-\d{2}-\d{6}$'),
    cik              text        NOT NULL    CHECK (cik ~ '^\d{10}$'),
    ticker           text        NOT NULL,
    company          text        NOT NULL,
    form             text        NOT NULL,
    -- From the filing's own cover data (dei:DocumentFiscalYearFocus and
    -- ...PeriodFocus), not from dates. Null only for a form with no fiscal period.
    fiscal_year      integer,
    fiscal_period    text,
    fiscal_year_end  text,                    -- "0927", from the submissions record
    period_end       date,                    -- the period the filing reports on
    filed            date        NOT NULL,
    primary_document text        NOT NULL,    -- "aapl-20250927.htm"
    sha256           text        NOT NULL    CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    ingested_at      timestamptz NOT NULL    DEFAULT now(),

    CHECK (form NOT IN ('10-K', '10-Q') OR (fiscal_year IS NOT NULL AND fiscal_period IS NOT NULL)),
    CHECK (form <> '10-K' OR fiscal_period = 'FY'),
    CHECK (form <> '10-Q' OR fiscal_period IN ('Q1', 'Q2', 'Q3')),
    -- The target of chunks' second foreign key, below.
    UNIQUE (accession, ticker, cik, form, fiscal_year, fiscal_period)
);

CREATE INDEX filings_company_period ON filings (ticker, fiscal_year, fiscal_period);


CREATE TABLE chunks (
    id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunk_id       text        NOT NULL UNIQUE,
    accession      text        NOT NULL REFERENCES filings ON DELETE CASCADE,
    -- Copied from the filing so a filtered search needs no join. The second
    -- foreign key keeps the copies equal to the filing's own values.
    ticker         text        NOT NULL,
    cik            text        NOT NULL,
    form           text        NOT NULL,
    fiscal_year    integer,
    fiscal_period  text,
    section        text,                      -- "MD&A", "Risk Factors"; filled from 4.1
    kind           text        NOT NULL DEFAULT 'text',   -- 'text' | 'table' (4.2); free text
    -- The span: character offsets into the filing's text, which the golden
    -- set's labels are matched against. -1 / -1 when it could not be located.
    page           integer     NOT NULL DEFAULT 1,
    start_char     integer     NOT NULL,
    end_char       integer     NOT NULL,
    start_word     integer     NOT NULL,
    end_word       integer     NOT NULL,
    text           text        NOT NULL,
    embedding      vector(384) NOT NULL,      -- all-MiniLM-L6-v2, unit length

    CHECK ((start_char = -1 AND end_char = -1) OR (0 <= start_char AND start_char < end_char)),
    CHECK (0 <= start_word AND start_word <= end_word),
    FOREIGN KEY (accession, ticker, cik, form, fiscal_year, fiscal_period)
        REFERENCES filings (accession, ticker, cik, form, fiscal_year, fiscal_period)
        ON DELETE CASCADE ON UPDATE CASCADE
);

-- No vector index: search is an exact scan (2.2), which is what makes parity
-- with FAISS's exact IndexFlatIP possible (2.6).
CREATE INDEX chunks_accession      ON chunks (accession);
CREATE INDEX chunks_company_period ON chunks (ticker, fiscal_year, fiscal_period);
CREATE INDEX chunks_form           ON chunks (form);


-- Tables extracted from filings as structured data (4.2).
CREATE TABLE tables (
    id          bigint  GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    accession   text    NOT NULL REFERENCES filings ON DELETE CASCADE,
    section     text,
    caption     text,
    start_char  integer,
    end_char    integer,
    data        jsonb   NOT NULL
);

CREATE INDEX tables_accession ON tables (accession);


-- One row per evaluation run: the headline numbers as columns to sort and
-- filter on, the whole results file alongside.
CREATE TABLE eval_runs (
    id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    configuration  text        NOT NULL,
    snapshot       text        NOT NULL,      -- "v1": the pinned snapshot it ran on
    git_commit     text        NOT NULL,
    run_at         timestamptz NOT NULL,
    metrics        jsonb       NOT NULL,
    result         jsonb       NOT NULL,
    recorded_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (configuration, run_at)
);

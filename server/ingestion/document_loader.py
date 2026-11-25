"""Turn files on disk into pages of plain text.

A "page" is (page_number, text). PDFs have real pages; text files are treated
as a single page 1. Keeping the page number lets us cite it later.

Two of the formats here do not give up their text easily:

A scanned PDF has no text layer at all - it is pictures of words - so it is
rasterised and read back with OCR, page by page. That is slow enough that it
only happens to pages that yielded nothing on their own.

A .docx has no pages. Pagination is decided by whatever renders the file, so
a page number would be invented. Only breaks the author actually inserted are
honoured; a document without them is one page, which is the truth.
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from utils import config, logs

log = logs.get(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".htm", ".html"}

# Below this many characters a PDF page has effectively no text on it, whether
# because it is a scan or because it is a full-page figure.
BLANK_PAGE_CHARS = 40


def load_pages(
    path: Path,
    allow_ocr: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> list[tuple[int, str]]:
    """Read one file into a list of (page_number, text)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {suffix!r}. Supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        return _load_pdf(path, allow_ocr, on_progress)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in (".htm", ".html"):
        text = html_text(path.read_bytes().decode("utf-8", errors="replace"))
        return [(1, text)] if text else []

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [(1, text)] if text else []


# --- PDF -------------------------------------------------------------------

def _load_pdf(path, allow_ocr, on_progress) -> list[tuple[int, str]]:
    """Extract what text there is, then OCR only the pages that had none.

    Doing it per page rather than per document matters for the common real
    case: a report that is mostly typed with a scanned appendix, or a letter
    with a photographed signature page.
    """
    reader = PdfReader(str(path))
    extracted: list[tuple[int, str]] = []
    empty: list[int] = []

    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) >= BLANK_PAGE_CHARS:
            extracted.append((number, text))
        else:
            empty.append(number)

    if empty and allow_ocr and config.OCR_ENABLED and ocr_available():
        extracted.extend(_ocr_pdf_pages(path, empty, on_progress))

    extracted.sort(key=lambda item: item[0])
    return [(number, text) for number, text in extracted if text.strip()]


def ocr_available() -> bool:
    """Whether the optional OCR dependencies are installed.

    They are optional on purpose: they add about 47 MB and are useless to
    anyone whose documents are already text.
    """
    try:
        import pymupdf  # noqa: F401
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


_reader = None


def _get_reader():
    """Built once; constructing it loads the detection and recognition models."""
    global _reader
    if _reader is None:
        from rapidocr_onnxruntime import RapidOCR

        _reader = RapidOCR()
    return _reader


def _ocr_pdf_pages(path, numbers, on_progress) -> list[tuple[int, str]]:
    """Rasterise the given pages and read them back. Slow: seconds per page."""
    import pymupdf

    reader = _get_reader()
    out = []

    with pymupdf.open(str(path)) as document:
        for position, number in enumerate(numbers, start=1):
            if on_progress:
                on_progress(
                    f"Reading page {number} with OCR ({position} of {len(numbers)})"
                )
            try:
                pixmap = document[number - 1].get_pixmap(dpi=config.OCR_DPI)
                result, _ = reader(pixmap.tobytes("png"))
            except Exception:
                # One unreadable page should not lose the rest of the document,
                # but it should not vanish either - a document that comes back
                # short is otherwise inexplicable.
                log.warning("OCR failed on %s page %d", path.name, number,
                            exc_info=True)
                continue

            lines = [line[1] for line in result] if result else []
            if lines:
                out.append((number, "\n".join(lines)))

    return out


# --- DOCX ------------------------------------------------------------------

def _load_docx(path) -> list[tuple[int, str]]:
    """Paragraphs and tables, split only on page breaks the author inserted.

    Tables are included because policy documents put their most citable facts
    in them - rates, thresholds, deadlines - and a reader that skipped tables
    would miss exactly the things people ask about.
    """
    import docx

    document = docx.Document(str(path))
    pages: list[list[str]] = [[]]

    for block in _docx_blocks(document):
        text, breaks_page = block
        if breaks_page and pages[-1]:
            pages.append([])
        if text:
            pages[-1].append(text)

    return [
        (number, "\n".join(lines).strip())
        for number, lines in enumerate(pages, start=1)
        if "".join(lines).strip()
    ]


def _docx_blocks(document):
    """Yield (text, starts_a_new_page) for each paragraph and table, in order.

    python-docx exposes paragraphs and tables as separate collections, which
    loses the order they appear in. The body's own XML children keep it, so a
    table between two paragraphs stays between them.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]

        if tag == "p":
            paragraph = Paragraph(child, document)
            # w:br with type="page" is an explicit page break.
            breaks_page = any(
                br.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type")
                == "page"
                for br in child.iter(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br"
                )
            )
            yield paragraph.text.strip(), breaks_page

        elif tag == "tbl":
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            yield "\n".join(rows), False


# --- HTML (SEC filings) ----------------------------------------------------
#
# Deliberately minimal: the baseline loader for ROADMAP 1.3, so that the
# layout-aware parser of 4.1 has something honest to be measured against.
# It knows block elements from inline ones, keeps table rows on one line with
# cells joined like the .docx tables above, and drops what a reader never
# sees: scripts, styles, and anything inside display:none, which in an
# inline-XBRL filing is the hidden block of machine-readable facts.

_BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
          "table", "ul", "ol", "section", "article", "header", "footer", "hr", "blockquote", "pre"}
_CELL = {"td", "th"}
_VOID = {"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr", "source"}
_NEVER_SHOWN = {"script", "style", "head", "title", "noscript"}
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.IGNORECASE)


class _TextOfHtml(HTMLParser):
    """Collects visible text. Rows end lines; cells are joined with " | "."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stack: list[tuple[str, bool, bool]] = []   # (tag, hides its content, is a counted cell)
        self.hidden = 0
        self.in_cell = 0
        self.cell_open = False                           # a cell has been written on this row

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        hides = tag in _NEVER_SHOWN or tag == "ix:header" or bool(
            _HIDDEN_STYLE.search(dict(attrs).get("style") or "")
        )
        cell = tag in _CELL and not self.hidden and not hides
        if tag not in _VOID:
            self.stack.append((tag, hides, cell))
            self.hidden += hides
        if self.hidden:
            return
        if tag in ("tr", "table"):
            self.parts.append("\n")
            self.cell_open = False
        elif cell:
            if self.cell_open:
                self.parts.append(" | ")
            self.cell_open = True
            self.in_cell += 1
        elif tag in _BLOCK:
            # EDGAR wraps cell contents in <div>/<p>; inside a cell that is not a new line.
            self.parts.append(" " if self.in_cell else "\n")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _VOID:      # <div/> opens and closes at once
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _VOID or not any(t == tag for t, _, _ in self.stack):
            return                        # a stray close tag: ignore it rather than unwind everything
        while self.stack:
            open_tag, hides, cell = self.stack.pop()
            self.hidden -= hides
            self.in_cell -= cell
            if open_tag == tag:
                break
        if self.hidden:
            return
        if tag in ("tr", "table"):
            self.parts.append("\n")
            self.cell_open = False
        elif tag in _BLOCK:
            self.parts.append(" " if self.in_cell else "\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def html_text(document: str) -> str:
    """The visible text of an HTML document: one block, or one table row, per line."""
    parser = _TextOfHtml()
    parser.feed(document)
    parser.close()
    lines = []
    for line in "".join(parser.parts).split("\n"):
        # str.split() with no argument also splits on no-break spaces (&#160;),
        # so this collapses every run of whitespace to one space. Empty cells,
        # which filings use for spacing columns, are dropped.
        cells = [" ".join(cell.split()) for cell in line.split(" | ")]
        line = " | ".join(cell for cell in cells if cell)
        if line:
            lines.append(line)
    return "\n".join(lines)


# --- Diagnosis -------------------------------------------------------------

def looks_scanned(path: Path, pages: list[tuple[int, str]]) -> bool:
    """True if a PDF yielded almost no text even after OCR was given a turn.

    Reaching here now means either the OCR dependencies are missing or the
    scan was too poor to read, so the message shown to the user depends on
    which - see ocr_available().
    """
    if Path(path).suffix.lower() != ".pdf":
        return False
    if not pages:
        return True
    total_chars = sum(len(text) for _, text in pages)
    return total_chars / max(len(pages), 1) < 100

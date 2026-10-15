"""Turn a file into clean, sectioned text, keeping the page each line came from.

PDFs need the most care. Their text arrives page by page with the running header, footer
and page number repeated on every page, words hyphenated across lines, and sentences cut in
half at page breaks. Each of those costs retrieval quality, so they are repaired here before
anything is chunked.
"""

import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from query_tool.core.exceptions import DocumentError

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".markdown", ".docx"})

Line = tuple[int | None, str]  # (page number, text); page is None for formats without pages

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)[\s#]*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SPACE_RUN_RE = re.compile(r"(?<=\S)[ \t]{2,}")
_DASH_RUN_RE = re.compile(r"-{4,}")
# "1. Scope", "4.2 Security", "A. Terms", "Annex A - Measures", "Section 3: Fees"
_PDF_HEADING_RE = re.compile(
    r"^(?:\d+\.(?:\d+\.?)*|[A-Z]\.|(?:Article|Section|Annex|Appendix|Schedule|Chapter|Part)"
    r"\s+[\dA-Z]+[.:]?(?:\s+-)?)\s+(?P<title>[A-Z].*)$"
)
_HYPHENATED_END_RE = re.compile(r"[a-z]-$")
# Header/footer lines are looked for among the first and last few lines of each page.
EDGE_LINES = 2


@dataclass(frozen=True)
class Section:
    heading: str  # heading path inside the document; "" before the first heading
    lines: list[Line] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractedDocument:
    title: str  # put at the top of every chunk, with the section heading
    sections: list[Section]


def tidy_line(line: str) -> str:
    """Collapse padding, such as aligned table columns, that dilutes relevance scores."""
    return _DASH_RUN_RE.sub("---", _SPACE_RUN_RE.sub(" ", line)).rstrip()


def _tidy_lines(lines: list[Line]) -> list[Line]:
    """Tidy each line, drop repeated blank lines and blank edges."""
    out: list[Line] = []
    for page, text in lines:
        text = tidy_line(text)
        if text or (out and out[-1][1]):
            out.append((page, text))
    while out and not out[-1][1]:
        out.pop()
    return out


def tidy_text(text: str) -> str:
    return "\n".join(line for _, line in _tidy_lines([(None, ln) for ln in text.splitlines()]))


# ---- Markdown (also used for Word files, converted to Markdown) ----------


def markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown at headings into (heading path, body) pairs.

    The path names every enclosing heading, e.g. "Pricing Guide > Trials", so a chunk keeps
    its context after it is separated from the rest of the document.
    """
    path: list[tuple[int, str]] = []
    blocks: list[tuple[str, list[str]]] = [("", [])]
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _MD_HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            path = [(lvl, title) for lvl, title in path if lvl < level]
            path.append((level, heading.group(2)))
            blocks.append((" > ".join(title for _, title in path), []))
        else:
            blocks[-1][1].append(line)
    return [(title, body) for title, lines in blocks if (body := "\n".join(lines).strip())]


def _markdown_document(text: str) -> ExtractedDocument:
    sections = [
        Section(heading, _tidy_lines([(None, ln) for ln in body.splitlines()]))
        for heading, body in markdown_sections(text)
    ]
    return ExtractedDocument(title="", sections=[s for s in sections if s.lines])


def _docx_document(data: bytes) -> ExtractedDocument:
    """Word paragraphs and tables in document order; headings become the section path and
    the Title style (or the file's title property) the document title."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    title = (doc.core_properties.title or "").strip()
    lines: list[str] = []
    for element in doc.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(element, doc)
            style = paragraph.style.name if paragraph.style is not None else ""
            level = style.removeprefix("Heading ")
            if style == "Title":
                title = title or paragraph.text.strip()
            elif style.startswith("Heading ") and level.isdigit():
                lines.append("#" * int(level) + " " + paragraph.text)
            else:
                lines.append(paragraph.text)
        elif tag == "tbl":
            for row in Table(element, doc).rows:
                lines.append("| " + " | ".join(cell.text.strip() for cell in row.cells) + " |")
            lines.append("")
    document = _markdown_document("\n".join(lines))
    return ExtractedDocument(title=title, sections=document.sections)


# ---- PDF ----------------------------------------------------------------


def _furniture_key(line: str) -> str:
    """Page furniture differs only in numbers: "Page 2 of 9" and "Page 3 of 9" match."""
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def strip_page_furniture(pages: list[list[str]]) -> list[list[str]]:
    """Remove lines that repeat at the top or bottom of at least half the pages."""
    if len(pages) < 2:
        return pages

    def edges(lines: list[str]) -> list[int]:
        filled = [i for i, line in enumerate(lines) if line.strip()]
        return sorted(set(filled[:EDGE_LINES] + filled[-EDGE_LINES:]))

    counts = Counter(
        key for lines in pages for key in {_furniture_key(lines[i]) for i in edges(lines)}
    )
    threshold = max(2, len(pages) / 2)
    repeated = {key for key, n in counts.items() if n >= threshold}
    cleaned: list[list[str]] = []
    for lines in pages:
        drop = {i for i in edges(lines) if _furniture_key(lines[i]) in repeated}
        cleaned.append([line for i, line in enumerate(lines) if i not in drop])
    return cleaned


def is_pdf_heading(line: str) -> bool:
    """A numbered or labelled short line in title case, e.g. "12. Liability and Law".

    Conservative on purpose: a missed heading costs a little context, a false one would cut
    a clause in half.
    """
    match = _PDF_HEADING_RE.match(line.strip())
    if not match or line.rstrip().endswith((".", ",", ";", ":")) or len(line) > 90:
        return False
    words = match.group("title").split()
    long_words = [w for w in words if len(w) >= 4]
    capitalized = sum(w[0].isupper() for w in long_words)
    return len(words) <= 8 and (not long_words or capitalized / len(long_words) >= 0.5)


def _join_hyphenated(lines: list[Line]) -> list[Line]:
    """'process-' + 'ing' -> 'processing', across line and page breaks."""
    out: list[Line] = []
    for page, text in lines:
        if out and _HYPHENATED_END_RE.search(out[-1][1]) and text[:1].islower():
            out[-1] = (out[-1][0], out[-1][1][:-1] + text)
        else:
            out.append((page, text))
    return out


def _pdf_document(filename: str, data: bytes) -> ExtractedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = strip_page_furniture(
        [(page.extract_text() or "").splitlines() for page in reader.pages]
    )
    lines = _join_hyphenated(
        [(number, line) for number, page in enumerate(pages, 1) for line in page]
    )

    sections: list[Section] = [Section("")]
    for page, text in lines:
        if is_pdf_heading(text):
            sections.append(Section(" ".join(text.split())))
        else:
            sections[-1].lines.append((page, text))
    sections = [Section(s.heading, tidied) for s in sections if (tidied := _tidy_lines(s.lines))]

    metadata_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    first_line = next((t.strip() for s in sections for _, t in s.lines if t.strip()), "")
    title = metadata_title or (first_line if len(first_line) <= 120 else "")
    return ExtractedDocument(title=title or Path(filename).stem, sections=sections)


# ---- entry point ----------------------------------------------------------


def extract_document(filename: str, data: bytes) -> ExtractedDocument:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentError(
            f"{filename}: unsupported file type. Use one of {sorted(SUPPORTED_EXTENSIONS)}."
        )
    try:
        if ext == ".pdf":
            document = _pdf_document(filename, data)
        elif ext == ".docx":
            document = _docx_document(data)
        elif ext in {".md", ".markdown"}:
            document = _markdown_document(data.decode("utf-8-sig", errors="replace"))
        else:
            text = data.decode("utf-8-sig", errors="replace")
            document = ExtractedDocument(
                "", [Section("", _tidy_lines([(None, ln) for ln in text.splitlines()]))]
            )
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"{filename}: could not be read ({exc}).") from exc

    if not any(text for s in document.sections for _, text in s.lines):
        hint = " (is it a scanned PDF? Only embedded text can be read.)" if ext == ".pdf" else ""
        raise DocumentError(f"{filename}: no extractable text{hint}")
    return document

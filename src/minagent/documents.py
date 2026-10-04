"""Reading data files, and writing PDFs out of what they hold.

Two tools live here, and they are two halves of one task: a report is built from
data somebody already has.

``read_document`` opens a PDF, a spreadsheet, a word processor file, a CSV or a
plain text file and returns it as text the model can reason about. ``read_file``
cannot do this. A PDF is a binary layout, not text; a CSV with 200000 rows and
an XLSX with fifteen sheets have no line-based shape to page through. So the
content never reaches the context until this tool asks for it.

``create_pdf`` is the other half. Once the data has been read, summarised and
shaped, the same model that could only describe the result in chat writes it as
a file: markdown or HTML in, a paginated PDF with real tables in
``salida/``. Output goes through the same folder rule as ``download_file`` -
``salida/`` only, numbered rather than overwritten - so a generated report
never lands among the source files and never silently replaces an earlier one.

Both depend on PyMuPDF for PDF work and use only the standard library for the
rest, so a session without the optional dependency still reads spreadsheets and
CSVs, and says so plainly when a PDF was asked for.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .download import resolve_destination, unique_output_path
from .errors import AgentError
from .markdown_html import markdown_to_html

READ_DOCUMENT_TOOL_NAME = "read_document"
CREATE_PDF_TOOL_NAME = "create_pdf"

OUTPUT_DIRNAME = "salida"

# A tool result over this is archived and can be recalled; the point here is to
# be generous enough that a real report is read whole in one call.
MAX_TEXT_CHARS = 60_000
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_PDF_PAGES_READ = 80
MAX_TABLE_ROWS_RETURNED = 200
MAX_TABLE_ROWS = 200
DEFAULT_TABLE_ROWS = 25
MAX_PROFILE_SAMPLES = 3

MAX_PDFS_PER_CALL = 10
MAX_PDF_CHARS = 2_000_000
MAX_PDF_PAGES = 500

PAGE_SIZES: dict[str, tuple[float, float]] = {
    "a4": (595.0, 842.0),
    "letter": (612.0, 792.0),
}
PAGE_MARGIN = 42.0

BASE_CSS = (
    "body { font-family: sans-serif; font-size: 10.5pt; color: #1a1a1a; }"
    "h1 { color: #14344f; }"
    "h2 { color: #1d4f70; }"
    "h3 { color: #2a5f80; }"
    "a { color: #1155aa; }"
    "code { font-family: monospace; background-color: #f0f0f0; }"
    "pre { background-color: #f4f4f4; padding: 6px; }"
    "blockquote { color: #555555; }"
    "table { border-collapse: collapse; }"
    "th { background-color: #e8eef3; border: 0.6pt solid #9aa7b1; padding: 3px; }"
    "td { border: 0.6pt solid #9aa7b1; padding: 3px; }"
)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log", ".jsonl", ".ini", ".yaml", ".yml"}
_TABULAR_SUFFIXES = {".csv", ".tsv", ".txt"}
_PYMUPDF_HINT = "Install it with `uv sync --extra documents`."


def create_document_tools() -> list[dict[str, Any]]:
    """The tool schemas for the ``documents`` capability."""
    return [
        {
            "type": "function",
            "function": {
                "name": READ_DOCUMENT_TOOL_NAME,
                "description": (
                    "Read a data file as text: PDF, CSV, TSV, JSON, TXT, MD, HTML, DOCX or XLSX. "
                    "Use this instead of read_file for any of those - read_file returns raw bytes, "
                    "which is useless for a PDF or a spreadsheet. Returns the text, or a table "
                    "preview with a column profile when the file holds tabular rows. For a PDF, "
                    "pages selects which ones ('1-5,9'); a PDF with no text layer is a scan and "
                    "needs OCR, which this tool does not do."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path, e.g. research_records/data-catalog.jsonl",
                        },
                        "pages": {
                            "type": "string",
                            "description": "PDF only: pages to read, e.g. '1-3' or '1,4-6'. Defaults to the first pages.",
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "First row to return for a tabular file; defaults to 1",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_TABLE_ROWS_RETURNED,
                            "description": f"Rows to return for a tabular file; defaults to {DEFAULT_TABLE_ROWS}",
                        },
                        "sheet": {
                            "type": "string",
                            "description": "XLSX only: sheet name or number, or 'all'. Defaults to the first sheet.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": CREATE_PDF_TOOL_NAME,
                "description": (
                    "Write one or more PDF files into the workspace, from text or from data already "
                    "read with read_document. Content is Markdown by default: headings, paragraphs, "
                    "lists, tables, code and links are rendered; pass format='html' for raw HTML. "
                    f"Files always land in {OUTPUT_DIRNAME}/ and never overwrite - a second file of "
                    "the same name gets a numbered suffix. Returns each file's page count and size."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_PDFS_PER_CALL,
                            "description": "The files to write, in one call.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "File name ending in .pdf, e.g. informe-enero.pdf",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The document body, in the format chosen below.",
                                    },
                                    "title": {
                                        "type": "string",
                                        "description": "Optional PDF title for the document metadata and page header.",
                                    },
                                },
                                "required": ["name", "content"],
                            },
                        },
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "html"],
                            "description": "How to read each document's content; defaults to markdown.",
                        },
                        "page_size": {
                            "type": "string",
                            "enum": sorted(PAGE_SIZES),
                            "description": "Paper size; defaults to a4.",
                        },
                    },
                    "required": ["documents"],
                },
            },
        },
    ]


# ------------------------------------------------------------------ reading


def _require_pymupdf() -> Any:
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AgentError(
            f"PyMuPDF is not installed, so this file cannot be handled. {_PYMUPDF_HINT}"
        ) from error
    return pymupdf


def _decode(data: bytes) -> tuple[str, str]:
    """Decode file bytes, and say which encoding it took.

    Spreadsheets and CSVs exported on Windows are latin-1 far more often than
    anyone admits, and a decode failure there is indistinguishable from an empty
    file unless the encoding is reported back to the model.
    """
    # A BOM is checked first so it does not show up as a stray character in the
    # first field; otherwise plain utf-8 is tried before the Windows encodings.
    candidates = (
        ("utf-16", "utf-16")
        if data[:2] in {b"\xff\xfe", b"\xfe\xff"}
        else ("utf-8-sig", "utf-16", "cp1252")
        if data[:3] == b"\xef\xbb\xbf"
        else ("utf-8", "cp1252")
    )
    for encoding in candidates:
        try:
            return data.decode(encoding), encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("latin-1"), "latin-1"


def _extension(path: str) -> str:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _kind(path: str, data: bytes) -> str:
    """The format to read by, from the extension and then the bytes themselves."""
    suffix = _extension(path)
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        return "pdf"
    if suffix in {".docx", ".docm"}:
        return "docx"
    if suffix in {".xlsx", ".xlsm"}:
        return "xlsx"
    if suffix in {".json", ".geojson"}:
        return "json"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix in {".html", ".htm", ".xhtml"}:
        return "html"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:2] == b"PK":
        if b"word/document.xml" in data[:4096] or _zip_names(data) & {"word/document.xml"}:
            return "docx"
        return "xlsx"
    if data.lstrip()[:1] in {b"{", b"["}:
        return "json"
    return "text"


def _zip_names(data: bytes) -> set[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return set(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return set()


async def run_read_document(
    args: dict[str, Any],
    workspace_access: Any,
    *,
    allow_outside: bool = True,
) -> str:
    """Read one data file and return it as text the model can work with."""
    path = args.get("path")
    if not isinstance(path, str) or not path.strip():
        raise AgentError("read_document needs a path.")
    path = path.strip()
    try:
        data = await workspace_access.read_raw_file(path, allow_outside=allow_outside)
    except FileNotFoundError:
        raise AgentError(f"There is no file at {path}.") from None
    except IsADirectoryError:
        raise AgentError(f"{path} is a directory; read_document needs a file.") from None
    if len(data) > MAX_DOCUMENT_BYTES:
        raise AgentError(f"{path} is larger than the {MAX_DOCUMENT_BYTES // (1024 * 1024)} MiB limit.")

    kind = _kind(path, data)
    offset = _positive_int(args.get("offset"), 1, "offset")
    limit = _bounded_int(args.get("limit"), DEFAULT_TABLE_ROWS, 1, MAX_TABLE_ROWS_RETURNED, "limit")

    if kind == "pdf":
        body = _read_pdf(data, args.get("pages"))
    elif kind == "csv":
        text, encoding = _decode(data)
        delimiter = "\t" if _extension(path) == ".tsv" else _sniff_delimiter(text)
        body = f"{_read_table(text, delimiter, offset, limit, 'row')}\n(encoding: {encoding})"
    elif kind == "json":
        text, encoding = _decode(data)
        body = f"{_read_json(text)}\n(encoding: {encoding})"
    elif kind == "jsonl":
        body = _read_jsonl(data, offset, limit)
    elif kind == "html":
        text, encoding = _decode(data)
        body = _read_html(text)
        body = f"{body}\n(stripped from HTML, encoding: {encoding})"
    elif kind == "docx":
        body = _read_docx(data)
    elif kind == "xlsx":
        body = _read_xlsx(data, args.get("sheet"), offset, limit)
    else:
        text, encoding = _decode(data)
        body = _read_text(text, offset, limit)
        body = f"{body}\n(encoding: {encoding})"

    return _bound(body, path, kind)


def _positive_int(value: Any, default: int, label: str) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise AgentError(f"read_document {label} must be a whole number.") from None
    return max(1, number)


def _bounded_int(value: Any, default: int, low: int, high: int, label: str) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise AgentError(f"read_document {label} must be a whole number.") from None
    return min(max(low, number), high)


def _bound(body: str, path: str, kind: str) -> str:
    """Keep the result inside what a tool result should carry."""
    header = f"# {path}\n"
    budget = MAX_TEXT_CHARS - len(header)
    if len(body) <= budget:
        return header + body
    advice = (
        "\n\n[Truncated. Read the rest with read_document and a narrower range: "
        "pages for a PDF, offset and limit for rows.]"
    )
    return header + body[: max(0, budget - len(advice))].rstrip() + advice


# --------------------------------------------------------------- PDF input


def _parse_pages(spec: Any, page_count: int) -> tuple[list[int], list[int]]:
    """The 0-based pages a request asks for, and the ones it left out."""
    if not isinstance(spec, str) or not spec.strip():
        wanted = list(range(min(page_count, MAX_PDF_PAGES_READ)))
        return wanted, []
    selected: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", chunk)
        if not match:
            raise AgentError(f"read_document cannot read '{chunk}' as a page range; use '1-3' or '1,4-6'.")
        first = int(match.group(1))
        last = int(match.group(2) or match.group(1))
        if first < 1 or last < first:
            raise AgentError(f"read_document pages start at 1 and count up: '{chunk}' is not a range.")
        selected.extend(range(first - 1, min(last, page_count)))
    selected = sorted({page for page in selected if 0 <= page < page_count})
    if not selected:
        raise AgentError(f"The PDF has {page_count} page(s); none of '{spec}' is inside it.")
    if len(selected) > MAX_PDF_PAGES_READ:
        raise AgentError(
            f"{len(selected)} pages were asked for, over the {MAX_PDF_PAGES_READ} limit. "
            "Ask for the pages you need in two calls."
        )
    return selected, [page for page in range(page_count) if page not in set(selected)]


def _read_pdf(data: bytes, spec: Any) -> str:
    pymupdf = _require_pymupdf()
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise AgentError("This PDF is password protected; unlock it before reading it here.")
            selected, missing = _parse_pages(spec, document.page_count)
            parts: list[str] = []
            empty_pages = 0
            for number in selected:
                page = document[number]
                text = page.get_text("text").strip()
                tables = _pdf_tables(pymupdf, page)
                if not text and not tables:
                    empty_pages += 1
                    parts.append(f"## Page {number + 1}\n[no text layer on this page]")
                    continue
                block = [f"## Page {number + 1}"]
                if text:
                    block.append(text)
                if tables:
                    block.append("\n### Tables on this page\n" + "\n\n".join(tables))
                parts.append("\n\n".join(block))
            skipped = len(missing)
    except AgentError:
        raise
    except Exception as error:  # a malformed file should read as a refusal, not a crash
        raise AgentError(f"This file does not open as a PDF: {error}") from None

    lines = [f"# {len(selected)} of {selected[-1] + 1} page(s) read"]
    if skipped:
        lines[0] += f", {skipped} not read"
    if empty_pages == len(selected) and selected:
        lines[0] += "\n[No text layer anywhere in the pages read: this is a scan. It needs OCR, which this tool does not do.]"
    elif empty_pages:
        lines[0] += f"\n[{empty_pages} page(s) had no text layer; they may be scans or images.]"
    return "\n\n".join([lines[0], *parts])


def _pdf_tables(pymupdf: Any, page: Any) -> list[str]:
    """Markdown for the ruled tables MuPDF can see on a page."""
    try:
        found = page.find_tables()
    except Exception:
        return []
    tables = []
    for found_table in getattr(found, "tables", []) or []:
        try:
            rows = found_table.extract()
        except Exception:
            continue
        rows = [row for row in rows if any(cell and str(cell).strip() for cell in row)]
        if len(rows) < 2:
            continue
        tables.append(_markdown_table([[str(cell or "") for cell in row] for row in rows[:MAX_TABLE_ROWS]]))
    return tables


# ---------------------------------------------------------- tabular input


def _sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {candidate: first.count(candidate) for candidate in (",", ";", "\t", "|")}
        best = max(counts, key=lambda key: counts[key])
        return best if counts[best] else ","


def _read_table(text: str, delimiter: str, offset: int, limit: int, label: str) -> str:
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        rows = list(reader)
    except csv.Error as error:
        raise AgentError(f"This file is not readable as delimited text: {error}") from None
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return "The file has no rows."
    header = rows[0]
    body = rows[1:]
    window = body[offset - 1 : offset - 1 + limit]
    lines = [
        f"# {len(body)} {label}(s) after the header, {len(header)} column(s)",
        "",
        _markdown_table([header, *window]),
    ]
    if offset + limit - 1 < len(body):
        lines.append(f"\n[Rows {offset} to {offset + len(window) - 1} shown; more follow. Re-read with offset={offset + len(window)}.]")
    lines.append("")
    lines.append(_profile(header, body))
    return "\n".join(lines)


_NUMBER = re.compile(r"^[+-]?\d{1,3}(?:[ .]\d{3})*(?:[.,]\d+)?$|^[+-]?\d+(?:[.,]\d+)?%?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|^\d{2}/\d{2}/\d{4}$")


def _to_number(value: str) -> float | None:
    if not _NUMBER.match(value.strip()):
        return None
    cleaned = value.strip().replace(" ", "").replace("%", "").replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _profile(header: list[str], body: list[list[str]]) -> str:
    """A per-column summary, so the report can be written from facts not guesses."""
    lines = [f"## Columns\n\n- **{_cell(header[0] or 'column 1')}** ..."]
    lines = ["## Columns"]
    for index, name in enumerate(header):
        values = [row[index].strip() for row in body if index < len(row)]
        values = [value for value in values if value]
        if not values:
            lines.append(f"- {_cell(name or f'column {index + 1}')}: empty")
            continue
        samples = ", ".join(_cell(value) for value in values[:MAX_PROFILE_SAMPLES])
        numbers = [number for number in (_to_number(value) for value in values) if number is not None]
        if len(numbers) == len(values):
            average = sum(numbers) / len(numbers)
            lines.append(
                f"- {_cell(name or f'column {index + 1}')}: number, {len(values)} filled, "
                f"min {min(numbers):g}, max {max(numbers):g}, mean {average:.3g}"
            )
            continue
        kinds = set()
        if len(numbers) > len(values) / 2:
            kinds.add("mostly numeric")
        if any(_DATE.match(value) for value in values):
            kinds.add("has dates")
        qualifier = f" ({', '.join(sorted(kinds))})" if kinds else ""
        lines.append(
            f"- {_cell(name or f'column {index + 1}')}: text, {len(values)} filled, "
            f"{len(set(values))} distinct{qualifier}; e.g. {samples}"
        )
    return "\n".join(lines)


def _cell(value: str) -> str:
    """One CSV field as one Markdown cell."""
    text = str(value).strip()
    return text.replace("|", "\\|").replace("\n", " ") if text else ""


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [[_cell(cell) for cell in row] + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(padded[0]) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def _read_text(text: str, offset: int, limit: int) -> str:
    lines = text.splitlines()
    total = len(lines)
    window = lines[offset - 1 : offset - 1 + limit * 40]
    body = "\n".join(window)
    if offset + len(window) - 1 < total:
        body += f"\n\n[{total} lines in total; showing from line {offset}. Re-read with offset={offset + len(window)}.]"
    return body


def _read_jsonl(data: bytes, offset: int, limit: int) -> str:
    text, _ = _decode(data)
    records: list[Any] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"line": number, "unparsable": line[:200]})
    return _render_records(records, offset, limit)


def _read_json(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        head = "\n".join(text.splitlines()[:40])
        return f"Not valid JSON ({error}). The first lines are:\n\n{head}"
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        return _render_records(data, 1, MAX_TABLE_ROWS_RETURNED)
    rendered = json.dumps(data, indent=2, ensure_ascii=False)
    return f"# {type(data).__name__}\n\n{rendered}"


def _render_records(records: list[Any], offset: int, limit: int) -> str:
    if not records:
        return "No records."
    window = records[offset - 1 : offset - 1 + limit]
    columns: list[str] = []
    for record in window:
        if isinstance(record, dict):
            for key in record:
                if key not in columns:
                    columns.append(str(key))
    lines = [f"# {len(records)} record(s)"]
    if columns:
        rows = [
            [str(record.get(column, "")) if isinstance(record, dict) else str(record) for column in columns]
            for record in window
        ]
        lines.append("")
        lines.append(_markdown_table([columns, *rows]))
    else:
        lines.extend(json.dumps(record, ensure_ascii=False) for record in window)
    if offset + len(window) - 1 < len(records):
        lines.append(
            f"\n[Records {offset} to {offset + len(window) - 1} of {len(records)} shown. Re-read with offset={offset + len(window)}.]"
        )
    return "\n".join(lines)


# -------------------------------------------------------------- HTML input


class _TextExtractor(HTMLParser):
    """The visible text of an HTML page, with its scripts and styles dropped."""

    _SKIP = {"script", "style", "head", "title", "meta", "link", "noscript"}
    _BREAK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _read_html(text: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return text
    body = parser.text()
    return body or text[:4000]


# ------------------------------------------------------------- DOCX input

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _read_docx(data: bytes) -> str:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError):
        raise AgentError("This file does not open as a .docx.") from None
    with archive:
        try:
            document = archive.read("word/document.xml")
        except KeyError:
            raise AgentError("This .docx has no word/document.xml, so it holds no readable text.") from None
    try:
        root = ElementTree.fromstring(document)
    except ElementTree.ParseError as error:
        raise AgentError(f"The .docx XML could not be parsed: {error}") from None

    body = root.find(f"{_W_NS}body")
    if body is None:
        return "The document body is empty."
    lines: list[str] = []
    for element in body:
        tag = element.tag
        if tag == f"{_W_NS}p":
            text = _docx_paragraph(element)
            if text:
                lines.append(text)
        elif tag == f"{_W_NS}tbl":
            rows = []
            for row_element in element.findall(f"{_W_NS}tr"):
                cells = [" ".join(_docx_paragraph(p) for p in cell.iter(f"{_W_NS}p")) for cell in row_element.findall(f"{_W_NS}tc")]
                rows.append(cells)
            rows = [row for row in rows if any(cell.strip() for cell in row)]
            if rows:
                lines.append(_markdown_table(rows[:MAX_TABLE_ROWS]))
    text = "\n\n".join(lines)
    return f"# {len(lines)} block(s)\n\n{text}" if text else "The document has no readable text."


def _docx_paragraph(element: ElementTree.Element) -> str:
    pieces: list[str] = []
    for node in element.iter():
        if node.tag == f"{_W_NS}t":
            pieces.append(node.text or "")
        elif node.tag == f"{_W_NS}tab":
            pieces.append("\t")
        elif node.tag in {f"{_W_NS}br", f"{_W_NS}cr"}:
            pieces.append(" ")
    return "".join(pieces).strip()


# ------------------------------------------------------------- XLSX input

_S_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_DATE_FORMAT_IDS = set(range(14, 23)) | set(range(45, 48)) | {27, 30, 36, 50, 57}
_DATE_FORMAT_TOKENS = {"d", "dd", "ddd", "m", "mm", "mmm", "y", "yy", "yyyy", "h", "hh", "s", "ss", "am/pm"}
_EXCEL_EPOCH = datetime(1899, 12, 30)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        data = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError:
        return []
    strings = []
    for item in root.findall(f"{_S_NS}si"):
        strings.append("".join(node.text or "" for node in item.iter(f"{_S_NS}t")))
    return strings


def _xlsx_sheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Sheet names and the archive path of each sheet part, in workbook order."""
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except (KeyError, ElementTree.ParseError):
        return []
    targets = {
        element.get("Id", ""): element.get("Target", "")
        for element in rels
        if _local(element.tag) == "Relationship"
    }
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.iter():
        if _local(sheet.tag) != "sheet":
            continue
        target = targets.get(sheet.get(f"{_R_NS}id", ""), "")
        path = target[1:] if target.startswith("/") else (f"xl/{target}" if target else "")
        sheets.append((sheet.get("name", f"sheet{len(sheets) + 1}"), path))
    return sheets


def _excel_date(serial: str) -> str:
    try:
        return (_EXCEL_EPOCH + timedelta(days=float(serial))).strftime("%Y-%m-%d")
    except (OverflowError, ValueError):
        return serial


def _xlsx_rows(data: bytes, name: str) -> list[list[str]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError):
        raise AgentError("This file does not open as an .xlsx.") from None
    with archive:
        shared = _xlsx_shared_strings(archive)
        try:
            root = ElementTree.fromstring(archive.read(name))
        except (KeyError, ElementTree.ParseError):
            return []
    styles = _xlsx_date_styles(_xlsx_styles(data))
    rows: list[list[str]] = []
    for row in root.iter():
        if _local(row.tag) != "row":
            continue
        values: dict[int, str] = {}
        for cell in row:
            if _local(cell.tag) != "c":
                continue
            column = _column_index(cell.get("r", ""))
            kind = cell.get("t", "")
            value_element = cell.find(f"{_S_NS}v")
            inline = cell.find(f"{_S_NS}is")
            if kind == "s" and value_element is not None:
                try:
                    text = shared[int(value_element.text or "0")]
                except (ValueError, IndexError):
                    text = value_element.text or ""
            elif kind == "inlineStr" and inline is not None:
                text = "".join(node.text or "" for node in inline.iter(f"{_S_NS}t"))
            elif kind == "b" and value_element is not None:
                text = "TRUE" if value_element.text == "1" else "FALSE"
            elif value_element is None:
                text = ""
            elif _xlsx_is_date(cell, styles):
                text = _excel_date(value_element.text or "")
            else:
                text = value_element.text or ""
            values[column] = text.strip()
        if not values:
            continue
        width = max(values) + 1
        rows.append([values.get(index, "") for index in range(width)])
    return rows


def _xlsx_is_date(cell: Any, date_styles: set[int]) -> bool:
    """Whether this cell carries a date number format.

    Per cell, never per column: one date in a column is not a column of dates,
    and reading it that way turns every other number in it into 1899.
    """
    try:
        style = int(cell.get("s", "0"))
    except ValueError:
        return False
    return style in date_styles


def _xlsx_styles(data: bytes) -> bytes:
    """Reopen the workbook just far enough to read its number formats."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        try:
            return archive.read("xl/styles.xml")
        except KeyError:
            return b""


def _xlsx_date_styles(styles: bytes) -> set[int]:
    """Which cell style ids carry a date format, so serials become dates."""
    if not styles:
        return set()
    try:
        root = ElementTree.fromstring(styles)
    except ElementTree.ParseError:
        return set()
    custom: dict[int, str] = {}
    for element in root.iter(f"{_S_NS}numFmt"):
        try:
            custom[int(element.get("numFmtId", "-1"))] = element.get("formatCode") or ""
        except ValueError:
            continue
    date_styles: set[int] = set()
    for cell_xfs in root.iter(f"{_S_NS}cellXfs"):
        for index, cell_xf in enumerate(cell_xfs):
            try:
                number = int(cell_xf.get("numFmtId", "0"))
            except ValueError:
                continue
            if number in _DATE_FORMAT_IDS or _is_date_format(custom.get(number, "")):
                date_styles.add(index)
    return date_styles


def _is_date_format(code: str) -> bool:
    """Whether a custom number format draws a date.

    Whole tokens, not a substring: the format for a red negative number is
    ``0.00;[Red]-0.00``, and a search for "d" would call it a date and turn
    every loss in a spreadsheet into 1899.
    """
    if not code or code.strip().lower() in {"general", "@", ""}:
        return False
    return bool(_DATE_FORMAT_TOKENS & set(re.findall(r"[a-z]+", code.lower())))


def _column_index(reference: str) -> int:
    letters = re.match(r"([A-Z]+)", reference.upper())
    if not letters:
        return 0
    index = 0
    for character in letters.group(1):
        index = index * 26 + (ord(character) - ord("A") + 1)
    return index - 1


def _read_xlsx(data: bytes, sheet: Any, offset: int, limit: int) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        sheets = _xlsx_sheets(archive)
    if not sheets:
        raise AgentError("This .xlsx lists no sheets, so it holds no readable rows.")
    available = [name for name, _ in sheets]
    wanted: list[tuple[str, str]]
    choice = str(sheet).strip() if sheet is not None else ""
    if not choice or choice.isdigit():
        index = int(choice) - 1 if choice else 0
        if not 0 <= index < len(sheets):
            raise AgentError(f"This workbook has {len(sheets)} sheet(s): {', '.join(available)}.")
        wanted = [sheets[index]]
    elif choice.lower() == "all":
        wanted = sheets
    else:
        match = [entry for entry in sheets if entry[0].lower() == choice.lower()]
        if not match:
            raise AgentError(f"No sheet named '{choice}'. This workbook has: {', '.join(available)}.")
        wanted = [match[0]]

    parts: list[str] = []
    for name, part in wanted:
        rows = [row for row in _xlsx_rows(data, part) if any(cell.strip() for cell in row)]
        if not rows:
            parts.append(f"## Sheet: {name}\n(empty)")
            continue
        header, body = rows[0], rows[1:]
        window = body[offset - 1 : offset - 1 + limit]
        block = [
            f"## Sheet: {name}",
            f"{len(body)} row(s) after the header, {len(header)} column(s)",
            "",
            _markdown_table([header, *window]),
            "",
            _profile(header, body),
        ]
        if offset + limit - 1 < len(body):
            block.insert(
                3,
                f"[Rows {offset} to {offset + len(window) - 1} shown; more follow. "
                f"Re-read with offset={offset + len(window)} or sheet='{name}'.]",
            )
        parts.append("\n".join(block))
    return "\n\n".join(parts)


# ----------------------------------------------------------------- writing


async def run_create_pdf(documents: Any, root_directory: str, *, page_size: str = "a4", body_format: str = "markdown") -> str:
    """Write every requested document as a PDF under ``salida/``."""
    if isinstance(documents, dict):
        documents = [documents]
    if not isinstance(documents, list) or not documents:
        raise AgentError("create_pdf needs a non-empty documents array.")
    if len(documents) > MAX_PDFS_PER_CALL:
        raise AgentError(f"create_pdf writes at most {MAX_PDFS_PER_CALL} files per call.")
    if page_size not in PAGE_SIZES:
        raise AgentError(f"create_pdf page_size must be one of {', '.join(sorted(PAGE_SIZES))}.")
    pymupdf = _require_pymupdf()
    width, height = PAGE_SIZES[page_size]
    root = Path(root_directory)

    lines: list[str] = []
    for entry in documents:
        if not isinstance(entry, dict):
            raise AgentError("Every create_pdf document must be an object with name and content.")
        name = str(entry.get("name", "")).strip()
        content = entry.get("content")
        if not name:
            raise AgentError("Every create_pdf document needs a name ending in .pdf.")
        if not isinstance(content, str) or not content.strip():
            raise AgentError(f"{name} has no content to write.")
        if len(content) > MAX_PDF_CHARS:
            raise AgentError(f"{name} is {len(content)} characters, over the {MAX_PDF_CHARS} limit.")
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"
        title = str(entry.get("title", "")).strip() or _title_from_content(content, name)

        html = content if body_format == "html" else markdown_to_html(content)
        requested = resolve_destination(root, name)
        destination = unique_output_path(requested.parent, requested.name)
        pages = _write_pdf(pymupdf, html, destination, width, height, title)
        size = destination.stat().st_size
        lines.append(
            f"Wrote {OUTPUT_DIRNAME}/{destination.name}: {pages} page(s), {size // 1024 or 1} KiB."
        )
    return "\n".join(lines)


def _title_from_content(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return fallback


def _write_pdf(pymupdf: Any, html: str, destination: Any, width: float, height: float, title: str) -> int:
    """Lay the HTML out over as many pages as it needs, then save it."""
    rect = pymupdf.Rect(PAGE_MARGIN, PAGE_MARGIN, width - PAGE_MARGIN, height - PAGE_MARGIN - 12)
    blocks = _split_blocks(html)
    if not blocks:
        raise AgentError("That content produced nothing to lay out.")

    document = pymupdf.open()
    scratch = pymupdf.open()
    pages = 0
    try:
        index = 0
        while index < len(blocks) and pages < MAX_PDF_PAGES:
            remaining = blocks[index:]
            count = _fit_prefix(scratch, remaining, BASE_CSS, width, height, rect)
            oversized = count == 0
            if oversized:
                head, tail = _split_block(remaining[0])
                if tail:
                    blocks[index : index + 1] = [head, tail]
                    continue
                # Nothing can be split further: one block taller than a page is
                # scaled down to fit rather than dropped.
                count = 1
            page = document.new_page(width=width, height=height)
            # The measurement above proved this prefix fits at full size, so the
            # scale floor only ever comes into play for the one block that could
            # not be split at all.
            scale_low = 0.35 if oversized else 1.0
            spare, _ = page.insert_htmlbox(rect, "\n".join(remaining[:count]), css=BASE_CSS, scale_low=scale_low)
            if spare < 0:
                page.insert_text(
                    (rect.x0, rect.y1 + 10),
                    "this block was scaled down to fit one page",
                    fontsize=6,
                    color=(0.45, 0.45, 0.45),
                )
            _stamp_page_number(page, pages + 1)
            pages += 1
            index += count
        if index < len(blocks):
            raise AgentError(f"The document is longer than the {MAX_PDF_PAGES} page limit; split it into files.")
        document.set_metadata({"title": title, "author": "MinAgent", "producer": "MinAgent (PyMuPDF)"})
        destination.parent.mkdir(parents=True, exist_ok=True)
        document.save(str(destination), garbage=3)
    finally:
        document.close()
        scratch.close()
    return pages


def _stamp_page_number(page: Any, number: int) -> None:
    try:
        page.insert_text(
            (page.rect.width / 2 - 12, page.rect.height - 26),
            str(number),
            fontsize=8,
            color=(0.45, 0.45, 0.45),
        )
    except Exception:
        return


_VOID_TAGS = {"br", "hr", "img", "meta", "link", "input", "col"}
_BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "ul",
    "ol",
    "table",
    "pre",
    "blockquote",
    "hr",
    "div",
    "section",
}
_TAG = re.compile(r"<(/?)([a-zA-Z0-9]+)[^>]*>")
_ROW = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)


def _split_blocks(html: str) -> list[str]:
    """Split HTML at its top-level elements, so a page break lands between them."""
    blocks: list[str] = []
    depth = 0
    start = 0
    for match in _TAG.finditer(html):
        closing, tag = match.group(1) == "/", match.group(2).lower()
        if tag in _VOID_TAGS:
            if depth == 0:
                blocks.append(html[start : match.end()])
                start = match.end()
            continue
        if closing:
            depth = max(0, depth - 1)
            if depth == 0:
                blocks.append(html[start : match.end()])
                start = match.end()
            continue
        if depth == 0 and tag in _BLOCK_TAGS:
            start = match.start()
        depth += 1
    tail = html[start:].strip()
    if tail:
        blocks.append(tail)
    return [block for block in (piece.strip() for piece in blocks) if block]


def _fit_prefix(scratch: Any, blocks: list[str], css: str, width: float, height: float, rect: Any) -> int:
    """How many leading blocks fit on one page, measured on a throwaway page.

    Spare height falls as content grows, so the boundary is a binary search
    rather than filling a page at a time: a long table would otherwise be
    re-measured once per row.
    """
    low, high = 1, len(blocks)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        if _spare_height(scratch, blocks[:middle], css, width, height, rect) >= 0:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _spare_height(scratch: Any, blocks: list[str], css: str, width: float, height: float, rect: Any) -> float:
    page = scratch.new_page(width=width, height=height)
    try:
        # scale_low=1 forbids shrinking, which is the only measurement that can
        # say "no": with the default the layout squeezes whatever it is given
        # into the page and reports zero spare height, which reads as a fit.
        spare, _ = page.insert_htmlbox(rect, "\n".join(blocks), css=css, scale_low=1.0)
        return spare
    finally:
        scratch.delete_page(-1)


def _split_block(block: str) -> tuple[str, str]:
    """Cut an over-tall block in half, or say that it cannot be cut.

    A page break in the middle of a paragraph reads as a bug, so a long one is
    cut at a sentence boundary instead of being scaled into illegibility.
    """
    if block.lower().startswith("<table"):
        return _split_table(block)
    opening = re.match(r"(<p\b[^>]*>)(.*)(</p>)", block, re.IGNORECASE | re.DOTALL)
    if not opening:
        return "", ""
    inner = opening.group(2)
    sentences = list(re.finditer(r"[^.!?\n]*[.!?]+\s+|\n{2,}", inner))
    if len(sentences) < 2:
        return "", ""
    # Cut near the middle sentence boundary, so both halves stay balanced.
    middle = len(inner) // 2
    boundary = min(sentences, key=lambda match: abs(match.end() - middle))
    head = opening.group(1) + inner[: boundary.start()].strip() + opening.group(3)
    tail = opening.group(1) + inner[boundary.start() :].strip() + opening.group(3)
    return head, tail


def _split_table(block: str) -> tuple[str, str]:
    """Halve a table's rows, repeating the header, so a long table can paginate."""
    if not block.lower().startswith("<table"):
        return "", ""
    rows = _ROW.findall(block)
    if len(rows) < 3:
        return "", ""
    header, body = rows[0], rows[1:]
    half = max(1, len(body) // 2)
    prefix = block[: block.index(header)]
    return (
        prefix + header + "".join(body[:half]) + "</table>",
        prefix + header + "".join(body[half:]) + "</table>",
    )
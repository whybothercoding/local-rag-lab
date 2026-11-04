"""Text extraction for source documents dropped into a workspace.

Each parser returns plain text. Callers decide how to chunk and index it.
Binary office formats are imported lazily so a missing optional library
fails that file only when someone actually uploads one.
"""
import csv
import json
import os
import re
import subprocess
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET


class ParseError(ValueError):
    """A document was refused or could not be read."""


PARSE_TIMEOUT_SEC = int(os.getenv("PARSE_TIMEOUT_SEC", "60"))
PARSER_MEMORY_LIMIT_BYTES = 1536 * 1024 * 1024
MAX_PDF_PAGES = 1000
ZIP_SUFFIXES = frozenset({".docx", ".pptx", ".xlsx", ".xlsm", ".odt", ".epub"})
# Formats whose whole "parser" is a decode: there is no structure to exploit, so no sandbox is needed. Each used
# to cost a Python start-up (about 43 ms) in a child process, which dominated re-syncs of unchanged workspaces.
PLAIN_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".rst", ".log", ".text"})
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100


class _HTMLText(HTMLParser):
    """Collect visible HTML text and skip script, style, and noscript blocks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        text = data.strip()
        if text:
            self.parts.append(text)


def _extract_plain(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="replace")


def _extract_html(file_path: Path) -> str:
    parser = _HTMLText()
    parser.feed(file_path.read_text(encoding="utf-8", errors="replace"))
    return " ".join(parser.parts)


def _extract_csv(file_path: Path) -> str:
    delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
    lines: list[str] = []
    with open(file_path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle, delimiter=delimiter):
            cells = [cell.strip() for cell in row if cell and cell.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def _extract_json(file_path: Path) -> str:
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def _extract_jsonl(file_path: Path) -> str:
    lines: list[str] = []
    for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            lines.append(json.dumps(json.loads(stripped), ensure_ascii=False))
        except json.JSONDecodeError:
            lines.append(stripped)
    return "\n".join(lines)


def _extract_xml(file_path: Path) -> str:
    root = ET.parse(file_path).getroot()
    return "\n".join(part.strip() for part in root.itertext() if part and part.strip())


def _extract_pdf(file_path: Path) -> str:
    import pypdf

    reader = pypdf.PdfReader(str(file_path))
    if reader.is_encrypted:
        raise ValueError("password-protected PDF; remove the password and upload it again")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ParseError(f"PDF has more than {MAX_PDF_PAGES} pages")
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_docx(file_path: Path) -> str:
    import docx

    document = docx.Document(str(file_path))
    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _extract_pptx(file_path: Path) -> str:
    from pptx import Presentation

    presentation = Presentation(str(file_path))
    slides: list[str] = []
    for index, slide in enumerate(presentation.slides, 1):
        bits: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text_frame.text.strip()
                if text:
                    bits.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                    if cells:
                        bits.append(" | ".join(cells))
        if bits:
            slides.append(f"Slide {index}\n" + "\n".join(bits))
    return "\n\n".join(slides)


def _extract_xlsx(file_path: Path) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(str(file_path), read_only=True, data_only=True)
    try:
        sheets: list[str] = []
        for worksheet in workbook.worksheets:
            rows: list[str] = []
            for row in worksheet.iter_rows(values_only=True):
                cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                sheets.append(f"Sheet {worksheet.title}\n" + "\n".join(rows))
        return "\n\n".join(sheets)
    finally:
        workbook.close()


def _extract_rtf(file_path: Path) -> str:
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    raw = raw.replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")
    raw = raw.replace("{", " ").replace("}", " ")
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    return re.sub(r"\n{3,}", "\n\n", raw)


def _zip_markup_text(file_path: Path, names: list[str] | None, suffixes: tuple[str, ...]) -> str:
    texts: list[str] = []
    with zipfile.ZipFile(file_path) as archive:
        if names is None:
            chosen = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(suffixes) and not name.startswith("__MACOSX")
            )
        else:
            chosen = [name for name in names if name in archive.namelist()]
        for name in chosen:
            try:
                root = ET.fromstring(archive.read(name))
            except ET.ParseError:
                continue
            parts = [part.strip() for part in root.itertext() if part and part.strip()]
            if parts:
                texts.append("\n".join(parts))
    return "\n\n".join(texts)


def _extract_odt(file_path: Path) -> str:
    return _zip_markup_text(file_path, ["content.xml"], ())


def _extract_epub(file_path: Path) -> str:
    return _zip_markup_text(file_path, None, (".xhtml", ".html", ".htm"))


PARSERS = {
    ".txt": _extract_plain,
    ".md": _extract_plain,
    ".markdown": _extract_plain,
    ".rst": _extract_plain,
    ".log": _extract_plain,
    ".text": _extract_plain,
    ".html": _extract_html,
    ".htm": _extract_html,
    ".csv": _extract_csv,
    ".tsv": _extract_csv,
    ".json": _extract_json,
    ".jsonl": _extract_jsonl,
    ".xml": _extract_xml,
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".pptx": _extract_pptx,
    ".xlsx": _extract_xlsx,
    ".xlsm": _extract_xlsx,
    ".rtf": _extract_rtf,
    ".odt": _extract_odt,
    ".epub": _extract_epub,
}

SUPPORTED_EXTENSIONS = frozenset(PARSERS)

# Formats people often drop in that this lab cannot read. The message is
# shown by the ingester instead of a generic "unsupported" line.
UNSUPPORTED_HINTS = {
    ".doc": "legacy Word .doc is not supported; save it as .docx",
    ".xls": "legacy Excel .xls is not supported; save it as .xlsx",
    ".ppt": "legacy PowerPoint .ppt is not supported; save it as .pptx",
    ".png": "images are not OCR'd; export a text, PDF, or Office file",
    ".jpg": "images are not OCR'd; export a text, PDF, or Office file",
    ".jpeg": "images are not OCR'd; export a text, PDF, or Office file",
    ".gif": "images are not OCR'd; export a text, PDF, or Office file",
    ".webp": "images are not OCR'd; export a text, PDF, or Office file",
    ".tif": "images are not OCR'd; export a text, PDF, or Office file",
    ".tiff": "images are not OCR'd; export a text, PDF, or Office file",
    ".bmp": "images are not OCR'd; export a text, PDF, or Office file",
    ".heic": "images are not OCR'd; export a text, PDF, or Office file",
}


def unsupported_reason(suffix: str) -> str:
    """Human-readable reason a file extension was not indexed."""
    ext = suffix.lower()
    if ext in UNSUPPORTED_HINTS:
        return UNSUPPORTED_HINTS[ext]
    return f"unsupported type {ext or '(none)'}"


def _check_archive(file_path: Path) -> None:
    """Refuse office files whose contents would expand far beyond their size on disk (zip bombs)."""
    try:
        with zipfile.ZipFile(file_path) as archive:
            expanded = 0
            for info in archive.infolist():
                expanded += info.file_size
                if expanded > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ParseError("file expands to more than 200 MB when unpacked")
                large = info.file_size > 1024 * 1024
                if large and info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                    raise ParseError("file has a suspicious compression ratio")
    except zipfile.BadZipFile as exc:
        raise ParseError("file is not a valid Office or EPUB archive") from exc


def _extract_in_process(file_path: Path) -> str:
    """Run the matching parser in this process. Called by parser_worker.py, never by the server directly."""
    suffix = file_path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        return ""
    if suffix in ZIP_SUFFIXES:
        _check_archive(file_path)
    return parser(file_path).strip()


def _limit_child_resources() -> None:
    """Cap CPU time and address space inside the parser child, where the OS supports it."""
    try:
        import resource
    except ImportError:
        return

    cpu_seconds = int(PARSE_TIMEOUT_SEC) + 5
    for kind, limit in (
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
        (resource.RLIMIT_AS, (PARSER_MEMORY_LIMIT_BYTES, PARSER_MEMORY_LIMIT_BYTES)),
    ):
        try:
            resource.setrlimit(kind, limit)
        except (ValueError, OSError):
            pass


def _extract_plain_bounded(file_path: Path, limit: int) -> str:
    """Read a plain-text file in this process: the same text the child would return, bounded by `limit`."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as handle:
            text = handle.read(limit + 1)
    except OSError as exc:
        raise ParseError(str(exc) or type(exc).__name__) from exc
    if len(text) > limit:
        raise ParseError(f"larger than {limit} bytes")
    return text.strip()


def extract_text_from_file(file_path: Path) -> str:
    """Extract plain text from a supported file. Unknown types return an empty string.

    Everything but plain text is parsed in a child process with a time limit and resource caps, so a hostile
    file cannot hang the server or exhaust its memory. Plain-text files (PLAIN_SUFFIXES) within the size limit
    are decoded in this process, with identical output. Failures raise ParseError.
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() not in PARSERS:
        return ""
    from rag_lab.config import settings

    if file_path.suffix.lower() in PLAIN_SUFFIXES:
        try:
            small = file_path.stat().st_size <= settings.max_source_bytes
        except OSError as exc:
            raise ParseError(str(exc) or type(exc).__name__) from exc
        if small:
            return _extract_plain_bounded(file_path, settings.max_source_bytes)
    return _extract_in_child(file_path)


def _extract_in_child(file_path: Path) -> str:
    # The worker runs as a module, so the child imports the package from the directory holding it.
    package_parent = str(Path(__file__).resolve().parents[2])
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "rag_lab.ingestion.parser_worker", str(file_path)],
            capture_output=True,
            text=True,
            timeout=PARSE_TIMEOUT_SEC,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": package_parent},
        )
    except subprocess.TimeoutExpired as exc:
        raise ParseError(f"parsing took longer than {PARSE_TIMEOUT_SEC} seconds") from exc
    if completed.returncode != 0:
        reasons = completed.stderr.strip().splitlines()
        raise ParseError(reasons[-1] if reasons else "the document could not be parsed")
    lines = completed.stdout.strip().splitlines()
    if not lines:
        raise ParseError("the parser returned no output")
    return str(json.loads(lines[-1])["text"])

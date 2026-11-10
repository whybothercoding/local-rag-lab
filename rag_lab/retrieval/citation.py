"""Source labels shown in citations: the file, plus the page for PDFs ("policy.pdf (p. 12)")."""
import re

_PAGE_SUFFIX = re.compile(r"\s*\(p\. \d+\)$")


def source_label(source: str, page: int | None = None) -> str:
    return f"{source} (p. {page})" if page else source


def source_file(label: str) -> str:
    """The file a label names, without its page."""
    return _PAGE_SUFFIX.sub("", label)

"""PII redaction shared by the input guardrail and the audit trail."""
import re

_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE = re.compile(r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b")


def redact_pii(text: str) -> str:
    """Replace email addresses and US-style phone numbers with placeholders."""
    redacted = _EMAIL.sub("[REDACTED_EMAIL]", text)
    return _PHONE.sub("[REDACTED_PHONE]", redacted)

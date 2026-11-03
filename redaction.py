"""PII redaction shared by the input guardrail, the outbound filter, and the audit trail.

Covers email addresses, formatted phone numbers (US-style and international), US social security
numbers, payment card numbers (Luhn-checked), and IBANs (mod-97 checked). Checksummed identifiers are
only redacted when the checksum passes, and a bare run of digits is never treated as a phone number,
so invoice numbers and amounts survive.
"""
import re

_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_US = re.compile(r"(?<![\w+])(?:\+?1[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
_PHONE_INTL = re.compile(r"(?<!\w)\+\d{1,3}(?:[-.\s]?\(?\d{1,4}\)?){2,5}(?!\w)")
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{3,4}){2,8}(?:[ ]?[A-Z0-9]{1,3})?\b")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def _iban_ok(text: str) -> bool:
    compact = text.replace(" ", "")
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(char, 36)) for char in rearranged)
    return int(numeric) % 97 == 1


def _redact_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group(0)


def _redact_iban(match: re.Match[str]) -> str:
    return "[REDACTED_IBAN]" if _iban_ok(match.group(0)) else match.group(0)


def redact_pii(text: str) -> str:
    """Replace emails, phone numbers, SSNs, card numbers, and IBANs with placeholders."""
    redacted = _EMAIL.sub("[REDACTED_EMAIL]", text)
    redacted = _IBAN.sub(_redact_iban, redacted)
    redacted = _CARD.sub(_redact_card, redacted)
    redacted = _SSN.sub("[REDACTED_SSN]", redacted)
    redacted = _PHONE_INTL.sub("[REDACTED_PHONE]", redacted)
    return _PHONE_US.sub("[REDACTED_PHONE]", redacted)

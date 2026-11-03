"""PII redaction shared by the input guardrail, the outbound filter, and the audit trail.

Covers email addresses, formatted phone numbers (US, international, Swiss and Greek national formats),
US social security numbers, Swiss AHV/AVS numbers (EAN-13 checked), Greek AMKA numbers (date and Luhn
checked), payment card numbers (Luhn-checked), and IBANs (mod-97 checked). Checksummed identifiers are
only redacted when the checksum passes, and a bare run of digits is never treated as a phone number,
so invoice numbers and amounts survive.
"""
import re

_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_US = re.compile(r"(?<![\w+])(?:\+?1[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
_PHONE_INTL = re.compile(r"(?<!\w)\+\d{1,3}(?:[-.\s]?\(?\d{1,4}\)?){2,5}(?!\w)")
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
# Swiss social insurance number: 756.XXXX.XXXX.XX, dotted or not.
_AHV = re.compile(r"(?<![\w.])756(?:\.?\d{4}\.?\d{4}\.?\d{2})(?![\w.])")
# Greek social security number: 11 digits, DDMMYY then 5 digits, Luhn-valid.
_AMKA = re.compile(r"(?<![\w.,-])\d{11}(?![\w.,-])")
# National phone formats, separators required so amounts and reference numbers are never touched:
# Swiss 044 123 45 67 / 079.123.45.67; Greek 210 1234567 / 694 123 4567.
_PHONE_CH = re.compile(r"(?<![\w+.])0\d{2}[ .]\d{3}[ .]\d{2}[ .]\d{2}(?![\w.])")
_PHONE_GR = re.compile(r"(?<![\w+.])(?:2\d{2}|69\d)(?:[ .-]\d{3}[ .-]?\d{4}|[ .-]\d{7})(?![\w.])")
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


def _ahv_ok(digits: str) -> bool:
    """EAN-13 check digit: weights 1 and 3 alternating over the first twelve digits."""
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == int(digits[12])


def _amka_ok(digits: str) -> bool:
    day, month = int(digits[0:2]), int(digits[2:4])
    return 1 <= day <= 31 and 1 <= month <= 12 and _luhn_ok(digits)


def _redact_ahv(match: re.Match[str]) -> str:
    digits = match.group(0).replace(".", "")
    return "[REDACTED_AHV]" if len(digits) == 13 and _ahv_ok(digits) else match.group(0)


def _redact_amka(match: re.Match[str]) -> str:
    return "[REDACTED_AMKA]" if _amka_ok(match.group(0)) else match.group(0)


def _redact_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group(0)


def _redact_iban(match: re.Match[str]) -> str:
    return "[REDACTED_IBAN]" if _iban_ok(match.group(0)) else match.group(0)


def redact_pii(text: str) -> str:
    """Replace emails, phone numbers, SSNs, AHV and AMKA numbers, card numbers, and IBANs with placeholders."""
    redacted = _EMAIL.sub("[REDACTED_EMAIL]", text)
    redacted = _IBAN.sub(_redact_iban, redacted)
    redacted = _AHV.sub(_redact_ahv, redacted)
    redacted = _AMKA.sub(_redact_amka, redacted)
    redacted = _CARD.sub(_redact_card, redacted)
    redacted = _SSN.sub("[REDACTED_SSN]", redacted)
    redacted = _PHONE_INTL.sub("[REDACTED_PHONE]", redacted)
    redacted = _PHONE_CH.sub("[REDACTED_PHONE]", redacted)
    redacted = _PHONE_GR.sub("[REDACTED_PHONE]", redacted)
    return _PHONE_US.sub("[REDACTED_PHONE]", redacted)

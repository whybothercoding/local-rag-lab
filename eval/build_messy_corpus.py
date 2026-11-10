#!/usr/bin/env python3
"""Regenerate eval/corpus/messy/: the non-Markdown fixtures (DOCX, XLSX, PDF) behind the `messy` suite.

The files are committed so CI needs no generator; run this only after changing the content below
(`python eval/build_messy_corpus.py`), then re-run `make eval`. Each file exercises one ingestion
failure mode: a Word table with an empty cell, a spreadsheet with sparse rows, a PDF whose first page
is blank (page numbers must not shift), and a PDF table that flattens to one line per row.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "corpus" / "messy"


def docx_travel() -> None:
    import docx

    doc = docx.Document()
    doc.add_heading("Travel Policy", level=1)
    doc.add_paragraph("Hotel caps are set per grade and city.")
    doc.add_heading("Hotel limits", level=2)
    rows = [
        ("Grade", "City", "Hotel cap EUR", "Notes"),
        ("G5", "Paris", "240", ""),
        ("G5", "Berlin", "190", ""),
        ("G7", "Paris", "310", "approval needed"),
        ("G7", "Berlin", "260", ""),
    ]
    table = doc.add_table(rows=len(rows), cols=4)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            table.cell(r, c).text = text
    doc.add_heading("Booking", level=2)
    doc.add_paragraph("Travel must be booked at least 14 days ahead.")
    doc.save(str(OUT / "travel_caps.docx"))


def xlsx_headcount() -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Q3 headcount"
    ws.append(["Team", "Open roles", "Hiring freeze", "Budget kEUR"])
    ws.append(["Platform", 3, None, 420])
    ws.append(["Support", None, "yes", 150])
    ws.append(["Data", 2, "no", None])
    wb.save(str(OUT / "headcount.xlsx"))


def pdf(path: Path, pages: list[list[str]]) -> None:
    """A minimal text PDF: one list of lines per page; an empty list is a page with no text layer."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, lines in enumerate(pages):
        body = "BT /F1 12 Tf 20 360 Td 18 TL " + " T* ".join(f"({line}) Tj" for line in lines) + " ET" if lines else ""
        stream = body.encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 500 400] /Contents {5 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(out)


def pdfs() -> None:
    pdf(OUT / "security_handbook.pdf", [
        [],  # blank cover page: page numbers below must still be right
        ["Password rotation", "Passwords must be rotated every 90 days."],
        ["Incident reporting", "Report security incidents to the SOC within 24 hours."],
        ["Backups", "Nightly backups are retained for 35 days."],
        ["Contacts", "The SOC desk is reachable on extension 4471."],
    ])
    pdf(OUT / "mileage.pdf", [[
        "Mileage rates",
        "Vehicle EUR per km",
        "Car 0.42",
        "Motorbike 0.21",
        "Bicycle 0.10",
    ]])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    docx_travel()
    xlsx_headcount()
    pdfs()
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")

import pytest
from pathlib import Path
from ingest import extract_text_from_file

def test_extract_text_and_md(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello plain text world", encoding="utf-8")
    assert extract_text_from_file(txt_file) == "Hello plain text world"

    md_file = tmp_path / "sample.md"
    md_file.write_text("# Title\nMarkdown content", encoding="utf-8")
    assert "Markdown content" in extract_text_from_file(md_file)


def test_extract_docx(tmp_path: Path):
    import docx
    doc_path = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_paragraph("First paragraph.")
    doc.add_paragraph("Second paragraph.")
    doc.save(str(doc_path))

    extracted = extract_text_from_file(doc_path)
    assert "First paragraph." in extracted
    assert "Second paragraph." in extracted


def test_extract_pdf(tmp_path: Path):
    import pypdf
    pdf_path = tmp_path / "sample.pdf"
    
    # Generate minimal valid PDF using pypdf writer
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Blank page returns empty string without error
    extracted = extract_text_from_file(pdf_path)
    assert isinstance(extracted, str)

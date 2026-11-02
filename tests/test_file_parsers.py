import zipfile
from pathlib import Path
import pytest

import parsers
from parsers import ParseError, extract_text_from_file, unsupported_reason


def test_extract_text_and_md(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello plain text world", encoding="utf-8")
    assert extract_text_from_file(txt_file) == "Hello plain text world"

    md_file = tmp_path / "sample.md"
    md_file.write_text("# Title\nMarkdown content", encoding="utf-8")
    assert "Markdown content" in extract_text_from_file(md_file)


def test_extract_html_strips_scripts_and_styles(tmp_path: Path):
    html_file = tmp_path / "page.html"
    html_file.write_text(
        "<html><head><style>body { color: red; }</style><script>alert('xss')</script></head>"
        "<body><h1>Policy Header</h1><noscript>Enable JS</noscript><p>Visible paragraph text.</p></body></html>",
        encoding="utf-8",
    )
    extracted = extract_text_from_file(html_file)
    assert "Policy Header" in extracted
    assert "Visible paragraph text." in extracted
    assert "alert" not in extracted
    assert "color: red" not in extracted
    assert "Enable JS" not in extracted


def test_extract_csv_and_tsv(tmp_path: Path):
    csv_file = tmp_path / "table.csv"
    csv_file.write_text("Name,Role,Clearance\nAlice,Admin,secret\n,,\nBob,Reviewer,internal\n", encoding="utf-8")
    csv_text = extract_text_from_file(csv_file)
    assert "Name | Role | Clearance" in csv_text
    assert "Alice | Admin | secret" in csv_text
    assert "Bob | Reviewer | internal" in csv_text

    tsv_file = tmp_path / "table.tsv"
    tsv_file.write_text("Dept\tBudget\nEngineering\t500000\n", encoding="utf-8")
    tsv_text = extract_text_from_file(tsv_file)
    assert "Dept | Budget" in tsv_text
    assert "Engineering | 500000" in tsv_text


def test_extract_json_and_jsonl(tmp_path: Path):
    json_file = tmp_path / "config.json"
    json_file.write_text('{"service":"rag","port":8501}', encoding="utf-8")
    json_text = extract_text_from_file(json_file)
    assert '"service": "rag"' in json_text

    bad_json = tmp_path / "malformed.json"
    bad_json.write_text("{not valid json but still text}", encoding="utf-8")
    assert "{not valid json but still text}" in extract_text_from_file(bad_json)

    jsonl_file = tmp_path / "events.jsonl"
    jsonl_file.write_text('{"seq":1,"event":"login"}\n\nraw_unparsed_line\n{"seq":2}\n', encoding="utf-8")
    jsonl_text = extract_text_from_file(jsonl_file)
    assert '{"seq": 1, "event": "login"}' in jsonl_text
    assert "raw_unparsed_line" in jsonl_text
    assert '{"seq": 2}' in jsonl_text


def test_extract_xml_and_rtf(tmp_path: Path):
    xml_file = tmp_path / "doc.xml"
    xml_file.write_text("<root><section><title>Security</title><body_text>Zero Trust</body_text></section></root>", encoding="utf-8")
    xml_text = extract_text_from_file(xml_file)
    assert "Security" in xml_text
    assert "Zero Trust" in xml_text

    rtf_file = tmp_path / "memo.rtf"
    rtf_file.write_text(r"{\rtf1\ansi\b Confidential Memo\b0\par Retain for 183 days. \{escaped\}}", encoding="utf-8")
    rtf_text = extract_text_from_file(rtf_file)
    assert "Confidential Memo" in rtf_text
    assert "Retain for 183 days." in rtf_text


def test_extract_odt_and_epub(tmp_path: Path):
    odt_file = tmp_path / "handbook.odt"
    with zipfile.ZipFile(odt_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "content.xml",
            '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
            "<text:p>ODT policy paragraph.</text:p></office:document-content>",
        )
    assert "ODT policy paragraph." in extract_text_from_file(odt_file)

    epub_file = tmp_path / "manual.epub"
    with zipfile.ZipFile(epub_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("__MACOSX/._chapter1.xhtml", "<html/>")
        zf.writestr("OEBPS/chapter1.xhtml", "<html><body><p>Chapter One EPUB content.</p></body></html>")
        zf.writestr("OEBPS/broken.html", "<unclosed")
    assert "Chapter One EPUB content." in extract_text_from_file(epub_file)


def test_extract_docx_paragraphs_and_tables(tmp_path: Path):
    import docx

    doc_path = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_paragraph("First paragraph.")
    doc.add_paragraph("Second paragraph.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Tier"
    table.cell(0, 1).text = "SLA"
    table.cell(1, 0).text = "Tier-1"
    table.cell(1, 1).text = "5 minutes"
    doc.save(str(doc_path))

    extracted = extract_text_from_file(doc_path)
    assert "First paragraph." in extracted
    assert "Second paragraph." in extracted
    assert "Tier | SLA" in extracted
    assert "Tier-1 | 5 minutes" in extracted


def test_extract_pdf_and_limits(tmp_path: Path, monkeypatch):
    import pypdf

    pdf_path = tmp_path / "sample.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    extracted = extract_text_from_file(pdf_path)
    assert isinstance(extracted, str)

    # Encrypted PDF is refused cleanly
    enc_pdf = tmp_path / "encrypted.pdf"
    enc_writer = pypdf.PdfWriter()
    enc_writer.add_blank_page(width=72, height=72)
    enc_writer.encrypt("secret-pass")
    with open(enc_pdf, "wb") as f:
        enc_writer.write(f)
    with pytest.raises(ParseError, match="password-protected PDF"):
        extract_text_from_file(enc_pdf)

    # PDF exceeding MAX_PDF_PAGES raises ParseError
    monkeypatch.setattr(parsers, "MAX_PDF_PAGES", 1)
    two_page_pdf = tmp_path / "two_pages.pdf"
    w2 = pypdf.PdfWriter()
    w2.add_blank_page(width=72, height=72)
    w2.add_blank_page(width=72, height=72)
    with open(two_page_pdf, "wb") as f:
        w2.write(f)
    with pytest.raises(ParseError, match="more than 1 pages"):
        parsers._extract_pdf(two_page_pdf)


def test_archive_max_uncompressed_bytes_and_unsupported_hints(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(parsers, "MAX_ARCHIVE_UNCOMPRESSED_BYTES", 100)
    oversized = tmp_path / "oversized.odt"
    with zipfile.ZipFile(oversized, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("content.xml", "<p>" + ("A" * 200) + "</p>")
    with pytest.raises(ParseError, match="200 MB"):
        parsers._check_archive(oversized)

    assert "save it as .docx" in unsupported_reason(".doc")
    assert "save it as .xlsx" in unsupported_reason(".xls")
    assert "save it as .pptx" in unsupported_reason(".ppt")
    assert "not OCR'd" in unsupported_reason(".png")
    assert "unsupported type .bin" in unsupported_reason(".bin")
    assert extract_text_from_file(tmp_path / "unknown.bin") == ""

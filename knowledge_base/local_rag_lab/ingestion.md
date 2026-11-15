# Workspaces and document ingestion

A workspace is a folder of documents. Place files under `knowledge_base/<workspace>/` or upload them under `uploads/<workspace>/`.
The two folders merge into one workspace, and when both hold the same relative path the `knowledge_base` copy wins.
Workspace names are sanitized, so `Accounting & Tax` becomes `accounting_tax`. Indexes are named `ws_<id>`.

## Commands

`make workspaces` lists workspaces with file and chunk counts.
`make create-workspace WS=<name>` and `make delete-workspace WS=<name> YES=1` manage them. Deleting needs an admin and prunes grants.
`make sync ADMIN=<admin>` indexes every workspace. Add `WS=<name>` for one, and `FORCE=1` to re-embed everything.
`FORCE=1` swaps the Chroma collection atomically.

## Supported formats

Markdown, plain text, PDF, DOCX, PPTX, XLSX and CSV are read.
Plain text decodes in-process. Other formats are parsed in a sandboxed child process with resource limits.
Table rows are written one per line with their column names, for example `Grade: G5 | Hotel cap EUR: 240`.
DOCX keeps paragraphs and tables in document order and turns heading styles into `#` lines.
PDFs are chunked page by page, and citations carry the page, for example `annual.pdf (p. 12)`.

## Limits by design

There is no OCR. A PDF with no text layer is refused with a note.
Tables inside a PDF flatten to running text. Images, charts and legacy `.doc`, `.xls` and `.ppt` files are not read.

## Incremental re-sync

Unchanged files are skipped using SHA-256 manifests.
A re-sync embeds only chunks whose exact text is new, and reuses stored vectors of the same workspace, model fingerprint and passage prefix.
Changing the parser or chunker output means bumping `CHUNKER_VERSION`, which re-indexes everything once.
Each index records its `embedding_fingerprint`. After a model change, dense retrieval is skipped until `make sync FORCE=1`.

## Model profiles

`MODEL_PROFILE` selects the embedding, reranker and NLI models. The default is `multilingual`. The others are `multilingual-large` and `en`.
Thresholds are calibrated per profile with `make calibrate-models`. Never reuse another model's numbers.

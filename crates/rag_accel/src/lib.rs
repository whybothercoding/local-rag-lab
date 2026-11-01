mod audit;
mod chunker;
mod sanitizer;

use pyo3::prelude::*;

#[pyfunction]
fn scan_chunk(text: &str) -> (bool, Option<String>) {
    let res = sanitizer::scan_chunk(text);
    (res.is_safe, res.reason)
}

#[pyfunction]
fn scan_chunks(
    chunks: Vec<String>,
    citations: Vec<String>,
) -> (Vec<String>, Vec<String>, Vec<(String, String)>) {
    let mut safe_chunks = Vec::new();
    let mut safe_citations = Vec::new();
    let mut rejected_alerts = Vec::new();

    let count = chunks.len();
    for i in 0..count {
        let chunk = &chunks[i];
        let citation = if i < citations.len() {
            &citations[i]
        } else {
            "document"
        };
        let res = sanitizer::scan_chunk(chunk);
        if res.is_safe {
            safe_chunks.push(chunk.clone());
            safe_citations.push(citation.to_string());
        } else {
            rejected_alerts.push((
                citation.to_string(),
                res.reason.unwrap_or_else(|| "unsafe_context".to_string()),
            ));
        }
    }

    (safe_chunks, safe_citations, rejected_alerts)
}

#[pyfunction]
fn split_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    chunker::split_text(text, chunk_size, chunk_overlap)
}

#[pymodule]
fn rag_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(scan_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(split_text, m)?)?;
    Ok(())
}

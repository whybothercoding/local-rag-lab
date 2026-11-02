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

    let count = chunks.len().max(citations.len());
    for i in 0..count {
        let chunk = chunks.get(i).map(|s| s.as_str()).unwrap_or("");
        let citation = citations
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("[CHUNK-{}] unknown", i));
        let res = sanitizer::scan_chunk(chunk);
        if res.is_safe {
            safe_chunks.push(chunk.to_string());
            safe_citations.push(citation);
        } else {
            rejected_alerts.push((
                citation,
                res.reason.unwrap_or_else(|| "unknown_violation".to_string()),
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


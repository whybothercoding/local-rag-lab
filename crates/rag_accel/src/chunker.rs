/// High-throughput character-based text splitter with configurable overlap.
pub fn split_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    if text.is_empty() || chunk_size == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let total_chars = chars.len();

    if total_chars <= chunk_size {
        return vec![text.to_string()];
    }

    let step = if chunk_size > chunk_overlap {
        chunk_size - chunk_overlap
    } else {
        1
    };

    let mut start = 0;
    while start < total_chars {
        let end = (start + chunk_size).min(total_chars);
        let chunk: String = chars[start..end].iter().collect();
        let trimmed = chunk.trim().to_string();
        if !trimmed.is_empty() {
            chunks.push(trimmed);
        }
        if end >= total_chars {
            break;
        }
        start += step;
    }

    chunks
}

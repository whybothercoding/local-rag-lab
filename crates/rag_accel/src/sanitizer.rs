use regex::{Regex, RegexSet};
use std::sync::LazyLock;

pub struct ScanResult {
    pub is_safe: bool,
    pub reason: Option<String>,
}

// The one pattern list both scanners use, compiled in from the Python package (scope, reason, regex).
// Every pattern runs case-insensitively; Python applies re.IGNORECASE to the same regexes.
const PATTERN_FILE: &str = include_str!("../../../rag_lab/security/injection_patterns.tsv");

static PATTERNS: LazyLock<Vec<(String, String)>> = LazyLock::new(|| {
    PATTERN_FILE
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.splitn(3, '\t');
            let _scope = fields.next().expect("pattern scope");
            let reason = fields.next().expect("pattern reason");
            let regex = fields.next().expect("pattern regex");
            (format!("(?i){regex}"), reason.to_string())
        })
        .collect()
});

static REGEX_SET: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new(PATTERNS.iter().map(|(pat, _)| pat.as_str())).expect("Failed to compile RegexSet")
});

static REGEX_RULES: LazyLock<Vec<(Regex, String)>> = LazyLock::new(|| {
    PATTERNS
        .iter()
        .map(|(pat, reason)| (Regex::new(pat).expect("Failed to compile Regex"), reason.clone()))
        .collect()
});

/// Pattern reasons in file order, so tests can prove both scanners load the same list.
pub fn pattern_reasons() -> Vec<String> {
    PATTERNS.iter().map(|(_, reason)| reason.clone()).collect()
}

fn is_invisible(ch: char) -> bool {
    matches!(
        ch as u32,
        0x00AD
            | 0x200B..=0x200F
            | 0x202A..=0x202E
            | 0x2060..=0x2063
            | 0x2066..=0x2069
            | 0xFEFF
    )
}

pub fn scan_chunk(text: &str) -> ScanResult {
    let cleaned: std::borrow::Cow<'_, str> = if text.chars().any(is_invisible) {
        std::borrow::Cow::Owned(text.chars().filter(|&c| !is_invisible(c)).collect())
    } else {
        std::borrow::Cow::Borrowed(text)
    };
    let target = cleaned.as_ref();

    // Single-pass automaton scan across all patterns simultaneously; respects word boundaries and priority order.
    let matches = REGEX_SET.matches(target);
    if let Some(idx) = matches.into_iter().next() {
        let (ref re, ref reason) = REGEX_RULES[idx];
        if let Some(mat) = re.find(target) {
            return ScanResult {
                is_safe: false,
                reason: Some(format!("{}:{}", reason, mat.as_str())),
            };
        }
    }

    ScanResult {
        is_safe: true,
        reason: None,
    }
}


use regex::{Regex, RegexSet};
use std::sync::LazyLock;

pub struct ScanResult {
    pub is_safe: bool,
    pub reason: Option<String>,
}

static PATTERNS: &[(&str, &str)] = &[
    (
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directives|rules|guidelines|prompts)\b",
        "instruction_override",
    ),
    (
        r"(?i)\bdisregard\s+(all\s+)?(prior|previous|above)\s+(directives|instructions|rules|prompts)\b",
        "directive_wipe",
    ),
    (
        r"(?i)\bsystem\s+prompt\s+override\b",
        "system_override",
    ),
    (
        r"(?i)\bbypass\s+(all\s+)?security\s+(rules|filters|guidelines)\b",
        "security_bypass",
    ),
    (
        r"(?i)\bnew\s+(system\s+)?instruction\s*:",
        "new_instruction",
    ),
    (
        r"(?i)\bdo\s+not\s+follow\s+(the\s+)?(original|initial|prior)\s+instructions\b",
        "instruction_rejection",
    ),
    (
        r"(?i)\byou\s+are\s+now\s+in\s+([a-z0-9_-]+\s+)?mode\b",
        "role_override",
    ),
    (
        r"(?i)\b(jailbreak|unfiltered\s+mode|dan\s+mode)\b",
        "jailbreak_signature",
    ),
    (
        r"(?i)<\|im_(start|end)\|>",
        "chatml_delimiter_spoofing",
    ),
    (
        r"(?i)\[(?:SYSTEM|ADMIN|DEVELOPER)[^\]]*\]",
        "role_bracket_spoofing",
    ),
    (
        r"(?i)</?SYS>>",
        "llama_sys_delimiter_spoofing",
    ),
    (
        r"(?i)###\s*(?:System|Instruction):",
        "markdown_role_spoofing",
    ),
    (
        r"(?i)!\[.*?\]\(https?://[^\s)]+\?[^\s)]*=(?:key|token|secret|password|history|context)",
        "image_exfiltration_beacon",
    ),
    (
        r"(?i)<script\b|javascript:",
        "script_payload",
    ),
];

static REGEX_SET: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new(PATTERNS.iter().map(|(pat, _)| *pat)).expect("Failed to compile RegexSet")
});

static REGEX_RULES: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    PATTERNS
        .iter()
        .map(|(pat, reason)| (Regex::new(pat).expect("Failed to compile Regex"), *reason))
        .collect()
});

pub fn scan_chunk(text: &str) -> ScanResult {
    // Single-pass automaton scan across all patterns simultaneously; respects word boundaries and priority order.
    let matches = REGEX_SET.matches(text);
    if let Some(idx) = matches.into_iter().next() {
        let (ref re, reason) = REGEX_RULES[idx];
        if let Some(mat) = re.find(text) {
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


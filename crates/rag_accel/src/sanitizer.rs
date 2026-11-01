use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use regex::Regex;
use std::sync::LazyLock;

pub struct ScanResult {
    pub is_safe: bool,
    pub reason: Option<String>,
}

// Substring signatures mapped directly to category reasons
static AHO_ITEMS: &[(&str, &str)] = &[
    ("system prompt override", "system_override"),
    ("<|im_start|>", "chatml_delimiter_spoofing"),
    ("<|im_end|>", "chatml_delimiter_spoofing"),
    ("<<sys>>", "llama_sys_delimiter_spoofing"),
    ("<</sys>>", "llama_sys_delimiter_spoofing"),
    ("dan mode", "jailbreak_signature"),
    ("unfiltered mode", "jailbreak_signature"),
    ("jailbreak", "jailbreak_signature"),
    ("javascript:", "script_payload"),
    ("<script", "script_payload"),
];

static AC: LazyLock<(AhoCorasick, Vec<&'static str>)> = LazyLock::new(|| {
    let patterns: Vec<&str> = AHO_ITEMS.iter().map(|(p, _)| *p).collect();
    let reasons: Vec<&'static str> = AHO_ITEMS.iter().map(|(_, r)| *r).collect();
    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .build(patterns)
        .expect("Failed to build Aho-Corasick automaton");
    (ac, reasons)
});

// Regex rules for patterns with variable whitespace or bracket syntax
static REGEX_RULES: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (
            Regex::new(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directives|rules|guidelines|prompts)\b").unwrap(),
            "instruction_override",
        ),
        (
            Regex::new(r"(?i)\bdisregard\s+(all\s+)?(prior|previous|above)\s+(directives|instructions|rules|prompts)\b").unwrap(),
            "directive_wipe",
        ),
        (
            Regex::new(r"(?i)\bsystem\s+prompt\s+override\b").unwrap(),
            "system_override",
        ),
        (
            Regex::new(r"(?i)\bbypass\s+(all\s+)?security\s+(rules|filters|guidelines)\b").unwrap(),
            "security_bypass",
        ),
        (
            Regex::new(r"(?i)\bnew\s+(system\s+)?instruction\s*:").unwrap(),
            "new_instruction",
        ),
        (
            Regex::new(r"(?i)\bdo\s+not\s+follow\s+(the\s+)?(original|initial|prior)\s+instructions\b").unwrap(),
            "instruction_rejection",
        ),
        (
            Regex::new(r"(?i)\byou\s+are\s+now\s+in\s+([a-z0-9_-]+\s+)?mode\b").unwrap(),
            "role_override",
        ),
        (
            Regex::new(r"(?i)\b(jailbreak|unfiltered\s+mode|dan\s+mode)\b").unwrap(),
            "jailbreak_signature",
        ),
        (
            Regex::new(r"(?i)<\|im_(start|end)\|>").unwrap(),
            "chatml_delimiter_spoofing",
        ),
        (
            Regex::new(r"(?i)\[(?:SYSTEM|ADMIN|DEVELOPER)[^\]]*\]").unwrap(),
            "role_bracket_spoofing",
        ),
        (
            Regex::new(r"(?i)</?SYS>>").unwrap(),
            "llama_sys_delimiter_spoofing",
        ),
        (
            Regex::new(r"(?i)###\s*(?:System|Instruction):").unwrap(),
            "markdown_role_spoofing",
        ),
        (
            Regex::new(r"(?i)!\[.*?\]\(https?://[^\s)]+\?[^\s)]*=(?:key|token|secret|password|history|context)").unwrap(),
            "image_exfiltration_beacon",
        ),
        (
            Regex::new(r"(?i)<script\b|javascript:").unwrap(),
            "script_payload",
        ),
    ]
});

pub fn scan_chunk(text: &str) -> ScanResult {
    // 1. Fast path: Aho-Corasick automaton with exact category reason mapping
    let (ref ac, ref reasons) = *AC;
    if let Some(mat) = ac.find(text) {
        let reason = reasons[mat.pattern()];
        let matched_text = &text[mat.start()..mat.end()];
        return ScanResult {
            is_safe: false,
            reason: Some(format!("{}:{}", reason, matched_text)),
        };
    }

    // 2. Full regex rules for syntax and boundary structures
    for (re, reason) in REGEX_RULES.iter() {
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

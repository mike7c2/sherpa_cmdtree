//! cmdtree — a voice command-tree grammar with free-text slots.
//!
//! * [`grammar`] — the YAML-backed command tree (raw types + compilation/validation).
//! * [`parser`] — match a transcript against the tree, extracting slot values.
//! * [`hotwords`] — derive a sherpa-onnx bias list from a grammar.
//! * `asr` (feature `sherpa`) — the sherpa-onnx recognizer.
//! * `pulse` (feature `daemon`) — blocking 16 kHz mono capture via libpulse.
//! * `daemon` (feature `daemon`) — push-to-talk capture + ASR + parse + emit.

pub mod grammar;
pub mod hotkey;
pub mod hotwords;
pub mod parser;
pub mod wav;

#[cfg(feature = "sherpa")]
pub mod asr;

#[cfg(feature = "daemon")]
pub mod daemon;

#[cfg(feature = "daemon")]
pub mod pulse;

pub use grammar::{Command, Comp, Grammar, GrammarError, OneOf};
pub use parser::Match;

/// Normalize free text into comparable word tokens: lowercase ASCII alphanumerics,
/// apostrophes kept inside words, everything else is a separator. The transducer
/// emits UPPERCASE with leading spaces; this collapses that to a clean word list.
pub fn normalize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            cur.push(c.to_ascii_lowercase());
        } else if c == '\'' && !cur.is_empty() {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

//! cmdtree — a voice command-tree grammar with free-text slots.
//!
//! * [`grammar`] — the YAML-backed command tree (raw types + compilation/validation).
//! * [`parser`] — match a transcript against the tree, extracting slot values.
//! * [`phonetics`] — espeak G2P + distinctive-feature-weighted phonetic distance.
//! * [`assess`]   — predict how reliably an ASR will pick up a given command tree.
//! * `asr` (feature `daemon` or `sherpa`) — ASR backends: external command + native sherpa-onnx.
//! * `daemon` (feature `daemon`) — push-to-talk: hotkey + mic capture + ASR + parse.

pub mod assess;
pub mod grammar;
pub mod hotkey;
pub mod hotwords;
pub mod parser;
pub mod phonetics;
pub mod wav;

#[cfg(any(feature = "daemon", feature = "sherpa"))]
pub mod asr;

#[cfg(feature = "sherpa")]
pub mod kws;

#[cfg(feature = "daemon")]
pub mod daemon;

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

//! Derive a sherpa-onnx **hotword / contextual-biasing** list straight from the
//! command tree, so the bias phrases always match the grammar with no separate list
//! to maintain. Feeding these to the recognizer boosts the log-probs of exactly the
//! words the parser cares about — routing keywords, enum option values, and the
//! free-text slot's end anchor.

use crate::grammar::{Comp, Grammar};

/// Single-word literal runs that are bare function words carry no useful acoustic
/// identity on their own, so we don't bias them (multi-word runs and enum options
/// are always kept — including short content words like `on`/`off`/`up`).
const RUN_STOPWORDS: &[&str] = &[
    "a", "an", "the", "to", "of", "for", "and", "or", "in", "at", "with",
];

fn keep_run(run: &[String]) -> bool {
    run.len() > 1 || run.first().is_some_and(|w| !RUN_STOPWORDS.contains(&w.as_str()))
}

/// The biasable surface phrases of a grammar: every maximal run of literal words
/// (minus bare stop-words) and every enum option, de-duplicated in first-seen order.
pub fn hotwords(grammar: &Grammar) -> Vec<String> {
    let mut phrases: Vec<String> = Vec::new();
    for cmd in &grammar.commands {
        let mut run: Vec<String> = Vec::new();
        for comp in &cmd.pattern {
            if let Comp::Word(w) = comp {
                run.push(w.clone());
                continue;
            }
            if keep_run(&run) {
                phrases.push(run.join(" "));
            }
            run.clear();
            if let Comp::Enum(en) = comp {
                for opt in &en.options {
                    phrases.push(opt.join(" "));
                }
            }
            // Comp::Slot contributes nothing — it's free text.
        }
        if keep_run(&run) {
            phrases.push(run.join(" "));
        }
    }

    let mut seen = std::collections::HashSet::new();
    phrases.into_iter().filter(|p| seen.insert(p.clone())).collect()
}

/// Render the sherpa-onnx hotwords file: one phrase per line. When `score` is given
/// it is appended as a per-phrase boost (`phrase :score`); otherwise the recognizer's
/// global `hotwords_score` applies.
pub fn hotwords_file(grammar: &Grammar, score: Option<f32>) -> String {
    let mut s = String::new();
    for phrase in hotwords(grammar) {
        match score {
            Some(v) => s.push_str(&format!("{phrase} :{v}\n")),
            None => {
                s.push_str(&phrase);
                s.push('\n');
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::Grammar;

    #[test]
    fn derives_phrases_and_options_but_not_stopwords_or_slot() {
        let y = r#"
commands:
  - name: light
    pattern:
      - {lit: switch}
      - {one_of: {name: state, options: [on, off]}}
      - {lit: the}
      - {one_of: {name: device, options: ["kitchen light", "desk lamp"]}}
  - name: note
    pattern:
      - {lit: "take a note"}
      - {slot: text}
      - {lit: "full stop"}
"#;
        let g = Grammar::from_yaml(y).unwrap();
        let hw = hotwords(&g);
        // multi-word runs, content keywords, enum options, and the slot anchor are in
        assert!(hw.contains(&"switch".to_string()));
        assert!(hw.contains(&"on".to_string())); // short enum option kept
        assert!(hw.contains(&"off".to_string()));
        assert!(hw.contains(&"kitchen light".to_string()));
        assert!(hw.contains(&"take a note".to_string()));
        assert!(hw.contains(&"full stop".to_string())); // the end anchor
        // the bare "the" between enums is dropped; the free-text slot contributes nothing
        assert!(!hw.contains(&"the".to_string()));
        assert!(!hw.iter().any(|p| p.contains("text")));
        // per-phrase score formatting
        assert!(hotwords_file(&g, Some(2.5)).contains("kitchen light :2.5"));
    }
}

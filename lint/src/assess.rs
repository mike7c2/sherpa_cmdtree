//! Predict how reliably an ASR will pick up a command tree, by checking the words
//! the parser must TELL APART for phonetic confusability (feature-weighted, via
//! [`crate::phonetics`]) plus anchor robustness (length + commonness).
//!
//! This is a PREDICTIVE PROXY, not a measurement of a specific model. For ground
//! truth, synthesize each command and run it back through the recognizer.

use serde::Serialize;

use cmdtree::grammar::{Command, Comp, Grammar};

use crate::phonetics::{available, distance, ipa, phone_count};

/// distances below this are "confusable"; below CLOSE are "somewhat close".
pub const CONFUSABLE: f64 = 0.28;
pub const CLOSE: f64 = 0.45;
const SHORT_PHONES: usize = 2;

/// A slot end-anchor drawn from here is likely to also be spoken inside the free
/// text, closing the slot early. (For a rigorous score use corpus frequencies.)
const COMMON: &[&str] = &[
    "a", "an", "the", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "is", "are", "was", "were", "be",
    "been", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "should",
    "to", "of", "in", "on", "off", "up", "down", "out", "over", "under", "for", "and", "or",
    "but", "so", "if", "then", "than", "as", "at", "by", "with", "from", "into", "onto", "about",
    "here", "there", "now", "back", "again", "just", "only", "also", "very", "more", "most",
    "some", "any", "one", "two", "three", "four", "five", "no", "yes", "ok", "okay", "please",
    "thanks", "go", "stop", "set", "get", "put", "make", "take", "give", "send", "show", "tell",
    "find", "open", "close", "turn", "play", "call", "add", "done", "next", "last", "new", "old",
    "all", "not", "what", "when", "where", "who", "how", "why", "like", "want", "need", "see",
    "look",
];

#[derive(Debug, Serialize)]
pub struct CommandScore {
    pub name: String,
    pub score: i32,
    pub issues: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct RoutingIssue {
    pub a: String,
    pub b: String,
    pub keys: (String, String),
    pub distance: f64,
    pub confusable: bool,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub commands: Vec<CommandScore>,
    pub routing: Vec<RoutingIssue>,
    pub overall: i32,
    pub espeak_available: bool,
}

/// Literal words before the first Slot/Enum — the command's "routing key".
fn leading_keywords(cmd: &Command) -> Vec<String> {
    let mut out = Vec::new();
    for c in &cmd.pattern {
        match c {
            Comp::Word(w) => out.push(w.clone()),
            _ => break,
        }
    }
    out
}

/// (slot_name, anchor_word) for each slot whose end anchor is a literal word.
fn slot_anchors(cmd: &Command) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (i, c) in cmd.pattern.iter().enumerate() {
        if let Comp::Slot(name) = c {
            if let Some(Comp::Word(w)) = cmd.pattern.get(i + 1) {
                out.push((name.clone(), w.clone()));
            }
        }
    }
    out
}

pub fn assess(grammar: &Grammar) -> Report {
    let espeak = available();

    // cross-command routing confusability
    let keys: Vec<(String, String)> = grammar
        .commands
        .iter()
        .map(|c| (c.name.clone(), leading_keywords(c).join(" ")))
        .collect();
    let mut routing = Vec::new();
    for i in 0..keys.len() {
        for j in (i + 1)..keys.len() {
            let (ka, kb) = (&keys[i].1, &keys[j].1);
            if ka.is_empty() || kb.is_empty() {
                continue;
            }
            let d = distance(ka, kb);
            if d <= CLOSE {
                routing.push(RoutingIssue {
                    a: keys[i].0.clone(),
                    b: keys[j].0.clone(),
                    keys: (ka.clone(), kb.clone()),
                    distance: d,
                    confusable: d <= CONFUSABLE,
                });
            }
        }
    }

    let mut commands = Vec::new();
    for cmd in &grammar.commands {
        let mut score: i32 = 100;
        let mut issues = Vec::new();

        // 1) enum options that sound alike
        for c in &cmd.pattern {
            if let Comp::Enum(en) = c {
                let opts: Vec<String> = en.options.iter().map(|o| o.join(" ")).collect();
                for a in 0..opts.len() {
                    for b in (a + 1)..opts.len() {
                        let d = distance(&opts[a], &opts[b]);
                        if d <= CONFUSABLE {
                            score -= 18;
                            issues.push(format!(
                                "enum '{}': '{}' ~ '{}' sound alike (dist {:.2}) — may misroute",
                                en.name, opts[a], opts[b], d
                            ));
                        } else if d <= CLOSE {
                            score -= 6;
                            issues.push(format!(
                                "enum '{}': '{}' ~ '{}' somewhat close (dist {:.2})",
                                en.name, opts[a], opts[b], d
                            ));
                        }
                    }
                }
            }
        }

        // 2) slot end-anchor quality
        for (sname, anchor) in slot_anchors(cmd) {
            if COMMON.contains(&anchor.as_str()) {
                score -= 15;
                issues.push(format!(
                    "slot '{sname}': end-anchor '{anchor}' is a common word — could occur inside \
                     the free text and end the slot early"
                ));
            }
            let n = phone_count(&anchor);
            if espeak && n > 0 && n <= SHORT_PHONES {
                score -= 8;
                issues.push(format!(
                    "slot '{sname}': end-anchor '{anchor}' is very short (/{}/) — ASR \
                     drops/inserts short words more often",
                    ipa(&anchor)
                ));
            }
            // anchor sounding like another literal in the same command self-collides
            for other in &cmd.pattern {
                if let Comp::Word(w) = other {
                    if *w != anchor && distance(&anchor, w) <= CONFUSABLE {
                        score -= 8;
                        issues.push(format!(
                            "slot '{sname}': end-anchor '{anchor}' ~ keyword '{w}' in same command"
                        ));
                    }
                }
            }
        }

        // 3) short leading keyword
        if let Some(first) = leading_keywords(cmd).first() {
            let n = phone_count(first);
            if espeak && n > 0 && n <= SHORT_PHONES {
                score -= 6;
                issues.push(format!(
                    "leading keyword '{first}' is very short (/{}/) — easy to miss",
                    ipa(first)
                ));
            }
        }

        commands.push(CommandScore {
            name: cmd.name.clone(),
            score: score.max(0),
            issues,
        });
    }

    let overall = if commands.is_empty() {
        0
    } else {
        commands.iter().map(|c| c.score).sum::<i32>() / commands.len() as i32
    };

    // worst-first
    commands.sort_by_key(|c| c.score);

    Report {
        commands,
        routing,
        overall,
        espeak_available: espeak,
    }
}

fn bar(score: i32) -> String {
    let n = ((score as f64) / 10.0).round() as usize;
    let n = n.min(10);
    format!("{}{}", "█".repeat(n), "░".repeat(10 - n))
}

impl Report {
    /// Render a human-readable report.
    pub fn render(&self) -> String {
        let rule = "─".repeat(66);
        let mut s = String::new();
        s.push_str("\nCOMMAND TREE QUALITY  (predictive proxy — feature-weighted phonetics)\n");
        s.push_str(&rule);
        s.push('\n');
        if !self.espeak_available {
            s.push_str("(espeak not found — phonetic checks disabled; only word-list checks ran)\n");
        }
        for c in &self.commands {
            s.push_str(&format!("{:<14} {} {:>3}/100\n", c.name, bar(c.score), c.score));
            if c.issues.is_empty() {
                s.push_str("   ✓ no phonetic red flags\n");
            }
            for it in &c.issues {
                s.push_str(&format!("   ⚠ {it}\n"));
            }
        }
        s.push_str(&rule);
        s.push_str("\nCross-command routing:\n");
        if self.routing.is_empty() {
            s.push_str("   ✓ all command-start keywords are phonetically distinct\n");
        } else {
            for r in &self.routing {
                let tag = if r.confusable { "CONFUSABLE" } else { "close" };
                s.push_str(&format!(
                    "   ⚠ [{tag}] '{}' ('{}') ~ '{}' ('{}')  dist {:.2}\n",
                    r.a, r.keys.0, r.b, r.keys.1, r.distance
                ));
            }
        }
        s.push_str(&rule);
        let weakest = self.commands.first();
        s.push_str(&format!(
            "\nOverall: {}/100 (mean)   weakest: {}\n",
            self.overall,
            weakest.map_or("-".into(), |c| format!("{} ({}/100)", c.name, c.score))
        ));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmdtree::grammar::Grammar;

    #[test]
    fn flags_common_anchor_and_confusable_enum() {
        if !available() {
            eprintln!("espeak unavailable; skipping");
            return;
        }
        let y = r#"
commands:
  - name: n
    pattern: [ {lit: "note that"}, {slot: text}, {lit: over} ]
  - name: dial
    pattern: [ {lit: dial}, {one_of: {name: d, options: [to, two, too]}}, {lit: "the number"} ]
"#;
        let g = Grammar::from_yaml(y).unwrap();
        let r = assess(&g);
        // 'over' is common -> n should lose points
        let n = r.commands.iter().find(|c| c.name == "n").unwrap();
        assert!(n.score < 100);
        // to/two/too are homophones -> dial should be flagged confusable
        let dial = r.commands.iter().find(|c| c.name == "dial").unwrap();
        assert!(dial.issues.iter().any(|i| i.contains("sound alike")));
    }
}

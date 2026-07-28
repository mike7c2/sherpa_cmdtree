//! Match a transcript against the compiled command tree, extracting slot values.
//!
//! The matcher is stateless: it takes a whole transcript string. That means the
//! same `parse()` can later be driven incrementally on a live token stream for an
//! always-on / wake-word front end.

use crate::grammar::{CompEnum, Command, Comp, Grammar};
use crate::normalize;

/// A successful command match.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub command: String,
    pub action: Option<String>,
    /// slot/enum captures, in pattern order
    pub slots: Vec<(String, String)>,
    /// literal/enum words matched (specificity; higher = more confident routing)
    pub score: usize,
    /// words left over after the match
    pub leftover: Vec<String>,
}

impl Match {
    pub fn slot(&self, name: &str) -> Option<&str> {
        self.slots
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// The slot captures as a JSON object.
    pub fn slots_json(&self) -> serde_json::Map<String, serde_json::Value> {
        self.slots
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect()
    }
}

/// Options controlling how a transcript is matched.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParseOpts {
    /// The utterance is known-complete (e.g. a push-to-talk key was released). A
    /// trailing free-text slot whose end anchor was *not* spoken then captures to
    /// the end of the utterance. The anchor stays required by the grammar itself and
    /// is still consumed whenever it *is* spoken — this only relaxes the trailing
    /// case for a bounded utterance.
    pub open_ended: bool,
}

/// Longest common subsequence length (character-level).
fn lcs_len(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev = vec![0usize; b.len() + 1];
    for &ca in &a {
        let mut cur = vec![0usize; b.len() + 1];
        for (j, &cb) in b.iter().enumerate() {
            cur[j + 1] = if ca == cb {
                prev[j] + 1
            } else {
                cur[j].max(prev[j + 1])
            };
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Fuzzy word equality: exact, or a matching-block similarity ratio >= `fuzzy`
/// (2·LCS / (|a|+|b|), the same measure Python's difflib uses). Handles minor
/// ASR variants like note/notes without matching on/off.
fn word_eq(a: &str, b: &str, fuzzy: f64) -> bool {
    if a == b {
        return true;
    }
    if fuzzy >= 1.0 {
        return false;
    }
    let len = a.chars().count() + b.chars().count(); // >= 1 since a != b
    let ratio = 2.0 * lcs_len(a, b) as f64 / len as f64;
    ratio >= fuzzy
}

/// If `en`'s longest matching option starts at `words[i]`, return its length.
fn enum_len_at(en: &CompEnum, words: &[String], i: usize, fuzzy: f64) -> Option<usize> {
    let mut best: Option<usize> = None;
    for opt in &en.options {
        if i + opt.len() <= words.len()
            && opt
                .iter()
                .enumerate()
                .all(|(k, w)| word_eq(&words[i + k], w, fuzzy))
        {
            best = Some(best.map_or(opt.len(), |b: usize| b.max(opt.len())));
        }
    }
    best
}

/// Does the anchor (a Word or Enum) match starting at `words[j]`?
fn anchor_matches(anchor: &Comp, words: &[String], j: usize, fuzzy: f64) -> bool {
    match anchor {
        Comp::Word(w) => j < words.len() && word_eq(&words[j], w, fuzzy),
        Comp::Enum(en) => enum_len_at(en, words, j, fuzzy).is_some(),
        Comp::Slot(_) => false, // never (validated away)
    }
}

fn match_command(cmd: &Command, words: &[String], fuzzy: f64, opts: ParseOpts) -> Option<Match> {
    let mut slots: Vec<(String, String)> = Vec::new();
    let mut i = 0usize; // position in words
    let mut score = 0usize;
    let mut p = 0usize; // position in pattern
    while p < cmd.pattern.len() {
        match &cmd.pattern[p] {
            Comp::Word(w) => {
                if i < words.len() && word_eq(&words[i], w, fuzzy) {
                    i += 1;
                    score += 1;
                    p += 1;
                } else {
                    return None;
                }
            }
            Comp::Enum(en) => {
                let len = enum_len_at(en, words, i, fuzzy)?;
                slots.push((en.name.clone(), words[i..i + len].join(" ")));
                score += len;
                i += len;
                p += 1;
            }
            Comp::Slot(name) => {
                let anchor = &cmd.pattern[p + 1]; // guaranteed by validation
                // free text is >= 1 word: search for the anchor from i+1 on
                let found = (i + 1..=words.len()).find(|&j| anchor_matches(anchor, words, j, fuzzy));
                match found {
                    Some(end) => {
                        slots.push((name.clone(), words[i..end].join(" ")));
                        i = end;
                        p += 1;
                    }
                    None => {
                        // The end anchor was not spoken. For a known-complete utterance
                        // (PTT), if this is the last slot — only literal/enum "end
                        // tokens" follow it — capture the rest to the utterance end and
                        // treat that unspoken tail as satisfied.
                        let trailing =
                            !cmd.pattern[p + 1..].iter().any(|c| matches!(c, Comp::Slot(_)));
                        if opts.open_ended && trailing && i < words.len() {
                            slots.push((name.clone(), words[i..].join(" ")));
                            i = words.len();
                            p = cmd.pattern.len();
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
    }
    Some(Match {
        command: cmd.name.clone(),
        action: cmd.action.clone(),
        slots,
        score,
        leftover: words[i..].to_vec(),
    })
}

impl Grammar {
    /// Parse a transcript, returning the best matching command (or `None`).
    /// "Best" = most specific (highest literal/enum score), then fewest leftover words.
    pub fn parse(&self, text: &str) -> Option<Match> {
        self.parse_with(text, ParseOpts::default())
    }

    /// Like [`parse`](Self::parse) but with matching [`ParseOpts`] (e.g. `open_ended`
    /// for a complete push-to-talk utterance).
    pub fn parse_with(&self, text: &str, opts: ParseOpts) -> Option<Match> {
        let words = normalize(text);
        let key = |m: &Match| (m.score, std::cmp::Reverse(m.leftover.len()));
        self.commands
            .iter()
            .filter_map(|cmd| match_command(cmd, &words, self.fuzzy, opts))
            .reduce(|best, m| if key(&m) > key(&best) { m } else { best })
    }
}

#[cfg(test)]
mod tests {
    use crate::grammar::Grammar;

    fn demo() -> Grammar {
        let y = r#"
commands:
  - name: add_note
    pattern: [ {lit: "note that"}, {slot: text}, {lit: over} ]
  - name: send_message
    pattern:
      - {lit: "send message to"}
      - {one_of: {name: who, options: [alice, bob, mum]}}
      - {lit: saying}
      - {slot: body}
      - {lit: send}
  - name: set_light
    pattern:
      - {lit: turn}
      - {one_of: {name: state, options: [on, off]}}
      - {lit: the}
      - {one_of: {name: device, options: ["kitchen light", "bedroom light", lamp]}}
  - name: search
    pattern: [ {lit: "search for"}, {slot: query}, {lit: go} ]
"#;
        Grammar::from_yaml(y).unwrap()
    }

    #[test]
    fn extracts_free_text_slot() {
        let m = demo().parse("note that buy milk and eggs over").unwrap();
        assert_eq!(m.command, "add_note");
        assert_eq!(m.slot("text"), Some("buy milk and eggs"));
    }

    #[test]
    fn enum_plus_slot() {
        let m = demo()
            .parse("send message to alice saying i'll be late send")
            .unwrap();
        assert_eq!(m.command, "send_message");
        assert_eq!(m.slot("who"), Some("alice"));
        assert_eq!(m.slot("body"), Some("i'll be late"));
    }

    #[test]
    fn multiword_enum() {
        let m = demo().parse("turn off the kitchen light").unwrap();
        assert_eq!(m.command, "set_light");
        assert_eq!(m.slot("state"), Some("off"));
        assert_eq!(m.slot("device"), Some("kitchen light"));
    }

    #[test]
    fn fuzzy_literal() {
        // "notes" ~ "note"
        let m = demo().parse("notes that call the dentist over").unwrap();
        assert_eq!(m.command, "add_note");
        assert_eq!(m.slot("text"), Some("call the dentist"));
    }

    #[test]
    fn no_match() {
        assert!(demo().parse("what's the weather like").is_none());
    }

    #[test]
    fn open_ended_ptt_drops_trailing_anchor_but_grammar_keeps_it() {
        use super::ParseOpts;
        let g = demo(); // add_note: "note that" <text> "over"
        // strict (default): the end anchor is still required by the grammar
        assert!(g.parse("note that buy milk and eggs").is_none());
        // open-ended (PTT complete): trailing slot captures to the utterance end
        let ptt = ParseOpts { open_ended: true };
        let m = g.parse_with("note that buy milk and eggs", ptt).unwrap();
        assert_eq!(m.command, "add_note");
        assert_eq!(m.slot("text"), Some("buy milk and eggs"));
        // the anchor is still consumed when it IS spoken, even open-ended
        let m2 = g.parse_with("note that buy milk over", ptt).unwrap();
        assert_eq!(m2.slot("text"), Some("buy milk"));
        assert!(m2.leftover.is_empty());
        // empty free text still doesn't match (a note needs content)
        assert!(g.parse_with("note that", ptt).is_none());
    }
}

//! Hotkey-combo parsing, kept free of any platform key type so it is unit-tested
//! in the default build. The daemon maps live key events to the same normalized
//! tokens ("ctrl", "alt", "shift", "meta", "space", "k", ...) and checks them
//! against a [`Combo`].

use std::collections::HashSet;

/// A key combo such as `ctrl+space`, stored as normalized lowercase tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Combo {
    pub keys: Vec<String>,
}

/// Normalize a single key token (aliases collapse to a canonical name).
pub fn normalize_key(tok: &str) -> String {
    match tok.trim().to_lowercase().as_str() {
        "control" | "ctrl" | "ctl" => "ctrl",
        "alt" | "option" | "altgr" => "alt",
        "shift" => "shift",
        "meta" | "super" | "win" | "windows" | "cmd" | "command" => "meta",
        "space" | "spacebar" | " " => "space",
        "return" | "enter" => "enter",
        "esc" | "escape" => "escape",
        other => return other.to_string(),
    }
    .to_string()
}

impl Combo {
    /// Parse a `+`-separated combo string, e.g. "ctrl+space".
    pub fn parse(s: &str) -> Combo {
        let keys = s
            .split('+')
            .map(normalize_key)
            .filter(|k| !k.is_empty())
            .collect();
        Combo { keys }
    }

    /// True when every key of the combo is currently held.
    pub fn is_satisfied(&self, pressed: &HashSet<String>) -> bool {
        !self.keys.is_empty() && self.keys.iter().all(|k| pressed.contains(k))
    }

    /// True when the combo involves `key` (used to detect release).
    pub fn contains(&self, key: &str) -> bool {
        self.keys.iter().any(|k| k == key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_and_normalizes_aliases() {
        assert_eq!(Combo::parse("Control+Space").keys, vec!["ctrl", "space"]);
        assert_eq!(Combo::parse("cmd+k").keys, vec!["meta", "k"]);
    }

    #[test]
    fn satisfied_only_when_all_held() {
        let c = Combo::parse("ctrl+space");
        assert!(!c.is_satisfied(&set(&["ctrl"])));
        assert!(c.is_satisfied(&set(&["ctrl", "space"])));
        assert!(c.is_satisfied(&set(&["ctrl", "space", "shift"]))); // extra keys ok
    }

    #[test]
    fn contains_detects_release_key() {
        assert!(Combo::parse("ctrl+alt+k").contains("k"));
        assert!(!Combo::parse("ctrl+alt+k").contains("space"));
    }
}

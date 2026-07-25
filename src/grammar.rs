//! The command tree: YAML-facing raw types, plus the compiled+validated form.

use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

use crate::normalize;

// --------------------------------------------------------------------------- //
// Raw, YAML-facing types
// --------------------------------------------------------------------------- //

/// A pattern element as written in YAML. Each element is a single-key map:
/// `{lit: "note that"}`, `{slot: text}`, or
/// `{one_of: {name: who, options: [alice, bob]}}`.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "ElementRepr")]
pub enum Element {
    Lit(String),
    Slot(String),
    OneOf(OneOf),
}

/// Deserialization shape for [`Element`]: exactly one field must be present.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ElementRepr {
    #[serde(default)]
    lit: Option<String>,
    #[serde(default)]
    slot: Option<String>,
    #[serde(default)]
    one_of: Option<OneOf>,
}

impl TryFrom<ElementRepr> for Element {
    type Error = String;
    fn try_from(r: ElementRepr) -> Result<Self, Self::Error> {
        match (r.lit, r.slot, r.one_of) {
            (Some(s), None, None) => Ok(Element::Lit(s)),
            (None, Some(s), None) => Ok(Element::Slot(s)),
            (None, None, Some(o)) => Ok(Element::OneOf(o)),
            _ => Err("each pattern element must have exactly one of: lit, slot, one_of".into()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OneOf {
    pub name: String,
    pub options: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawCommand {
    pub name: String,
    pub pattern: Vec<Element>,
    #[serde(default)]
    pub action: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawGrammar {
    pub commands: Vec<RawCommand>,
    /// Fuzzy literal-match threshold in [0,1]; 1.0 = exact. Default 0.82.
    #[serde(default)]
    pub fuzzy: Option<f64>,
}

// --------------------------------------------------------------------------- //
// Errors
// --------------------------------------------------------------------------- //

#[derive(Debug, Error)]
pub enum GrammarError {
    #[error("failed to read grammar file {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("command '{command}': {message}")]
    Invalid { command: String, message: String },
}

// --------------------------------------------------------------------------- //
// Compiled form
// --------------------------------------------------------------------------- //

/// A compiled pattern element: literal words are expanded to one token each.
#[derive(Debug, Clone)]
pub enum Comp {
    Word(String),
    Enum(CompEnum),
    Slot(String),
}

#[derive(Debug, Clone)]
pub struct CompEnum {
    pub name: String,
    /// each option is its list of normalized words (options may be multi-word)
    pub options: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Command {
    pub name: String,
    pub action: Option<String>,
    pub pattern: Vec<Comp>,
}

#[derive(Debug, Clone)]
pub struct Grammar {
    pub commands: Vec<Command>,
    pub fuzzy: f64,
}

impl RawGrammar {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, GrammarError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| GrammarError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_yaml(&text)
    }

    pub fn from_yaml(text: &str) -> Result<Self, GrammarError> {
        Ok(serde_yaml::from_str(text)?)
    }

    /// Compile and validate. The key rule: a free-text `Slot` must be followed by
    /// a literal/enum END ANCHOR so we always know where the free text stops.
    pub fn compile(self) -> Result<Grammar, GrammarError> {
        let fuzzy = self.fuzzy.unwrap_or(0.82);
        let mut commands = Vec::with_capacity(self.commands.len());
        for raw in self.commands {
            let mut pattern: Vec<Comp> = Vec::new();
            for el in &raw.pattern {
                match el {
                    Element::Lit(s) => {
                        let words = normalize(s);
                        if words.is_empty() {
                            return Err(GrammarError::Invalid {
                                command: raw.name.clone(),
                                message: "empty literal in pattern".into(),
                            });
                        }
                        pattern.extend(words.into_iter().map(Comp::Word));
                    }
                    Element::Slot(name) => pattern.push(Comp::Slot(name.clone())),
                    Element::OneOf(o) => {
                        let options: Vec<Vec<String>> =
                            o.options.iter().map(|s| normalize(s)).collect();
                        if options.is_empty() || options.iter().any(|o| o.is_empty()) {
                            return Err(GrammarError::Invalid {
                                command: raw.name.clone(),
                                message: format!("one_of '{}' has empty options", o.name),
                            });
                        }
                        pattern.push(Comp::Enum(CompEnum {
                            name: o.name.clone(),
                            options,
                        }));
                    }
                }
            }
            // validate slot anchors
            for (i, c) in pattern.iter().enumerate() {
                if let Comp::Slot(name) = c {
                    match pattern.get(i + 1) {
                        Some(Comp::Word(_)) | Some(Comp::Enum(_)) => {}
                        _ => {
                            return Err(GrammarError::Invalid {
                                command: raw.name.clone(),
                                message: format!(
                                    "slot '{name}' needs a literal/enum END ANCHOR right after it \
                                     (free text must be bounded on both sides)"
                                ),
                            });
                        }
                    }
                }
            }
            commands.push(Command {
                name: raw.name,
                action: raw.action,
                pattern,
            });
        }
        Ok(Grammar { commands, fuzzy })
    }
}

impl Grammar {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, GrammarError> {
        RawGrammar::load(path)?.compile()
    }

    pub fn from_yaml(text: &str) -> Result<Self, GrammarError> {
        RawGrammar::from_yaml(text)?.compile()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unanchored_slot_is_rejected() {
        let y = "commands:\n  - name: bad\n    pattern:\n      - lit: remember\n      - slot: x\n";
        let err = Grammar::from_yaml(y).unwrap_err();
        assert!(matches!(err, GrammarError::Invalid { .. }));
    }

    #[test]
    fn compiles_and_expands_literals() {
        let y = "commands:\n  - name: n\n    pattern:\n      - lit: note that\n      - slot: text\n      - lit: over\n";
        let g = Grammar::from_yaml(y).unwrap();
        assert_eq!(g.commands.len(), 1);
        // "note that" -> two Word tokens, then Slot, then "over"
        assert_eq!(g.commands[0].pattern.len(), 4);
    }
}

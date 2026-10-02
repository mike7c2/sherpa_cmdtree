//! `tokens.txt` — the model's sub-word inventory, and turning ids back into text.

use std::path::Path;

/// id → token piece, as listed in the model's `tokens.txt` (`piece id` per line).
pub struct Tokens(Vec<String>);

impl Tokens {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join("tokens.txt");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        let mut pieces: Vec<String> = Vec::new();
        for line in text.lines() {
            // `piece id`; the piece itself never contains a space, but a blank
            // piece (the space symbol) can make the split degenerate.
            let mut parts = line.rsplitn(2, ' ');
            let (Some(id), Some(piece)) = (parts.next(), parts.next()) else {
                continue;
            };
            let id: usize = id.trim().parse().map_err(|_| {
                format!("{}: line {line:?} does not end with an integer id", path.display())
            })?;
            if pieces.len() <= id {
                pieces.resize(id + 1, String::new());
            }
            pieces[id] = piece.to_string();
        }
        if pieces.is_empty() {
            return Err(format!("{} is empty", path.display()));
        }
        Ok(Self(pieces))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn piece(&self, id: i64) -> &str {
        self.0.get(id as usize).map(String::as_str).unwrap_or("")
    }

    /// Join token ids into text. BPE marks a word start with `▁`; sherpa's C API
    /// renders that as a leading space, and we match it so transcripts compare
    /// byte-for-byte against `cmdtree transcribe`.
    pub fn decode(&self, ids: &[i64]) -> String {
        let mut out = String::new();
        for &id in ids {
            let piece = self.piece(id);
            if let Some(rest) = piece.strip_prefix('\u{2581}') {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(rest);
            } else {
                out.push_str(piece);
            }
        }
        out.trim().to_string()
    }
}

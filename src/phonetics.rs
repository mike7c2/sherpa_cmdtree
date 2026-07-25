//! Grapheme→phoneme via espeak, then a distinctive-feature-weighted phonetic
//! distance between words.
//!
//! Character-level edit distance on IPA treats every phoneme swap as equal.
//! Human/ASR confusions are not equal: Miller & Nicely (1955) showed voicing and
//! nasality survive noise while place of articulation collapses first. So here a
//! voicing-only difference is cheap (phonemes stay confusable) and a manner or
//! vowel/consonant difference is expensive. This makes the distance track *acoustic
//! confusability* far better than letters do.
//!
//! It shells out to `espeak -q --ipa`. If espeak is missing, [`ipa`] returns an
//! empty string and [`distance`] returns 1.0 (treated as "not confusable").

use std::collections::HashMap;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

// --------------------------------------------------------------------------- //
// espeak G2P
// --------------------------------------------------------------------------- //

fn cache() -> &'static Mutex<HashMap<String, String>> {
    static C: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// IPA for `text`, with stress marks, length marks, and spaces stripped.
pub fn ipa(text: &str) -> String {
    let key = text.to_lowercase();
    if let Some(v) = cache().lock().unwrap().get(&key) {
        return v.clone();
    }
    let out = Command::new("espeak")
        .args(["-q", "--ipa", &key])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    let cleaned: String = out
        .trim()
        .chars()
        .filter(|c| !matches!(c, 'ˈ' | 'ˌ' | 'ː' | ' ' | '\n' | '\r' | '\t'))
        .collect();
    cache().lock().unwrap().insert(key, cleaned.clone());
    cleaned
}

/// Is espeak usable at all?
pub fn available() -> bool {
    !ipa("test").is_empty()
}

// --------------------------------------------------------------------------- //
// Phone inventory + distinctive features
// --------------------------------------------------------------------------- //

#[derive(Clone, Copy, PartialEq)]
enum Manner {
    Plosive,
    Affricate,
    Fricative,
    Nasal,
    Approximant,
    Lateral,
}

/// Place ordered front→back so we can measure distance along it.
#[derive(Clone, Copy, PartialEq)]
enum Place {
    Bilabial = 0,
    Labiodental = 1,
    Dental = 2,
    Alveolar = 3,
    PostAlveolar = 4,
    Palatal = 5,
    Velar = 6,
    Glottal = 7,
}

#[derive(Clone, Copy)]
enum Feat {
    /// height/back in [0,1] (close→open, front→back)
    Vowel { height: f64, back: f64, round: bool },
    Cons { voiced: bool, manner: Manner, place: Place },
}

/// Multi-codepoint phones that must be segmented as a unit.
const DIGRAPHS: &[&str] = &[
    "eɪ", "aɪ", "ɔɪ", "əʊ", "oʊ", "aʊ", "ɪə", "eə", "ʊə", "tʃ", "dʒ",
];

fn segment(ipa: &str) -> Vec<String> {
    let chars: Vec<char> = ipa.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let mut matched = None;
        if i + 1 < chars.len() {
            let two: String = chars[i..i + 2].iter().collect();
            if DIGRAPHS.contains(&two.as_str()) {
                matched = Some(two);
            }
        }
        match matched {
            Some(d) => {
                out.push(d);
                i += 2;
            }
            None => {
                out.push(chars[i].to_string());
                i += 1;
            }
        }
    }
    out
}

fn feat(p: &str) -> Option<Feat> {
    use Manner::*;
    use Place::*;
    let v = |height: f64, back: f64, round: bool| Some(Feat::Vowel { height, back, round });
    let c = |voiced: bool, manner: Manner, place: Place| Some(Feat::Cons { voiced, manner, place });
    match p {
        // monophthongs (length already stripped)
        "i" | "ɪ" => v(0.1, 0.05, false),
        "e" => v(0.3, 0.05, false),
        "ɛ" => v(0.4, 0.05, false),
        "æ" => v(0.75, 0.1, false),
        "a" => v(0.85, 0.2, false),
        "ʌ" => v(0.6, 0.5, false),
        "ə" | "ɜ" => v(0.5, 0.5, false),
        "ɑ" => v(0.95, 0.9, false),
        "ɒ" => v(0.9, 0.95, true),
        "ɔ" => v(0.6, 0.95, true),
        "ʊ" => v(0.2, 0.85, true),
        "u" | "ʉ" => v(0.05, 0.95, true),
        "o" => v(0.35, 0.9, true),
        // diphthongs (represented by an approximate nucleus)
        "eɪ" => v(0.3, 0.1, false),
        "aɪ" => v(0.8, 0.3, false),
        "ɔɪ" => v(0.6, 0.8, true),
        "əʊ" | "oʊ" => v(0.4, 0.6, true),
        "aʊ" => v(0.8, 0.5, true),
        "ɪə" => v(0.2, 0.2, false),
        "eə" => v(0.4, 0.1, false),
        "ʊə" => v(0.3, 0.7, true),
        // consonants
        "p" => c(false, Plosive, Bilabial),
        "b" => c(true, Plosive, Bilabial),
        "t" => c(false, Plosive, Alveolar),
        "d" => c(true, Plosive, Alveolar),
        "k" => c(false, Plosive, Velar),
        "g" | "ɡ" => c(true, Plosive, Velar),
        "tʃ" => c(false, Affricate, PostAlveolar),
        "dʒ" => c(true, Affricate, PostAlveolar),
        "f" => c(false, Fricative, Labiodental),
        "v" => c(true, Fricative, Labiodental),
        "θ" => c(false, Fricative, Dental),
        "ð" => c(true, Fricative, Dental),
        "s" => c(false, Fricative, Alveolar),
        "z" => c(true, Fricative, Alveolar),
        "ʃ" => c(false, Fricative, PostAlveolar),
        "ʒ" => c(true, Fricative, PostAlveolar),
        "h" => c(false, Fricative, Glottal),
        "m" => c(true, Nasal, Bilabial),
        "n" => c(true, Nasal, Alveolar),
        "ŋ" => c(true, Nasal, Velar),
        "l" => c(true, Lateral, Alveolar),
        "r" | "ɹ" => c(true, Approximant, Alveolar),
        "w" => c(true, Approximant, Bilabial),
        "j" => c(true, Approximant, Palatal),
        _ => None,
    }
}

/// Substitution cost between two phones, in [0,1].
fn sub_cost(a: &str, b: &str) -> f64 {
    if a == b {
        return 0.0;
    }
    match (feat(a), feat(b)) {
        (Some(Feat::Cons { voiced: v1, manner: m1, place: p1 }),
         Some(Feat::Cons { voiced: v2, manner: m2, place: p2 })) => {
            let manner = if m1 == m2 { 0.0 } else { 1.0 };
            let voice = if v1 == v2 { 0.0 } else { 1.0 };
            let place = ((p1 as i32 - p2 as i32).abs() as f64) / 7.0;
            (0.45 * manner + 0.20 * voice + 0.35 * place).min(1.0)
        }
        (Some(Feat::Vowel { height: h1, back: b1, round: r1 }),
         Some(Feat::Vowel { height: h2, back: b2, round: r2 })) => {
            let round = if r1 == r2 { 0.0 } else { 1.0 };
            (0.55 * (h1 - h2).abs() + 0.35 * (b1 - b2).abs() + 0.10 * round).min(1.0)
        }
        // vowel vs consonant, or an unknown phone: maximally different
        _ => 1.0,
    }
}

/// Distinctive-feature-weighted phonetic distance between two words, in [0,1].
/// 0.0 = homophones. Uses a weighted Levenshtein over segmented phones.
pub fn distance(word_a: &str, word_b: &str) -> f64 {
    let pa = segment(&ipa(word_a));
    let pb = segment(&ipa(word_b));
    if pa.is_empty() || pb.is_empty() {
        return 1.0;
    }
    const INDEL: f64 = 0.8;
    let mut prev: Vec<f64> = (0..=pb.len()).map(|j| j as f64 * INDEL).collect();
    for (i, a) in pa.iter().enumerate() {
        let mut cur = vec![(i as f64 + 1.0) * INDEL];
        for (j, b) in pb.iter().enumerate() {
            let sub = prev[j] + sub_cost(a, b);
            let del = prev[j + 1] + INDEL;
            let ins = cur[j] + INDEL;
            cur.push(sub.min(del).min(ins));
        }
        prev = cur;
    }
    let raw = prev[pb.len()];
    (raw / (pa.len().max(pb.len()) as f64)).min(1.0)
}

/// Number of phones in a word (0 if espeak unavailable).
pub fn phone_count(word: &str) -> usize {
    segment(&ipa(word)).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homophones_are_zero() {
        if !available() {
            eprintln!("espeak unavailable; skipping");
            return;
        }
        assert!(distance("to", "two") < 0.05, "to/two = {}", distance("to", "two"));
        assert!(distance("for", "four") < 0.05);
    }

    #[test]
    fn unrelated_words_are_far() {
        if !available() {
            return;
        }
        assert!(distance("alice", "bob") > 0.6);
    }

    #[test]
    fn voicing_only_is_closer_than_manner() {
        if !available() {
            return;
        }
        // p/b differ only in voicing (cheap); p/m differ in manner (dearer)
        assert!(distance("pat", "bat") < distance("pat", "mat"));
    }
}

//! In-process sherpa-onnx recognition. [`SherpaTransducer`] turns a mono f32
//! utterance into an [`AsrResult`] — the text plus sub-word tokens with timestamps.
//! It is driven **directly through `sherpa-rs-sys`** so we can read the full result
//! (tokens + per-token timestamps), which `sherpa-rs`'s high-level `transcribe()`
//! discards, and it supports hotword biasing.

/// A recognized sub-word token and its start time, in seconds.
#[derive(Debug, Clone)]
pub struct AsrToken {
    pub text: String,
    pub start: f32,
}

/// A recognition result: the text plus sub-word tokens with timestamps.
#[derive(Debug, Clone, Default)]
pub struct AsrResult {
    pub text: String,
    pub tokens: Vec<AsrToken>,
}

impl AsrResult {
    /// Merge sub-word tokens into words. sherpa marks a word start with a leading
    /// space (the C API translates BPE's `▁`); other pieces attach to the current
    /// word. Returns `(word, start_s)` with the start time of the word's first token.
    pub fn words(&self) -> Vec<(String, f32)> {
        let mut words: Vec<(String, f32)> = Vec::new();
        for tok in &self.tokens {
            let starts_word = tok.text.starts_with([' ', '\u{2581}']);
            let piece = tok.text.trim_start_matches([' ', '\u{2581}']);
            if starts_word || words.is_empty() {
                words.push((piece.to_string(), tok.start));
            } else if let Some(w) = words.last_mut() {
                w.0.push_str(piece);
            }
        }
        words.retain(|(w, _)| !w.is_empty());
        words
    }
}

pub use sherpa_backend::{SherpaOpts, SherpaTransducer};

/// In-process sherpa-onnx offline transducer, driven directly through the C API
/// (`sherpa-rs-sys`) so the full result — tokens + timestamps — is available.
mod sherpa_backend {
    use super::{AsrResult, AsrToken};
    use sherpa_rs::sherpa_rs_sys as sys;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    /// Construction options for [`SherpaTransducer`].
    #[derive(Default)]
    pub struct SherpaOpts {
        pub num_threads: Option<i32>,
        /// sherpa-onnx execution provider (`"cpu"`, or `"cuda"` with the cuda feature).
        pub provider: Option<String>,
        /// hotword (contextual-biasing) file — one phrase per line. Requires a
        /// `bpe.vocab` in the model dir; ignored (with a warning) if that is missing.
        pub hotwords_file: Option<PathBuf>,
        pub hotwords_score: f32,
    }

    /// Owns the raw recognizer pointer; freed on drop. Only ever touched under the
    /// [`SherpaTransducer`] mutex, so it is safe to move across threads.
    struct OwnedRecognizer(*const sys::SherpaOnnxOfflineRecognizer);
    unsafe impl Send for OwnedRecognizer {}
    impl Drop for OwnedRecognizer {
        fn drop(&mut self) {
            unsafe { sys::SherpaOnnxDestroyOfflineRecognizer(self.0) }
        }
    }

    pub struct SherpaTransducer {
        inner: Mutex<OwnedRecognizer>,
    }

    impl SherpaTransducer {
        /// Build from a model directory containing the encoder/decoder/joiner ONNX
        /// files and `tokens.txt` (auto-discovered, preferring full-precision over
        /// `int8`). If `opts.hotwords_file` is set and the dir has a `bpe.vocab`,
        /// contextual biasing is enabled (which forces `modified_beam_search`).
        pub fn from_dir(dir: &Path, opts: SherpaOpts) -> Result<Self, String> {
            let tokens = dir.join("tokens.txt");
            if !tokens.is_file() {
                return Err(format!("tokens.txt not found in {}", dir.display()));
            }

            // Hotword biasing needs a BPE vocab (`piece score` per line) to tokenize
            // the phrase text, and only works under beam search.
            let (hotwords_file, modeling_unit, bpe_vocab) = match &opts.hotwords_file {
                Some(hw) => {
                    let bpe = dir.join("bpe.vocab");
                    if bpe.is_file() {
                        (
                            hw.to_string_lossy().into_owned(),
                            "bpe".to_string(),
                            bpe.to_string_lossy().into_owned(),
                        )
                    } else {
                        eprintln!(
                            "warning: hotword biasing requested but no bpe.vocab in {} — biasing disabled",
                            dir.display()
                        );
                        (String::new(), String::new(), String::new())
                    }
                }
                None => (String::new(), String::new(), String::new()),
            };
            let decoding_method = if hotwords_file.is_empty() {
                "greedy_search"
            } else {
                "modified_beam_search"
            };

            // Every CString must outlive SherpaOnnxCreateOfflineRecognizer (sherpa
            // copies the strings during create), so bind them all here first.
            let encoder = cstr(&find_model_file(dir, "encoder")?);
            let decoder = cstr(&find_model_file(dir, "decoder")?);
            let joiner = cstr(&find_model_file(dir, "joiner")?);
            let tokens_c = cstr(&tokens.to_string_lossy());
            let provider = cstr(opts.provider.as_deref().unwrap_or("cpu"));
            let model_type = cstr("transducer");
            let modeling_unit_c = cstr(&modeling_unit);
            let bpe_vocab_c = cstr(&bpe_vocab);
            let hotwords_c = cstr(&hotwords_file);
            let decoding_c = cstr(decoding_method);

            let recognizer = unsafe {
                let mut model: sys::SherpaOnnxOfflineModelConfig = std::mem::zeroed();
                model.transducer = sys::SherpaOnnxOfflineTransducerModelConfig {
                    encoder: encoder.as_ptr(),
                    decoder: decoder.as_ptr(),
                    joiner: joiner.as_ptr(),
                };
                model.tokens = tokens_c.as_ptr();
                model.num_threads = opts.num_threads.unwrap_or(2);
                model.debug = 0;
                model.provider = provider.as_ptr();
                model.model_type = model_type.as_ptr();
                model.modeling_unit = modeling_unit_c.as_ptr();
                model.bpe_vocab = bpe_vocab_c.as_ptr();

                let mut config: sys::SherpaOnnxOfflineRecognizerConfig = std::mem::zeroed();
                config.model_config = model;
                config.feat_config = sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                };
                config.decoding_method = decoding_c.as_ptr();
                config.hotwords_file = hotwords_c.as_ptr();
                config.hotwords_score = opts.hotwords_score;

                sys::SherpaOnnxCreateOfflineRecognizer(&config)
            };
            if recognizer.is_null() {
                return Err("SherpaOnnxCreateOfflineRecognizer failed (see stderr for details)".into());
            }
            Ok(Self {
                inner: Mutex::new(OwnedRecognizer(recognizer)),
            })
        }
    }

    impl SherpaTransducer {
        /// Recognise one mono f32 utterance at `sample_rate` Hz.
        pub fn recognize(&self, samples: &[f32], sample_rate: u32) -> Result<AsrResult, String> {
            let rec = self
                .inner
                .lock()
                .map_err(|_| "recognizer mutex poisoned".to_string())?;
            unsafe {
                let stream = sys::SherpaOnnxCreateOfflineStream(rec.0);
                sys::SherpaOnnxAcceptWaveformOffline(
                    stream,
                    sample_rate as i32,
                    samples.as_ptr(),
                    samples.len() as i32,
                );
                sys::SherpaOnnxDecodeOfflineStream(rec.0, stream);
                let result = sys::SherpaOnnxGetOfflineStreamResult(stream);
                let out = read_result(result);
                sys::SherpaOnnxDestroyOfflineRecognizerResult(result);
                sys::SherpaOnnxDestroyOfflineStream(stream);
                Ok(out)
            }
        }
    }

    /// Copy text + per-token strings + timestamps out of a sherpa result before it
    /// is destroyed.
    unsafe fn read_result(result: *const sys::SherpaOnnxOfflineRecognizerResult) -> AsrResult {
        if result.is_null() {
            return AsrResult::default();
        }
        let r = &*result;
        let text = cstr_to_string(r.text).trim().to_string();
        let mut tokens = Vec::new();
        let count = r.count.max(0) as usize;
        if count > 0 && !r.tokens_arr.is_null() {
            let arr = std::slice::from_raw_parts(r.tokens_arr, count);
            let times: &[f32] = if r.timestamps.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(r.timestamps, count)
            };
            for (k, &tp) in arr.iter().enumerate() {
                if tp.is_null() {
                    continue;
                }
                tokens.push(AsrToken {
                    text: cstr_to_string(tp),
                    start: times.get(k).copied().unwrap_or(0.0),
                });
            }
        }
        AsrResult { text, tokens }
    }

    fn cstr(s: &str) -> CString {
        CString::new(s).unwrap_or_default()
    }

    unsafe fn cstr_to_string(p: *const c_char) -> String {
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }

    /// Find the single `*keyword*.onnx` file in `dir`, preferring a full-precision
    /// build over an `int8` one when both exist.
    fn find_model_file(dir: &Path, keyword: &str) -> Result<String, String> {
        let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let name = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                name.contains(keyword) && name.ends_with(".onnx")
            })
            .collect();
        matches.sort(); // deterministic order first ...
        matches.sort_by_key(|p| p.to_string_lossy().contains("int8")); // ... full-precision before int8
        matches
            .into_iter()
            .next()
            .map(|p| p.to_string_lossy().into_owned())
            .ok_or_else(|| format!("no *{keyword}*.onnx model file in {}", dir.display()))
    }
}

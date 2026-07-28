//! Streaming keyword spotter for wake-word (and optional end-word) detection.
//!
//! Driven directly through `sherpa-rs-sys` so we can feed a continuous mic stream
//! and poll for hits — unlike sherpa-rs's high-level `extract_keyword`, which calls
//! `InputFinished` on every chunk (wrong for an always-on stream). A tiny sherpa-onnx
//! streaming zipformer KWS model listens for the phrases in a tokenized `keywords`
//! file; on a hit we reset the stream so the same utterance doesn't re-fire.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};

use sherpa_rs::sherpa_rs_sys as sys;

pub struct KwsOpts {
    /// tokenized keywords file (one phrase per line as space-separated BPE tokens).
    pub keywords_file: PathBuf,
    /// detection threshold in (0,1]; lower = more sensitive (more false wakes).
    pub threshold: f32,
    /// per-keyword boost.
    pub score: f32,
    pub num_threads: Option<i32>,
    pub provider: Option<String>,
}

impl Default for KwsOpts {
    fn default() -> Self {
        Self {
            keywords_file: PathBuf::new(),
            threshold: 0.25,
            score: 3.0,
            num_threads: None,
            provider: None,
        }
    }
}

/// Owns the spotter + its stream; freed on drop. Touched only from one thread (the
/// capture loop), so `Send` is enough.
pub struct KeywordSpotter {
    spotter: *const sys::SherpaOnnxKeywordSpotter,
    stream: *const sys::SherpaOnnxOnlineStream,
}
unsafe impl Send for KeywordSpotter {}

impl Drop for KeywordSpotter {
    fn drop(&mut self) {
        unsafe {
            sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sys::SherpaOnnxDestroyKeywordSpotter(self.spotter);
        }
    }
}

impl KeywordSpotter {
    /// Build from a sherpa-onnx streaming KWS model dir (encoder/decoder/joiner ONNX
    /// + tokens.txt, auto-discovered) and a tokenized keywords file.
    pub fn from_dir(dir: &Path, opts: KwsOpts) -> Result<Self, String> {
        let tokens = dir.join("tokens.txt");
        if !tokens.is_file() {
            return Err(format!("tokens.txt not found in {}", dir.display()));
        }
        if !opts.keywords_file.is_file() {
            return Err(format!(
                "keywords file not found: {}",
                opts.keywords_file.display()
            ));
        }

        // Bind every CString until after create returns (sherpa copies them).
        let encoder = cstr(&crate::asr::find_onnx(dir, "encoder")?);
        let decoder = cstr(&crate::asr::find_onnx(dir, "decoder")?);
        let joiner = cstr(&crate::asr::find_onnx(dir, "joiner")?);
        let tokens_c = cstr(&tokens.to_string_lossy());
        let keywords_c = cstr(&opts.keywords_file.to_string_lossy());
        let provider = cstr(opts.provider.as_deref().unwrap_or("cpu"));

        let spotter = unsafe {
            let mut model: sys::SherpaOnnxOnlineModelConfig = std::mem::zeroed();
            model.transducer = sys::SherpaOnnxOnlineTransducerModelConfig {
                encoder: encoder.as_ptr(),
                decoder: decoder.as_ptr(),
                joiner: joiner.as_ptr(),
            };
            model.tokens = tokens_c.as_ptr();
            model.num_threads = opts.num_threads.unwrap_or(1);
            model.provider = provider.as_ptr();
            model.debug = 0;

            let mut config: sys::SherpaOnnxKeywordSpotterConfig = std::mem::zeroed();
            config.feat_config = sys::SherpaOnnxFeatureConfig {
                sample_rate: 16000,
                feature_dim: 80,
            };
            config.model_config = model;
            config.keywords_file = keywords_c.as_ptr();
            config.max_active_paths = 4;
            config.keywords_score = opts.score;
            config.keywords_threshold = opts.threshold;
            config.num_trailing_blanks = 1;

            sys::SherpaOnnxCreateKeywordSpotter(&config)
        };
        if spotter.is_null() {
            return Err("failed to create keyword spotter (see stderr)".into());
        }
        let stream = unsafe { sys::SherpaOnnxCreateKeywordStream(spotter) };
        if stream.is_null() {
            unsafe { sys::SherpaOnnxDestroyKeywordSpotter(spotter) };
            return Err("failed to create keyword stream".into());
        }
        Ok(Self { spotter, stream })
    }

    /// Feed a chunk of mono f32 audio; returns the phrase if a keyword fired within
    /// it. On a hit the stream is reset so the same utterance won't re-fire.
    pub fn accept(&mut self, samples: &[f32], sample_rate: u32) -> Option<String> {
        unsafe {
            sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
            while sys::SherpaOnnxIsKeywordStreamReady(self.spotter, self.stream) == 1 {
                sys::SherpaOnnxDecodeKeywordStream(self.spotter, self.stream);
            }
            let result = sys::SherpaOnnxGetKeywordResult(self.spotter, self.stream);
            let mut fired = None;
            if !result.is_null() {
                let kw = cstr_to_string((*result).keyword);
                if !kw.trim().is_empty() {
                    fired = Some(kw.trim().to_string());
                    sys::SherpaOnnxResetKeywordStream(self.spotter, self.stream);
                }
                sys::SherpaOnnxDestroyKeywordResult(result);
            }
            fired
        }
    }
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

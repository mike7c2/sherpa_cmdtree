//! The three ONNX graphs of a sherpa-onnx offline transducer, driven directly.
//!
//! The decoding contract is read from the graph shapes, not from metadata: the
//! zipformer2 checkpoints carry only descriptive metadata (`model_type`, `version`),
//! so `context_size` comes from the decoder's `y: [N, 2]` input and `vocab_size`
//! from the joiner's `logit: [N, 500]` output. `inspect` dumps all of it.

use std::path::Path;

use ort::session::Session;
use ort::value::{Value, ValueType};

/// The parts of a transducer needed to decode: three graphs plus the contract.
pub struct Transducer {
    pub encoder: Session,
    pub decoder: Session,
    pub joiner: Session,
    /// prediction-network context width (zipformer: 2)
    pub context_size: usize,
    pub vocab_size: usize,
}

/// Find the single `*keyword*.onnx` in `dir`, preferring full precision over int8 —
/// mirrors `cmdtree::asr::find_onnx` so both pick the same file.
pub fn find_onnx(dir: &Path, keyword: &str) -> Result<std::path::PathBuf, String> {
    let mut matches: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
            name.contains(keyword) && name.ends_with(".onnx")
        })
        .collect();
    matches.sort();
    matches.sort_by_key(|p| p.to_string_lossy().contains("int8"));
    matches
        .into_iter()
        .next()
        .ok_or_else(|| format!("no *{keyword}*.onnx in {}", dir.display()))
}

fn open(path: &Path, threads: usize) -> Result<Session, String> {
    let build = || -> Result<Session, Box<dyn std::error::Error>> {
        Ok(Session::builder()?
            .with_intra_threads(threads)?
            .commit_from_file(path)?)
    };
    build().map_err(|e| format!("loading {}: {e}", path.display()))
}

/// The last (innermost) dimension of an outlet's declared shape, when it is static.
fn trailing_dim(dtype: &ValueType, what: &str) -> Result<usize, String> {
    let ValueType::Tensor { shape, .. } = dtype else {
        return Err(format!("{what} is not a tensor"));
    };
    match shape.iter().last() {
        Some(&d) if d > 0 => Ok(d as usize),
        _ => Err(format!("{what} has no static trailing dimension (shape {shape:?})")),
    }
}

impl Transducer {
    pub fn load(dir: &Path, threads: usize) -> Result<Self, String> {
        let encoder = open(&find_onnx(dir, "encoder")?, threads)?;
        let decoder = open(&find_onnx(dir, "decoder")?, threads)?;
        let joiner = open(&find_onnx(dir, "joiner")?, threads)?;
        let context_size = trailing_dim(decoder.inputs()[0].dtype(), "decoder input `y`")?;
        let vocab_size = trailing_dim(joiner.outputs()[0].dtype(), "joiner output `logit`")?;
        Ok(Self { encoder, decoder, joiner, context_size, vocab_size })
    }

    /// Encoder forward for one utterance. `features` is [frames][80] row-major.
    /// Returns `(encoder_out, frames, encoder_dim)`, row-major over frames.
    pub fn encode(&mut self, features: &[f32], frames: usize) -> Result<(Vec<f32>, usize, usize), String> {
        let dim = features.len() / frames.max(1);
        let x = Value::from_array(([1usize, frames, dim], features.to_vec()))
            .map_err(|e| format!("encoder input: {e}"))?;
        let x_lens = Value::from_array(([1usize], vec![frames as i64]))
            .map_err(|e| format!("encoder input lens: {e}"))?;

        // names must be read before `run` borrows the session mutably
        let (in0, in1) = (
            self.encoder.inputs()[0].name().to_string(),
            self.encoder.inputs()[1].name().to_string(),
        );
        let name = self.encoder.outputs()[0].name().to_string();
        let outputs = self
            .encoder
            .run(ort::inputs![in0 => x, in1 => x_lens])
            .map_err(|e| format!("encoder run: {e}"))?;

        let (shape, data) = outputs[name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("encoder output: {e}"))?;
        // [1, T, encoder_dim]
        let out_frames = shape[1] as usize;
        let enc_dim = shape[2] as usize;
        Ok((data.to_vec(), out_frames, enc_dim))
    }

    /// Prediction network for one context window; returns the decoder state vector.
    pub fn decode_step(&mut self, context: &[i64]) -> Result<Vec<f32>, String> {
        let y = Value::from_array(([1usize, context.len()], context.to_vec()))
            .map_err(|e| format!("decoder input: {e}"))?;
        let input = self.decoder.inputs()[0].name().to_string();
        let name = self.decoder.outputs()[0].name().to_string();
        let outputs = self
            .decoder
            .run(ort::inputs![input => y])
            .map_err(|e| format!("decoder run: {e}"))?;
        let (_, data) = outputs[name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("decoder output: {e}"))?;
        Ok(data.to_vec())
    }

    /// Joiner: one encoder frame + one decoder state → logits over the vocab.
    pub fn join(&mut self, enc: &[f32], dec: &[f32]) -> Result<Vec<f32>, String> {
        let e = Value::from_array(([1usize, enc.len()], enc.to_vec()))
            .map_err(|e| format!("joiner encoder input: {e}"))?;
        let d = Value::from_array(([1usize, dec.len()], dec.to_vec()))
            .map_err(|e| format!("joiner decoder input: {e}"))?;
        let (in0, in1) = (
            self.joiner.inputs()[0].name().to_string(),
            self.joiner.inputs()[1].name().to_string(),
        );
        let name = self.joiner.outputs()[0].name().to_string();
        let outputs = self
            .joiner
            .run(ort::inputs![in0 => e, in1 => d])
            .map_err(|e| format!("joiner run: {e}"))?;
        let (_, data) = outputs[name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("joiner output: {e}"))?;
        Ok(data.to_vec())
    }

    /// Greedy RNN-T decode: walk the encoder frames, emitting the argmax token
    /// whenever it is not blank (id 0) and re-running the prediction network only
    /// when the context actually changes.
    pub fn greedy(&mut self, enc: &[f32], frames: usize, dim: usize) -> Result<Vec<i64>, String> {
        let mut context: Vec<i64> = vec![0; self.context_size]; // blank-padded start
        let mut state = self.decode_step(&context)?;
        let mut tokens = Vec::new();
        for t in 0..frames {
            let frame = &enc[t * dim..(t + 1) * dim];
            let logits = self.join(frame, &state)?;
            let best = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(0);
            if best != 0 {
                tokens.push(best);
                context.remove(0);
                context.push(best);
                state = self.decode_step(&context)?;
            }
        }
        Ok(tokens)
    }
}

/// Dump a model's graph contract — input/output names, shapes, and the custom
/// metadata that drives decoding. Run this first on any new model.
pub fn inspect(dir: &Path) -> Result<(), String> {
    for part in ["encoder", "decoder", "joiner"] {
        let path = find_onnx(dir, part)?;
        let session = open(&path, 1)?;
        println!("=== {part}: {}", path.file_name().unwrap_or_default().to_string_lossy());
        for i in session.inputs() {
            println!("   in   {:<24} {:?}", i.name(), i.dtype());
        }
        for o in session.outputs() {
            println!("   out  {:<24} {:?}", o.name(), o.dtype());
        }
        if part == "encoder" {
            let meta = session.metadata().map_err(|e| e.to_string())?;
            let mut keys = meta.custom_keys().map_err(|e| e.to_string())?;
            keys.sort();
            for k in keys {
                if let Some(v) = meta.custom(&k) {
                    println!("   meta {k} = {v}");
                }
            }
        }
    }
    Ok(())
}

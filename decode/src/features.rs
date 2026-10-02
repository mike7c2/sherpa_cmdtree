//! Audio in, sherpa-compatible fbank out.
//!
//! Features must match sherpa-onnx *exactly* or the encoder silently produces
//! garbage, so this calls the same C++ implementation sherpa does —
//! kaldi-native-fbank, via `knf-rs-sys` — rather than reimplementing it.
//!
//! The `knf-rs` safe wrapper is deliberately bypassed: it subtracts the feature
//! mean, which sherpa's offline transducer does not do.
//!
//! **Pinned options.** `knf-rs-sys` hard-codes its options in a C shim: 16 kHz,
//! dither 0, 80 mel bins, `snip_edges = true`, and Kaldi's default `high_freq`
//! (Nyquist). sherpa's own defaults differ on two of those — `snip_edges = false`
//! and `high_freq = -400` (⇒ 7600 Hz) — so if parity ever falls short, building a
//! configurable shim to vary exactly those two is the first thing to try. The pure
//! Rust port (`kaldi-native-fbank` 0.1) was evaluated for this and rejected: it
//! agrees with the C++ code on the first frame (~1e-5) but diverges later
//! (max |Δ| 5.55 over a 0.5 s tone) and yields garbage transcripts on real audio.

use std::path::Path;

pub const SAMPLE_RATE: u32 = 16_000;
pub const NUM_BINS: usize = 80;

/// Read a 16 kHz mono PCM16 WAV as f32 in [-1, 1]. Minimal RIFF walk — enough for
/// the corpus WAVs `cmdtree daemon --save-audio` writes and for espeak output.
pub fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{} is not a RIFF/WAVE file", path.display()));
    }
    let u16at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u32at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    let (mut pos, mut fmt) = (12usize, None);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32at(pos + 4) as usize;
        let body = pos + 8;
        match id {
            b"fmt " if body + 16 <= bytes.len() => {
                fmt = Some((u16at(body), u16at(body + 2), u32at(body + 4), u16at(body + 14)));
            }
            b"data" => {
                let (format, channels, rate, bits) =
                    fmt.ok_or_else(|| format!("{}: data before fmt", path.display()))?;
                if format != 1 || bits != 16 || channels != 1 || rate != SAMPLE_RATE {
                    return Err(format!(
                        "{}: need 16 kHz mono PCM16, got {rate} Hz, {channels} ch, {bits}-bit \
                         (format {format}). Convert with: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav",
                        path.display()
                    ));
                }
                let end = (body + size).min(bytes.len());
                return Ok(bytes[body..end]
                    .chunks_exact(2)
                    .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
                    .collect());
            }
            _ => {}
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    Err(format!("{}: no data chunk", path.display()))
}

/// Kaldi frame geometry at 16 kHz: 25 ms window, 10 ms shift.
const FRAME_LEN: usize = 400;
const FRAME_SHIFT: usize = 160;
/// `snip_edges=false` centres frame `t` at `t*shift + shift/2`, i.e. its window
/// starts at `t*shift − 120`, so mirror-padding by 120 either side turns the
/// pinned `snip_edges=true` extractor into the `false` one.
const EDGE_PAD: usize = (FRAME_LEN - FRAME_SHIFT) / 2;

/// Mirror-pad both ends the way Kaldi does for out-of-range indices (reflect about
/// the boundary sample), so a `snip_edges=true` extractor produces the frames a
/// `snip_edges=false` one would. See [`fbank`].
fn mirror_pad(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    let pad = EDGE_PAD.min(n);
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend(samples[..pad].iter().rev());
    out.extend_from_slice(samples);
    out.extend(samples[n - pad..].iter().rev());
    out
}

/// 80-dim log-mel fbank, row-major `[frames][80]`. Returns `(features, frames)`.
///
/// `snip_edges_false` emulates sherpa's framing over the pinned `snip_edges=true`
/// extractor (see the module note) by mirror-padding the signal.
pub fn fbank(samples: &[f32], snip_edges_false: bool) -> Result<(Vec<f32>, usize), String> {
    if samples.is_empty() {
        return Err("no audio samples".into());
    }
    let padded;
    let samples = if snip_edges_false {
        padded = mirror_pad(samples);
        &padded[..]
    } else {
        samples
    };
    let mut result = unsafe { knf_rs_sys::ComputeFbank(samples.as_ptr(), samples.len() as i32) };
    let frames = result.num_frames as usize;
    let bins = result.num_bins as usize;
    if frames == 0 || result.frames.is_null() {
        unsafe { knf_rs_sys::DestroyFbankResult(&mut result as *mut _) };
        return Err("fbank produced no frames (audio shorter than one 25 ms window?)".into());
    }
    let features = unsafe { std::slice::from_raw_parts(result.frames, frames * bins) }.to_vec();
    unsafe { knf_rs_sys::DestroyFbankResult(&mut result as *mut _) };
    if bins != NUM_BINS {
        return Err(format!("fbank returned {bins} bins, expected {NUM_BINS}"));
    }
    Ok((features, frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Framing check: 0.5 s at snip_edges=true ⇒ (8000−400)/160+1 = 48 frames of 80
    /// bins. Catches a layout or option regression in the C shim.
    #[test]
    fn fbank_shape_matches_kaldi_framing() {
        let samples: Vec<f32> = (0..8000)
            .map(|i| (i as f32 / SAMPLE_RATE as f32 * 440.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect();
        let (features, frames) = fbank(&samples, false).unwrap();
        assert_eq!(frames, 48);
        assert_eq!(features.len(), frames * NUM_BINS);
        assert!(features.iter().all(|f| f.is_finite()));
    }
}

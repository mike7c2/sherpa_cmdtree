//! Minimal PCM-16 mono WAV encoder (no external crate), used to hand a captured
//! utterance to an external ASR command.

/// Encode mono 16-bit PCM samples as a WAV byte buffer at `sample_rate` Hz.
pub fn encode_pcm16_mono(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let byte_rate = sample_rate * 2; // mono, 2 bytes/sample
    let data_len = (samples.len() * 2) as u32;
    let mut v = Vec::with_capacity(44 + data_len as usize);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVE");
    v.extend_from_slice(b"fmt ");
    v.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    v.extend_from_slice(&1u16.to_le_bytes()); // PCM
    v.extend_from_slice(&1u16.to_le_bytes()); // channels = 1
    v.extend_from_slice(&sample_rate.to_le_bytes());
    v.extend_from_slice(&byte_rate.to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes()); // block align
    v.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

/// Convert f32 samples in [-1,1] to i16 PCM.
pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect()
}

/// Naive linear resample from `from_hz` to `to_hz`.
pub fn resample_linear(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_hz as f64 / from_hz as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let j = src.floor() as usize;
        let frac = src - j as f64;
        let a = input[j.min(input.len() - 1)];
        let b = input[(j + 1).min(input.len() - 1)];
        out.push(a + (b - a) * frac as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_44_bytes_and_riff() {
        let w = encode_pcm16_mono(&[0, 1, -1, 100], 16000);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(&w[8..12], b"WAVE");
        assert_eq!(w.len(), 44 + 4 * 2);
        // data length field
        let data_len = u32::from_le_bytes([w[40], w[41], w[42], w[43]]);
        assert_eq!(data_len, 8);
    }

    #[test]
    fn resample_halves_length() {
        let input = vec![0.0f32; 1000];
        let out = resample_linear(&input, 32000, 16000);
        assert!((out.len() as i32 - 500).abs() <= 1);
    }
}

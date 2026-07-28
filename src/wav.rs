//! PCM-16 mono WAV encoding, f32→i16 conversion, and streaming linear resampling
//! for captured audio.

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

/// Streaming linear resampler. Carries the fractional read position and the last
/// input sample across calls, so a chunked stream resamples identically to one
/// continuous buffer (no phase reset or edge clamp at frame boundaries).
pub struct StreamResampler {
    from_hz: u32,
    to_hz: u32,
    /// source position of the next output sample, relative to the start of the
    /// next input frame (negative: between `prev` and that frame's first sample).
    pos: f64,
    /// last sample of the previous frame.
    prev: f32,
}

impl StreamResampler {
    pub fn new(from_hz: u32, to_hz: u32) -> Self {
        Self { from_hz, to_hz, pos: 0.0, prev: 0.0 }
    }

    /// Resample one frame, continuing from where the previous call left off.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.from_hz == self.to_hz || input.is_empty() {
            return input.to_vec();
        }
        let step = self.from_hz as f64 / self.to_hz as f64;
        let mut out = Vec::with_capacity((input.len() as f64 / step) as usize + 2);
        while self.pos <= input.len() as f64 - 1.0 {
            let (a, b, frac) = if self.pos < 0.0 {
                (self.prev, input[0], self.pos + 1.0)
            } else {
                let j = self.pos as usize;
                (input[j], input[(j + 1).min(input.len() - 1)], self.pos - j as f64)
            };
            out.push(a + (b - a) * frac as f32);
            self.pos += step;
        }
        self.pos -= input.len() as f64;
        self.prev = input[input.len() - 1];
        out
    }
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
        let out = StreamResampler::new(32000, 16000).process(&vec![0.0f32; 1000]);
        assert!((out.len() as i32 - 500).abs() <= 1);
    }

    #[test]
    fn identity_rate_passes_through() {
        assert_eq!(StreamResampler::new(16000, 16000).process(&[0.5, -0.5]), [0.5, -0.5]);
    }

    #[test]
    fn chunked_matches_one_shot() {
        // continuity: resampling in arbitrary chunks must equal one whole-buffer pass
        let input: Vec<f32> = (0..4410).map(|i| (i as f32 * 0.01).sin()).collect();
        let expect = StreamResampler::new(44100, 16000).process(&input);
        let mut r = StreamResampler::new(44100, 16000);
        let got: Vec<f32> = input.chunks(447).flat_map(|c| r.process(c)).collect();
        assert_eq!(got.len(), expect.len());
        for (x, y) in got.iter().zip(&expect) {
            assert!((x - y).abs() < 1e-4);
        }
    }
}

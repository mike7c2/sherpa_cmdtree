//! One blocking 16 kHz mono f32 record stream, via `libpulse-simple-binding`.
//! The PulseAudio-compatible server (pipewire-pulse / pulseaudio) owns resampling,
//! downmixing, and mic sharing — cmdtree just reads frames.

use libpulse_binding::def::BufferAttr;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

pub const SAMPLE_RATE: u32 = 16_000;

/// A blocking 16 kHz mono f32 record stream from the server's default source.
pub struct Recorder {
    simple: Simple,
    /// byte-side scratch for `Simple::read`, decoded into f32 per frame.
    bytes: Vec<u8>,
}

impl Recorder {
    /// Open the stream, asking the server for `frag_samples`-sized delivery so
    /// capture latency stays at one fragment.
    pub fn open(frag_samples: usize) -> Result<Self, String> {
        let spec = Spec {
            format: Format::F32le,
            channels: 1,
            rate: SAMPLE_RATE,
        };
        let frag_bytes = frag_samples * std::mem::size_of::<f32>();
        // u32::MAX leaves a field at the server default; only fragsize matters here.
        let attr = BufferAttr {
            maxlength: u32::MAX,
            tlength: u32::MAX,
            prebuf: u32::MAX,
            minreq: u32::MAX,
            fragsize: frag_bytes as u32,
        };
        let simple = Simple::new(
            None,              // default server
            "cmdtree",         // application name (shown in pavucontrol)
            Direction::Record,
            None,              // default source
            "capture",         // stream description
            &spec,
            None,              // default channel map
            Some(&attr),
        )
        .map_err(|e| {
            format!(
                "cannot open pulse record stream: {e} (is a PulseAudio-compatible server \
                 running? PipeWire desktops have one; headless boxes need pipewire-pulse)"
            )
        })?;
        Ok(Self { simple, bytes: vec![0; frag_bytes] })
    }

    /// Blocking read of exactly `out.len()` samples (`out` must be the fragment size).
    pub fn read(&mut self, out: &mut [f32]) -> Result<(), String> {
        debug_assert_eq!(out.len() * 4, self.bytes.len());
        self.simple
            .read(&mut self.bytes)
            .map_err(|e| format!("pulse read failed: {e}"))?;
        for (s, b) in out.iter_mut().zip(self.bytes.chunks_exact(4)) {
            *s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        Ok(())
    }
}

//! Push-to-talk daemon: hold a global hotkey, capture the mic, transcribe, parse.
//!
//! Feature-gated (`--features daemon`) because it links native audio (cpal/ALSA).
//! The hotkey is read from `/dev/input` via evdev, so it works under both X11 and
//! Wayland (unlike an X11 key-grab); it needs read access to the input devices,
//! i.e. membership in the `input` group. ASR is pluggable via the [`Recognizer`]
//! trait; the default backend shells out to an external command (so it works today
//! with the Python transducer). A native sherpa-onnx backend can be added behind
//! its own feature.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::asr::{ExternalCommand, Recognizer};
use crate::grammar::Grammar;
use crate::hotkey::Combo;
use crate::wav;

pub struct DaemonConfig {
    pub hotkey: String,
    /// external ASR command template; `{wav}` is replaced with a temp wav path,
    /// transcript is read from stdout. Ignored when `sherpa_model` is set.
    pub asr_cmd: Option<String>,
    /// directory of a sherpa-onnx transducer model; selects the native backend
    /// (feature `sherpa`) in preference to `asr_cmd`.
    pub sherpa_model: Option<PathBuf>,
    pub sherpa_provider: Option<String>,
    pub sherpa_threads: Option<i32>,
    /// if set, every processed utterance is written here as a 16 kHz mono WAV plus
    /// a `manifest.jsonl` line (transcript + parsed command) — for post-processing
    /// or building a dataset of the user's own voice.
    pub save_audio: Option<PathBuf>,
    /// auto-derive a hotword list from the grammar and bias the (sherpa) recognizer
    /// toward it. On by default; needs the sherpa backend + a `bpe.model` in the dir.
    pub bias: bool,
    /// boost applied to biased hotword phrases.
    pub bias_score: f32,
    /// PTT: let a trailing free-text slot omit its end anchor (the key-release bounds
    /// the utterance). The anchor stays in the grammar and is still used when spoken.
    pub open_ended: bool,
    pub sample_rate: u32,
}

/// Milliseconds since the Unix epoch.
fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build the NDJSON event emitted on stdout for one recognised utterance: the
/// structured intent a downstream executor acts on (`command`/`action`/`slots` are
/// null/empty when nothing matched).
fn event_json(transcript: &str, matched: Option<&crate::parser::Match>, ts: u64) -> String {
    let slots: serde_json::Map<String, serde_json::Value> = matched
        .map(|m| {
            m.slots
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect()
        })
        .unwrap_or_default();
    let event = serde_json::json!({
        "ts": ts,
        "command": matched.map(|m| m.command.clone()),
        "action": matched.and_then(|m| m.action.clone()),
        "slots": slots,
        "transcript": transcript,
    });
    serde_json::to_string(&event).unwrap_or_default()
}

/// Write one processed utterance to `dir`: a timestamped 16 kHz mono WAV and an
/// appended `manifest.jsonl` row tying the audio to its transcript and parsed command.
fn save_snippet(
    dir: &std::path::Path,
    samples: &[f32],
    sample_rate: u32,
    transcript: &str,
    matched: Option<&crate::parser::Match>,
    words: &[(String, f32)],
) -> std::io::Result<()> {
    use std::io::Write;
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let stem = format!("utterance-{ms}");
    let pcm = wav::f32_to_i16(samples);
    std::fs::write(dir.join(format!("{stem}.wav")), wav::encode_pcm16_mono(&pcm, sample_rate))?;

    let slots: serde_json::Map<String, serde_json::Value> = matched
        .map(|m| {
            m.slots
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect()
        })
        .unwrap_or_default();
    // per-word recognition detail (which words came out, and when) for post-analysis
    let words_json: Vec<serde_json::Value> = words
        .iter()
        .map(|(w, t)| serde_json::json!({ "w": w, "t": t }))
        .collect();
    let entry = serde_json::json!({
        "wav": format!("{stem}.wav"),
        "epoch_ms": ms as u64,
        "transcript": transcript,
        "command": matched.map(|m| m.command.clone()),
        "action": matched.and_then(|m| m.action.clone()),
        "slots": slots,
        "words": words_json,
    });
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("manifest.jsonl"))?;
    writeln!(f, "{entry}")?;
    Ok(())
}

/// Pick the ASR backend from the config: native sherpa-onnx if a model dir is
/// given, otherwise the external command. `hotwords_file` (when set) biases the
/// native recognizer toward the grammar's phrases.
fn build_recognizer(
    cfg: &DaemonConfig,
    hotwords_file: Option<&std::path::Path>,
) -> Result<Box<dyn Recognizer>, String> {
    if let Some(dir) = &cfg.sherpa_model {
        #[cfg(feature = "sherpa")]
        {
            return crate::asr::SherpaTransducer::from_dir(
                dir,
                crate::asr::SherpaOpts {
                    num_threads: cfg.sherpa_threads,
                    provider: cfg.sherpa_provider.clone(),
                    hotwords_file: hotwords_file.map(|p| p.to_path_buf()),
                    hotwords_score: cfg.bias_score,
                },
            )
            .map(|r| Box::new(r) as Box<dyn Recognizer>);
        }
        #[cfg(not(feature = "sherpa"))]
        {
            let _ = (dir, hotwords_file);
            return Err("--sherpa-model needs the `sherpa` feature \
                 (rebuild: cargo build --features daemon,sherpa)"
                .to_string());
        }
    }
    if let Some(template) = &cfg.asr_cmd {
        return Ok(Box::new(ExternalCommand {
            template: template.clone(),
        }));
    }
    Err("no ASR backend selected: pass --sherpa-model DIR (native) \
         or --asr-cmd CMD (external, e.g. mic_asr.py --wav {wav})"
        .to_string())
}

/// Map a live evdev key to the same normalized token space as [`crate::hotkey`].
/// evdev's `Debug` prints the kernel constant name (`KEY_LEFTCTRL`, `KEY_SPACE`,
/// `KEY_A`, `KEY_1`, ...), so we strip the `KEY_` prefix and fold left/right
/// variants together: "KEY_LEFTCTRL"→ctrl, "KEY_A"→a, "KEY_1"→1, "KEY_SPACE"→space.
fn norm_key(k: evdev::Key) -> String {
    let d = format!("{k:?}").to_lowercase();
    let d = d.strip_prefix("key_").unwrap_or(&d);
    if d.contains("ctrl") {
        "ctrl".into()
    } else if d.contains("shift") {
        "shift".into()
    } else if d.contains("alt") {
        "alt".into()
    } else if d.contains("meta") || d.contains("super") {
        "meta".into()
    } else if d == "space" {
        "space".into()
    } else if d == "enter" || d == "kpenter" {
        "enter".into()
    } else if d == "esc" {
        "escape".into()
    } else {
        d.to_string()
    }
}

/// A device is treated as a keyboard if it can produce Enter — enough to exclude
/// mice/touchpads (which report `BTN_*`, not `KEY_ENTER`) while catching laptop
/// keyboards, external keyboards, and most programmable input devices.
fn is_keyboard(dev: &evdev::Device) -> bool {
    dev.supported_keys()
        .is_some_and(|keys| keys.contains(evdev::Key::KEY_ENTER))
}

/// Open every keyboard under `/dev/input`. We read the nodes directly (rather than
/// `evdev::enumerate`, which silently skips devices it can't open) so a permission
/// problem produces an actionable message instead of "no keyboards".
fn open_keyboards() -> Result<Vec<evdev::Device>, String> {
    let mut keyboards = Vec::new();
    let mut permission_denied = false;
    let mut last_err: Option<std::io::Error> = None;

    let entries = std::fs::read_dir("/dev/input")
        .map_err(|e| format!("cannot read /dev/input: {e}"))?;
    let mut paths: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("event"))
        })
        .collect();
    paths.sort();

    for path in paths {
        match evdev::Device::open(&path) {
            Ok(dev) if is_keyboard(&dev) => keyboards.push(dev),
            Ok(_) => {}
            Err(e) => {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    permission_denied = true;
                }
                last_err = Some(e);
            }
        }
    }

    if keyboards.is_empty() {
        if permission_denied {
            return Err(
                "no readable keyboard devices in /dev/input (permission denied).\n\
                 Add yourself to the `input` group and log back in:\n    \
                 sudo usermod -aG input $USER"
                    .to_string(),
            );
        }
        return Err(format!(
            "no keyboard devices found in /dev/input{}",
            last_err
                .map(|e| format!(" (last error: {e})"))
                .unwrap_or_default()
        ));
    }
    Ok(keyboards)
}

/// Read key events from one device forever, maintaining the shared `pressed` set
/// and driving the record/stop transition when the combo becomes (un)satisfied.
/// One of these runs per keyboard so a modifier and key on different devices still
/// combine. The `pressed` lock serializes transitions across devices.
fn monitor_device(
    mut dev: evdev::Device,
    combo: Combo,
    pressed: Arc<Mutex<HashSet<String>>>,
    recording: Arc<AtomicBool>,
    buffer: Arc<Mutex<Vec<f32>>>,
    tx: mpsc::Sender<Vec<f32>>,
) {
    loop {
        let events = match dev.fetch_events() {
            Ok(events) => events,
            Err(e) => {
                eprintln!("input device closed ({e}); stopping monitor for it");
                return;
            }
        };
        for ev in events {
            let evdev::InputEventKind::Key(key) = ev.kind() else {
                continue;
            };
            let token = norm_key(key);
            let mut p = pressed.lock().unwrap();
            match ev.value() {
                0 => {
                    p.remove(&token);
                } // release
                1 => {
                    p.insert(token);
                } // press
                _ => continue, // 2 = autorepeat
            }
            let now = combo.is_satisfied(&p);
            let was = recording.load(Ordering::SeqCst);
            if now && !was {
                buffer.lock().unwrap().clear();
                recording.store(true, Ordering::SeqCst);
                eprintln!("🎤 listening…");
            } else if !now && was {
                recording.store(false, Ordering::SeqCst);
                let samples = buffer.lock().unwrap().clone();
                let _ = tx.send(samples);
            }
        }
    }
}

fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Build a continuously-running input stream that appends mono f32 to `buffer`
/// whenever `recording` is true. Returns the stream (keep alive) and its rate.
fn build_input_stream(
    recording: Arc<AtomicBool>,
    buffer: Arc<Mutex<Vec<f32>>>,
) -> Result<(cpal::Stream, u32), String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "no default input device".to_string())?;
    let supported = device
        .default_input_config()
        .map_err(|e| e.to_string())?;
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let config: cpal::StreamConfig = supported.clone().into();
    let err_fn = |e| eprintln!("audio stream error: {e}");

    macro_rules! push {
        ($to_f32:expr) => {{
            let rec = recording.clone();
            let buf = buffer.clone();
            move |data: &[_], _: &cpal::InputCallbackInfo| {
                if rec.load(Ordering::SeqCst) {
                    let f: Vec<f32> = data.iter().map(|&s| $to_f32(s)).collect();
                    buf.lock().unwrap().extend_from_slice(&downmix(&f, channels));
                }
            }
        }};
    }

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            push!(|s: f32| s),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            push!(|s: i16| s as f32 / 32768.0),
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            push!(|s: u16| (s as f32 - 32768.0) / 32768.0),
            err_fn,
            None,
        ),
        other => return Err(format!("unsupported sample format: {other:?}")),
    }
    .map_err(|e| e.to_string())?;

    Ok((stream, sample_rate))
}

pub fn run(grammar: Grammar, cfg: DaemonConfig) -> Result<(), String> {
    let combo = Combo::parse(&cfg.hotkey);
    if combo.keys.is_empty() {
        return Err(format!("could not parse hotkey {:?}", cfg.hotkey));
    }

    // Auto-derive a hotword bias list from the command tree so biasing always
    // matches the grammar (written to a temp file the sherpa backend reads).
    let hotwords_path: Option<PathBuf> = if cfg.bias && cfg.sherpa_model.is_some() {
        let phrases = crate::hotwords::hotwords(&grammar);
        let path = std::env::temp_dir().join(format!("cmdtree-hotwords-{}.txt", std::process::id()));
        std::fs::write(&path, crate::hotwords::hotwords_file(&grammar, None))
            .map_err(|e| format!("cannot write hotwords file {}: {e}", path.display()))?;
        eprintln!(
            "biasing {} hotword phrases derived from the grammar (score {})",
            phrases.len(),
            cfg.bias_score
        );
        Some(path)
    } else {
        None
    };

    let recognizer = build_recognizer(&cfg, hotwords_path.as_deref())?;

    if let Some(dir) = &cfg.save_audio {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create --save-audio dir {}: {e}", dir.display()))?;
        eprintln!(
            "saving processed audio + manifest.jsonl to {}",
            dir.display()
        );
    }

    let recording = Arc::new(AtomicBool::new(false));
    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let (stream, in_rate) = build_input_stream(recording.clone(), buffer.clone())?;
    stream.play().map_err(|e| e.to_string())?;

    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    // hotkey monitoring: evdev reads /dev/input directly, so it works on X11 and
    // Wayland alike. One thread per keyboard, all sharing the `pressed` set.
    let keyboards = open_keyboards()?;
    eprintln!(
        "monitoring {} keyboard device(s) for [{}]",
        keyboards.len(),
        cfg.hotkey
    );
    let pressed = Arc::new(Mutex::new(HashSet::<String>::new()));
    for dev in keyboards {
        let combo = combo.clone();
        let pressed = pressed.clone();
        let recording = recording.clone();
        let buffer = buffer.clone();
        let tx = tx.clone();
        std::thread::spawn(move || monitor_device(dev, combo, pressed, recording, buffer, tx));
    }
    // The monitor threads hold their own senders; drop ours so the loop below only
    // ends if every keyboard monitor exits.
    drop(tx);

    // stdout carries ONE NDJSON event per utterance (machine-readable, for a
    // downstream executor); everything human-readable goes to stderr, so the two
    // can be piped apart.
    eprintln!(
        "cmdtree daemon ready. Hold [{}] and speak; release to recognise. Ctrl-C to quit.",
        cfg.hotkey
    );

    for samples in rx {
        if samples.is_empty() {
            continue;
        }
        let resampled = wav::resample_linear(&samples, in_rate, cfg.sample_rate);
        let (transcript, matched, words) = match recognizer.recognize(&resampled, cfg.sample_rate) {
            Ok(res) => {
                eprintln!("heard: {:?}", res.text);
                let words = res.words();
                if !words.is_empty() {
                    let line: Vec<String> =
                        words.iter().map(|(w, t)| format!("{w}@{t:.2}")).collect();
                    eprintln!("  words: {}", line.join(" "));
                }
                let m = grammar
                    .parse_with(&res.text, crate::parser::ParseOpts { open_ended: cfg.open_ended });
                match &m {
                    Some(m) => {
                        let slots: Vec<String> =
                            m.slots.iter().map(|(k, v)| format!("{k}={v:?}")).collect();
                        eprintln!(
                            "  → {}{}  [{}]",
                            m.command,
                            m.action.as_ref().map(|a| format!(" ({a})")).unwrap_or_default(),
                            slots.join(", ")
                        );
                    }
                    None => eprintln!("  → (no command recognised)"),
                }
                // machine-readable intent → stdout (NDJSON). stdout is line-buffered,
                // so each event flushes on its newline.
                println!("{}", event_json(&res.text, m.as_ref(), epoch_ms()));
                (res.text, m, words)
            }
            Err(e) => {
                eprintln!("asr error: {e}");
                (String::new(), None, Vec::new())
            }
        };
        if let Some(dir) = &cfg.save_audio {
            if let Err(e) = save_snippet(
                dir,
                &resampled,
                cfg.sample_rate,
                &transcript,
                matched.as_ref(),
                &words,
            ) {
                eprintln!("could not save audio snippet: {e}");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{event_json, norm_key, save_snippet};
    use evdev::Key;

    #[test]
    fn event_json_shape() {
        // matched command → populated intent
        let m = crate::parser::Match {
            command: "light_power".into(),
            action: Some("light.power".into()),
            slots: vec![("state".into(), "on".into()), ("device".into(), "kitchen light".into())],
            score: 3,
            leftover: vec![],
        };
        let j = event_json("SWITCH ON THE KITCHEN LIGHT", Some(&m), 42);
        assert!(j.contains(r#""ts":42"#));
        assert!(j.contains(r#""command":"light_power""#));
        assert!(j.contains(r#""action":"light.power""#));
        assert!(j.contains(r#""state":"on""#));
        assert!(j.contains(r#""transcript":"SWITCH ON THE KITCHEN LIGHT""#));
        // no match → nulls + empty slots, still one clean line
        let none = event_json("HELLO THERE", None, 7);
        assert!(none.contains(r#""command":null"#));
        assert!(none.contains(r#""action":null"#));
        assert!(none.contains(r#""slots":{}"#));
        assert_eq!(none.lines().count(), 1);
    }

    #[test]
    fn save_snippet_writes_wav_and_manifest_row() {
        let dir = std::env::temp_dir().join(format!("cmdtree-savetest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let samples = vec![0.0f32; 1600]; // 0.1 s @ 16 kHz
        let words = [("TURN".to_string(), 0.1), ("ON".to_string(), 0.4)];
        save_snippet(&dir, &samples, 16000, "TURN ON THE LAMP", None, &words).unwrap();

        // exactly one WAV, with a valid RIFF header
        let wavs: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
            .collect();
        assert_eq!(wavs.len(), 1);
        assert_eq!(&std::fs::read(wavs[0].path()).unwrap()[0..4], b"RIFF");

        // one manifest row linking audio → transcript
        let manifest = std::fs::read_to_string(dir.join("manifest.jsonl")).unwrap();
        assert_eq!(manifest.lines().count(), 1);
        assert!(manifest.contains(r#""transcript":"TURN ON THE LAMP""#));
        assert!(manifest.contains(r#""epoch_ms":"#));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn maps_evdev_keys_to_combo_tokens() {
        // left/right modifier variants fold to one token, matching hotkey::Combo.
        assert_eq!(norm_key(Key::KEY_LEFTCTRL), "ctrl");
        assert_eq!(norm_key(Key::KEY_RIGHTALT), "alt");
        assert_eq!(norm_key(Key::KEY_LEFTSHIFT), "shift");
        assert_eq!(norm_key(Key::KEY_LEFTMETA), "meta");
        assert_eq!(norm_key(Key::KEY_SPACE), "space");
        // both Enter keys and Escape line up with the hotkey aliases.
        assert_eq!(norm_key(Key::KEY_ENTER), "enter");
        assert_eq!(norm_key(Key::KEY_KPENTER), "enter");
        assert_eq!(norm_key(Key::KEY_ESC), "escape");
        // plain letters/digits pass through.
        assert_eq!(norm_key(Key::KEY_A), "a");
        assert_eq!(norm_key(Key::KEY_1), "1");
    }

    #[test]
    fn evdev_token_satisfies_parsed_combo() {
        use crate::hotkey::Combo;
        use std::collections::HashSet;
        // a right-ctrl + space press must satisfy a "ctrl+space" combo.
        let combo = Combo::parse("ctrl+space");
        let pressed: HashSet<String> = [Key::KEY_RIGHTCTRL, Key::KEY_SPACE]
            .into_iter()
            .map(norm_key)
            .collect();
        assert!(combo.is_satisfied(&pressed));
    }
}

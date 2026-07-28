# cmdtree — Specification

A lean, offline **voice command** recogniser: hold a hotkey, speak, release, and a
transcript from a local sherpa-onnx model is matched against a YAML command tree.
Free-text dictation is available as an explicit command. On-device, **tiny latency**,
no cloud, no LLM.

Status: working end-to-end (pure push-to-talk; wake-word support was built, then
removed — see §2). Audio is native libpulse (§7). Anything under **Deferred** is
out of scope until explicitly pulled in.

---

## 1. Purpose & core elements

Three things, deliberately kept small:

1. **sherpa** — recognition by a local sherpa-onnx offline transducer.
2. **a command grammar** — a YAML tree of commands with typed enum choices and free-text slots.
3. **tools that make it effective** — a confusability linter, auto-derived hotword biasing,
   and per-utterance audio+transcript capture.

## 2. Scope

**In scope**
- **Pure push-to-talk**: press starts capture, release ends it and triggers recognition
  immediately. The key is the only authority — no VAD, no silence thresholds, no
  timeouts, no caps. Latency is a feature: release → NDJSON event ≈ decode time.
- Deterministic grammar matching → a typed command (`name`, `action`, `slots`).
- Exactly one style of free text: a **slot**, bounded by an end anchor (or by
  key-release when open-ended).
- Tooling: grammar quality assessment, hotword generation, voice-data capture.

**Out of scope (Non-goals)**
- **Wake-word support, in any form.** In-process KWS was built and removed; an external
  trigger port (`start`/`stop` on stdin) was specced and rejected — both trade the
  simplicity and latency of pure PTT for end-pointing machinery (VAD, timeouts, pre-roll).
  Git history has both designs if the need ever really returns.
- **Audio routing/sharing.** cmdtree opens one capture stream from the sound server;
  anything fancier (fan-out, echo cancel, device juggling) is the server's job.
- Streaming / always-on recognition as the engine.
- LLM intent parsing; cloud services.
- Automatic "unrecognised → dictation" fallback. Dictation is an **explicit** command.

## 3. Architecture

A Cargo **workspace**: the recognizer crate `cmdtree` (lib + `cmdtree` CLI) and a
separate grammar-authoring crate `cmdtree-lint` (`lint/`). The runtime crate carries
no quality-linting code.

`cmdtree` modules:

| Module | Responsibility | Deps |
|---|---|---|
| `grammar` | YAML command tree → compiled + validated form | core |
| `parser` | match a transcript → `Match{command, action, slots, score, leftover}` | core |
| `hotwords` | derive the sherpa hotword bias list from the grammar | core |
| `wav`, `hotkey` | PCM16 WAV encode (for `--save-audio`); hotkey-combo parsing | core |
| `asr` | `SherpaTransducer` (sherpa-onnx recognizer) → `AsrResult` | feature `sherpa` |
| `pulse` | blocking 16 kHz mono capture (libpulse-simple-binding) | feature `daemon` |
| `daemon` | hotkey-gated capture → ASR → parse → emit | feature `daemon` |

`cmdtree-lint` modules (dev tool, depends on `cmdtree` for the grammar types):
`phonetics` (espeak G2P + feature-weighted phone distance) and `assess` (confusability
linter). Needs espeak at runtime; not part of the recognizer.

**Cargo features (cmdtree):** `default = []` · `sherpa = [sherpa-rs]` ·
`daemon = [evdev, libpulse-binding, libpulse-simple-binding, sherpa]` ·
`sherpa-cuda = [sherpa, sherpa-rs/cuda]`. Core (grammar/parser/hotwords) builds with no
native deps.

**CLI:** `cmdtree` → `parse` · `hotwords` · `daemon` · `transcribe`.
`cmdtree-lint <grammar> [--json]` → the quality report.

## 4. Command grammar

YAML. Elements: `{lit: "words"}` (fixed), `{one_of: {name, options}}` (bounded enum →
captured slot), `{slot: name}` (free text). Rules:

- A `{slot}` **must be followed by a literal/enum end anchor** — validated at load time,
  so free text is bounded on both sides. This keeps the matcher stateless.
- **Design idiom for low confusability:** factor a shared enum suffix into a literal
  (`{one_of: [two,five,ten]} {lit: minutes}`, not options `"five minutes"…`). Verified
  with `cmdtree-lint`; the shipped `grammar.yaml` scores ≥98/100 with zero confusable pairs.

`grammar.yaml` is the reference tree (11 commands: light_power, brightness, scene, volume,
play_music, pause, timer, weather, navigate, phone, dictate_note).

**Parsing (`ParseOpts`):** `parse()` is strict (anchors required). `open_ended` (PTT) lets a
**trailing** slot capture to the utterance end when its anchor wasn't spoken — the anchor
stays in the grammar and is still consumed when present.

## 5. ASR

One backend: `asr::SherpaTransducer` (feature `sherpa`).
`recognize(samples, rate) -> AsrResult{ text, tokens:[{text,start}] }`.

- sherpa-onnx **offline** transducer driven **directly via `sherpa-rs-sys`** (not the
  high-level wrapper, so tokens + timestamps are available). 16 kHz mono. `words()`
  merges sub-words → words (boundary = leading space).
- No trait / no external-command backend: the recognizer is concrete, so `daemon`
  implies `sherpa` and `--sherpa-model` is required.

**Hotword biasing** (auto, on by default): the daemon derives a bias list from the grammar
(`hotwords`), writes it, and enables contextual biasing. Requires a `bpe.vocab` in the model
dir and forces `modified_beam_search`; without `bpe.vocab` it warns and skips.

**No token probabilities** — the sherpa-onnx C API exposes tokens + timestamps only.
Grammar-level confidence, if needed, comes from `Match.score`.

## 6. Interfaces & data

**Daemon run:**
```
cmdtree daemon grammar.yaml --hotkey ctrl+space \
    --sherpa-model <onnx-transducer-dir> [--open-ended] [--save-audio DIR] \
    [--no-bias] [--bias-score N]
```

**Capture model.** The mic stream runs continuously; the hotkey gates it. Press starts
buffering — including the frame the press landed in, so speech beginning with the press
isn't clipped — and release hands the utterance straight to the recognizer. Hold and
pause as long as you like.

**stdout / stderr split (the integration contract).** cmdtree stays a recognizer and
just *emits*; a separate executor acts (see §9). Per utterance:
- **stdout** — exactly one **NDJSON** event (machine-readable, line-buffered):
  `{ts, command, action, slots, transcript}`. `command`/`action` are `null` and `slots`
  `{}` when nothing matched. This is the only thing on stdout — pipe it to an executor.
- **stderr** — all human-readable diagnostics: capture duration, `heard: …`,
  `words: w@t …`, the matched command, startup/biasing/errors. Pipe/redirect separately.

**`--save-audio DIR`** writes `utterance-<epoch_ms>.wav` (16 kHz mono) + appends
`manifest.jsonl`: `{wav, epoch_ms, transcript, command, action, slots, words:[{w,t}]}` —
the dataset for seeing which words work on the user's voice. The WAV stem, manifest
`epoch_ms`, and event `ts` agree.

## 7. Constraints & prerequisites

- **Model:** a sherpa-onnx **offline** transducer (encoder/decoder/joiner `.onnx` +
  `tokens.txt`), auto-discovered (full-precision preferred over int8). This is an **ONNX**
  model, distinct from the repo's TorchScript streaming model. `bpe.vocab` needed for biasing.
- **Hotkey:** evdev reads `/dev/input` → works on X11 **and** Wayland, but needs
  membership in the `input` group.
- **Audio:** libpulse via `libpulse-simple-binding` (built with `default-features = false`
  = oldest PA API baseline, so it runs against any newer server), requiring a
  PulseAudio-compatible server at runtime (PipeWire's `pipewire-pulse`, or PulseAudio).
  The server delivers 16 kHz mono f32 directly — no client-side resampling/downmix —
  in 10 ms fragments, and mic sharing with other consumers is the server's job.
- **espeak** required by `cmdtree-lint` only (predictive proxy; not a runtime dep).
- **Latency budget (the core goal):** press → capture is immediate (the stream never
  stops; the press frame is included). Release → NDJSON event ≈ offline decode ≈ 20 ms
  for a short command, model resident. Model load ≈ 2 s once at daemon start.

## 8. Quality bar

`cargo test` green (core and `--features daemon`); `cargo clippy` clean across all feature
combos; `cargo package --list` clean.

## 9. Deferred (out of scope until pulled in)

- **Executor:** a **separate program** consuming the stdout NDJSON stream and performing
  effects (HA REST service calls, `wl-copy`/`wtype`, `wpctl`, dbus via `playerctl`/`gdbus`,
  custom scripts). cmdtree never grows built-in integrations; `action`+`slots` is the contract.
  Optional extra-hotwords list (names/jargon, e.g. "Tuvok") merged with the grammar-derived
  bias list is the one biasing add still worth doing.
- `--source NAME` pulse source selector (trivial via `pa_simple`; add when needed).
- Surfacing `Match.score` as a confidence readout / manifest field.
- Fine-tuning the transducer on the user's voice (uses the `--save-audio` corpus).
- CUDA execution provider (`sherpa-cuda`) — not needed for command latency.

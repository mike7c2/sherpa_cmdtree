# cmdtree — Specification

A lean, offline **voice command** recogniser: hold a hotkey, speak, and a transcript
from a local sherpa-onnx model is matched against a YAML command tree. Free-text
dictation is available as an explicit command. On-device, low-latency, no cloud, no LLM.

Status: working end-to-end (offline push-to-talk). This spec locks the scope and the
design decisions; anything under **Deferred** is out of scope until explicitly pulled in.

---

## 1. Purpose & core elements

Three things, deliberately kept small:

1. **sherpa** — recognition by a local sherpa-onnx offline transducer.
2. **a command grammar** — a YAML tree of commands with typed enum choices and free-text slots.
3. **tools that make it effective** — a confusability linter, auto-derived hotword biasing,
   and per-utterance audio+transcript capture.

## 2. Scope

**In scope**
- Offline **push-to-talk** (PTT): hold a global hotkey, speak one utterance, release.
- Deterministic grammar matching → a typed command (`name`, `action`, `slots`).
- Exactly one style of free text: a **slot**, bounded by an end anchor (or by key-release in PTT).
- Tooling: grammar quality assessment, hotword generation, voice-data capture.

**Out of scope (Non-goals)**
- Streaming / always-on recognition as the engine (PTT only for now).
- Wake-word (deferred — see §7).
- LLM intent parsing; cloud services.
- Automatic "unrecognised → dictation" fallback. Dictation is an **explicit** command.

## 3. Architecture

Rust crate `cmdtree` (lib + `cmdtree` CLI). Modules:

| Module | Responsibility | Deps |
|---|---|---|
| `grammar` | YAML command tree → compiled + validated form | core |
| `parser` | match a transcript → `Match{command, action, slots, score, leftover}` | core |
| `phonetics` | espeak G2P + distinctive-feature-weighted phone distance | espeak (runtime) |
| `assess` | predict ASR reliability of a grammar (confusability linter) | core |
| `hotwords` | derive the sherpa hotword bias list from the grammar | core |
| `wav`, `hotkey` | PCM16 WAV encode/resample; hotkey-combo parsing | core |
| `asr` | `Recognizer` trait + backends (see §5) | feature `daemon`/`sherpa` |
| `daemon` | evdev hotkey → cpal mic → ASR → parse → act/log | feature `daemon` |

**Cargo features:** `default = []` · `daemon = [cpal, evdev]` · `sherpa = [sherpa-rs]` ·
`sherpa-cuda = [sherpa, sherpa-rs/cuda]`. Core (grammar/parser/assess/hotwords) builds
with no native deps.

**CLI:** `assess` · `parse` · `hotwords` · `daemon` · `transcribe`.

## 4. Command grammar

YAML. Elements: `{lit: "words"}` (fixed), `{one_of: {name, options}}` (bounded enum →
captured slot), `{slot: name}` (free text). Rules:

- A `{slot}` **must be followed by a literal/enum end anchor** — validated at load time,
  so free text is bounded on both sides. This keeps the matcher stateless.
- **Design idiom for low confusability:** factor a shared enum suffix into a literal
  (`{one_of: [two,five,ten]} {lit: minutes}`, not options `"five minutes"…`). Verified
  with `assess`; the shipped `grammar.yaml` scores ≥98/100 with zero confusable pairs.

`grammar.yaml` is the reference tree (11 commands: light_power, brightness, scene, volume,
play_music, pause, timer, weather, navigate, phone, dictate_note).

**Parsing (`ParseOpts`):** `parse()` is strict (anchors required). `open_ended` (PTT) lets a
**trailing** slot capture to the utterance end when its anchor wasn't spoken — the anchor
stays in the grammar and is still consumed when present.

## 5. ASR

`asr::Recognizer::recognize(samples, rate) -> AsrResult{ text, tokens:[{text,start}] }`.

- **`SherpaTransducer`** (feature `sherpa`) — sherpa-onnx **offline** transducer driven
  **directly via `sherpa-rs-sys`** (not the high-level wrapper, so tokens + timestamps are
  available). 16 kHz mono. `words()` merges sub-words → words (boundary = leading space).
- **`ExternalCommand`** — shells out to any ASR command (`{wav}` → transcript on stdout);
  text only. Works with the repo's Python transducer.

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

**stdout / stderr split (the integration contract).** cmdtree stays a recognizer and
just *emits*; a separate executor acts (see §9). Per utterance:
- **stdout** — exactly one **NDJSON** event (machine-readable, line-buffered):
  `{ts, command, action, slots, transcript}`. `command`/`action` are `null` and `slots`
  `{}` when nothing matched. This is the only thing on stdout — pipe it to an executor.
- **stderr** — all human-readable diagnostics (`heard: …`, `words: w@t …`, the matched
  command, startup/biasing/errors). Pipe/redirect separately.

**`--save-audio DIR`** writes `utterance-<epoch_ms>.wav` (16 kHz mono) + appends
`manifest.jsonl`: `{wav, epoch_ms, transcript, command, action, slots, words:[{w,t}]}` —
the dataset for seeing which words work on the user's voice.

## 7. Constraints & prerequisites

- **Model:** a sherpa-onnx **offline** transducer (encoder/decoder/joiner `.onnx` +
  `tokens.txt`), auto-discovered (full-precision preferred over int8). This is an **ONNX**
  model, distinct from the repo's TorchScript streaming model. `bpe.vocab` needed for biasing.
- **Hotkey:** evdev reads `/dev/input` → works on X11 **and** Wayland, but needs
  membership in the `input` group.
- **Audio:** cpal default input device, resampled to 16 kHz mono. No device picker yet.
- **espeak** required for `assess`/`phonetics` (predictive proxy only).
- **Latency (offline, model resident):** decode ≈ 20 ms for a short command; one-time model
  load ≈ 2 s at daemon start.

## 8. Quality bar

`cargo test` (core) + `cargo test --features daemon,sherpa` green (currently 21 tests);
`cargo clippy` clean across all feature combos; `cargo package --list` clean.

## 9. Deferred (out of scope until pulled in)

- **Executor:** a **separate program** consuming the stdout NDJSON stream and performing
  effects (HA REST service calls, `wl-copy`/`wtype`, `wpctl`, dbus via `playerctl`/`gdbus`,
  custom scripts). cmdtree never grows built-in integrations; `action`+`slots` is the contract.
  Optional extra-hotwords list (names/jargon, e.g. "Tuvok") merged with the grammar-derived
  bias list is the one biasing add still worth doing.
- **Wake-word:** a small **streaming** keyword-spotter in front → hand off to this offline
  recogniser. Streaming would require the sys online recognizer.
- Mic `--device` selector.
- Surfacing `Match.score` as a confidence readout / manifest field.
- Fine-tuning the transducer on the user's voice (uses the `--save-audio` corpus).
- CUDA execution provider (`sherpa-cuda`) — not needed for command latency.

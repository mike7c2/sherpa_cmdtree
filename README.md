# cmdtree

A lean, low-latency, offline **voice command** recogniser: hold a hotkey, speak, release —
a transcript from a local sherpa-onnx model is matched against a YAML command tree and
emitted as a structured intent. Free-text dictation is available within specific commands.

No cloud, no LLM, no wake word. Push-to-talk means the key delimits the utterance, so
there is no VAD, no silence timeout, and nothing between key-release and the decode.

See [SPEC.md](SPEC.md) for the full design and its rationale.

## Layout

A Cargo workspace: the `cmdtree` recognizer crate, and a separate `cmdtree-lint`
crate (`lint/`) for grammar-quality authoring — so the runtime stays minimal.

**`cmdtree`** (the recognizer):

| Module | What |
|---|---|
| `grammar` | YAML command tree → compiled + validated form |
| `parser`  | match a transcript → `{command, action, slots, score, leftover}` |
| `hotwords` | derive a sherpa-onnx bias list from the grammar |
| `wav`, `hotkey` | PCM16 WAV encode + hotkey-combo parsing (pure, unit-tested) |
| `asr` *(feature `sherpa`)* | `SherpaTransducer` → text + per-word timings |
| `pulse` *(feature `daemon`)* | blocking 16 kHz mono capture via libpulse |
| `daemon` *(feature `daemon`)* | hotkey-gated capture → ASR → parse → emit NDJSON |

**`cmdtree-lint`** (dev tool): `phonetics` (espeak G2P + feature-weighted distance) and
`assess` (confusability linter). Run it on a grammar; not part of the runtime.

## Build & test

```bash
cargo test                              # core: grammar, parser, hotwords (no native deps)
cargo build --release                   # the `cmdtree` CLI (parse/hotwords)
cargo build --release --features sherpa  # + in-process ASR and `transcribe`
cargo build --release --features daemon  # + push-to-talk daemon (implies `sherpa`)
cargo run -p cmdtree-lint -- grammar.yaml  # grammar-quality report (needs espeak)
```

Features are additive. `sherpa` links an in-process sherpa-onnx recognizer via
[`sherpa-rs`] (its `download-binaries` feature fetches the prebuilt native lib — no local
C++/CMake build). `daemon` adds mic capture (libpulse) and the evdev hotkey, and implies
`sherpa`. `sherpa-cuda` swaps in the CUDA execution provider.

[`sherpa-rs`]: https://crates.io/crates/sherpa-rs

## Grammar (YAML)

```yaml
fuzzy: 0.82          # optional fuzzy literal-match threshold
commands:
  - name: add_note
    action: notes.add                 # optional, opaque to cmdtree
    pattern:
      - {lit: "note that"}            # fixed literal words
      - {slot: text}                  # FREE TEXT (needs an end anchor after it)
      - {lit: over}                   # <- the end anchor
  - name: set_light
    pattern:
      - {lit: turn}
      - {one_of: {name: state, options: [on, off]}}     # bounded enum → slot
      - {lit: the}
      - {one_of: {name: device, options: ["kitchen light", lamp]}}
```

**Rule:** a `{slot}` must be followed by a literal/enum **end anchor**, so free
text is bounded on both sides and the parser always knows where it stops. This is
validated at load time.

`grammar.yaml` is the reference tree (11 commands); `grammar.example.yaml` is a smaller
starting point.

## CLI

```bash
cmdtree parse grammar.example.yaml --text "turn off the kitchen light"   # add --json
cmdtree hotwords grammar.yaml --out hw.txt          # sherpa bias list from the grammar

# push-to-talk daemon (feature `daemon`):
cmdtree daemon grammar.yaml --sherpa-model models/onnx-zipformer-en --hotkey ctrl+space

# transcribe a file — no mic or hotkey needed (feature `sherpa`):
cmdtree transcribe --wav utterance.wav --sherpa-model models/onnx-zipformer-en \
        --grammar grammar.yaml

cargo run -p cmdtree-lint -- grammar.yaml           # quality report (add --json)
```

`cmdtree-lint` scores each command on phonetic confusability (feature-weighted, so a
voicing-only difference counts as *closer* than a manner/vowel difference — per
Miller & Nicely), plus anchor length/commonness. It is a **predictive proxy**;
for ground truth, synthesize each command and run it back through your recogniser.

## Daemon / push-to-talk

Hold the hotkey, speak, release. The captured audio is transcribed by
`asr::SherpaTransducer`, an in-process sherpa-onnx offline transducer driven directly
through [`sherpa-rs`]'s sys layer (so tokens + per-word timestamps are available), then
parsed against the grammar. The mic stream runs continuously and the key gates it, so
capture starts the instant you press — including the frame the press landed in.

Useful flags:

- `--open-ended` — let a trailing free-text slot omit its end anchor, since key-release
  already bounds the utterance (dictate a note without saying "full stop"). The anchor
  stays in the grammar and is still consumed when spoken.
- `--save-audio DIR` — write every processed utterance as a 16 kHz mono WAV plus a
  `manifest.jsonl` row (transcript, parsed command, per-word timings): a dataset of your
  own voice for seeing which words actually work.
- `--no-bias` / `--bias-score N` — hotword biasing toward the grammar is on by default.

### Output contract

The daemon is a recogniser, not an executor: **stdout carries exactly one NDJSON event per
utterance**, so you can pipe it straight into whatever performs the effects.

```json
{"ts":1730000000000,"command":"light_power","action":"light.power","slots":{"state":"on","device":"kitchen light"},"transcript":"SWITCH ON THE KITCHEN LIGHT"}
```

`command`/`action` are `null` and `slots` `{}` when nothing matched. Everything
human-readable — capture duration, what was heard, per-word timings, errors — goes to
**stderr**, so the two can be redirected apart:

```bash
cmdtree daemon grammar.yaml --sherpa-model MODEL 2>/dev/null | ./my-executor
```

### The model

`--sherpa-model` takes a directory holding a sherpa-onnx zipformer transducer —
`encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx` (full-precision preferred over `int8`
when both are present) and `tokens.txt`. Note this is an **ONNX** model, not a TorchScript
`cpu_jit.pt`; grab one from the
[sherpa-onnx model zoo](https://github.com/k2-fsa/sherpa-onnx/releases). A `bpe.vocab` in
the same directory enables hotword biasing (without it, biasing warns and disables itself).

### Prerequisites

| Piece | Requirement |
|---|---|
| Mic capture | a PulseAudio-compatible server — PipeWire's `pipewire-pulse`, or PulseAudio |
| Global hotkey | read access to `/dev/input`, i.e. membership in the `input` group |
| `cmdtree-lint` | `espeak` on `PATH` (authoring only, not a runtime dep) |

The hotkey is read from `/dev/input` via [`evdev`], which sits **below** the display
server — so the same code works on X11 and Wayland alike (Wayland forbids reading global
key state through the display server, which is why an X11 key-grab would not port). Every
keyboard device is monitored, so a modifier and key pressed on different keyboards still
combine, and left/right modifier variants fold to one token (`ctrl+space` fires from
either Ctrl).

If the daemon reports a permission error reading `/dev/input`:

```bash
sudo usermod -aG input $USER    # then log out and back in
```

Audio goes through the sound server, which owns resampling, downmixing, and sharing the
mic with other applications — cmdtree just asks for 16 kHz mono and reads frames.

[`evdev`]: https://crates.io/crates/evdev

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

# cmdtree

A voice **command-tree** grammar with free-text slots, plus tooling to judge how
reliably an ASR will recognise it, and an optional push-to-talk daemon.

It is ASR-agnostic: recognition is produced by any speech recogniser (e.g. the
streaming transducer in this repo, or sherpa-onnx). cmdtree turns a transcript
into a structured command with typed slots, and lints your grammar for
acoustic confusability.

## Layout

A Cargo workspace: the `cmdtree` recognizer crate, and a separate `cmdtree-lint`
crate (`lint/`) for grammar-quality authoring — so the runtime stays minimal.

**`cmdtree`** (the recognizer):

| Module | What |
|---|---|
| `grammar` | YAML command tree → compiled + validated form |
| `parser`  | match a transcript → `{command, action, slots, leftover}` |
| `hotwords` | derive a sherpa-onnx bias list from the grammar |
| `wav`, `hotkey` | PCM16 WAV encode/resample + hotkey-combo parsing (pure, unit-tested) |
| `asr` *(feature `sherpa`)* | `SherpaTransducer` → text + per-word timings |
| `kws` *(feature `sherpa`)* | streaming keyword spotter (wake / end word) |
| `daemon` *(feature)* | hotkey or wake-word capture → ASR → parse → emit NDJSON |

**`cmdtree-lint`** (dev tool): `phonetics` (espeak G2P + feature-weighted distance) and
`assess` (confusability linter). Run it on a grammar; not part of the runtime.

## Build & test

```bash
cargo test                       # core: grammar, parser, hotwords (no native deps)
cargo build --release            # the `cmdtree` CLI (parse/hotwords)
cargo build --release --features daemon          # + push-to-talk daemon (mic + hotkey)
cargo build --release --features sherpa          # + native sherpa-onnx ASR, transcribe, spot
cargo build --release --features daemon,sherpa   # daemon + in-process recognizer + wake word
cargo run -p cmdtree-lint -- grammar.yaml        # grammar-quality report (needs espeak)
```

Features are additive: `daemon` links mic capture (cpal) and the evdev hotkey;
`sherpa` links an in-process sherpa-onnx recognizer via [`sherpa-rs`] (its
`download-binaries` feature fetches the prebuilt native lib — no local C++/CMake
build). `sherpa-cuda` swaps in the CUDA execution provider.

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

## CLI

```bash
cargo run -p cmdtree-lint -- grammar.example.yaml   # grammar-quality report (add --json)
cmdtree parse  grammar.example.yaml --text "turn off the kitchen light"

# push-to-talk daemon (feature `daemon`, which pulls in the sherpa recognizer):
cmdtree daemon grammar.example.yaml --hotkey ctrl+space --sherpa-model models/onnx-zipformer-en

# transcribe a file (no mic/hotkey needed; feature `sherpa`):
cmdtree transcribe --wav utterance.wav --sherpa-model models/onnx-zipformer-en \
        --grammar grammar.example.yaml
```

`cmdtree-lint` scores each command on phonetic confusability (feature-weighted, so a
voicing-only difference counts as *closer* than a manner/vowel difference — per
Miller & Nicely), plus anchor length/commonness. It is a **predictive proxy**;
for ground truth, synthesize each command and run it back through your recogniser.

## Daemon / push-to-talk

Hold the hotkey (or say a wake word), speak, release — the captured audio is
transcribed by `asr::SherpaTransducer`, an in-process sherpa-onnx offline transducer
driven directly through [`sherpa-rs`]'s sys layer (so tokens + per-word timestamps are
available). The `daemon` feature pulls in the recognizer, so `--sherpa-model` is required.

`--sherpa-model` takes a directory holding a sherpa-onnx zipformer transducer —
`encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx` (full-precision preferred over
`int8` when both are present) and `tokens.txt`. Note this is an **ONNX** model, not
the TorchScript `cpu_jit.pt` the Python side uses; grab one from the
[sherpa-onnx model zoo](https://github.com/k2-fsa/sherpa-onnx/releases). The
`transcribe` subcommand runs a WAV through the same backend without a mic or
hotkey — handy for testing a grammar against real audio.

### Platform support

| Piece | X11 | Wayland |
|---|---|---|
| Mic capture (cpal/ALSA/PipeWire) | ✅ | ✅ |
| Transcribe + parse | ✅ | ✅ |
| **Global hotkey** (`evdev`, `/dev/input`) | ✅ | ✅ |

The global hotkey is read from `/dev/input` via [`evdev`], which sits **below** the
display server — so the same code works on X11 and Wayland alike (Wayland forbids
reading global key state through the display server, which is why an X11 key-grab
would not port). Every keyboard device is monitored, so a modifier and key pressed
on different keyboards still combine, and left/right modifier variants fold to one
token (`ctrl+space` fires from either Ctrl).

**Access:** reading `/dev/input` requires membership in the `input` group. If the
daemon reports a permission error, run:

```bash
sudo usermod -aG input $USER    # then log out and back in
```

[`evdev`]: https://crates.io/crates/evdev

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

# cmdtree

A voice **command-tree** grammar with free-text slots, plus tooling to judge how
reliably an ASR will recognise it, and an optional push-to-talk daemon.

It is ASR-agnostic: recognition is produced by any speech recogniser (e.g. the
streaming transducer in this repo, or sherpa-onnx). cmdtree turns a transcript
into a structured command with typed slots, and lints your grammar for
acoustic confusability.

## Layout

| Module | What |
|---|---|
| `grammar` | YAML command tree → compiled + validated form |
| `parser`  | match a transcript → `{command, slots, leftover}` |
| `phonetics` | espeak G2P + **distinctive-feature-weighted** phonetic distance |
| `assess`  | predict recognition reliability of a grammar |
| `wav`, `hotkey` | PCM16 WAV encoder + hotkey-combo parsing (pure, unit-tested) |
| `daemon` *(feature)* | hold-a-hotkey → capture mic → ASR → parse |

## Build & test

```bash
cargo test                       # core: grammar, parser, phonetics, assess (no native deps)
cargo build --release            # the `cmdtree` CLI (assess/parse)
cargo build --release --features daemon          # + push-to-talk daemon (mic + hotkey)
cargo build --release --features sherpa          # + native sherpa-onnx ASR & `transcribe`
cargo build --release --features daemon,sherpa   # daemon using the in-process recognizer
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
cmdtree assess grammar.example.yaml            # quality report (add --json)
cmdtree parse  grammar.example.yaml --text "turn off the kitchen light"

# push-to-talk, external ASR (the Python transducer in this repo):
cmdtree daemon grammar.example.yaml --hotkey ctrl+space \
        --asr-cmd '/path/.venv/bin/python /path/mic_asr.py --wav {wav}'

# push-to-talk, native in-process ASR (feature `sherpa`):
cmdtree daemon grammar.example.yaml --hotkey ctrl+space --sherpa-model models/onnx-zipformer-en

# transcribe a file with the native ASR (no mic/hotkey needed; feature `sherpa`):
cmdtree transcribe --wav utterance.wav --sherpa-model models/onnx-zipformer-en \
        --grammar grammar.example.yaml
```

`assess` scores each command on phonetic confusability (feature-weighted, so a
voicing-only difference counts as *closer* than a manner/vowel difference — per
Miller & Nicely), plus anchor length/commonness. It is a **predictive proxy**;
for ground truth, synthesize each command and run it back through your recogniser.

## Daemon / push-to-talk

Hold the hotkey, speak, release — the captured audio is transcribed and parsed.
ASR is pluggable via the `asr::Recognizer` trait; two backends ship:

- **`ExternalCommand`** (`--asr-cmd`) — writes the utterance to a temp WAV and runs
  your command (with `{wav}` substituted), reading the transcript from stdout. Needs
  no native library, and works today with the Python transducer in this repo.
- **`SherpaTransducer`** (`--sherpa-model`, feature `sherpa`) — an in-process
  sherpa-onnx offline transducer via [`sherpa-rs`], no subprocess per utterance.

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

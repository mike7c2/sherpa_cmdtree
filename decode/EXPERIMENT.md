# cmdtree-decode — grammar-trellis rescoring (experiment spec)

**Standalone brief: implementable without prior context. Read `../SPEC.md` §4–5,
`../src/grammar.rs`, and `../src/asr.rs` first.**

## 0. Background & hypothesis

cmdtree V2 recognises commands by *general transcription then text matching*: sherpa-onnx
decodes an unconstrained 1-best transcript, and `parser` fuzzy-matches it against the grammar.
That is hard-decision decoding followed by error correction. The optimal decision rule is
soft-decision sequence decoding: `argmax over grammar paths of P(path | audio)` — Viterbi over
the grammar trellis, the way classical WFST recognisers (and trellis codes) work.

**Hypothesis:** rescoring grammar candidates directly against the acoustics beats
transcribe-then-match on command accuracy, and a likelihood-ratio against the unconstrained
decode gives calibrated rejection (no "everything maps to the nearest valid command").

### Measured motivation (2026-07-28)

Failures of the current architecture, measured with espeak TTS → `cmdtree transcribe` against
`models/sherpa-onnx-zipformer-en-2023-06-26`. **Caveat: espeak is out-of-distribution for the
model, so absolute rates mean little — the value is in the failure *classes*, each of which
must be re-confirmed on human audio.**

| utterance | transcript | note |
|---|---|---|
| `pour five milliliters` | `FOUR FIVE MILLION ETERNS` | identical at bias 2.0 / 5.0 / **12.0** |
| `pour five liters` | `FOUR FIVE LEADERS` | rare word, never reaches the beam |
| `set a timer for two hours` | `AT A TIME OF FORTY HOURS` | cross-boundary: "for two" → "forty" |
| `set a timer for two hours` (biased) | `AT A TIME OF FOR TWO HOURS` | biasing *did* fix this one |
| 10× confusable numerals | **10/10 correct** | `fifteen`/`fifty`, `thirty`/`forty`, … |

Two conclusions that shape this experiment:

1. **The rare-word failures are architecturally unreachable today.** Hotword biasing reweights
   candidates *within the beam*; if `MILLILITERS` never appears as a candidate, no bias score
   can boost it — confirmed unchanged from 2.0 to 12.0. Rescoring does not have this problem:
   it scores a *given* token sequence directly, so a candidate never has to win an
   open-vocabulary search. **This is the clearest predicted win and the primary thing to
   measure.**
2. **Phonetic confusability was not the problem.** `cmdtree-lint` flagged 22 "confusable" pairs
   in the 1–60 numerals; zero materialised, because the transducer's language-model prior
   disambiguates `fifteen minutes` from `fifty minutes` in context. The linter compares words
   in isolation and cannot see this. Do **not** treat lint scores as ground truth here.

Note also that under constrained decoding, post-routing decisions become closed-set (choose
among one enum's options) rather than open-vocabulary — so gains should concentrate in
routing-phrase robustness and rare in-grammar vocabulary, not in telling similar numbers apart.

This experiment **measures that gap**. No cmdtree changes until the numbers justify it.

## 1. Approach — enumerate + rescore (no constrained beam search in this phase)

The transducer checkpoint is three ONNX graphs (encoder / decoder a.k.a. prediction network,
context 2 / joiner). Run them directly via `ort`:

1. **One encoder pass** per utterance: 80-dim fbank → encoder frames (~25 Hz).
2. **Candidate set**: expand the grammar's slot-free commands (cartesian product over enums)
   → normalise → UPPERCASE → BPE-encode with the model's `bpe.model`. Report the count; cap
   and warn above ~2 000.
3. **Score each candidate** with the standard RNN-T forward algorithm over the (T × U) lattice
   (Graves 2012): with `b(t,u)` = log-prob of blank and `y(t,u)` = log-prob of token u+1 from
   the joiner at frame t after u emitted tokens,
   `α(t,u) = logaddexp(α(t−1,u) + b(t−1,u), α(t,u−1) + y(t,u−1))`,
   final score `α(T−1,U) + b(T−1,U)`. Batch the joiner over the whole lattice — it is a small
   dense net; T·U ≈ 50×10 per candidate is trivial.
4. **Rejection**: also run our own greedy decode (unconstrained 1-best), then **re-score its
   token sequence with the same forward algorithm** so the comparison is apples-to-apples:
   `LLR = score(best candidate) − score(free 1-best)`. Near 0 ⇒ accept; large negative ⇒
   reject. Report both raw and per-frame-normalised LLR (length normalisation is a calibration
   knob, not a given).

**Slots (phase B, only if phase A wins):** slots make the candidate set infinite. Hybrid:
constrained scoring routes the command and locates the end anchor in time (best-alignment
timestamps); the free-text content is spliced from the unconstrained transcript between the
anchors, using the token timestamps `asr.rs` already demonstrates. Both passes share the one
encoder run.

## 2. Build plan

- **This crate** (`decode/`, `cmdtree-decode`) is a workspace member alongside `lint/`,
  depending on `cmdtree` by path for grammar types only. A CLI skeleton with the §2 command
  shape already exists; fill it in milestone by milestone. Nothing here is compiled into the
  runtime crate, and **no file outside `decode/` may change** until M4 proposes integration.
- Deps: `ort` (CPU EP; mirror the `--sherpa-threads` idea), a sentencepiece binding for
  `bpe.model` (or exact BPE re-implementation — must round-trip: decode(encode(s)) == s),
  fbank either via kaldi-native-fbank bindings or pure Rust — **must match sherpa's features
  exactly** (16 kHz, 25 ms / 10 ms, 80 mel bins, dither 0; mismatch here is the most likely
  total failure and is why M1 comes first).
- Reference implementations: sherpa-onnx `offline-transducer-model.cc` and
  `offline-transducer-greedy-search-decoder.cc` (I/O names, decoder context handling,
  blank id). Introspect the ONNX graphs for tensor names rather than trusting docs.
- Test model: `sherpa-onnx-zipformer-en-2023-06-26` (has `bpe.vocab`/`bpe.model`; vocab is
  UPPERCASE with `▁` word-start markers) — `../../models/…` from this crate, i.e. a sibling
  of the repo root. Pass it explicitly via `--model`; never hard-code the path.

**CLI:**
```
cmdtree-decode parity  --model DIR --wav F                 # our greedy vs `cmdtree transcribe`
cmdtree-decode rescore --model DIR --grammar G --wav F     # candidate table: score, LLR, rank
cmdtree-decode eval    --model DIR --grammar G --corpus DIR [--negatives DIR] [--json]
```

## 3. Milestones (ordered; each independently verifiable)

- **M1 — parity.** Feature extraction + encoder + greedy decode reproduce `cmdtree transcribe`
  (greedy, no hotwords) on the corpus WAVs — ≥95% exact-transcript match, diffs explained.
  Nothing else proceeds until M1 passes.
- **M2 — rescoring.** Forward-algorithm scores; on hand-checked positive WAVs the spoken
  command ranks first. `rescore` prints the top-N table.
- **M3 — evaluation.** `eval` reports: command accuracy (rescoring vs the V2 text pipeline),
  FA/FR curve over the LLR threshold using the negatives dir, wall-time per utterance
  (budget: ≤30 ms added over the encoder pass for ≤500 candidates, CPU).

  **Binary win conditions** (from §0; each is a specific WAV + expected rank):
  - **W1 (primary).** `pour five milliliters` → the `pour … milliliters` candidate ranks
    **first**. Open decoding cannot produce this word at *any* bias score, so a win here is a
    capability the current architecture provably lacks. Same for `liters`.
  - **W2.** `set a timer for two hours` → the timer candidate ranks first despite the routing
    phrase transcribing as `AT A TIME OF FOR`.
  - **W3 (regression guard).** The 10 confusable numerals stay correct — rescoring must not
    *lose* accuracy where transcribe-then-match already scores 10/10.
  - **W4.** On negatives, LLR separates from positives well enough to pick a threshold with
    no false accepts at a usable false-reject rate.

  If W1 fails, report why before proceeding — that failure would undercut the whole hypothesis.
- **M4 — only if M3 wins:** phase-B slot hybrid + a written integration proposal for cmdtree
  (do not modify cmdtree in this experiment).

## 4. Data & metrics

- **Positives (primary):** the `--save-audio` corpus (`manifest.jsonl`: wav, transcript,
  parsed command). Note the manifest's `command` is what V2 *decided*, not ground truth —
  `eval` must emit a review file for a quick human labelling pass; report accuracy against
  corrected labels.
- **Bootstrap set (before human audio exists):** synthesise with espeak, which is how §0 was
  measured and is enough to exercise M1/M2 and the W1–W3 cases:
  ```bash
  espeak -w out.wav -s 145 "pour five milliliters"
  ffmpeg -y -i out.wav -ar 16000 -ac 1 out16k.wav   # sherpa requires exactly 16 kHz
  ```
  **TTS is out-of-distribution** — treat it as a smoke test that makes failures reproducible,
  never as the accuracy result. Every headline number in the M3 report must come from human
  audio.
- **Negatives:** a directory of PTT-held non-command speech (record conversation, reading,
  coughs). The FA/FR curve is computed over these.
- Report JSON + a human-readable table; keep the raw per-utterance scores for offline analysis.

## 5. Risks / notes

- Fbank mismatch ⇒ silently garbage encoder output (hence M1).
- Candidate tokenisation must produce the model's exact token ids (UPPERCASE + `▁`).
- Never compare a greedy path score against a forward score — forward vs forward only.
- Prediction-network context is 2 tokens, blank-padded at start; copy sherpa's init exactly.
- Hotword biasing is irrelevant here (candidates *are* the prior); parity runs use plain
  greedy so the baselines match.
- Rescoring always returns *some* candidate — decoding noise still yields a codeword. The LLR
  in §1.4 is the only rejection mechanism; W4 is therefore not optional.

## 6. Non-goals (this experiment)

Constrained beam search, CTC variant, streaming, cmdtree runtime changes, CUDA. All gated on
the M3 report.

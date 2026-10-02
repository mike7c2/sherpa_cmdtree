//! cmdtree-decode — grammar-trellis rescoring experiment. See `EXPERIMENT.md`.
//!
//! Skeleton only: the CLI shape is fixed so the milestones have a stable interface,
//! but no milestone is implemented. M1 (parity) must land before anything else.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod features;
mod model;
mod tokens;

#[derive(Parser)]
#[command(
    name = "cmdtree-decode",
    about = "Score grammar candidates against the acoustics (experiment; see EXPERIMENT.md)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dump a model's graph contract: input/output names, dtypes, and the custom
    /// metadata (context_size, vocab_size) that drives decoding.
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
    /// M1: our own fbank+encoder+greedy decode vs `cmdtree transcribe` (must match first).
    Parity {
        /// sherpa-onnx offline transducer model dir
        #[arg(long)]
        model: PathBuf,
        /// WAV to decode (16 kHz mono); repeat or pass --dir for a batch
        #[arg(long)]
        wav: Vec<PathBuf>,
        /// decode every *.wav in this directory
        #[arg(long)]
        dir: Option<PathBuf>,
        /// ONNX Runtime intra-op threads
        #[arg(long, default_value_t = 2)]
        threads: usize,
        /// use Kaldi's snip_edges=true framing instead of emulating sherpa's false
        #[arg(long)]
        snip_edges: bool,
    },
    /// M2: rank grammar candidates for one utterance (score, LLR, rank).
    Rescore {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        grammar: PathBuf,
        #[arg(long)]
        wav: PathBuf,
        /// how many candidates to print
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    /// M3: accuracy vs the cmdtree text pipeline, LLR separation, timing.
    Eval {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        grammar: PathBuf,
        /// dir of positives (WAVs + manifest.jsonl from `cmdtree daemon --save-audio`)
        #[arg(long)]
        corpus: PathBuf,
        /// dir of non-command speech, for the false-accept curve
        #[arg(long)]
        negatives: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    let outcome = match Cli::parse().cmd {
        Cmd::Inspect { model } => model::inspect(&model),
        Cmd::Parity { model, wav, dir, threads, snip_edges } => run_parity(&model, wav, dir, threads, !snip_edges),
        Cmd::Rescore { .. } => Err("M2 (rescoring) is not implemented yet — see EXPERIMENT.md".into()),
        Cmd::Eval { .. } => Err("M3 (evaluation) is not implemented yet — see EXPERIMENT.md".into()),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// M1: decode WAVs with our own stack and print `path\ttranscript`, so the batch can
/// be diffed against `cmdtree transcribe` (greedy, no hotwords) to get the parity rate.
fn run_parity(
    model: &PathBuf,
    mut wavs: Vec<PathBuf>,
    dir: Option<PathBuf>,
    threads: usize,
    snip_edges_false: bool,
) -> Result<(), String> {
    if let Some(dir) = dir {
        let entries =
            std::fs::read_dir(&dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
        let mut found: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wav"))
            .collect();
        found.sort();
        wavs.extend(found);
    }
    if wavs.is_empty() {
        return Err("no WAVs: pass --wav FILE (repeatable) or --dir DIR".into());
    }

    let tokens = tokens::Tokens::load(model)?;
    let mut transducer = model::Transducer::load(model, threads)?;
    if tokens.len() < transducer.vocab_size {
        return Err(format!(
            "tokens.txt has {} entries, fewer than the joiner's {} logits",
            tokens.len(),
            transducer.vocab_size
        ));
    }

    // One unreadable file must not abandon a corpus run: report it and carry on.
    let mut failed = 0;
    for wav in &wavs {
        match decode_one(&mut transducer, &tokens, wav, snip_edges_false) {
            Ok(text) => println!("{}\t{text}", wav.display()),
            Err(e) => {
                failed += 1;
                eprintln!("skipped {}: {e}", wav.display());
            }
        }
    }
    if failed > 0 {
        eprintln!("{failed} of {} file(s) skipped", wavs.len());
    }
    Ok(())
}

fn decode_one(
    transducer: &mut model::Transducer,
    tokens: &tokens::Tokens,
    wav: &PathBuf,
    snip_edges_false: bool,
) -> Result<String, String> {
    let samples = features::read_wav_16k_mono(wav)?;
    let (feats, frames) = features::fbank(&samples, snip_edges_false)?;
    let (enc, enc_frames, dim) = transducer.encode(&feats, frames)?;
    let ids = transducer.greedy(&enc, enc_frames, dim)?;
    eprintln!(
        "{}: {:.2}s → {frames} fbank → {enc_frames} encoder frames (dim {dim})",
        wav.display(),
        samples.len() as f32 / features::SAMPLE_RATE as f32
    );
    Ok(tokens.decode(&ids))
}

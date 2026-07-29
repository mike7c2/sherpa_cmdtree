//! cmdtree-decode — grammar-trellis rescoring experiment. See `EXPERIMENT.md`.
//!
//! Skeleton only: the CLI shape is fixed so the milestones have a stable interface,
//! but no milestone is implemented. M1 (parity) must land before anything else.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

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
    /// M1: our own fbank+encoder+greedy decode vs `cmdtree transcribe` (must match first).
    Parity {
        /// sherpa-onnx offline transducer model dir
        #[arg(long)]
        model: PathBuf,
        /// WAV to decode (16 kHz mono)
        #[arg(long)]
        wav: PathBuf,
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
    let milestone = match Cli::parse().cmd {
        Cmd::Parity { .. } => "M1 (parity)",
        Cmd::Rescore { .. } => "M2 (rescoring)",
        Cmd::Eval { .. } => "M3 (evaluation)",
    };
    eprintln!("cmdtree-decode: {milestone} is not implemented yet — see EXPERIMENT.md");
    ExitCode::FAILURE
}

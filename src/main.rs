use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use cmdtree::grammar::Grammar;

#[derive(Parser)]
#[command(name = "cmdtree", about = "Voice command-tree grammar with free-text slots")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse a transcript against the grammar and print the matched command.
    Parse {
        grammar: PathBuf,
        /// text to parse
        #[arg(long)]
        text: String,
        #[arg(long)]
        json: bool,
    },
    /// Emit the sherpa-onnx hotword (contextual-biasing) list derived from a grammar.
    Hotwords {
        grammar: PathBuf,
        /// per-phrase boost score (appended as `phrase :score`); default: global score
        #[arg(long)]
        score: Option<f32>,
        /// write to this file instead of stdout
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run the push-to-talk daemon (requires building with `--features daemon`).
    Daemon(DaemonArgs),
    /// Transcribe a WAV file with the native sherpa-onnx backend, and optionally
    /// parse it against a grammar (requires building with `--features sherpa`).
    Transcribe(TranscribeArgs),
}

#[derive(clap::Args)]
struct DaemonArgs {
    grammar: PathBuf,
    /// hotkey combo to hold while speaking, e.g. "ctrl+space" or "ctrl+alt+k"
    #[arg(long, default_value = "ctrl+space")]
    hotkey: String,
    /// sherpa-onnx transducer model dir (encoder/decoder/joiner ONNX + tokens.txt)
    #[arg(long)]
    sherpa_model: PathBuf,
    /// sherpa-onnx execution provider ("cpu", or "cuda" with --features sherpa-cuda)
    #[arg(long)]
    sherpa_provider: Option<String>,
    /// sherpa-onnx intra-op threads
    #[arg(long)]
    sherpa_threads: Option<i32>,
    /// save every processed utterance (16 kHz mono WAV) + a manifest.jsonl line to
    /// this directory, for post-processing or collecting a dataset of your own voice
    #[arg(long)]
    save_audio: Option<PathBuf>,
    /// disable automatic hotword biasing toward the grammar (sherpa backend)
    #[arg(long)]
    no_bias: bool,
    /// boost applied to auto-derived hotword phrases
    #[arg(long, default_value_t = 1.5)]
    bias_score: f32,
    /// allow a trailing free-text slot to omit its end anchor (key-release ends
    /// the utterance), e.g. dictate a note without saying "full stop". The anchor
    /// stays in the grammar and is still honoured when spoken.
    #[arg(long)]
    open_ended: bool,
}

#[derive(clap::Args)]
struct TranscribeArgs {
    /// WAV file to transcribe (16 kHz mono PCM16)
    #[arg(long)]
    wav: PathBuf,
    /// directory of a sherpa-onnx transducer model (encoder/decoder/joiner ONNX + tokens.txt)
    #[arg(long)]
    sherpa_model: PathBuf,
    /// sherpa-onnx execution provider ("cpu", or "cuda" with --features sherpa-cuda)
    #[arg(long)]
    sherpa_provider: Option<String>,
    /// sherpa-onnx intra-op threads
    #[arg(long)]
    sherpa_threads: Option<i32>,
    /// optional grammar to also parse the transcript against
    #[arg(long)]
    grammar: Option<PathBuf>,
    /// hotword bias file (one phrase per line; needs bpe.vocab in the model dir).
    /// Tip: `cmdtree hotwords <grammar> --out hw.txt` to generate one.
    #[arg(long)]
    hotwords: Option<PathBuf>,
    /// boost applied to hotword phrases
    #[arg(long, default_value_t = 1.5)]
    bias_score: f32,
    /// print per-word recognition detail (word + start time)
    #[arg(long)]
    tokens: bool,
}

fn load(grammar: &Path) -> Result<Grammar, String> {
    Grammar::load(grammar).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let outcome = match Cli::parse().cmd {
        Cmd::Parse { grammar, text, json } => run_parse(grammar, text, json),
        Cmd::Hotwords { grammar, score, out } => run_hotwords(grammar, score, out),
        Cmd::Daemon(args) => run_daemon(args),
        Cmd::Transcribe(args) => run_transcribe(args),
    };
    outcome.unwrap_or_else(|e| {
        eprintln!("error: {e}");
        ExitCode::FAILURE
    })
}

fn run_parse(grammar: PathBuf, text: String, json: bool) -> Result<ExitCode, String> {
    let g = load(&grammar)?;
    let Some(m) = g.parse(&text) else {
        println!("(no command recognised)");
        return Ok(ExitCode::FAILURE);
    };
    if json {
        let obj = serde_json::json!({
            "command": m.command, "action": m.action, "slots": m.slots_json(), "leftover": m.leftover,
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap());
    } else {
        print_match(&m);
    }
    Ok(ExitCode::SUCCESS)
}

fn run_hotwords(grammar: PathBuf, score: Option<f32>, out: Option<PathBuf>) -> Result<ExitCode, String> {
    let text = cmdtree::hotwords::hotwords_file(&load(&grammar)?, score);
    match out {
        Some(path) => {
            std::fs::write(&path, &text).map_err(|e| format!("writing {}: {e}", path.display()))?;
            eprintln!("wrote {} hotword phrases to {}", text.lines().count(), path.display());
        }
        None => print!("{text}"),
    }
    Ok(ExitCode::SUCCESS)
}

/// Print a matched command as a human-readable block (shared by `parse` and `transcribe`).
fn print_match(m: &cmdtree::parser::Match) {
    println!("command: {}", m.command);
    if let Some(a) = &m.action {
        println!("action : {a}");
    }
    for (k, v) in &m.slots {
        println!("   {k} = {v:?}");
    }
    if !m.leftover.is_empty() {
        println!("   (leftover: {})", m.leftover.join(" "));
    }
}

#[cfg(feature = "daemon")]
fn run_daemon(args: DaemonArgs) -> Result<ExitCode, String> {
    let cfg = cmdtree::daemon::DaemonConfig {
        hotkey: args.hotkey,
        sherpa_model: args.sherpa_model,
        sherpa_provider: args.sherpa_provider,
        sherpa_threads: args.sherpa_threads,
        save_audio: args.save_audio,
        bias: !args.no_bias,
        bias_score: args.bias_score,
        open_ended: args.open_ended,
    };
    cmdtree::daemon::run(load(&args.grammar)?, cfg)?;
    Ok(ExitCode::SUCCESS)
}

#[cfg(feature = "sherpa")]
fn run_transcribe(args: TranscribeArgs) -> Result<ExitCode, String> {
    use cmdtree::asr::{SherpaOpts, SherpaTransducer};

    let recognizer = SherpaTransducer::from_dir(
        &args.sherpa_model,
        SherpaOpts {
            num_threads: args.sherpa_threads,
            provider: args.sherpa_provider,
            hotwords_file: args.hotwords,
            hotwords_score: args.bias_score,
        },
    )?;
    let (samples, sample_rate) = sherpa_rs::read_audio_file(&args.wav.to_string_lossy())
        .map_err(|e| format!("reading {}: {e}", args.wav.display()))?;
    let res = recognizer.recognize(&samples, sample_rate)?;
    println!("transcript: {:?}", res.text);
    if args.tokens {
        for (w, t) in res.words() {
            println!("  {t:6.2}s  {w}");
        }
    }
    if let Some(grammar) = &args.grammar {
        match load(grammar)?.parse(&res.text) {
            Some(m) => print_match(&m),
            None => println!("(no command recognised)"),
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(not(feature = "daemon"))]
fn run_daemon(_args: DaemonArgs) -> Result<ExitCode, String> {
    Err("the `daemon` subcommand needs the daemon feature (rebuild with --features daemon)".into())
}

#[cfg(not(feature = "sherpa"))]
fn run_transcribe(_args: TranscribeArgs) -> Result<ExitCode, String> {
    Err("the `transcribe` subcommand needs the sherpa feature (rebuild with --features sherpa)".into())
}

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use cmdtree::assess::assess;
use cmdtree::grammar::Grammar;

#[derive(Parser)]
#[command(name = "cmdtree", about = "Voice command-tree grammar with free-text slots")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Assess how reliably an ASR should pick up a command tree.
    Assess {
        /// path to the YAML grammar
        grammar: PathBuf,
        /// emit JSON instead of a text report
        #[arg(long)]
        json: bool,
    },
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
    /// external ASR command; {wav} is replaced by a temp wav path, transcript read from stdout.
    /// Example: ".venv/bin/python mic_asr.py --wav {wav}"
    #[arg(long)]
    asr_cmd: Option<String>,
    /// native ASR: directory of a sherpa-onnx transducer model (encoder/decoder/joiner
    /// ONNX + tokens.txt). Takes precedence over --asr-cmd. Needs --features sherpa.
    #[arg(long)]
    sherpa_model: Option<PathBuf>,
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
    /// PTT: allow a trailing free-text slot to omit its end anchor (key-release ends
    /// the utterance), e.g. dictate a note without saying "full stop". The anchor
    /// stays in the grammar and is still honoured when spoken.
    #[arg(long)]
    open_ended: bool,
    /// mic sample rate
    #[arg(long, default_value_t = 16000)]
    sample_rate: u32,
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
    /// hotword bias file (one phrase per line; needs bpe.model in the model dir).
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

fn load(grammar: &PathBuf) -> Result<Grammar, String> {
    Grammar::load(grammar).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Assess { grammar, json } => {
            let g = match load(&grammar) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let report = assess(&g);
            if json {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                println!("{}", report.render());
            }
            ExitCode::SUCCESS
        }
        Cmd::Parse { grammar, text, json } => {
            let g = match load(&grammar) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match g.parse(&text) {
                Some(m) => {
                    if json {
                        let mut slots = serde_json::Map::new();
                        for (k, v) in &m.slots {
                            slots.insert(k.clone(), serde_json::Value::String(v.clone()));
                        }
                        let obj = serde_json::json!({
                            "command": m.command,
                            "action": m.action,
                            "slots": slots,
                            "leftover": m.leftover,
                        });
                        println!("{}", serde_json::to_string_pretty(&obj).unwrap());
                    } else {
                        print_match(&m);
                    }
                    ExitCode::SUCCESS
                }
                None => {
                    println!("(no command recognised)");
                    ExitCode::FAILURE
                }
            }
        }
        Cmd::Hotwords { grammar, score, out } => {
            let g = match load(&grammar) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let text = cmdtree::hotwords::hotwords_file(&g, score);
            match out {
                Some(path) => {
                    if let Err(e) = std::fs::write(&path, &text) {
                        eprintln!("error writing {}: {e}", path.display());
                        return ExitCode::FAILURE;
                    }
                    eprintln!("wrote {} hotword phrases to {}", text.lines().count(), path.display());
                }
                None => print!("{text}"),
            }
            ExitCode::SUCCESS
        }
        Cmd::Daemon(args) => run_daemon(args),
        Cmd::Transcribe(args) => run_transcribe(args),
    }
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
fn run_daemon(args: DaemonArgs) -> ExitCode {
    let g = match load(&args.grammar) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cfg = cmdtree::daemon::DaemonConfig {
        hotkey: args.hotkey,
        asr_cmd: args.asr_cmd,
        sherpa_model: args.sherpa_model,
        sherpa_provider: args.sherpa_provider,
        sherpa_threads: args.sherpa_threads,
        save_audio: args.save_audio,
        bias: !args.no_bias,
        bias_score: args.bias_score,
        open_ended: args.open_ended,
        sample_rate: args.sample_rate,
    };
    match cmdtree::daemon::run(g, cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("daemon error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "daemon"))]
fn run_daemon(_args: DaemonArgs) -> ExitCode {
    eprintln!(
        "the `daemon` subcommand needs the daemon feature.\n\
         rebuild with:  cargo build --release --features daemon"
    );
    ExitCode::FAILURE
}

#[cfg(feature = "sherpa")]
fn run_transcribe(args: TranscribeArgs) -> ExitCode {
    use cmdtree::asr::{Recognizer, SherpaOpts, SherpaTransducer};

    let recognizer = match SherpaTransducer::from_dir(
        &args.sherpa_model,
        SherpaOpts {
            num_threads: args.sherpa_threads,
            provider: args.sherpa_provider,
            hotwords_file: args.hotwords,
            hotwords_score: args.bias_score,
        },
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (samples, sample_rate) = match sherpa_rs::read_audio_file(&args.wav.to_string_lossy()) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error reading {}: {e}", args.wav.display());
            return ExitCode::FAILURE;
        }
    };
    let res = match recognizer.recognize(&samples, sample_rate) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("asr error: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("transcript: {:?}", res.text);
    if args.tokens {
        for (w, t) in res.words() {
            println!("  {t:6.2}s  {w}");
        }
    }

    if let Some(grammar) = &args.grammar {
        let g = match load(grammar) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        match g.parse(&res.text) {
            Some(m) => print_match(&m),
            None => println!("(no command recognised)"),
        }
    }
    ExitCode::SUCCESS
}

#[cfg(not(feature = "sherpa"))]
fn run_transcribe(_args: TranscribeArgs) -> ExitCode {
    eprintln!(
        "the `transcribe` subcommand needs the sherpa feature.\n\
         rebuild with:  cargo build --release --features sherpa"
    );
    ExitCode::FAILURE
}

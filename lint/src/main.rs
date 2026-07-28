use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use cmdtree::grammar::Grammar;
use cmdtree_lint::assess::assess;

/// Predict how reliably an ASR will pick up a cmdtree command tree (a predictive
/// proxy over espeak phonetics; for ground truth, synthesize each command and run it
/// back through your recognizer).
#[derive(Parser)]
#[command(name = "cmdtree-lint", about = "Grammar-quality linter for cmdtree command trees")]
struct Cli {
    /// path to the YAML grammar
    grammar: PathBuf,
    /// emit JSON instead of a text report
    #[arg(long)]
    json: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let g = match Grammar::load(&cli.grammar) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let report = assess(&g);
    if cli.json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!("{}", report.render());
    }
    ExitCode::SUCCESS
}

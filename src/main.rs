use anyhow::{Context, Result};
use clap::Parser;
use hound;
use log::{debug, info};
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

mod cache;
mod commands;
mod config;
mod ffmpeg;
mod llama;
mod utils;

use cache::{Cache, Clip, Timestamp};
use commands::{Cli, Commands};
use config::Config;
use ffmpeg::FFmpeg;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input video file
    #[arg(short, long)]
    input: PathBuf,

    /// Path to output directory
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to config file
    #[arg(long)]
    config: Option<PathBuf>,

    /// Whisper model to use (base, tiny, small, medium, large)
    #[arg(short, long)]
    model: Option<String>,

    /// Audio tracks to process (1-based indexing)
    #[arg(short, long)]
    tracks: Option<Vec<u32>>,

    /// Keywords to search for in the audio
    #[arg(short, long, num_args = 1.., value_delimiter = ' ')]
    clips: Option<Vec<String>>,

    /// Don't clean up intermediate files
    #[arg(long)]
    no_cleanup: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::WordClip(args) => commands::wordclip::run(args),
    }
}

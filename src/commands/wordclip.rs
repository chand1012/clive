use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use crate::cache::Cache;
use crate::config::Config;
use crate::ffmpeg::FFmpeg;
use crate::utils::wordclip;

#[derive(Parser, Debug)]
pub struct WordClipArgs {
    /// Path to the input video file
    #[arg(short, long)]
    pub input: PathBuf,

    /// Path to the output directory
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Path to the config file
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Model to use for transcription (base, tiny, small, medium, large)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Audio tracks to process
    #[arg(short, long)]
    pub tracks: Option<Vec<u32>>,

    /// Keywords to search for in the audio
    #[arg(short, long)]
    pub clips: Option<Vec<String>>,

    /// Don't clean up temporary files
    #[arg(long)]
    pub no_cleanup: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run(args: WordClipArgs) -> Result<()> {
    // Initialize logging
    if args.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .format_timestamp(None)
            .init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .format_timestamp(None)
            .init();
    }

    // Initialize configuration
    let mut config = if let Some(config_path) = args.config {
        Config::from_file(&config_path)?
    } else {
        Config::default()
    };

    // Merge CLI arguments with config
    if let Some(keywords) = args.clips {
        let cli_config = Config::from_cli(
            args.input.clone(),
            args.output,
            args.model,
            args.tracks,
            keywords,
        );
        config.merge_cli(cli_config);
    } else {
        config.input_file = Some(args.input.clone());
        if let Some(output) = args.output {
            config.output.directory = output;
        }
        if let Some(model) = args.model {
            config.clive.model = model;
        }
        if let Some(tracks) = args.tracks {
            config.tracks.audio_tracks = tracks;
        }
    }

    // Validate configuration
    config.validate()?;

    // Initialize cache
    let cache = Cache::default();
    cache.init()?;

    // Check FFmpeg availability
    FFmpeg::check_ffmpeg()?;

    // Process the video
    wordclip::process_video(&config, &cache)?;

    // Clean up if requested
    if !args.no_cleanup {
        cache.cleanup_for_input(config.input_file.as_ref().unwrap())?;
    }

    Ok(())
}

use anyhow::{Context, Result};
use clap::Parser;
use hound;
use log::{debug, info};
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use clive::utils::fetch;
use clive::{Cache, Clip, Config, FFmpeg, Timestamp};

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
    let args = Args::parse();

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
    process_video(&config, &cache)?;

    // Clean up if requested
    if !args.no_cleanup {
        cache.cleanup_for_input(config.input_file.as_ref().unwrap())?;
    }

    Ok(())
}

fn process_video(config: &Config, cache: &Cache) -> Result<()> {
    let input_path = config.input_file.as_ref().unwrap();
    info!("Processing video: {}", input_path.display());

    // Step 1: Check/Download model
    debug!("Step 1: Checking/Downloading model");
    fetch::download_whisper_model_if_needed(
        &config.clive.model,
        &cache.model_path(&config.clive.model),
    )?;

    // Step 2: Extract audio tracks
    debug!("Step 2: Extracting audio tracks");
    let audio_paths = extract_audio_tracks(config, cache)?;
    debug!("Extracted {} audio tracks", audio_paths.len());

    // Step 3: Transcribe audio and combine results
    debug!("Step 3: Transcribing audio");
    let timestamps = transcribe_audio_tracks(&config.clive.model, &audio_paths, cache)?;
    debug!("Found {} timestamp segments", timestamps.len());

    // Step 3.5: Save timestamps to cache
    debug!("Step 3.5: Saving timestamps to cache");
    cache.save_transcription(input_path, timestamps.clone())?;
    debug!("Successfully saved timestamps to cache");

    // Step 4: Find clips based on keywords
    debug!("Step 4: Finding clips based on keywords");
    let clips = find_clips(&timestamps, config)?;
    debug!("Found {} clips matching keywords", clips.len());

    // Step 5: Create output clips
    debug!("Step 5: Creating output clips");
    create_output_clips(input_path, &clips, &config.output.directory)?;
    info!("Successfully created {} clips", clips.len());

    Ok(())
}

fn extract_audio_tracks(config: &Config, cache: &Cache) -> Result<Vec<PathBuf>> {
    let input_path = config.input_file.as_ref().unwrap();
    debug!("Extracting audio tracks from {}", input_path.display());
    let mut audio_paths = Vec::new();

    for &track in &config.tracks.audio_tracks {
        debug!("Processing audio track {}", track);
        let output_path = cache.audio_path(input_path, track);
        debug!("Extracting to {}", output_path.display());
        FFmpeg::extract_audio_tracks(input_path, &output_path, &[track])?;
        audio_paths.push(output_path);
        debug!("Successfully extracted track {}", track);
    }

    Ok(audio_paths)
}

fn load_audio(path: &PathBuf) -> Result<Vec<f32>> {
    debug!("Loading WAV file: {}", path.display());
    let reader = hound::WavReader::open(path).context("Failed to open WAV file")?;

    let samples: Vec<i16> = reader.into_samples().map(|s| s.unwrap()).collect();
    let mut float_samples = vec![0.0; samples.len()];

    // Convert to float samples
    whisper_rs::convert_integer_to_float_audio(&samples, &mut float_samples)
        .context("Failed to convert audio to float")?;

    Ok(float_samples)
}

fn transcribe_audio_tracks(
    model_name: &str,
    audio_paths: &[PathBuf],
    cache: &Cache,
) -> Result<Vec<Timestamp>> {
    debug!("Loading Whisper model: {}", model_name);
    let ctx = WhisperContext::new_with_params(
        &cache.model_path(model_name).to_string_lossy(),
        WhisperContextParameters::default(),
    )
    .context("Failed to load Whisper model")?;
    debug!("Successfully loaded Whisper model");

    let mut all_timestamps: Vec<Timestamp> = Vec::new();

    for (i, audio_path) in audio_paths.iter().enumerate() {
        debug!("Processing audio file {} of {}", i + 1, audio_paths.len());
        let samples = load_audio(audio_path)?;
        debug!("Loaded {} samples", samples.len());

        // Create parameters for transcription
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);

        // Create state and run transcription
        let mut state = ctx.create_state().context("Failed to create state")?;
        debug!("Running transcription on entire audio file");
        state
            .full(params, &samples)
            .context("Failed to process audio")?;

        let num_segments = state
            .full_n_segments()
            .context("Failed to get number of segments")?;
        debug!("Found {} segments", num_segments);

        for i in 0..num_segments {
            let text = state
                .full_get_segment_text(i)
                .context("Failed to get segment text")?;
            let start = state
                .full_get_segment_t0(i)
                .context("Failed to get segment start")? as f64
                * 0.01;
            let end = state
                .full_get_segment_t1(i)
                .context("Failed to get segment end")? as f64
                * 0.01;

            // Get token-level timestamps for this segment
            let num_tokens = state
                .full_n_tokens(i)
                .context("Failed to get number of tokens")?;

            // If there are no tokens, just add the segment
            if num_tokens == 0 {
                debug!("Segment {}: {}s -> {}s: {}", i, start, end, text);
                // check if the last text and the new text are the same
                // if they are the same, don't add the segment
                // if they are different, add the segment
                if all_timestamps.last().is_some() {
                    if all_timestamps.last().unwrap().text == text {
                        continue;
                    }
                }
                all_timestamps.push(Timestamp { start, end, text });
                continue;
            }

            // Process each token in the segment
            let mut token_start = None;
            let mut current_text = String::new();

            for t in 0..num_tokens {
                let token = state
                    .full_get_token_text(i, t)
                    .context("Failed to get token text")?;
                let token_data = state
                    .full_get_token_data(i, t)
                    .context("Failed to get token data")?;

                // Skip special tokens and empty tokens
                if token_data.id >= 50258 || token.trim().is_empty() {
                    continue;
                }

                // Get token time from whisper
                let token_time = state
                    .full_get_segment_t0(i)
                    .context("Failed to get segment start")?
                    as f64
                    * 0.01;

                if token_start.is_none() {
                    token_start = Some(token_time);
                }

                // Add the token text
                current_text.push_str(&token);

                // If this is the last token or the next token is a new word/sentence
                let is_last_token = t == num_tokens - 1;
                let is_word_end = token.ends_with(' ') || token.ends_with('\n');

                if is_last_token || is_word_end {
                    let trimmed_text = current_text.trim();
                    if !trimmed_text.is_empty() {
                        debug!(
                            "Adding word: '{}' ({} -> {})",
                            trimmed_text,
                            token_start.unwrap(),
                            token_time
                        );
                        all_timestamps.push(Timestamp {
                            start: token_start.unwrap(),
                            end: token_time,
                            text: trimmed_text.to_string(),
                        });
                    }
                    token_start = None;
                    current_text.clear();
                }
            }

            // Add any remaining text as a segment
            if !current_text.trim().is_empty() {
                let trimmed_text = current_text.trim();
                debug!(
                    "Adding remaining word: '{}' ({} -> {})",
                    trimmed_text,
                    token_start.unwrap_or(start),
                    end
                );
                all_timestamps.push(Timestamp {
                    start: token_start.unwrap_or(start),
                    end,
                    text: trimmed_text.to_string(),
                });
            }
        }
    }

    debug!("Total timestamps found: {}", all_timestamps.len());
    Ok(all_timestamps)
}

fn find_clips(timestamps: &[Timestamp], config: &Config) -> Result<Vec<Clip>> {
    let mut clips: Vec<Clip> = Vec::new();

    for (keyword, clip_config) in &config.clips {
        for timestamp in timestamps {
            if timestamp
                .text
                .to_lowercase()
                .split_whitespace()
                .map(|word| word.trim_matches(|c: char| c.is_ascii_punctuation()))
                .any(|word| word == keyword.to_lowercase())
            {
                clips.push(Clip {
                    start: (timestamp.start - clip_config.start_time as f64).max(0.0),
                    end: timestamp.end + clip_config.end_time as f64,
                    keyword: keyword.clone(),
                });
            }
        }
    }

    // Merge overlapping clips
    clips.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    let mut merged_clips: Vec<Clip> = Vec::new();

    for clip in clips {
        if let Some(last) = merged_clips.last_mut() {
            if clip.start <= last.end {
                last.end = last.end.max(clip.end);
                last.keyword = format!("{}, {}", last.keyword, clip.keyword);
                continue;
            }
        }
        merged_clips.push(clip);
    }

    Ok(merged_clips)
}

fn create_output_clips(input_path: &PathBuf, clips: &[Clip], output_dir: &PathBuf) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    for (i, clip) in clips.iter().enumerate() {
        let output_path = output_dir.join(format!(
            "clip_{}_{}_{}.mp4",
            i + 1,
            clip.keyword.replace([' ', ','], "_"),
            input_path.file_stem().unwrap().to_string_lossy()
        ));

        FFmpeg::create_clip(input_path, &output_path, clip.start, clip.end)?;
    }

    Ok(())
}

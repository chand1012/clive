use anyhow::{Context, Result};
use clap::Parser;
use hound;
use log::{debug, info, warn};
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use clive::utils::fetch;
use clive::{Cache, Clip, Config, FFmpeg, Llama, Timestamp, VectorDB};

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
    #[arg(long)]
    whisper_model: Option<String>,

    /// Language model to use (base, tiny, small, medium, large)
    #[arg(long)]
    language_model: Option<String>,

    /// Embedding model to use (base, tiny, small, medium, large)
    #[arg(long)]
    embedding_model: Option<String>,

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
            args.whisper_model,
            args.tracks,
            keywords,
        );
        config.merge_cli(cli_config);
    } else {
        config.input_file = Some(args.input.clone());
        if let Some(output) = args.output {
            config.output.directory = output;
        }
        if let Some(whisper_model) = args.whisper_model {
            config.clive.whisper_model = whisper_model;
        }
        if let Some(language_model) = args.language_model {
            config.clive.language_model = language_model;
        }
        if let Some(embedding_model) = args.embedding_model {
            config.clive.embedding_model = embedding_model;
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
    debug!("Step 1: Checking/Downloading models");
    fetch::download_whisper_model_if_needed(
        &config.clive.whisper_model,
        &cache.model_path(&config.clive.whisper_model),
    )?;

    fetch::download_embedding_model_if_needed(
        &config.clive.embedding_model,
        &cache.embedding_model_path(&config.clive.embedding_model),
    )?;

    // Step 2: Extract audio tracks
    debug!("Step 2: Extracting audio tracks");
    let audio_paths = extract_audio_tracks(config, cache)?;
    debug!("Extracted {} audio tracks", audio_paths.len());

    // Step 3: Transcribe audio and combine results
    debug!("Step 3: Transcribing audio");
    let timestamps = transcribe_audio_tracks(&config.clive.whisper_model, &audio_paths, cache)?;
    debug!("Found {} timestamp segments", timestamps.len());

    // Step 3.5: Save timestamps to cache
    debug!("Step 3.5: Saving timestamps to cache");
    cache.save_transcription(input_path, timestamps.clone())?;
    debug!("Successfully saved timestamps to cache");

    // Step 4: Find clips based on keywords
    debug!("Step 4: Finding clips based on keywords");
    let clips = find_clips(&timestamps, config, cache)?;
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
            // Try to get all required segment data, skip if any fails
            let text = match state.full_get_segment_text(i) {
                Ok(text) => text,
                Err(e) => {
                    warn!("Skipping segment {}: Failed to get text: {}", i, e);
                    continue;
                }
            };

            let start = match state.full_get_segment_t0(i) {
                Ok(t) => t as f64 * 0.01,
                Err(e) => {
                    warn!("Skipping segment {}: Failed to get start time: {}", i, e);
                    continue;
                }
            };

            let end = match state.full_get_segment_t1(i) {
                Ok(t) => t as f64 * 0.01,
                Err(e) => {
                    warn!("Skipping segment {}: Failed to get end time: {}", i, e);
                    continue;
                }
            };

            let num_tokens = match state.full_n_tokens(i) {
                Ok(n) => n,
                Err(e) => {
                    warn!(
                        "Skipping segment {}: Failed to get number of tokens: {}",
                        i, e
                    );
                    continue;
                }
            };

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
                let token = match state.full_get_token_text(i, t) {
                    Ok(text) => text,
                    Err(e) => {
                        debug!(
                            "Skipping token {} in segment {}: Failed to get text: {}",
                            t, i, e
                        );
                        continue;
                    }
                };

                let token_data = match state.full_get_token_data(i, t) {
                    Ok(data) => data,
                    Err(e) => {
                        debug!(
                            "Skipping token {} in segment {}: Failed to get data: {}",
                            t, i, e
                        );
                        continue;
                    }
                };

                // Skip special tokens and empty tokens
                if token_data.id >= 50258 || token.trim().is_empty() {
                    continue;
                }

                // Get token time from whisper
                let token_time = match state.full_get_segment_t0(i) {
                    Ok(t) => t as f64 * 0.01,
                    Err(e) => {
                        debug!(
                            "Skipping token {} in segment {}: Failed to get time: {}",
                            t, i, e
                        );
                        continue;
                    }
                };

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

fn find_clips(timestamps: &[Timestamp], config: &Config, cache: &Cache) -> Result<Vec<Clip>> {
    let mut clips: Vec<Clip> = Vec::new();

    // for now we'll just implement the vector db search
    // first initialize the embedding model
    let mut embedding_model = Llama::new(
        cache.embedding_model_path(&config.clive.embedding_model),
        None,
        config.llama.n_threads,
        config.llama.n_batch,
        config.llama.seed,
        None,
        true,
        false,
        false,
    )?;

    // only embedding model we support is 1024 dimensions
    let vector_db = VectorDB::new_in_memory(1024)?;

    // use the batch add to add all the timestamps to the vector db
    // loop through the clips and add them to the vector db
    for clip in timestamps {
        vector_db.add_clip(&mut embedding_model, clip)?;
    }

    // now we can search the vector db for each moment
    for moment in &config.moments {
        let text = &moment.text;

        debug!("Searching for moment: {}", text);
        let results = vector_db.search(&mut embedding_model, text, 3)?;
        debug!("Found {} results", results.len());

        for result in results {
            debug!("Result: {}", result.transcript);
            let neighboring_clips = vector_db.get_neighboring_clips(
                result.id,
                config.line_buffer.before as usize,
                config.line_buffer.after as usize,
            )?;
            debug!("Found {} neighboring clips", neighboring_clips.len());
            // get the start of the first clip and the end of the last clip
            let start = neighboring_clips.first().unwrap().start_time;
            let end = neighboring_clips.last().unwrap().end_time;
            let keyword = neighboring_clips
                .iter()
                .map(|c| c.transcript.clone())
                .collect::<Vec<String>>()
                .join("\n");
            clips.push(Clip {
                start,
                end,
                keyword,
            });
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
            "clip_{}_{}.mp4",
            i + 1,
            input_path.file_stem().unwrap().to_string_lossy()
        ));

        FFmpeg::create_clip(input_path, &output_path, clip.start, clip.end)?;
    }

    Ok(())
}

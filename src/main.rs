use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

mod cache;
mod config;
mod ffmpeg;

use cache::{Cache, Clip, Timestamp};
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
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Whisper model to use (base, tiny, small, medium, large)
    #[arg(short, long)]
    model: Option<String>,

    /// Audio tracks to process (1-based indexing)
    #[arg(short, long)]
    tracks: Option<Vec<u32>>,

    /// Keywords to search for in the audio
    #[arg(short, long)]
    clips: Option<Vec<String>>,

    /// Don't clean up intermediate files
    #[arg(long)]
    no_cleanup: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

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

    // Step 1: Check/Download model
    download_model_if_needed(config, cache)?;

    // Step 2: Extract audio tracks
    let audio_paths = extract_audio_tracks(config, cache)?;

    // Step 3: Transcribe audio and combine results
    let timestamps = transcribe_audio_tracks(&config.clive.model, &audio_paths, cache)?;

    // Step 4: Find clips based on keywords
    let clips = find_clips(&timestamps, config)?;

    // Step 5: Create output clips
    create_output_clips(input_path, &clips, &config.output.directory)?;

    Ok(())
}

fn download_model_if_needed(config: &Config, cache: &Cache) -> Result<()> {
    if !cache.model_exists(&config.clive.model) {
        println!("Downloading {} model...", config.clive.model);
        let url = get_model_url(&config.clive.model)?;

        let response = ureq::get(&url).call().context("Failed to download model")?;

        let mut file = std::fs::File::create(cache.model_path(&config.clive.model))?;
        std::io::copy(&mut response.into_reader(), &mut file)?;
    }
    Ok(())
}

fn get_model_url(model_name: &str) -> Result<String> {
    let base_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
    let url = match model_name {
        "base" => format!("{}ggml-base.en-q8_0.bin?download=true", base_url),
        "tiny" => format!("{}ggml-tiny.en-q8_0.bin?download=true", base_url),
        "small" => format!("{}ggml-small.en-q8_0.bin?download=true", base_url),
        "medium" => format!("{}ggml-medium.en-q5_0.bin?download=true", base_url),
        "large" => format!("{}ggml-large-v3-turbo-q8_0.bin?download=true", base_url),
        _ => anyhow::bail!("Invalid model name: {}", model_name),
    };
    Ok(url)
}

fn extract_audio_tracks(config: &Config, cache: &Cache) -> Result<Vec<PathBuf>> {
    let input_path = config.input_file.as_ref().unwrap();
    let mut audio_paths = Vec::new();

    for &track in &config.tracks.audio_tracks {
        let output_path = cache.audio_path(input_path, track);
        FFmpeg::extract_audio_tracks(input_path, &output_path, &[track])?;
        audio_paths.push(output_path);
    }

    Ok(audio_paths)
}

fn load_audio(path: &PathBuf) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("mp3");

    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: DecoderOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .context("Failed to probe audio format")?;

    let mut format = probed.format;
    let track = format.default_track().context("No default track found")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .context("Failed to create decoder")?;

    let track_id = track.id;
    let mut sample_buf = None;
    let mut samples = Vec::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;
        if sample_buf.is_none() {
            sample_buf = Some(SampleBuffer::<f32>::new(
                decoded.capacity() as u64,
                decoded.spec().clone(),
            ));
        }

        if let Some(buf) = &mut sample_buf {
            buf.copy_interleaved_ref(decoded);
            samples.extend_from_slice(buf.samples());
        }
    }

    Ok(samples)
}

fn transcribe_audio_tracks(
    model_name: &str,
    audio_paths: &[PathBuf],
    cache: &Cache,
) -> Result<Vec<Timestamp>> {
    let ctx = WhisperContext::new_with_params(
        &cache.model_path(model_name).to_string_lossy(),
        WhisperContextParameters::default(),
    )
    .context("Failed to load Whisper model")?;

    let mut all_timestamps = Vec::new();

    for audio_path in audio_paths {
        let samples = load_audio(audio_path)?;

        // Convert samples to i16 for whisper processing
        let mut i16_samples: Vec<i16> = samples.iter().map(|&x| (x * 32767.0) as i16).collect();

        // Process audio in chunks to avoid memory issues
        let chunk_size = 16000 * 30; // 30 seconds chunks
        for chunk in i16_samples.chunks(chunk_size) {
            let mut state = ctx.create_state().context("Failed to create state")?;

            // Convert chunk to f32 for whisper
            let mut inter_samples = vec![0.0; chunk.len()];
            whisper_rs::convert_integer_to_float_audio(chunk, &mut inter_samples)
                .context("Failed to convert audio to float")?;

            // Create parameters for transcription
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_language(Some("en"));
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);

            // Run transcription
            state
                .full(params, &inter_samples)
                .context("Failed to process audio")?;

            // Get segments
            let num_segments = state
                .full_n_segments()
                .context("Failed to get number of segments")?;

            for i in 0..num_segments {
                let text = state
                    .full_get_segment_text(i)
                    .context("Failed to get segment text")?;
                let start = state
                    .full_get_segment_t0(i)
                    .context("Failed to get segment start")? as f64;
                let end = state
                    .full_get_segment_t1(i)
                    .context("Failed to get segment end")? as f64;

                all_timestamps.push(Timestamp { start, end, text });
            }
        }
    }

    // Sort timestamps by start time
    all_timestamps.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    Ok(all_timestamps)
}

fn find_clips(timestamps: &[Timestamp], config: &Config) -> Result<Vec<Clip>> {
    let mut clips: Vec<Clip> = Vec::new();

    for (keyword, clip_config) in &config.clips {
        for timestamp in timestamps {
            if timestamp
                .text
                .to_lowercase()
                .contains(&keyword.to_lowercase())
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

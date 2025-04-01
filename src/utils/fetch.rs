use anyhow::{Context, Result};
use log::{debug, info};
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::PathBuf;
//https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q8_0.bin
/// Base URL for Whisper models on HuggingFace
const HUGGINGFACE_WHISPER_BASE_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

/// Get the download URL for a specific Whisper model
pub fn get_whisper_model_url(model_name: &str) -> Result<String> {
    let url = match model_name {
        "base" => format!(
            "{}ggml-base-q8_0.bin?download=true",
            HUGGINGFACE_WHISPER_BASE_URL
        ),
        "base.en" => format!(
            "{}ggml-base-en-q8_0.bin?download=true",
            HUGGINGFACE_WHISPER_BASE_URL
        ),
        "tiny" => format!(
            "{}ggml-tiny-q8_0.bin?download=true",
            HUGGINGFACE_WHISPER_BASE_URL
        ),
        "tiny.en" => format!(
            "{}ggml-tiny-en-q8_0.bin?download=true",
            HUGGINGFACE_WHISPER_BASE_URL
        ),
        "small" => format!(
            "{}ggml-small-q8_0.bin?download=true",
            HUGGINGFACE_WHISPER_BASE_URL
        ),
        "small.en" => format!(
            "{}ggml-small-en-q8_0.bin?download=true",
            HUGGINGFACE_WHISPER_BASE_URL
        ),
        "medium" => format!(
            "{}ggml-medium-q5_0.bin?download=true",
            HUGGINGFACE_WHISPER_BASE_URL
        ),
        "medium.en" => format!(
            "{}ggml-medium-en-q5_0.bin?download=true",
            HUGGINGFACE_WHISPER_BASE_URL
        ),
        "large" => format!(
            "{}ggml-large-v3-turbo-q8_0.bin?download=true",
            HUGGINGFACE_WHISPER_BASE_URL
        ),
        _ => anyhow::bail!("Invalid model name: {}", model_name),
    };
    Ok(url)
}

// Models are subject to change as we test, this is a guess on what's going to work
/// Get the download URL for a specific Llama model
pub fn get_llama_model_url(model_name: &str) -> Result<String> {
    let url = match model_name {
        "tiny" => "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf".to_string(),
        "small" => "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf".to_string(),
        "base" => "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf".to_string(),
        "medium" => "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_0.gguf".to_string(),
        "large" => "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-IQ4_XS.gguf".to_string(),
        _ => anyhow::bail!("Invalid model name: {}", model_name),
    };
    Ok(url)
}

// Models are subject to change as we test, this is the best open source option I could find
/// Get the download URL for a specific embedding model
pub fn get_embedding_model_url(model_name: &str) -> Result<String> {
    let url = match model_name {
        "base" => "https://huggingface.co/bbvch-ai/bge-m3-GGUF/resolve/main/bge-m3-q4_k_m.gguf"
            .to_string(),
        _ => anyhow::bail!("Invalid model name: {}", model_name),
    };

    Ok(url)
}

const BUFFER_SIZE: usize = 8192; // 8KB buffer size

/// Download a file from a URL to a specific path
///
/// Uses buffered I/O to efficiently handle large files without loading them entirely into memory.
/// The download is processed in chunks of BUFFER_SIZE bytes.
pub fn download_file(url: &str, output_path: &PathBuf) -> Result<()> {
    debug!("Downloading from URL: {}", url);
    let mut response = ureq::get(url).call().context("Failed to download file")?;
    debug!("Got response from server");

    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let mut reader = response.body_mut().as_reader();
    let mut buffer = vec![0; BUFFER_SIZE];

    loop {
        match reader.read(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(n) => {
                writer.write_all(&buffer[..n])?;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }

    writer.flush()?;
    debug!("Successfully downloaded file to {}", output_path.display());
    Ok(())
}

/// Download a Whisper model if it doesn't exist in the cache
pub fn download_whisper_model_if_needed(model_name: &str, model_path: &PathBuf) -> Result<()> {
    if !model_path.exists() {
        info!("Downloading {} model...", model_name);
        let url = get_whisper_model_url(model_name)?;
        download_file(&url, model_path)?;
        info!("Successfully downloaded model");
    } else {
        debug!("Model already exists at {}", model_path.display());
    }
    Ok(())
}

/// Download an embedding model if it doesn't exist in the cache
pub fn download_embedding_model_if_needed(model_name: &str, model_path: &PathBuf) -> Result<()> {
    if !model_path.exists() {
        info!("Downloading {} model...", model_name);
        let url = get_embedding_model_url(model_name)?;
        download_file(&url, model_path)?;
        info!("Successfully downloaded model");
    } else {
        debug!("Model already exists at {}", model_path.display());
    }
    Ok(())
}

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Represents a clip configuration with start and end times
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClipConfig {
    /// Seconds before the moment to start the clip
    pub start_time: u32,
    /// Seconds after the moment to end the clip
    pub end_time: u32,
}

/// Represents a moment to find in the video
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Moment {
    /// The text to search for in the video
    pub text: String,
    /// The clip configuration for this moment
    #[serde(default)]
    pub clip: ClipConfig,
}

/// Configuration for line buffering when searching for clips
#[derive(Debug, Serialize, Deserialize)]
pub struct LineBufferConfig {
    /// Number of lines to look before a potential clip
    #[serde(default = "default_line_buffer")]
    pub before: u32,
    /// Number of lines to look after a potential clip
    #[serde(default = "default_line_buffer")]
    pub after: u32,
}

/// Configuration for Llama model parameters
#[derive(Debug, Serialize, Deserialize)]
pub struct LlamaConfig {
    /// Context size for the model (default: 2048)
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,
    /// Number of threads to use for inference (default: number of available CPU threads)
    #[serde(default)]
    pub n_threads: Option<i32>,
    /// Number of threads to use for batch processing (default: same as n_threads)
    #[serde(default)]
    pub n_batch: Option<i32>,
    /// Random seed for sampling
    #[serde(default)]
    pub seed: Option<u32>,
    /// Whether to disable GPU acceleration (default: false)
    #[serde(default)]
    pub disable_gpu: bool,
}

/// Main configuration structure for the Clive application
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    /// Whisper model configuration
    pub clive: CliveConfig,
    /// Llama model configuration
    #[serde(default)]
    pub llama: LlamaConfig,
    /// Audio track configuration
    pub tracks: TracksConfig,
    /// List of moments to find in the video
    #[serde(default)]
    pub moments: Vec<Moment>,
    /// Output configuration
    pub output: OutputConfig,
    /// Line buffer configuration
    #[serde(default)]
    pub line_buffer: LineBufferConfig,
    /// Input file path (from CLI)
    #[serde(skip)]
    pub input_file: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CliveConfig {
    /// Whisper model to use (base, tiny, small, medium, large)
    #[serde(default = "default_model")]
    pub whisper_model: String,
    /// Language model to use (base, tiny, small, medium, large)
    #[serde(default = "default_model")]
    pub language_model: String,
    /// Embedding model to use (base, tiny, small, medium, large)
    #[serde(default = "default_model")]
    pub embedding_model: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TracksConfig {
    /// Audio track numbers to process (1-based indexing)
    #[serde(default = "default_audio_tracks")]
    pub audio_tracks: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Directory where output files will be saved
    #[serde(default = "default_output_dir")]
    pub directory: PathBuf,
}

fn default_audio_tracks() -> Vec<u32> {
    vec![1, 2]
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("output")
}

fn default_model() -> String {
    String::from("base")
}

fn default_line_buffer() -> u32 {
    5
}

fn default_ctx_size() -> u32 {
    2048
}

impl Default for LineBufferConfig {
    fn default() -> Self {
        Self {
            before: default_line_buffer(),
            after: default_line_buffer(),
        }
    }
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            ctx_size: default_ctx_size(),
            n_threads: None,
            n_batch: None,
            seed: None,
            disable_gpu: false,
        }
    }
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            start_time: 30,
            end_time: 30,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            clive: CliveConfig {
                whisper_model: default_model(),
                language_model: default_model(),
                embedding_model: default_model(),
            },
            llama: LlamaConfig::default(),
            tracks: TracksConfig {
                audio_tracks: default_audio_tracks(),
            },
            moments: Vec::new(),
            output: OutputConfig {
                directory: default_output_dir(),
            },
            line_buffer: LineBufferConfig::default(),
            input_file: None,
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    ///
    /// # Arguments
    /// * `path` - Path to the configuration file
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path).context("Failed to read config file")?;
        let config: Config = toml::from_str(&contents).context("Failed to parse config file")?;

        // Ensure output directory exists
        fs::create_dir_all(&config.output.directory)
            .context("Failed to create output directory")?;

        Ok(config)
    }

    /// Save configuration to a TOML file
    ///
    /// # Arguments
    /// * `path` - Path where the configuration will be saved
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let toml = toml::to_string_pretty(self).context("Failed to serialize config")?;
        fs::write(path, toml).context("Failed to write config file")?;
        Ok(())
    }

    /// Create a new configuration from command line arguments
    ///
    /// # Arguments
    /// * `input` - Path to the input video file
    /// * `output` - Path to the output directory
    /// * `model` - Whisper model to use
    /// * `tracks` - Audio tracks to process
    /// * `moments` - Moments to search for in the video
    pub fn from_cli(
        input: PathBuf,
        output: Option<PathBuf>,
        model: Option<String>,
        tracks: Option<Vec<u32>>,
        moments: Vec<String>,
    ) -> Self {
        let mut config = Config::default();

        config.input_file = Some(input);

        if let Some(output_dir) = output {
            config.output.directory = output_dir;
        }

        if let Some(model_name) = model {
            config.clive.whisper_model = model_name;
        }

        if let Some(track_list) = tracks {
            config.tracks.audio_tracks = track_list;
        }

        // Create moment configs for each moment with default clip timings
        config.moments = moments
            .into_iter()
            .map(|text| Moment {
                text,
                clip: ClipConfig::default(),
            })
            .collect();

        config
    }

    /// Merge command line arguments into an existing configuration
    ///
    /// # Arguments
    /// * `cli_config` - Configuration from command line arguments
    pub fn merge_cli(&mut self, cli_config: Config) {
        self.input_file = cli_config.input_file;

        if !cli_config.moments.is_empty() {
            self.moments = cli_config.moments;
        }

        if cli_config.tracks.audio_tracks != default_audio_tracks() {
            self.tracks.audio_tracks = cli_config.tracks.audio_tracks;
        }

        if cli_config.clive.whisper_model != "base" {
            self.clive.whisper_model = cli_config.clive.whisper_model;
        }

        if cli_config.output.directory != default_output_dir() {
            self.output.directory = cli_config.output.directory;
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Check if input file is specified
        if self.input_file.is_none() {
            anyhow::bail!("Input file not specified");
        }

        // Check if input file exists
        if !self.input_file.as_ref().unwrap().exists() {
            anyhow::bail!("Input file does not exist");
        }

        // Validate model name
        match self.clive.whisper_model.as_str() {
            "base" | "tiny" | "small" | "medium" | "large" | "base.en" | "tiny.en" | "small.en"
            | "medium.en" | "large.en" => (),
            _ => anyhow::bail!("Invalid model name: {}", self.clive.whisper_model),
        }

        // Validate audio tracks
        if self.tracks.audio_tracks.is_empty() {
            anyhow::bail!("No audio tracks specified");
        }

        // Validate moments
        if self.moments.is_empty() {
            anyhow::bail!("No moments specified");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.clive.whisper_model, "base");
        assert_eq!(config.tracks.audio_tracks, vec![1, 2]);
        assert!(config.moments.is_empty());
        assert_eq!(config.output.directory, PathBuf::from("output"));
    }

    #[test]
    fn test_config_from_cli() {
        let input = PathBuf::from("test.mp4");
        let moments = vec!["test1".to_string(), "test2".to_string()];

        let config = Config::from_cli(input.clone(), None, None, None, moments);

        assert_eq!(config.input_file.unwrap(), input);
        assert_eq!(config.moments.len(), 2);
        assert!(config.moments.iter().any(|m| m.text == "test1"));
        assert!(config.moments.iter().any(|m| m.text == "test2"));
    }

    #[test]
    fn test_config_save_and_load() -> Result<()> {
        let mut config = Config::default();
        config.moments.push(Moment {
            text: "test".to_string(),
            clip: ClipConfig {
                start_time: 10,
                end_time: 20,
            },
        });

        let temp_file = NamedTempFile::new()?;
        config.save_to_file(temp_file.path())?;

        let loaded_config = Config::from_file(temp_file.path())?;
        assert_eq!(loaded_config.moments.len(), 1);
        assert_eq!(loaded_config.moments[0].clip.start_time, 10);
        assert_eq!(loaded_config.moments[0].clip.end_time, 20);

        Ok(())
    }
}

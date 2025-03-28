use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Represents a clip configuration with start and end times
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClipConfig {
    /// Seconds before the keyword to start the clip
    pub start_time: u32,
    /// Seconds after the keyword to end the clip
    pub end_time: u32,
}

/// Main configuration structure for the Clive application
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    /// Whisper model configuration
    pub clive: CliveConfig,
    /// Audio track configuration
    pub tracks: TracksConfig,
    /// Clip configurations keyed by keyword
    pub clips: HashMap<String, ClipConfig>,
    /// Output configuration
    pub output: OutputConfig,
    /// Input file path (from CLI)
    #[serde(skip)]
    pub input_file: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CliveConfig {
    /// Whisper model to use (base, tiny, small, medium, large)
    pub model: String,
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

impl Default for Config {
    fn default() -> Self {
        Self {
            clive: CliveConfig {
                model: String::from("base"),
            },
            tracks: TracksConfig {
                audio_tracks: default_audio_tracks(),
            },
            clips: HashMap::new(),
            output: OutputConfig {
                directory: default_output_dir(),
            },
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
        let mut config: Config =
            toml::from_str(&contents).context("Failed to parse config file")?;

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
    /// * `keywords` - Keywords to search for in the audio
    pub fn from_cli(
        input: PathBuf,
        output: Option<PathBuf>,
        model: Option<String>,
        tracks: Option<Vec<u32>>,
        keywords: Vec<String>,
    ) -> Self {
        let mut config = Config::default();

        config.input_file = Some(input);

        if let Some(output_dir) = output {
            config.output.directory = output_dir;
        }

        if let Some(model_name) = model {
            config.clive.model = model_name;
        }

        if let Some(track_list) = tracks {
            config.tracks.audio_tracks = track_list;
        }

        // Create clip configs for each keyword with default timings (30 seconds)
        for keyword in keywords {
            config.clips.insert(
                keyword,
                ClipConfig {
                    start_time: 30,
                    end_time: 30,
                },
            );
        }

        config
    }

    /// Merge command line arguments into an existing configuration
    ///
    /// # Arguments
    /// * `cli_config` - Configuration from command line arguments
    pub fn merge_cli(&mut self, cli_config: Config) {
        self.input_file = cli_config.input_file;

        if !cli_config.clips.is_empty() {
            self.clips = cli_config.clips;
        }

        if cli_config.tracks.audio_tracks != default_audio_tracks() {
            self.tracks.audio_tracks = cli_config.tracks.audio_tracks;
        }

        if cli_config.clive.model != "base" {
            self.clive.model = cli_config.model;
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
        match self.clive.model.as_str() {
            "base" | "tiny" | "small" | "medium" | "large" => (),
            _ => anyhow::bail!("Invalid model name: {}", self.clive.model),
        }

        // Validate audio tracks
        if self.tracks.audio_tracks.is_empty() {
            anyhow::bail!("No audio tracks specified");
        }

        // Validate clip configurations
        if self.clips.is_empty() {
            anyhow::bail!("No clips specified");
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
        assert_eq!(config.clive.model, "base");
        assert_eq!(config.tracks.audio_tracks, vec![1, 2]);
        assert!(config.clips.is_empty());
        assert_eq!(config.output.directory, PathBuf::from("output"));
    }

    #[test]
    fn test_config_from_cli() {
        let input = PathBuf::from("test.mp4");
        let keywords = vec!["test1".to_string(), "test2".to_string()];

        let config = Config::from_cli(input.clone(), None, None, None, keywords);

        assert_eq!(config.input_file.unwrap(), input);
        assert_eq!(config.clips.len(), 2);
        assert!(config.clips.contains_key("test1"));
        assert!(config.clips.contains_key("test2"));
    }

    #[test]
    fn test_config_save_and_load() -> Result<()> {
        let mut config = Config::default();
        config.clips.insert(
            "test".to_string(),
            ClipConfig {
                start_time: 10,
                end_time: 20,
            },
        );

        let temp_file = NamedTempFile::new()?;
        config.save_to_file(temp_file.path())?;

        let loaded_config = Config::from_file(temp_file.path())?;
        assert_eq!(loaded_config.clips.len(), 1);
        assert_eq!(loaded_config.clips["test"].start_time, 10);
        assert_eq!(loaded_config.clips["test"].end_time, 20);

        Ok(())
    }
}

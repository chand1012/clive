use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
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
    pub clips: ClipsConfig,
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
    pub audio_tracks: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClipsConfig {
    #[serde(flatten)]
    pub keywords: std::collections::HashMap<String, ClipConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Directory where output files will be saved
    pub directory: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            clive: CliveConfig {
                model: "base".to_string(),
            },
            tracks: TracksConfig {
                audio_tracks: vec![1],
            },
            clips: ClipsConfig {
                keywords: std::collections::HashMap::new(),
            },
            output: OutputConfig {
                directory: PathBuf::from("output"),
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
        let content = std::fs::read_to_string(path).context("Failed to read config file")?;
        toml::from_str(&content).context("Failed to parse config file")
    }

    /// Save configuration to a TOML file
    ///
    /// # Arguments
    /// * `path` - Path where the configuration will be saved
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let toml = toml::to_string_pretty(self).context("Failed to serialize config")?;
        std::fs::write(path, toml).context("Failed to write config file")?;
        Ok(())
    }

    /// Create configuration from CLI arguments
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

        if let Some(output) = output {
            config.output.directory = output;
        }

        if let Some(model) = model {
            config.clive.model = model;
        }

        if let Some(tracks) = tracks {
            config.tracks.audio_tracks = tracks;
        }

        // Create clip configs for each keyword with default timings
        for keyword in keywords {
            config.clips.keywords.insert(
                keyword,
                ClipConfig {
                    start_time: 10,
                    end_time: 10,
                },
            );
        }

        config
    }

    /// Merge CLI configuration into this configuration
    ///
    /// # Arguments
    /// * `cli_config` - Configuration from command line arguments
    pub fn merge_cli(&mut self, cli_config: Config) {
        self.input_file = cli_config.input_file;
        self.output = cli_config.output;
        self.clive = cli_config.clive;
        self.tracks = cli_config.tracks;
        self.clips = cli_config.clips;
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.input_file.is_none() {
            anyhow::bail!("Input file is required");
        }

        if self.tracks.audio_tracks.is_empty() {
            anyhow::bail!("At least one audio track must be specified");
        }

        if self.clips.keywords.is_empty() {
            anyhow::bail!("At least one clip keyword must be specified");
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
        assert_eq!(config.tracks.audio_tracks, vec![1]);
        assert!(config.clips.keywords.is_empty());
        assert_eq!(config.output.directory, PathBuf::from("output"));
    }

    #[test]
    fn test_config_from_cli() {
        let input = PathBuf::from("test.mp4");
        let keywords = vec!["test1".to_string(), "test2".to_string()];

        let config = Config::from_cli(input.clone(), None, None, None, keywords);

        assert_eq!(config.input_file.unwrap(), input);
        assert_eq!(config.clips.keywords.len(), 2);
        assert!(config.clips.keywords.contains_key("test1"));
        assert!(config.clips.keywords.contains_key("test2"));
    }

    #[test]
    fn test_config_save_and_load() -> Result<()> {
        let mut config = Config::default();
        config.clips.keywords.insert(
            "test".to_string(),
            ClipConfig {
                start_time: 10,
                end_time: 20,
            },
        );

        let temp_file = NamedTempFile::new()?;
        config.save_to_file(temp_file.path())?;

        let loaded_config = Config::from_file(temp_file.path())?;
        assert_eq!(loaded_config.clips.keywords.len(), 1);
        assert_eq!(loaded_config.clips.keywords["test"].start_time, 10);
        assert_eq!(loaded_config.clips.keywords["test"].end_time, 20);

        Ok(())
    }
}

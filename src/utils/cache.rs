use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Represents a timestamp in the transcription
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Timestamp {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// The transcribed text
    pub text: String,
}

/// Represents a clip with its timing information
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Clip {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// The keyword that triggered this clip
    pub keyword: String,
}

/// Manages cache directories and intermediate files
#[derive(Debug)]
pub struct Cache {
    /// Base cache directory
    cache_dir: PathBuf,
    /// Directory for model files
    models_dir: PathBuf,
    /// Directory for audio files
    audio_dir: PathBuf,
    /// Directory for transcription files
    transcription_dir: PathBuf,
    /// Directory for clip metadata
    clips_dir: PathBuf,
}

impl Default for Cache {
    fn default() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("clive");

        Self::new(cache_dir)
    }
}

impl Cache {
    /// Create a new cache instance with the specified base directory
    pub fn new(cache_dir: PathBuf) -> Self {
        let models_dir = cache_dir.join("models");
        let audio_dir = cache_dir.join("audio");
        let transcription_dir = cache_dir.join("transcriptions");
        let clips_dir = cache_dir.join("clips");

        Self {
            cache_dir,
            models_dir,
            audio_dir,
            transcription_dir,
            clips_dir,
        }
    }

    /// Initialize cache directories
    pub fn init(&self) -> Result<()> {
        fs::create_dir_all(&self.models_dir).context("Failed to create models directory")?;
        fs::create_dir_all(&self.audio_dir).context("Failed to create audio directory")?;
        fs::create_dir_all(&self.transcription_dir)
            .context("Failed to create transcription directory")?;
        fs::create_dir_all(&self.clips_dir).context("Failed to create clips directory")?;
        Ok(())
    }

    /// Get the path for a model file
    pub fn model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(format!("whisper-{}.bin", model_name))
    }

    /// Get the path for an embedding model file
    pub fn embedding_model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir
            .join(format!("embedding-{}.bin", model_name))
    }

    /// Get the path for a language model file
    pub fn language_model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(format!("llm-{}.bin", model_name))
    }

    /// Check if a model file exists
    pub fn model_exists(&self, model_name: &str) -> bool {
        self.model_path(model_name).exists()
    }

    /// Get the path for an extracted audio file
    pub fn audio_path(&self, input_path: &Path, track: u32) -> PathBuf {
        let file_stem = input_path.file_stem().unwrap_or_default();
        self.audio_dir.join(format!(
            "{}_track_{}.wav",
            file_stem.to_string_lossy(),
            track
        ))
    }

    /// Get the path for a transcription file
    pub fn transcription_path(&self, input_path: &Path) -> PathBuf {
        let file_stem = input_path.file_stem().unwrap_or_default();
        self.transcription_dir
            .join(format!("{}.json", file_stem.to_string_lossy()))
    }

    /// Get the path for a clips metadata file
    pub fn clips_path(&self, input_path: &Path) -> PathBuf {
        let file_stem = input_path.file_stem().unwrap_or_default();
        self.clips_dir
            .join(format!("{}_clips.json", file_stem.to_string_lossy()))
    }

    /// Save transcription data to cache
    pub fn save_transcription(&self, input_path: &Path, timestamps: Vec<Timestamp>) -> Result<()> {
        let path = self.transcription_path(input_path);
        let json = serde_json::to_string_pretty(&timestamps)
            .context("Failed to serialize transcription")?;
        fs::write(&path, json).context("Failed to write transcription file")?;
        Ok(())
    }

    /// Load transcription data from cache
    pub fn load_transcription(&self, input_path: &Path) -> Result<Vec<Timestamp>> {
        let path = self.transcription_path(input_path);
        let json = fs::read_to_string(&path).context("Failed to read transcription file")?;
        let timestamps =
            serde_json::from_str(&json).context("Failed to parse transcription file")?;
        Ok(timestamps)
    }

    /// Save clips metadata to cache
    pub fn save_clips(&self, input_path: &Path, clips: Vec<Clip>) -> Result<()> {
        let path = self.clips_path(input_path);
        let json = serde_json::to_string_pretty(&clips).context("Failed to serialize clips")?;
        fs::write(&path, json).context("Failed to write clips file")?;
        Ok(())
    }

    /// Load clips metadata from cache
    pub fn load_clips(&self, input_path: &Path) -> Result<Vec<Clip>> {
        let path = self.clips_path(input_path);
        let json = fs::read_to_string(&path).context("Failed to read clips file")?;
        let clips = serde_json::from_str(&json).context("Failed to parse clips file")?;
        Ok(clips)
    }

    /// Clean up all cache files
    pub fn cleanup(&self) -> Result<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir).context("Failed to remove cache directory")?;
        }
        Ok(())
    }

    /// Clean up cache files for a specific input file
    pub fn cleanup_for_input(&self, input_path: &Path) -> Result<()> {
        // Remove audio files
        for entry in fs::read_dir(&self.audio_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with(input_path.file_stem().unwrap().to_string_lossy().as_ref()))
                .unwrap_or(false)
            {
                fs::remove_file(path)?;
            }
        }

        // Remove transcription file
        let transcription_path = self.transcription_path(input_path);
        if transcription_path.exists() {
            fs::remove_file(transcription_path)?;
        }

        // Remove clips file
        let clips_path = self.clips_path(input_path);
        if clips_path.exists() {
            fs::remove_file(clips_path)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_test_cache() -> (Cache, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let cache = Cache::new(temp_dir.path().to_path_buf());
        cache.init().unwrap();
        (cache, temp_dir)
    }

    #[test]
    fn test_cache_initialization() {
        let (cache, _temp_dir) = setup_test_cache();
        assert!(cache.models_dir.exists());
        assert!(cache.audio_dir.exists());
        assert!(cache.transcription_dir.exists());
        assert!(cache.clips_dir.exists());
    }

    #[test]
    fn test_model_path() {
        let (cache, _temp_dir) = setup_test_cache();
        let model_path = cache.model_path("base");
        assert_eq!(
            model_path.file_name().unwrap().to_string_lossy(),
            "ggml-base.bin"
        );
    }

    #[test]
    fn test_save_and_load_transcription() -> Result<()> {
        let (cache, _temp_dir) = setup_test_cache();
        let input_path = Path::new("test.mp4");

        let timestamps = vec![Timestamp {
            start: 0.0,
            end: 1.0,
            text: "Hello".to_string(),
        }];

        cache.save_transcription(input_path, timestamps.clone())?;
        let loaded = cache.load_transcription(input_path)?;

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].text, "Hello");

        Ok(())
    }

    #[test]
    fn test_save_and_load_clips() -> Result<()> {
        let (cache, _temp_dir) = setup_test_cache();
        let input_path = Path::new("test.mp4");

        let clips = vec![Clip {
            start: 0.0,
            end: 1.0,
            keyword: "test".to_string(),
        }];

        cache.save_clips(input_path, clips.clone())?;
        let loaded = cache.load_clips(input_path)?;

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].keyword, "test");

        Ok(())
    }

    #[test]
    fn test_cleanup() -> Result<()> {
        let (cache, _temp_dir) = setup_test_cache();

        // Create some test files
        let input_path = Path::new("test.mp4");
        fs::write(cache.audio_path(input_path, 1), "test")?;

        let clips = vec![Clip {
            start: 0.0,
            end: 1.0,
            keyword: "test".to_string(),
        }];
        cache.save_clips(input_path, clips)?;

        // Test cleanup for specific input
        cache.cleanup_for_input(input_path)?;
        assert!(!cache.audio_path(input_path, 1).exists());
        assert!(!cache.clips_path(input_path).exists());

        // Test full cleanup
        cache.cleanup()?;
        assert!(!cache.cache_dir.exists());

        Ok(())
    }
}

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

/// Handles all FFMPEG-related operations for video and audio processing
pub struct FFmpeg;

impl FFmpeg {
    /// Checks if FFmpeg is available on the system
    pub fn check_ffmpeg() -> Result<()> {
        Command::new("ffmpeg")
            .arg("-version")
            .output()
            .context("FFmpeg is not installed or not available in system PATH")?;
        Ok(())
    }

    /// Extracts specific audio tracks from a video file
    ///
    /// # Arguments
    /// * `input_path` - Path to the input video file
    /// * `output_path` - Path where the extracted audio will be saved
    /// * `tracks` - Vector of track numbers to extract (1-based indexing)
    pub fn extract_audio_tracks(
        input_path: &Path,
        output_path: &Path,
        tracks: &[u32],
    ) -> Result<()> {
        let track_map = tracks
            .iter()
            .map(|t| format!("0:{}", t - 1))
            .collect::<Vec<_>>()
            .join(";");

        Command::new("ffmpeg")
            .args([
                "-i",
                input_path.to_str().unwrap(),
                "-map",
                &track_map,
                "-c:a",
                "libmp3lame",
                "-q:a",
                "4",
                output_path.to_str().unwrap(),
                "-y",
            ])
            .output()
            .context("Failed to extract audio tracks")?;

        Ok(())
    }

    /// Creates a clip from the video file based on start and end timestamps
    ///
    /// # Arguments
    /// * `input_path` - Path to the input video file
    /// * `output_path` - Path where the clip will be saved
    /// * `start_time` - Start time in seconds
    /// * `end_time` - End time in seconds
    pub fn create_clip(
        input_path: &Path,
        output_path: &Path,
        start_time: f64,
        end_time: f64,
    ) -> Result<()> {
        Command::new("ffmpeg")
            .args([
                "-i",
                input_path.to_str().unwrap(),
                "-ss",
                &start_time.to_string(),
                "-t",
                &(end_time - start_time).to_string(),
                "-c:v",
                "copy", // Copy video stream without re-encoding
                "-c:a",
                "copy", // Copy audio stream without re-encoding
                output_path.to_str().unwrap(),
                "-y",
            ])
            .output()
            .context("Failed to create video clip")?;

        Ok(())
    }

    /// Combines multiple clips into a single video file
    ///
    /// # Arguments
    /// * `clip_paths` - Vector of paths to input clips
    /// * `output_path` - Path where the combined video will be saved
    pub fn combine_clips(clip_paths: &[&Path], output_path: &Path) -> Result<()> {
        // Create a temporary file listing all clips
        let temp_file = tempfile::NamedTempFile::new()?;
        let mut file_content = String::new();

        for path in clip_paths {
            file_content.push_str(&format!("file '{}'\n", path.to_str().unwrap()));
        }

        std::fs::write(&temp_file, file_content)?;

        Command::new("ffmpeg")
            .args([
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                temp_file.path().to_str().unwrap(),
                "-c",
                "copy",
                output_path.to_str().unwrap(),
                "-y",
            ])
            .output()
            .context("Failed to combine video clips")?;

        Ok(())
    }

    /// Gets the duration of a video file in seconds
    ///
    /// # Arguments
    /// * `input_path` - Path to the input video file
    pub fn get_duration(input_path: &Path) -> Result<f64> {
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_path.to_str().unwrap(),
            ])
            .output()
            .context("Failed to get video duration")?;

        let duration_str = String::from_utf8(output.stdout)?;
        let duration = duration_str.trim().parse::<f64>()?;

        Ok(duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_ffmpeg_available() {
        assert!(FFmpeg::check_ffmpeg().is_ok());
    }
}

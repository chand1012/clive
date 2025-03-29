<p align="center">
  <img src="assets/logo.png" alt="Clive Logo" />
  <h1>Clive</h1>
  <p>Clive is a CLI tool for transcribing and clipping audio using keywords.</p>
</p>

## Toolchain

- Rust
- Whisper CPP (via whisper-rs)
  - Will require [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) on Windows and Linux.
  - Uses Metal on MacOS.
- FFMPEG for video and audio processing.
  - Will require [FFMPEG](https://ffmpeg.org/download.html) on Windows, Linux, and MacOS to be available on the system path.
- GGML files from [HuggingFace](https://huggingface.co/ggerganov/whisper.cpp/tree/main).
  - Will download at runtime if not present. Allow the user to choose the model via the config or the command line.

## Configuration

Configuration will be done via either a config toml file or via command line arguments.

```toml
[clive]
model = "base"

[tracks]
# if there are multiple audio tracks in a video, specify which ones to use.
# Ideally users should record voices in separate tracks for easier processing.
audio_tracks = [1, 2] # default 

[clips.keyword1]
start_time = 10 # seconds before the keyword
end_time = 10 # seconds after the keyword

[clips.keyword2]
start_time = 10 # seconds before the keyword
end_time = 10 # seconds after the keyword

[output]
directory = "output"
```

```bash
# using the config file
clive --config config.toml

# using the command line
clive --input input.mp4 --output output.mp4 --model base --tracks 1 2 --clips keyword1 keyword2 keyword3 # default for CLI is 30 seconds before and after the keyword
```

## Processing Flow

1. Check if the GGML model is present.

- If not, download it from HuggingFace.
- Save it to ~/.cache/clive/models/ggml-<model-name>.bin

2. Process the input video

- Extract the audio tracks specified in the config.
- Convert the audio to MP3 format. (saved to ~/.cache/clive/audio)
- Transcribe the audio.
- Save the transcription as a JSON file to the cache directory. (saved to ~/.cache/clive/transcriptions)

3. Process the transcription

- Read the transcription JSON file.
- For each keyword, add the timestamps to the transcription object.
- Once all the keywords are added, combine any overlapping clips to form a single clip.
- Save the clips as a JSON file to the cache directory. (saved to ~/.cache/clive/clips)

4. Process the clips

- Read the clips JSON file.
- For each clip, create a new video file with the specified start and end times from the original video.
- Save the new video files to the output directory.

### Notes

- The outputs between steps is primarily for debugging purposes.
- There should be a cleanup step at the end to remove any intermediate files.
  - This can be disabled via a `--no-cleanup` flag.
- The config file is optional, but either config or command line arguments must be provided.

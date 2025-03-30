# Clive: CLI tool for transcribing and clipping audio

<div align="center">
  <img src="assets/logo.png" alt="Clive Logo" width="200">
  <h1>Clive CLI: Audio transcription and clipping for the AI era</h1>
</div>

<div align="center">
  <a href="https://github.com/chand1012/clive">
    <img src="https://img.shields.io/badge/Clive-CLI-blue" alt="Clive CLI">
  </a>
  <a href="https://github.com/chand1012/clive/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
</div>

Clive is a command-line tool for transcribing audio and creating clips based on keywords. It's designed to make working with audio transcription and clipping as human-friendly as possible, providing a simple and natural syntax with formatted output.

## Installing from source

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install)
- [LLVM](https://llvm.org/docs/GettingStarted.html)
- [FFMPEG](https://ffmpeg.org/download.html)
- [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) (Windows and Linux)
- [XCode](https://developer.apple.com/xcode/) (MacOS)

### Clone the repository

```bash
git clone https://github.com/chand1012/clive.git
cd clive
cargo install --path .
```

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

## Roadmap

- [x] Get initial clipping working
- [x] Add a step to merge overlapping clips
- [ ] Add the ability to cut out silence
- [ ] Add LLM support for more complex operations

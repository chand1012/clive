<div align="center">
  <img src="assets/logo.png" alt="Clive Logo" width="200">
  <h1>Clive CLI: Natural language video processing</h1>
</div>

<div align="center">
  <a href="https://github.com/chand1012/clive">
    <img src="https://img.shields.io/badge/Clive-CLI-blue" alt="Clive CLI">
  </a>
  <a href="https://github.com/chand1012/clive/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
</div>

Clive is a command-line tool for transcribing audio and creating clips based on keywords and natural language. It's designed to make working with audio transcription and clipping as human-friendly as possible, providing a simple and natural syntax with formatted output.

## Why?

I've been working on YouTube videos as a hobby project of mine, and one of the tasks I've had to do a lot of is finding certain moments in long swathes of video that I want to use in my videos.
Rather than sitting in Davinci Resolve for hours and manually finding the moments, I thought it would be interesting to see if it's possible to automate this process using AI.
This is also being used to see how far I can push the limits of a local AI model, and how much I can squeeze into a single CLI tool.
Rather than using Python and tons and tons of libraries that users would have to install, I wanted to see if I could condense a full AI pipeline into a single executable.
Normally I would've chosen Go for this, but my friend [Kyle](https://x.com/kylevasulka) and I are working on a [Tauri app](https://github.com/TimeSurgeLabs/twitch-tools) for our creator tooling, so by using Rust we can easily share code between the two projects.

## Installation

### Prerequisites

Before installing Clive, ensure you have the following dependencies installed on your system:

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| [Rust](https://www.rust-lang.org/tools/install) | Core build toolchain | Follow the official Rust installation guide |
| [FFMPEG](https://ffmpeg.org/download.html) | Audio/video processing | Must be available in system PATH |
| [LLVM](https://llvm.org/docs/GettingStarted.html) | Required for building | Follow OS-specific instructions |

#### Platform-Specific Requirements

- **Windows & Linux**
  - [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) - Required for GPU acceleration
- **macOS**
  - Xcode - Required for Metal support
  - Install via: `xcode-select --install`

### Building from Source

1. Clone the repository:
```bash
git clone https://github.com/chand1012/clive.git
cd clive
```

2. Build and install:
```bash
cargo install --path .
```

The binary will be installed to your Cargo bin directory (usually `~/.cargo/bin/`).

## Usage

Clive can be configured either through a TOML configuration file or command-line arguments.

### Command Line Interface

```bash
# Basic usage with config file
clive --config config.toml

# Direct command line usage
clive --input input.mp4 \
      --output output.mp4 \
      --model base \
      --tracks 1 2 \
      --clips keyword1 keyword2 keyword3
```

### Configuration File

Create a `config.toml` file with your desired settings:

```toml
[clive]
model = "base"  # Choose from: tiny, base, small, medium, large

[tracks]
# Specify which audio tracks to process
audio_tracks = [1, 2]  # Default tracks

[clips]
# Define keywords and their clip boundaries
[clips.keyword1]
start_time = 10  # Seconds before keyword
end_time = 10   # Seconds after keyword

[clips.keyword2]
start_time = 10
end_time = 10

[output]
directory = "output"  # Output directory for processed clips
```

## How It Works

Clive processes your audio/video files in four main stages:

### 1. Model Preparation
- Checks for Whisper GGML model in `~/.cache/clive/models/`
- Downloads from HuggingFace if not present
- Supports multiple model sizes (tiny to large)

### 2. Audio Processing
- Extracts specified audio tracks
- Converts to WAV format
- Stores temporary files in `~/.cache/clive/audio/`
- Performs transcription
- Saves transcription JSON to `~/.cache/clive/transcriptions/`

### 3. Keyword Processing
- Analyzes transcription for keywords
- Identifies timestamps for each keyword
- Merges overlapping clip segments
- Saves clip data to `~/.cache/clive/clips/`

### 4. Video Generation
- Creates individual clips based on timestamps
- Exports to specified output directory
- Optionally cleans up temporary files

## Advanced Features

### Cache Management
- Temporary files stored in `~/.cache/clive/`
- Use `--no-cleanup` to preserve intermediate files
- Useful for debugging or reprocessing

### Model Selection
Available models from HuggingFace:
- `tiny`: Fastest, lowest accuracy
- `base`: Good balance of speed/accuracy
- `small`: Better accuracy, slower
- `medium`: High accuracy, slower
- `large`: Best accuracy, slowest

## Roadmap

- [x] Basic clip extraction and merging
- [x] Overlapping clip management
- [ ] LLM integration for advanced operations
  - [x] Basic vector based clip search
  - [ ] Analyze the transcript for moments that the user wants to clip
  - [ ] Make the vector based search looser, it should find a lot more clips
  - [ ] Make modes of "lite" and "full" configurable. "lite" is vector only, "full" is vector + LLM

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

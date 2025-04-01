pub mod utils {
    mod cache;
    mod config;
    pub mod fetch;
    mod ffmpeg;
    mod llama;
    mod vec_db;
    pub use cache::{Cache, Clip, Timestamp};
    pub use config::Config;
    pub use ffmpeg::FFmpeg;
    pub use llama::Llama;
    pub use vec_db::VectorDB;
}

// Re-export commonly used types at the crate root for convenience
pub use utils::{Cache, Clip, Config, FFmpeg, Llama, Timestamp, VectorDB};

use clap::{Parser, Subcommand};

pub mod wordclip;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Clip videos based on spoken words
    WordClip(wordclip::WordClipArgs),
}

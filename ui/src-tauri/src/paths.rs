//! Where the shell keeps its own files. The state folder is the one the
//! engine uses for its logs, so both logs end up side by side:
//!
//! * Linux: `$XDG_STATE_HOME/betasuite` (default `~/.local/state/betasuite`)
//! * Windows: `%LOCALAPPDATA%\BetaSuite\state`
//! * macOS: `~/Library/Application Support/BetaSuite/state`
//!
//! An environment variable that is empty or relative is ignored, as the
//! XDG base directory rules ask.

use std::env;
use std::path::PathBuf;

fn absolute_env(name: &str) -> Option<PathBuf> {
    let value = PathBuf::from(env::var_os(name)?);
    value.is_absolute().then_some(value)
}

#[cfg(windows)]
pub fn state_dir() -> Option<PathBuf> {
    Some(
        absolute_env("LOCALAPPDATA")?
            .join("BetaSuite")
            .join("state"),
    )
}

#[cfg(target_os = "macos")]
pub fn state_dir() -> Option<PathBuf> {
    Some(
        absolute_env("HOME")?
            .join("Library")
            .join("Application Support")
            .join("BetaSuite")
            .join("state"),
    )
}

#[cfg(all(unix, not(target_os = "macos")))]
pub fn state_dir() -> Option<PathBuf> {
    let base = absolute_env("XDG_STATE_HOME")
        .or_else(|| Some(absolute_env("HOME")?.join(".local").join("state")))?;
    Some(base.join("betasuite"))
}

pub fn log_dir() -> Option<PathBuf> {
    Some(state_dir()?.join("logs"))
}

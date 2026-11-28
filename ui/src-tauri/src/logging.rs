//! The shell's log: five levels (trace, debug, info, warn, error), one file
//! at the file level and the terminal at the console level. Lines look
//! like the engine's log:
//!
//! ```text
//! YYYY-MM-DD hh:mm:ss,mmm [INFO ] app: BetaSuite <version> starting
//! ```
//!
//! The shell's own records (its modules, and the targets in OWN_TARGETS)
//! follow the chosen levels. Other crates (tauri, wry, tao) are held to
//! warn and above unless the file level is trace, so their chatter never
//! drowns the app's own debug output.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use log::{Level, LevelFilter, Log, Metadata, Record};

use crate::cli::Options;
use crate::paths;

/// A log file this big is moved to `<name>.1` when the app starts.
const ROTATE_BYTES: u64 = 10 * 1024 * 1024;

const CRATE: &str = "betasuite_lib";

/// Targets the shell logs under on purpose: the page, the engine's stderr,
/// and panics.
const OWN_TARGETS: &[&str] = &["page", "engine", "panic"];

struct Logger {
    file: Mutex<Option<File>>,
    file_level: LevelFilter,
    console_level: LevelFilter,
}

fn is_own(target: &str) -> bool {
    target == CRATE || target.starts_with("betasuite_lib::") || OWN_TARGETS.contains(&target)
}

/// The level a record from `target` is held to, given the level chosen for
/// the shell's own records.
fn cap(target: &str, chosen: LevelFilter) -> LevelFilter {
    if is_own(target) || chosen == LevelFilter::Trace {
        chosen
    } else {
        chosen.min(LevelFilter::Warn)
    }
}

fn display_target(target: &str) -> &str {
    match target.strip_prefix("betasuite_lib::") {
        Some(rest) => rest,
        None if target == CRATE => "app",
        None => target,
    }
}

fn level_name(level: Level) -> &'static str {
    match level {
        Level::Trace => "TRACE",
        Level::Debug => "DEBUG",
        Level::Info => "INFO",
        Level::Warn => "WARN",
        Level::Error => "ERROR",
    }
}

fn format_line(record: &Record) -> String {
    format!(
        "{} [{:<5}] {}: {}\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S,%3f"),
        level_name(record.level()),
        display_target(record.target()),
        record.args()
    )
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        let target = metadata.target();
        metadata.level() <= cap(target, self.file_level).max(cap(target, self.console_level))
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format_line(record);
        let target = record.target();
        if record.level() <= cap(target, self.file_level) {
            if let Ok(mut file) = self.file.lock() {
                if let Some(file) = file.as_mut() {
                    // Written and flushed per line, so the file is complete
                    // even when the process aborts right after.
                    let _ = file.write_all(line.as_bytes()).and_then(|()| file.flush());
                }
            }
        }
        if record.level() <= cap(target, self.console_level) {
            let _ = io::stdout().lock().write_all(line.as_bytes());
        }
    }

    fn flush(&self) {
        if let Ok(mut file) = self.file.lock() {
            if let Some(file) = file.as_mut() {
                let _ = file.flush();
            }
        }
        let _ = io::stdout().flush();
    }
}

/// What `init` did about the log file, for the first lines of the log.
pub struct LogFile {
    pub path: Option<PathBuf>,
    pub rotated: bool,
    pub error: Option<String>,
}

fn open(options: &Options) -> (Option<File>, LogFile) {
    let path = match options
        .log_file
        .clone()
        .or_else(|| Some(paths::log_dir()?.join("ui.log")))
    {
        Some(path) => path,
        None => {
            let error = Some("no state folder (HOME or LOCALAPPDATA is not set)".to_owned());
            return (
                None,
                LogFile {
                    path: None,
                    rotated: false,
                    error,
                },
            );
        }
    };
    let mut rotated = false;
    let result = (|| -> io::Result<File> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if fs::metadata(&path)
            .map(|m| m.len() > ROTATE_BYTES)
            .unwrap_or(false)
        {
            let mut old = path.clone().into_os_string();
            old.push(".1");
            fs::rename(&path, old)?;
            rotated = true;
        }
        OpenOptions::new().create(true).append(true).open(&path)
    })();
    match result {
        Ok(file) => (
            Some(file),
            LogFile {
                path: Some(path),
                rotated,
                error: None,
            },
        ),
        Err(error) => (
            None,
            LogFile {
                path: Some(path),
                rotated,
                error: Some(error.to_string()),
            },
        ),
    }
}

/// Installs the logger and a panic hook that logs the panic before the
/// default hook runs. Call once, first thing.
pub fn init(options: &Options) -> LogFile {
    let (file, outcome) = open(options);
    let logger = Logger {
        file: Mutex::new(file),
        file_level: options.log_level,
        console_level: options.console_level,
    };
    if log::set_boxed_logger(Box::new(logger)).is_ok() {
        log::set_max_level(options.log_level.max(options.console_level));
    }
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        log::error!(target: "panic", "{info}");
        log::logger().flush();
        default_hook(info);
    }));
    outcome
}

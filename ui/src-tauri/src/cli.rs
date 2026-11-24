//! Command-line options. Every option has a default, so a launch from a
//! desktop icon needs none.

use std::ffi::OsString;
use std::path::PathBuf;

use log::LevelFilter;

pub const USAGE: &str = "\
usage: betasuite [--log-level LEVEL] [--console-level LEVEL] [--log-file PATH]

  --log-level LEVEL      log file level (default info)
  --console-level LEVEL  terminal level (default info)
  --log-file PATH        log file (default: logs/ui.log in the app's state folder)

LEVEL is trace, debug, info, warn or error.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub log_level: LevelFilter,
    pub console_level: LevelFilter,
    pub log_file: Option<PathBuf>,
    pub help: bool,
    /// Arguments this version does not know. Logged and ignored, so a
    /// launcher that adds its own arguments cannot stop the app starting.
    pub unknown: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            log_level: LevelFilter::Info,
            console_level: LevelFilter::Info,
            log_file: None,
            help: false,
            unknown: Vec::new(),
        }
    }
}

pub fn parse_level(name: &str) -> Option<LevelFilter> {
    match name {
        "trace" => Some(LevelFilter::Trace),
        "debug" => Some(LevelFilter::Debug),
        "info" => Some(LevelFilter::Info),
        "warn" => Some(LevelFilter::Warn),
        "error" => Some(LevelFilter::Error),
        _ => None,
    }
}

/// Parses the arguments after the program name. Both `--flag value` and
/// `--flag=value` are accepted. A known flag with a missing or bad value
/// is an error; an unknown argument is not.
pub fn parse<I>(args: I) -> Result<Options, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut options = Options::default();
    let mut args = args.into_iter().map(|a| a.to_string_lossy().into_owned());
    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") => {
                (flag.to_owned(), Some(value.to_owned()))
            }
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            match inline.clone().or_else(|| args.next()) {
                Some(v) if !v.is_empty() => Ok(v),
                _ => Err(format!("{name} needs a value")),
            }
        };
        match flag.as_str() {
            "--log-level" => options.log_level = level_value(&value("--log-level")?)?,
            "--console-level" => options.console_level = level_value(&value("--console-level")?)?,
            "--log-file" => options.log_file = Some(PathBuf::from(value("--log-file")?)),
            "-h" | "--help" => options.help = true,
            _ => options.unknown.push(arg),
        }
    }
    Ok(options)
}

fn level_value(name: &str) -> Result<LevelFilter, String> {
    parse_level(name).ok_or_else(|| format!("unknown level {name:?}"))
}

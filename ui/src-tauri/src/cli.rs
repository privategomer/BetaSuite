//! Command-line options. Every option has a default, so a launch from a
//! desktop icon needs none.

use std::ffi::OsString;
use std::path::PathBuf;

use log::LevelFilter;

pub const USAGE: &str = "\
usage: betasuite [--log-level LEVEL] [--console-level LEVEL] [--log-file PATH]
                 [--locale TAG] [--theme light|dark] [--open VIDEO]
                 [--measure-startup FILE] [--view player|what-to-hide|models|settings|all-settings]
                 [--search TEXT] [--help-topic ANCHOR]

  --log-level LEVEL      log file level (default info)
  --console-level LEVEL  terminal level (default info)
  --log-file PATH        log file (default: logs/ui.log in the app's state folder)
  --locale TAG           UI language, a BCP 47 tag such as en-US, or qps-ploc
                         for the pseudo-locale (default: the system's)
  --theme light|dark     force a theme (default: follow the system)
  --open VIDEO           play this video at once, muted
  --measure-startup FILE write the time of each startup step to FILE as JSON,
                         then close (with --open, after its first frame)
  --view VIEW            open on this view: player (default), what-to-hide, models, settings or all-settings
  --search TEXT          start All Settings with this search filled in
  --help-topic ANCHOR    open the help panel at this section of the reference

LEVEL is trace, debug, info, warn or error.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub log_level: LevelFilter,
    pub console_level: LevelFilter,
    pub log_file: Option<PathBuf>,
    /// The UI language asked for; the page picks the closest catalog.
    pub locale: Option<String>,
    /// "light" or "dark"; None follows the system.
    pub theme: Option<String>,
    pub open: Option<PathBuf>,
    pub measure_startup: Option<PathBuf>,
    /// The view the page opens on; None is the player.
    pub view: Option<String>,
    /// Text All Settings starts with in its search box.
    pub search: Option<String>,
    /// A help section to open at start.
    pub help_topic: Option<String>,
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
            locale: None,
            theme: None,
            open: None,
            measure_startup: None,
            view: None,
            search: None,
            help_topic: None,
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
            "--locale" => options.locale = Some(value("--locale")?),
            "--theme" => match value("--theme")?.as_str() {
                theme @ ("light" | "dark") => options.theme = Some(theme.to_owned()),
                other => return Err(format!("unknown theme {other:?}; use light or dark")),
            },
            "--open" => options.open = Some(PathBuf::from(value("--open")?)),
            "--measure-startup" => {
                options.measure_startup = Some(PathBuf::from(value("--measure-startup")?))
            }
            "--view" => match value("--view")?.as_str() {
                view @ ("player" | "what-to-hide" | "models" | "settings" | "all-settings") => {
                    options.view = Some(view.to_owned())
                }
                other => {
                    return Err(format!(
                        "unknown view {other:?}; use player, what-to-hide, models, settings or all-settings"
                    ))
                }
            },
            "--search" => options.search = Some(value("--search")?),
            "--help-topic" => options.help_topic = Some(value("--help-topic")?),
            "-h" | "--help" => options.help = true,
            _ => options.unknown.push(arg),
        }
    }
    Ok(options)
}

fn level_value(name: &str) -> Result<LevelFilter, String> {
    parse_level(name).ok_or_else(|| format!("unknown level {name:?}"))
}

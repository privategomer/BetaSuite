//! The BetaSuite desktop shell.
//!
//! The shell owns the window and the native side. It never produces text
//! for the user: what it reports to the page are codes and parameters,
//! and the page words them from the message catalog.

mod cli;
#[cfg(target_os = "linux")]
mod linux;
mod logging;
mod paths;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tauri::{Manager, WebviewWindow};

/// The release version, from the repository's VERSION file.
pub const VERSION: &str = env!("BETASUITE_VERSION");

/// How long the page gets to show the window itself. After this the shell
/// shows it anyway, so a page that fails to start never leaves an
/// invisible process behind.
const SHOW_FALLBACK: Duration = Duration::from_secs(5);

static STARTED: OnceLock<Instant> = OnceLock::new();

fn elapsed_ms() -> u128 {
    STARTED.get().map_or(0, |start| start.elapsed().as_millis())
}

/// Whether the main window has been shown, by the page or the fallback.
struct Shown(AtomicBool);

fn show_window(window: &WebviewWindow, shown: &Shown, why: &str) {
    if shown.0.swap(true, Ordering::SeqCst) {
        return;
    }
    match window.show() {
        Ok(()) => log::info!("window shown {} ms after start ({why})", elapsed_ms()),
        Err(error) => log::error!("could not show the window: {error}"),
    }
}

/// The page's log records, at the level the page gives.
#[tauri::command]
fn ui_log(level: &str, message: &str) {
    match cli::parse_level(level).and_then(|filter| filter.to_level()) {
        Some(level) => log::log!(target: "page", level, "{message}"),
        None => log::warn!(target: "page", "(unknown level {level:?}) {message}"),
    }
}

/// Called by the page once its first render is in the DOM. The window is
/// created hidden: shown before the page has content, WebKitGTK stalls its
/// first draws for about a second.
#[tauri::command]
fn ui_ready(window: WebviewWindow, shown: tauri::State<'_, Shown>, title: Option<String>) {
    if let Some(title) = title {
        if let Err(error) = window.set_title(&title) {
            log::warn!("could not set the window title: {error}");
        }
    }
    show_window(&window, &shown, "page ready");
}

fn setup(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let window = app
        .get_webview_window("main")
        .ok_or("tauri.conf.json defines no main window")?;
    #[cfg(target_os = "linux")]
    linux::allow_local_files(&window)?;

    let handle = app.handle().clone();
    std::thread::spawn(move || {
        std::thread::sleep(SHOW_FALLBACK);
        let shown = handle.state::<Shown>();
        if !shown.0.load(Ordering::SeqCst) {
            log::warn!(
                "the page did not show the window within {} s; showing it anyway",
                SHOW_FALLBACK.as_secs()
            );
            if let Some(window) = handle.get_webview_window("main") {
                show_window(&window, &shown, "fallback");
            }
        }
    });
    Ok(())
}

/// Runs the app and returns its exit code.
pub fn run() -> i32 {
    STARTED.get_or_init(Instant::now);
    let options = match cli::parse(std::env::args_os().skip(1)) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("betasuite: {message}\n\n{}", cli::USAGE);
            return 2;
        }
    };
    if options.help {
        println!("{}", cli::USAGE);
        return 0;
    }

    let log_file = logging::init(&options);
    log::info!(
        "BetaSuite {VERSION} starting on {} {}, pid {}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::process::id()
    );
    match (&log_file.path, &log_file.error) {
        (Some(path), None) => log::info!("log file {}", path.display()),
        (Some(path), Some(error)) => {
            log::warn!("cannot write the log file {}: {error}", path.display())
        }
        (None, Some(error)) => log::warn!("no log file: {error}"),
        (None, None) => {}
    }
    if log_file.rotated {
        log::info!("the previous log was over its size limit and was moved to .1");
    }
    log::debug!(
        "log level {}, console level {}",
        options.log_level,
        options.console_level
    );
    for arg in &options.unknown {
        log::warn!("ignoring unknown argument {arg:?}");
    }

    let app = tauri::Builder::default()
        .manage(Shown(AtomicBool::new(false)))
        .setup(setup)
        .invoke_handler(tauri::generate_handler![ui_log, ui_ready])
        .build(tauri::generate_context!());
    let app = match app {
        Ok(app) => app,
        Err(error) => {
            log::error!("the app failed to start: {error}");
            return 1;
        }
    };
    let code = app.run_return(|_handle, _event| {});
    log::info!("exiting with code {code}");
    log::logger().flush();
    code
}

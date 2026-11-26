//! The BetaSuite desktop shell.
//!
//! The shell owns the window and the native side. It never produces text
//! for the user: what it reports to the page are codes and parameters,
//! and the page words them from the message catalog.

mod cli;
mod engine;
mod files;
#[cfg(target_os = "linux")]
mod linux;
mod logging;
mod look;
mod paths;
mod startup;
mod tray;
mod unsaved;
mod video;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};
use tauri::{Emitter, Manager, RunEvent, WebviewWindow, WindowEvent};

use engine::{Engine, ErrorCode};
use startup::Startup;

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

/// What the page needs before its first render.
#[derive(Clone, Serialize)]
struct StartupInfo {
    version: &'static str,
    /// The `--locale` option, if given.
    locale: Option<String>,
    /// The `--theme` option: "light", "dark", or None to follow the system.
    theme: Option<String>,
    /// The `--view` option: "library", "player", "all-settings", ..., or None for the library.
    view: Option<String>,
    /// The `--search` option: All Settings' starting search text.
    search: Option<String>,
    /// The `--help-topic` option: a help section to open at start.
    help_topic: Option<String>,
    /// The `--measure-scrub` option.
    measure_scrub: bool,
    /// The `--open` video, already allowed, so the page's first render can
    /// start it; or why it could not be opened.
    open: Option<video::PlayableVideo>,
    open_error: Option<ErrorCode>,
}

#[tauri::command]
fn startup_info(info: tauri::State<'_, StartupInfo>) -> StartupInfo {
    info.inner().clone()
}

fn show_window(window: &WebviewWindow, shown: &Shown, why: &str) {
    if shown.0.swap(true, Ordering::SeqCst) {
        return;
    }
    match window.show() {
        Ok(()) => {
            log::info!("window shown {} ms after start ({why})", elapsed_ms());
            mark(window.app_handle(), "window_shown");
        }
        Err(error) => log::error!("could not show the window: {error}"),
    }
}

/// Records a startup step; closes the app once `--measure-startup` has its
/// report.
fn mark(app: &tauri::AppHandle, name: &str) {
    if app.state::<Startup>().mark(name) {
        // The same path as the user closing the window.
        if let Some(window) = app.get_webview_window("main") {
            if let Err(error) = window.close() {
                log::error!("could not close the window: {error}");
            }
        }
    }
}

/// A startup step the page saw: its first script, or the first video frame.
#[tauri::command]
fn ui_mark(app: tauri::AppHandle, name: &str) {
    if matches!(name, "page_script" | "first_frame") {
        mark(&app, name);
    } else {
        log::warn!("ignoring unknown startup step {name:?}");
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

/// The engine's current status, for the page's first read. Changes after
/// that arrive as `engine-status` events.
#[tauri::command]
fn engine_status(engine: tauri::State<'_, Engine>) -> engine::Status {
    engine.status()
}

/// Sends one command to the engine and returns its result data, or an error
/// code: the engine's own, or a `ui.engine.*` one.
#[tauri::command]
async fn engine_request(
    engine: tauri::State<'_, Engine>,
    name: String,
    data: Option<Value>,
) -> Result<Value, ErrorCode> {
    let engine = engine.inner().clone();
    tauri::async_runtime::spawn_blocking(move || engine.request(&name, data.unwrap_or(json!({}))))
        .await
        .unwrap_or_else(|_| Err(ErrorCode::new("ui.engine.write_failed", json!({}))))
}

/// Starts the engine again after it stopped, exited or failed to start.
#[tauri::command]
fn engine_restart(engine: tauri::State<'_, Engine>) {
    engine.restart();
}

/// Set once the app has begun closing, so a second close request does not
/// start a second shutdown.
struct Closing(AtomicBool);

/// Set when the page has asked the user about unapplied changes and closes
/// the window itself, so that close goes through.
struct CloseConfirmed(AtomicBool);

/// The page holds unapplied changes (true) or none (false).
#[tauri::command]
fn set_unsaved(unsaved: tauri::State<'_, unsaved::Unsaved>, held: bool) {
    log::debug!(
        "page {} unapplied changes",
        if held { "holds" } else { "has no" }
    );
    unsaved.set(held);
}

/// Closes the window after the page has asked about unapplied changes.
#[tauri::command]
fn close_window(app: tauri::AppHandle) {
    confirm_close(&app);
    log::info!("page confirmed the close");
    close(&app);
}

/// The next close request goes through without asking the page.
pub(crate) fn confirm_close(app: &tauri::AppHandle) {
    app.state::<CloseConfirmed>()
        .0
        .store(true, Ordering::SeqCst);
}

/// Closing the window lets it close at once and shuts the engine down
/// (shutdown, then a kill after the timeout) on a thread; the app's exit
/// waits for that thread. The window is never hidden and kept: on Windows,
/// exiting with a hidden window left the event loop stuck.
fn close(app: &tauri::AppHandle) {
    if app.state::<Closing>().0.swap(true, Ordering::SeqCst) {
        return;
    }
    log::info!("window closed; shutting the engine down");
    let app = app.clone();
    std::thread::spawn(move || {
        app.state::<Engine>().shutdown();
        log::debug!("asking the event loop to exit");
        app.exit(0);
        // The engine is already down, so a stuck event loop must not keep a
        // windowless process alive: it would hold the webview profile and
        // stall the next start.
        std::thread::sleep(EXIT_WATCHDOG);
        log::warn!(
            "the event loop did not exit within {} s; ending the process with code {EXIT_FORCED}",
            EXIT_WATCHDOG.as_secs()
        );
        log::logger().flush();
        std::process::exit(EXIT_FORCED);
    });
}

/// How long the event loop gets to finish after the window closes.
const EXIT_WATCHDOG: Duration = Duration::from_secs(3);

/// The exit code when the watchdog ended the process: the engine stopped
/// cleanly, but the app did not exit the normal way, so 0 would hide it.
const EXIT_FORCED: i32 = 2;

fn create_engine(app: &tauri::AppHandle, transcript: Option<std::path::PathBuf>) -> Engine {
    let mut config = engine::Config::new(VERSION);
    config.transcript = transcript;
    let status_app = app.clone();
    let event_app = app.clone();
    Engine::new(
        config,
        Arc::new(move |status: &engine::Status| {
            if matches!(status, engine::Status::Ready { .. }) {
                mark(&status_app, "engine_ready");
            }
            if let Err(error) = status_app.emit("engine-status", status) {
                log::warn!("cannot send the engine status to the page: {error}");
            }
        }),
        Arc::new(move |event: &Value| {
            if let Err(error) = event_app.emit("engine-event", event) {
                log::warn!("cannot send an engine event to the page: {error}");
            }
        }),
    )
}

fn setup(
    app: &mut tauri::App,
    transcript: Option<std::path::PathBuf>,
    mut startup: StartupInfo,
    open: Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_engine(app.handle(), transcript);
    engine.start();
    app.manage(engine);

    if let Some(path) = open {
        match video::allow(app.handle(), &path) {
            Ok(playable) => startup.open = Some(playable),
            Err(error) => startup.open_error = Some(error),
        }
    }
    app.manage(startup);

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

    let startup = StartupInfo {
        version: VERSION,
        locale: options.locale.clone(),
        theme: options.theme.clone(),
        view: options.view.clone(),
        search: options.search.clone(),
        help_topic: options.help_topic.clone(),
        measure_scrub: options.measure_scrub,
        open: None,
        open_error: None,
    };
    let open = options.open.clone();
    let startup_timing = Startup::new(options.measure_startup.clone(), options.open.is_some());
    let transcript = log_file
        .path
        .as_ref()
        .and_then(|path| path.parent())
        .map(|dir| dir.join("engine-transcript.jsonl"));
    let app = tauri::Builder::default()
        .manage(Shown(AtomicBool::new(false)))
        .manage(Closing(AtomicBool::new(false)))
        .manage(CloseConfirmed(AtomicBool::new(false)))
        .manage(unsaved::Unsaved::new())
        .manage(startup_timing)
        .manage(tray::Tray::default())
        // First, so a second start exits here: before its webview (which
        // would share this one's profile) or its engine exists.
        .plugin(tauri_plugin_single_instance::init(|app, args, _cwd| {
            log::info!(
                "a second start ({} argument(s)) brought this window back",
                args.len().saturating_sub(1)
            );
            tray::bring_back(app, "second start");
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .setup(move |app| setup(app, transcript, startup, open))
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let app = window.app_handle();
                let held = app.state::<unsaved::Unsaved>().held();
                let confirmed = app.state::<CloseConfirmed>().0.load(Ordering::SeqCst);
                match unsaved::on_close(held, confirmed) {
                    unsaved::OnClose::Close => close(app),
                    unsaved::OnClose::AskPage => {
                        api.prevent_close();
                        log::info!("close held: the page has unapplied changes");
                        if let Err(error) = window.emit("close-requested", ()) {
                            log::warn!("could not ask the page about closing: {error}; closing");
                            close(app);
                        }
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            startup_info,
            ui_log,
            ui_ready,
            engine_status,
            engine_request,
            engine_restart,
            ui_mark,
            video::pick_video,
            look::look_image,
            files::strip_image,
            files::pick_files,
            files::pick_folder,
            files::save_as,
            files::open_video,
            tray::tray_setup,
            tray::tray_status,
            tray::hide_to_tray,
            tray::notify,
            set_unsaved,
            close_window
        ])
        .build(tauri::generate_context!());
    let app = match app {
        Ok(app) => app,
        Err(error) => {
            log::error!("the app failed to start: {error}");
            return 1;
        }
    };
    let code = app.run_return(|app, event| {
        if let RunEvent::ExitRequested { code, api, .. } = &event {
            // The last window closing asks to exit with no code; the
            // shutdown thread asks again with code 0 once the engine is down.
            if code.is_none() && app.state::<Closing>().0.load(Ordering::SeqCst) {
                log::debug!("window gone; waiting for the engine to stop");
                api.prevent_exit();
            } else {
                log::debug!("exit requested (code {code:?})");
            }
        }
        if let RunEvent::Exit = event {
            log::debug!("event loop exiting");
            // Returns at once when the window close already stopped it.
            if let Some(engine) = app.try_state::<Engine>() {
                engine.shutdown();
            }
        }
    });
    log::info!("exiting with code {code}");
    log::logger().flush();
    code
}

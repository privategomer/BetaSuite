//! Running in the background: a tray icon while a job goes on with the
//! window hidden, and a notice when the queue finishes.
//!
//! Every word comes from the page's message catalog: the page sends the
//! menu's labels and the tooltip (`tray_setup`, `tray_status`) and the
//! notice's text (`notify`), and the tray is made only once it has them.
//! Its menu shows the window, pauses the running job (the page is told,
//! `tray-pause`, since it knows which job runs), or quits.
//!
//! Quit shows the window again before closing it: on Windows, exiting with
//! a hidden window once left the event loop stuck (see lib.rs, close).
//! The tray shows while the app is open unless `tray_icon_shown` is off
//! (`tray_shown`); while the window is hidden for a job it shows either
//! way. Its image is the chosen app icon (app_icon.rs).
//!
//! Where the desktop has no tray (a Linux desktop without the
//! AppIndicator library), `tray_setup` answers false, and the page does
//! not offer to keep running in the background.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde::Deserialize;
use serde_json::json;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_notification::NotificationExt;

use crate::engine::ErrorCode;

const SHOW: &str = "show";
const PAUSE: &str = "pause";
const QUIT: &str = "quit";

/// The tray icon, once made, and whether it should show.
pub struct Tray {
    icon: Mutex<Option<TrayIcon>>,
    /// The `tray_icon_shown` setting, as the page last sent it.
    shown: AtomicBool,
    /// The window is hidden while a job runs, so the tray must show.
    background: AtomicBool,
}

impl Default for Tray {
    fn default() -> Self {
        Tray {
            icon: Mutex::new(None),
            shown: AtomicBool::new(true),
            background: AtomicBool::new(false),
        }
    }
}

/// Whether the tray shows: always while in the background, else as set.
pub fn visible(shown: bool, background: bool) -> bool {
    shown || background
}

impl Tray {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<TrayIcon>> {
        self.icon
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn apply_visibility(&self) {
        let show = visible(
            self.shown.load(Ordering::SeqCst),
            self.background.load(Ordering::SeqCst),
        );
        if let Some(icon) = self.lock().as_ref() {
            if let Err(error) = icon.set_visible(show) {
                log::warn!(
                    "cannot {} the tray icon: {error}",
                    if show { "show" } else { "hide" }
                );
            }
        }
    }

    /// Gives the tray a new image (the app icon changed).
    pub fn set_image(&self, image: tauri::image::Image<'static>) {
        if let Some(icon) = self.lock().as_ref() {
            if let Err(error) = icon.set_icon(Some(image)) {
                log::warn!("cannot change the tray icon: {error}");
            }
        }
    }
}

/// The tray's words, from the page's catalog.
#[derive(Clone, Debug, Deserialize)]
pub struct TrayTexts {
    tooltip: String,
    show: String,
    pause: String,
    quit: String,
}

/// What a menu item asks for.
#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    Show,
    Pause,
    Quit,
    Unknown,
}

pub fn action_for(id: &str) -> Action {
    match id {
        SHOW => Action::Show,
        PAUSE => Action::Pause,
        QUIT => Action::Quit,
        _ => Action::Unknown,
    }
}

/// Shows the main window again and gives it focus.
pub fn bring_back(app: &AppHandle, why: &str) {
    let tray = app.state::<Tray>();
    tray.background.store(false, Ordering::SeqCst);
    tray.apply_visibility();
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        if let Err(error) = window.show() {
            log::warn!("cannot show the window ({why}): {error}");
        }
        let _ = window.set_focus();
        log::info!("window brought back ({why})");
    }
}

fn on_menu(app: &AppHandle, id: &str) {
    match action_for(id) {
        Action::Show => bring_back(app, "tray"),
        Action::Pause => {
            log::info!("tray: pause asked");
            if let Err(error) = app.emit("tray-pause", ()) {
                log::warn!("cannot tell the page to pause: {error}");
            }
        }
        Action::Quit => {
            log::info!("tray: quit asked");
            bring_back(app, "quit");
            crate::confirm_close(app);
            if let Some(window) = app.get_webview_window("main") {
                if let Err(error) = window.close() {
                    log::warn!("cannot close the window: {error}");
                }
            }
        }
        Action::Unknown => log::debug!("tray: unknown menu item {id}"),
    }
}

fn menu(app: &AppHandle, texts: &TrayTexts) -> tauri::Result<Menu<tauri::Wry>> {
    let show = MenuItem::with_id(app, SHOW, &texts.show, true, None::<&str>)?;
    let pause = MenuItem::with_id(app, PAUSE, &texts.pause, true, None::<&str>)?;
    let quit = MenuItem::with_id(app, QUIT, &texts.quit, true, None::<&str>)?;
    Menu::with_items(app, &[&show, &pause, &quit])
}

fn build(app: &AppHandle, texts: &TrayTexts) -> tauri::Result<TrayIcon> {
    let mut builder = TrayIconBuilder::with_id("main")
        .menu(&menu(app, texts)?)
        .show_menu_on_left_click(false)
        .tooltip(&texts.tooltip)
        .on_menu_event(|app, event| on_menu(app, event.id().as_ref()))
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                bring_back(tray.app_handle(), "tray click");
            }
        });
    let scale = app
        .get_webview_window("main")
        .and_then(|window| window.scale_factor().ok())
        .unwrap_or(1.0);
    if let Some(image) = app.state::<crate::app_icon::AppIcon>().tray_image(scale) {
        builder = builder.icon(image);
    } else if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder.build(app)
}

/// Makes the tray icon with the page's words, or gives it new ones.
/// Answers whether there is a tray to hide the window to.
#[tauri::command]
pub fn tray_setup(app: AppHandle, tray: tauri::State<'_, Tray>, texts: TrayTexts) -> bool {
    let mut icon = tray
        .icon
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(existing) = icon.as_ref() {
        match menu(&app, &texts) {
            Ok(menu) => {
                let _ = existing.set_menu(Some(menu));
                let _ = existing.set_tooltip(Some(&texts.tooltip));
            }
            Err(error) => log::warn!("cannot update the tray menu: {error}"),
        }
        return true;
    }
    match build(&app, &texts) {
        Ok(made) => {
            log::info!("tray icon ready");
            let show = visible(
                tray.shown.load(Ordering::SeqCst),
                tray.background.load(Ordering::SeqCst),
            );
            if !show {
                let _ = made.set_visible(false);
            }
            *icon = Some(made);
            true
        }
        Err(error) => {
            log::warn!("no tray icon on this desktop: {error}");
            false
        }
    }
}

/// The `tray_icon_shown` setting: show the tray while the app is open, or
/// only while a job runs in the background.
#[tauri::command]
pub fn tray_shown(tray: tauri::State<'_, Tray>, shown: bool) {
    tray.shown.store(shown, Ordering::SeqCst);
    tray.apply_visibility();
    log::info!(
        "tray icon {}",
        if shown {
            "shown"
        } else {
            "shown only in the background"
        }
    );
}

/// The tray's tooltip: the queue's progress, worded by the page.
#[tauri::command]
pub fn tray_status(tray: tauri::State<'_, Tray>, tooltip: String) {
    let icon = tray
        .icon
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(icon) = icon.as_ref() {
        let _ = icon.set_tooltip(Some(&tooltip));
    }
}

/// Keep running in the background: hides the window, which the tray, a
/// second start of the app, or Quit brings back.
#[tauri::command]
pub fn hide_to_tray(app: AppHandle, tray: tauri::State<'_, Tray>) -> Result<(), ErrorCode> {
    let has_tray = tray
        .icon
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_some();
    if !has_tray {
        return Err(ErrorCode::new("ui.tray.unavailable", json!({})));
    }
    let Some(window) = app.get_webview_window("main") else {
        return Err(ErrorCode::new("ui.tray.unavailable", json!({})));
    };
    tray.background.store(true, Ordering::SeqCst);
    tray.apply_visibility();
    window.hide().map_err(|error| {
        log::warn!("cannot hide the window: {error}");
        tray.background.store(false, Ordering::SeqCst);
        tray.apply_visibility();
        ErrorCode::new("ui.tray.unavailable", json!({}))
    })?;
    log::info!("window hidden; running in the background");
    Ok(())
}

/// A desktop notice (the queue finished while the window was hidden).
#[tauri::command]
pub fn notify(app: AppHandle, title: String, body: String) {
    match app.notification().builder().title(title).body(body).show() {
        Ok(()) => log::info!("notice shown"),
        Err(error) => log::warn!("cannot show a notice: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tray_shows_in_the_background_whatever_the_setting() {
        assert!(visible(true, false));
        assert!(visible(true, true));
        assert!(visible(false, true));
        assert!(!visible(false, false));
    }

    #[test]
    fn menu_ids_map_to_actions() {
        assert_eq!(action_for(SHOW), Action::Show);
        assert_eq!(action_for(PAUSE), Action::Pause);
        assert_eq!(action_for(QUIT), Action::Quit);
        assert_eq!(action_for("other"), Action::Unknown);
    }
}

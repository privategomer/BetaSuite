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
//! Where the desktop has no tray (a Linux desktop without the
//! AppIndicator library), `tray_setup` answers false, and the page does
//! not offer to keep running in the background.

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

/// The tray icon, once made.
#[derive(Default)]
pub struct Tray {
    icon: Mutex<Option<TrayIcon>>,
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
    if let Some(icon) = app.default_window_icon() {
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
            *icon = Some(made);
            true
        }
        Err(error) => {
            log::warn!("no tray icon on this desktop: {error}");
            false
        }
    }
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
    window.hide().map_err(|error| {
        log::warn!("cannot hide the window: {error}");
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
    fn menu_ids_map_to_actions() {
        assert_eq!(action_for(SHOW), Action::Show);
        assert_eq!(action_for(PAUSE), Action::Pause);
        assert_eq!(action_for(QUIT), Action::Quit);
        assert_eq!(action_for("other"), Action::Unknown);
    }
}

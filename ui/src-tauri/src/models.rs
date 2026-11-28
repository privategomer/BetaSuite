//! The native file dialog for "I already have the file" on the Models page.
//! The page gets no dialog permission, so the shell opens it and hands back
//! the chosen path; the engine reads, checks and copies the file
//! (models.import). The page never reads it itself.

use std::sync::mpsc;

use serde_json::json;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;

use crate::engine::ErrorCode;

/// Model file names the engine can install.
const EXTENSIONS: [&str; 1] = ["onnx"];

/// Opens the native file dialog for one model file. Ok(None) when cancelled.
/// The title and filter name come from the page's catalog.
#[tauri::command]
pub async fn pick_model(
    app: AppHandle,
    title: String,
    filter_name: String,
) -> Result<Option<String>, ErrorCode> {
    let (sender, receiver) = mpsc::channel();
    let mut dialog = app
        .dialog()
        .file()
        .set_title(title)
        .add_filter(filter_name, &EXTENSIONS);
    if let Some(window) = app.get_webview_window("main") {
        dialog = dialog.set_parent(&window);
    }
    dialog.pick_file(move |picked| {
        let _ = sender.send(picked);
    });
    let picked = tauri::async_runtime::spawn_blocking(move || receiver.recv().ok().flatten())
        .await
        .map_err(|_| ErrorCode::new("ui.model.dialog_failed", json!({})))?;
    let Some(picked) = picked else {
        log::debug!("model dialog cancelled");
        return Ok(None);
    };
    let path = picked
        .into_path()
        .map_err(|_| ErrorCode::new("ui.model.dialog_failed", json!({})))?;
    log::info!("model file chosen for import");
    Ok(Some(path.to_string_lossy().into_owned()))
}

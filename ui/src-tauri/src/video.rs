//! Local video for the page. The shell, not the page, opens the file
//! dialog, so the page holds no file-system permission; each chosen file is
//! then allowed on its own, never its folder.
//!
//! Linux plays video from `file://` URLs (see linux.rs), so the page gets
//! one. Elsewhere the file joins the asset protocol's scope and the page
//! addresses it through that protocol.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use serde::Serialize;
use serde_json::json;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;

use crate::engine::ErrorCode;

#[derive(Clone, Debug, Serialize)]
pub struct PlayableVideo {
    path: String,
    name: String,
    /// Set where video plays from file:// (Linux).
    file_url: Option<String>,
}

/// `canonicalize` on Windows returns `\\?\C:\...`, which the asset protocol
/// and WebView2 do not take.
fn plain(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) => PathBuf::from(rest),
        None => path,
    }
}

/// Allows one file for playback and says how the page should address it.
pub fn allow(app: &AppHandle, path: &Path) -> Result<PlayableVideo, ErrorCode> {
    let path = match path.canonicalize() {
        Ok(path) if path.is_file() => plain(path),
        _ => {
            log::warn!("cannot open {}: not a readable file", path.display());
            return Err(ErrorCode::new("ui.video.not_found", json!({})));
        }
    };
    if let Err(error) = app.asset_protocol_scope().allow_file(&path) {
        log::warn!("cannot allow {} for playback: {error}", path.display());
    }
    let file_url = if cfg!(target_os = "linux") {
        tauri::Url::from_file_path(&path).ok().map(String::from)
    } else {
        None
    };
    log::info!("video allowed: {}", path.display());
    Ok(PlayableVideo {
        name: path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        path: path.to_string_lossy().into_owned(),
        file_url,
    })
}

/// Opens the native file dialog for one video. Ok(None) when cancelled.
/// The title and filter name come from the page's catalog.
#[tauri::command]
pub async fn pick_video(
    app: AppHandle,
    title: String,
    filter_name: String,
    extensions: Vec<String>,
) -> Result<Option<PlayableVideo>, ErrorCode> {
    let (sender, receiver) = mpsc::channel();
    let extensions: Vec<&str> = extensions.iter().map(String::as_str).collect();
    let mut dialog = app
        .dialog()
        .file()
        .set_title(title)
        .add_filter(filter_name, &extensions);
    if let Some(window) = app.get_webview_window("main") {
        dialog = dialog.set_parent(&window);
    }
    dialog.pick_file(move |picked| {
        let _ = sender.send(picked);
    });
    let picked = tauri::async_runtime::spawn_blocking(move || receiver.recv().ok().flatten())
        .await
        .map_err(|_| ErrorCode::new("ui.video.dialog_failed", json!({})))?;
    let Some(picked) = picked else {
        log::debug!("video dialog cancelled");
        return Ok(None);
    };
    let path = picked
        .into_path()
        .map_err(|_| ErrorCode::new("ui.video.not_found", json!({})))?;
    allow(&app, &path).map(Some)
}

//! Files for the library and the player: choosing files and folders,
//! saving a result under a name the person picks, opening a library video
//! for playback, and reading a scrub strip the engine made.
//!
//! The shell opens every dialog, so the page holds no file-system
//! permission of its own. The strip reader hands over only what the
//! engine could have written: a canonical path in a `strips` folder whose
//! name is 32 lowercase hex digits and `.jpg` (betautils_catalog_strip).

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use serde_json::json;
use tauri::ipc::Response;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::{DialogExt, FileDialogBuilder};

use crate::engine::ErrorCode;
use crate::video::{self, PlayableVideo};

/// The engine's cache folder for strips (betautils_catalog_strip.FOLDER_NAME).
const STRIP_FOLDER: &str = "strips";
/// A strip of 600 small tiles is a few hundred KiB.
const STRIP_MAX_BYTES: u64 = 16 * 1024 * 1024;

fn is_strip_name(name: &str) -> bool {
    match name.strip_suffix(".jpg") {
        Some(stem) => {
            stem.len() == 32
                && stem
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }
        None => false,
    }
}

/// The canonical path, when it names a strip image the engine wrote.
pub fn checked_strip(path: &Path) -> Option<PathBuf> {
    let path = path.canonicalize().ok()?;
    let name = path.file_name()?.to_str()?;
    let folder = path.parent()?.file_name()?.to_str()?;
    if folder != STRIP_FOLDER || !is_strip_name(name) {
        return None;
    }
    let meta = path.metadata().ok()?;
    if !meta.is_file() || meta.len() > STRIP_MAX_BYTES {
        return None;
    }
    Some(path)
}

/// The bytes of one scrub strip, for the page to show from a `data:` URL.
#[tauri::command]
pub fn strip_image(path: String) -> Result<Response, ErrorCode> {
    let Some(path) = checked_strip(Path::new(&path)) else {
        log::warn!("refused a strip path: {path}");
        return Err(ErrorCode::new("ui.strip.unreadable", json!({})));
    };
    match std::fs::read(&path) {
        Ok(bytes) => {
            log::trace!("strip {} ({} bytes)", path.display(), bytes.len());
            Ok(Response::new(bytes))
        }
        Err(error) => {
            log::warn!("cannot read strip {}: {error}", path.display());
            Err(ErrorCode::new("ui.strip.unreadable", json!({})))
        }
    }
}

fn parented(
    app: &AppHandle,
    dialog: FileDialogBuilder<tauri::Wry>,
) -> FileDialogBuilder<tauri::Wry> {
    match app.get_webview_window("main") {
        Some(window) => dialog.set_parent(&window),
        None => dialog,
    }
}

/// Waits for a dialog's answer off the async runtime's threads.
async fn answer<T: Send + 'static>(receiver: mpsc::Receiver<T>) -> Result<T, ErrorCode> {
    tauri::async_runtime::spawn_blocking(move || receiver.recv())
        .await
        .ok()
        .and_then(Result::ok)
        .ok_or_else(|| ErrorCode::new("ui.video.dialog_failed", json!({})))
}

fn strings(paths: Vec<tauri_plugin_dialog::FilePath>) -> Vec<String> {
    paths
        .into_iter()
        .filter_map(|path| path.into_path().ok())
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

/// The native dialog for adding videos to the library: one or more files.
/// An empty list when cancelled.
#[tauri::command]
pub async fn pick_files(
    app: AppHandle,
    title: String,
    filter_name: String,
    extensions: Vec<String>,
) -> Result<Vec<String>, ErrorCode> {
    let (sender, receiver) = mpsc::channel();
    let extensions: Vec<&str> = extensions.iter().map(String::as_str).collect();
    let dialog = parented(
        &app,
        app.dialog()
            .file()
            .set_title(title)
            .add_filter(filter_name, &extensions),
    );
    dialog.pick_files(move |picked| {
        let _ = sender.send(picked.unwrap_or_default());
    });
    let picked = strings(answer(receiver).await?);
    log::info!("{} file(s) chosen to add", picked.len());
    Ok(picked)
}

/// The native dialog for a folder to watch. None when cancelled.
#[tauri::command]
pub async fn pick_folder(app: AppHandle, title: String) -> Result<Option<String>, ErrorCode> {
    let (sender, receiver) = mpsc::channel();
    let dialog = parented(&app, app.dialog().file().set_title(title));
    dialog.pick_folder(move |picked| {
        let _ = sender.send(picked);
    });
    let picked = answer(receiver)
        .await?
        .and_then(|path| path.into_path().ok())
        .map(|path| path.to_string_lossy().into_owned());
    log::info!("folder chosen: {}", picked.is_some());
    Ok(picked)
}

/// Copies source to target through a temporary file beside the target, so
/// an interrupted copy never leaves a partial file under the chosen name.
pub fn copy_into_place(source: &Path, target: &Path) -> std::io::Result<u64> {
    let folder = target.parent().unwrap_or_else(|| Path::new("."));
    let name = target
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temporary = folder.join(format!(".{name}.{}.part", std::process::id()));
    let copied = std::fs::copy(source, &temporary).and_then(|bytes| {
        std::fs::rename(&temporary, target)?;
        Ok(bytes)
    });
    if copied.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    copied
}

/// Save as: the native save dialog, then a copy of the result under the
/// chosen name. The result's own keyed name stays where it is, as the
/// cache identity. None when cancelled.
#[tauri::command]
pub async fn save_as(
    app: AppHandle,
    source: String,
    title: String,
    default_name: String,
) -> Result<Option<String>, ErrorCode> {
    let source = PathBuf::from(source);
    if !source.is_file() {
        log::warn!("save as: {} is not a file", source.display());
        return Err(ErrorCode::new("ui.video.not_found", json!({})));
    }
    let (sender, receiver) = mpsc::channel();
    let dialog = parented(
        &app,
        app.dialog()
            .file()
            .set_title(title)
            .set_file_name(default_name),
    );
    dialog.save_file(move |picked| {
        let _ = sender.send(picked);
    });
    let Some(target) = answer(receiver)
        .await?
        .and_then(|path| path.into_path().ok())
    else {
        log::debug!("save as cancelled");
        return Ok(None);
    };
    let to = target.clone();
    let copied = tauri::async_runtime::spawn_blocking(move || copy_into_place(&source, &to)).await;
    match copied {
        Ok(Ok(bytes)) => {
            log::info!("saved a result as {} ({bytes} bytes)", target.display());
            Ok(Some(target.to_string_lossy().into_owned()))
        }
        Ok(Err(error)) => {
            log::warn!("save as {} failed: {error}", target.display());
            Err(ErrorCode::new(
                "ui.files.save_failed",
                json!({ "os_error": error.to_string() }),
            ))
        }
        Err(_) => Err(ErrorCode::new(
            "ui.files.save_failed",
            json!({ "os_error": "" }),
        )),
    }
}

/// Opens a library video (a result or a source the engine listed) for the
/// player: the one file is allowed, as a chosen one is.
#[tauri::command]
pub fn open_video(app: AppHandle, path: String) -> Result<PlayableVideo, ErrorCode> {
    video::allow(&app, Path::new(&path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const NAME: &str = "0123456789abcdef0123456789abcdef.jpg";

    fn folder(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("betasuite-files-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join(STRIP_FOLDER)).unwrap();
        fs::create_dir_all(dir.join("elsewhere")).unwrap();
        dir
    }

    #[test]
    fn strip_names() {
        assert!(is_strip_name(NAME));
        assert!(!is_strip_name(&NAME.replace(".jpg", ".json")));
        assert!(!is_strip_name(&NAME.to_uppercase().replace(".JPG", ".jpg")));
        assert!(!is_strip_name(&NAME[1..]));
        assert!(!is_strip_name("../x.jpg"));
    }

    #[test]
    fn accepts_only_strips_in_their_folder() {
        let dir = folder("strips");
        let good = dir.join(STRIP_FOLDER).join(NAME);
        fs::write(&good, b"jpg").unwrap();
        assert!(checked_strip(&good).is_some());
        let elsewhere = dir.join("elsewhere").join(NAME);
        fs::write(&elsewhere, b"jpg").unwrap();
        assert!(checked_strip(&elsewhere).is_none());
        let climbing = dir
            .join(STRIP_FOLDER)
            .join("..")
            .join("elsewhere")
            .join(NAME);
        assert!(checked_strip(&climbing).is_none());
        let missing = dir.join(STRIP_FOLDER).join(NAME.replace('0', "1"));
        assert!(checked_strip(&missing).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn copy_into_place_leaves_the_whole_file_or_nothing() {
        let dir = folder("copy");
        let source = dir.join("elsewhere").join("result.mp4");
        fs::write(&source, b"video bytes").unwrap();
        let target = dir.join("saved.mp4");
        assert_eq!(copy_into_place(&source, &target).unwrap(), 11);
        assert_eq!(fs::read(&target).unwrap(), b"video bytes");
        assert!(source.exists(), "the result stays where it is");

        let nowhere = dir.join("no-such-folder").join("saved.mp4");
        assert!(copy_into_place(&source, &nowhere).is_err());
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".part"))
            .collect();
        assert!(leftovers.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}

//! The app's own icon on its window and tray, from the folder the
//! `app_icon` setting names under the engine folder's `assets/icons/`
//! (the folders `betautils_icons.py` lists as the setting's choices).
//!
//! The page reads the setting and calls `set_app_icon` at start and after
//! every change, so a new icon shows without a restart. An id is only ever
//! a folder name of lowercase letters, digits and underscores that holds
//! an `icon.json`, so the commands cannot be pointed at any other file.
//! Each size is a pre-drawn PNG (sizes up to 32 px are drawn on the pixel
//! grid), so the shell picks the size the platform shows instead of
//! scaling one image.
//!
//! What follows the setting: the tray everywhere; the window's own icon
//! on X11 and in the Windows title bar. On Wayland the compositor takes
//! the window and taskbar icon from the app's desktop entry, and the
//! Windows taskbar keeps the executable's icon; both are the default icon
//! (`src-tauri/icons`, drawn from `censored_play`).

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::json;
use tauri::image::Image;
use tauri::ipc::Response;
use tauri::{AppHandle, Manager};

use crate::engine::ErrorCode;

/// The sizes every icon folder holds (tools/art/icons.py SIZES).
pub const SIZES: [u32; 9] = [16, 22, 24, 32, 48, 64, 128, 256, 512];
/// The tray's size at a scale of 1: Windows draws 16 px tray icons; KDE
/// Plasma and GNOME's AppIndicator panels draw 22 px.
#[cfg(windows)]
const TRAY_BASE: f64 = 16.0;
#[cfg(not(windows))]
const TRAY_BASE: f64 = 22.0;
/// The window icon's size at a scale of 1: the Windows title bar's small
/// icon is 16 px; X11 window managers scale one image for every place.
#[cfg(windows)]
const WINDOW_BASE: f64 = 16.0;
#[cfg(not(windows))]
const WINDOW_BASE: f64 = 64.0;

/// The icon in use and where the icon folders are.
pub struct AppIcon {
    root: Option<PathBuf>,
    current: Mutex<Option<String>>,
}

impl AppIcon {
    /// `engine_dir` is the folder holding betaengine.py, when found.
    pub fn new(engine_dir: Option<&Path>) -> Self {
        AppIcon {
            root: engine_dir.map(|dir| dir.join("assets").join("icons")),
            current: Mutex::new(None),
        }
    }

    fn current(&self) -> Option<String> {
        self.current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// The chosen icon for the tray at this scale, once the page has
    /// named one; until then the tray uses the built-in default.
    pub fn tray_image(&self, scale: f64) -> Option<Image<'static>> {
        let root = self.root.as_deref()?;
        let path = png_path(root, &self.current()?, nearest_size(TRAY_BASE * scale))?;
        load(&path).ok()
    }
}

/// An icon id: a folder name, never a path.
pub fn is_icon_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

/// The smallest pre-drawn size at least `want` pixels, else the largest.
pub fn nearest_size(want: f64) -> u32 {
    SIZES
        .iter()
        .copied()
        .find(|&size| f64::from(size) >= want.round())
        .unwrap_or(SIZES[SIZES.len() - 1])
}

/// One icon's PNG at one size, when the id is an icon folder holding it.
pub fn png_path(root: &Path, id: &str, size: u32) -> Option<PathBuf> {
    if !is_icon_id(id) || !SIZES.contains(&size) {
        return None;
    }
    let folder = root.join(id);
    if !folder.join("icon.json").is_file() {
        return None;
    }
    let path = folder.join(format!("icon-{size}.png"));
    path.is_file().then_some(path)
}

/// A PNG as the RGBA image the window and tray take.
pub fn load(path: &Path) -> Result<Image<'static>, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|error| error.to_string())?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| "image too large".to_string())?;
    let mut buf = vec![0; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|error| error.to_string())?;
    buf.truncate(info.buffer_size());
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => buf
            .chunks_exact(3)
            .flat_map(|px| [px[0], px[1], px[2], 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf
            .chunks_exact(2)
            .flat_map(|px| [px[0], px[0], px[0], px[1]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        other => return Err(format!("unsupported PNG color type {other:?}")),
    };
    Ok(Image::new_owned(rgba, info.width, info.height))
}

fn unknown(id: &str) -> ErrorCode {
    ErrorCode::new("ui.icon.unknown", json!({ "id": id }))
}

/// Uses the icon `id` for the window and the tray from now on.
#[tauri::command]
pub fn set_app_icon(
    app: AppHandle,
    icons: tauri::State<'_, AppIcon>,
    tray: tauri::State<'_, crate::tray::Tray>,
    id: String,
) -> Result<(), ErrorCode> {
    let Some(root) = icons.root.as_deref() else {
        log::warn!("no engine folder, so no icon folders; keeping the built-in icon");
        return Err(unknown(&id));
    };
    if png_path(root, &id, SIZES[0]).is_none() {
        log::warn!("refused app icon {id:?}: not an icon folder");
        return Err(unknown(&id));
    }
    *icons
        .current
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(id.clone());
    if let Some(window) = app.get_webview_window("main") {
        let scale = window.scale_factor().unwrap_or(1.0);
        match png_path(root, &id, nearest_size(WINDOW_BASE * scale)).map(|path| load(&path)) {
            Some(Ok(image)) => {
                if let Err(error) = window.set_icon(image) {
                    log::warn!("cannot set the window icon: {error}");
                }
            }
            Some(Err(error)) => log::warn!("cannot read the window icon of {id}: {error}"),
            None => log::warn!("icon {id} has no window-sized PNG"),
        }
        if let Some(image) = icons.tray_image(scale) {
            tray.set_image(image);
        }
    }
    log::info!("app icon: {id}");
    Ok(())
}

/// One icon's PNG bytes, for the page's picker and its tray preview.
#[tauri::command]
pub fn app_icon_image(
    icons: tauri::State<'_, AppIcon>,
    id: String,
    size: u32,
) -> Result<Response, ErrorCode> {
    let path = icons
        .root
        .as_deref()
        .and_then(|root| png_path(root, &id, size))
        .ok_or_else(|| unknown(&id))?;
    std::fs::read(&path).map(Response::new).map_err(|error| {
        log::warn!("cannot read {}: {error}", path.display());
        unknown(&id)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_icons() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/icons")
    }

    #[test]
    fn ids_are_folder_names_only() {
        assert!(is_icon_id("censored_play"));
        assert!(is_icon_id("icon2"));
        for bad in [
            "",
            "../art",
            "a/b",
            "a\\b",
            "Censored",
            "a.b",
            "a b",
            &"x".repeat(65),
        ] {
            assert!(!is_icon_id(bad), "{bad:?}");
        }
    }

    #[test]
    fn nearest_size_rounds_up_to_a_drawn_size() {
        assert_eq!(nearest_size(16.0), 16);
        assert_eq!(nearest_size(22.0), 22);
        assert_eq!(nearest_size(16.0 * 1.25), 22);
        assert_eq!(nearest_size(16.0 * 1.5), 24);
        assert_eq!(nearest_size(22.0 * 1.5), 48);
        assert_eq!(nearest_size(22.0 * 2.0), 48);
        assert_eq!(nearest_size(16.0 * 2.0), 32);
        assert_eq!(nearest_size(2000.0), 512);
        assert_eq!(nearest_size(1.0), 16);
    }

    #[test]
    fn png_path_needs_an_icon_folder_and_a_drawn_size() {
        let root = repo_icons();
        assert!(png_path(&root, "censored_play", 16).is_some());
        assert!(png_path(&root, "censored_play", 17).is_none());
        assert!(png_path(&root, "no_such_icon", 16).is_none());
        assert!(png_path(&root, "..", 16).is_none());
        assert!(png_path(&root.join("censored_play"), "..", 16).is_none());
    }

    #[test]
    fn every_shipped_icon_loads_at_every_size() {
        let root = repo_icons();
        for id in [
            "censored_play",
            "bars_shield",
            "focus_blur",
            "pixel_play",
            "pixel_b",
        ] {
            for size in SIZES {
                let path = png_path(&root, id, size).unwrap_or_else(|| panic!("{id} {size}"));
                let image = load(&path).unwrap_or_else(|error| panic!("{id} {size}: {error}"));
                assert_eq!((image.width(), image.height()), (size, size));
                assert_eq!(image.rgba().len(), (size * size * 4) as usize);
            }
        }
    }

    #[test]
    fn tray_image_waits_for_a_choice() {
        let icons = AppIcon::new(Some(&repo_icons().join("../..")));
        assert!(icons.tray_image(1.0).is_none());
        *icons.current.lock().unwrap() = Some("pixel_b".into());
        let image = icons.tray_image(1.0).expect("tray image");
        assert_eq!(image.width(), nearest_size(TRAY_BASE));
    }
}

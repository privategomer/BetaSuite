//! Look previews for the page. The engine's `preview.look` writes each one
//! as a PNG in its cache folder and answers with the path; the page asks
//! the shell for the bytes and shows them from a `blob:` URL. The shell
//! hands over only a file the engine could have written there: a canonical
//! path whose folder is `look_previews` and whose name is 64 lowercase hex
//! digits and `.png`, so the command cannot be used to read anything else.

use std::path::{Path, PathBuf};

use serde_json::json;
use tauri::ipc::Response;

use crate::engine::ErrorCode;

/// The engine's cache folder for previews (betaconst.look_preview_dir).
const FOLDER: &str = "look_previews";
/// Previews are a few hundred KiB; anything far larger is not one.
const MAX_BYTES: u64 = 16 * 1024 * 1024;

fn is_preview_name(name: &str) -> bool {
    match name.strip_suffix(".png") {
        Some(stem) => {
            stem.len() == 64
                && stem
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }
        None => false,
    }
}

/// The canonical path, when it names a look preview the engine wrote.
pub fn checked(path: &Path) -> Option<PathBuf> {
    let path = path.canonicalize().ok()?;
    let name = path.file_name()?.to_str()?;
    let folder = path.parent()?.file_name()?.to_str()?;
    if folder != FOLDER || !is_preview_name(name) {
        return None;
    }
    let meta = path.metadata().ok()?;
    if !meta.is_file() || meta.len() > MAX_BYTES {
        return None;
    }
    Some(path)
}

/// The bytes of one look preview PNG, for the page to show.
#[tauri::command]
pub fn look_image(path: String) -> Result<Response, ErrorCode> {
    let Some(path) = checked(Path::new(&path)) else {
        log::warn!("refused a look preview path: {path}");
        return Err(ErrorCode::new("ui.look.unreadable", json!({})));
    };
    match std::fs::read(&path) {
        Ok(bytes) => {
            log::trace!("look preview {} ({} bytes)", path.display(), bytes.len());
            Ok(Response::new(bytes))
        }
        Err(error) => {
            log::warn!("cannot read look preview {}: {error}", path.display());
            Err(ErrorCode::new("ui.look.unreadable", json!({})))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const NAME: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.png";

    fn folder(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("betasuite-look-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join(FOLDER)).unwrap();
        fs::create_dir_all(dir.join("elsewhere")).unwrap();
        dir
    }

    #[test]
    fn names() {
        assert!(is_preview_name(NAME));
        assert!(!is_preview_name(&NAME.replace(".png", ".jpg")));
        assert!(!is_preview_name(
            &NAME.to_uppercase().replace(".PNG", ".png")
        ));
        assert!(!is_preview_name(&NAME[1..]));
        assert!(!is_preview_name("../x.png"));
    }

    #[test]
    fn accepts_only_previews_in_their_folder() {
        let dir = folder("folder");
        let good = dir.join(FOLDER).join(NAME);
        fs::write(&good, b"png").unwrap();
        assert!(checked(&good).is_some());

        let elsewhere = dir.join("elsewhere").join(NAME);
        fs::write(&elsewhere, b"png").unwrap();
        assert!(checked(&elsewhere).is_none());

        let other = dir.join(FOLDER).join("notes.png");
        fs::write(&other, b"png").unwrap();
        assert!(checked(&other).is_none());

        let missing = dir.join(FOLDER).join(NAME.replace('0', "1"));
        assert!(checked(&missing).is_none());

        // a path that climbs out after naming the folder
        let climbing = dir.join(FOLDER).join("..").join("elsewhere").join(NAME);
        assert!(checked(&climbing).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_link_is_judged_by_what_it_points_at() {
        let dir = folder("link");
        let secret = dir.join("elsewhere").join("secret.txt");
        fs::write(&secret, b"secret").unwrap();
        let link = dir.join(FOLDER).join(NAME);
        std::os::unix::fs::symlink(&secret, &link).unwrap();
        assert!(checked(&link).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_a_file_too_large_to_be_a_preview() {
        let dir = folder("large");
        let path = dir.join(FOLDER).join(NAME);
        let file = fs::File::create(&path).unwrap();
        file.set_len(MAX_BYTES).unwrap();
        assert!(checked(&path).is_some());
        file.set_len(MAX_BYTES + 1).unwrap();
        assert!(checked(&path).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_a_folder() {
        let dir = folder("dir");
        let as_dir = dir.join(FOLDER).join(NAME);
        fs::create_dir_all(&as_dir).unwrap();
        assert!(checked(&as_dir).is_none());
        let _ = fs::remove_dir_all(&dir);
    }
}

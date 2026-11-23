//! WebKitGTK setup.
//!
//! WebKitGTK's media player cannot read Tauri's asset protocol: it rejects
//! custom schemes, and allowing one only moves the failure into GStreamer.
//! So on Linux a video plays from a `file://` URL, and the app's own page
//! scheme is registered as local, which lets that page load local files as
//! subresources. Windows keeps the asset protocol, which WebView2 handles.

use tauri::WebviewWindow;

pub fn allow_local_files(window: &WebviewWindow) -> tauri::Result<()> {
    window.with_webview(|webview| {
        use webkit2gtk::{SecurityManagerExt, WebContextExt, WebViewExt};
        let Some(context) = webview.inner().context() else {
            log::error!("WebKitGTK gave no web context; local video will not play");
            return;
        };
        let Some(manager) = context.security_manager() else {
            log::error!("WebKitGTK gave no security manager; local video will not play");
            return;
        };
        manager.register_uri_scheme_as_local("tauri");
        log::debug!("tauri:// registered as a local scheme");
    })
}

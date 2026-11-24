//! Startup timing for `--measure-startup FILE`: the wall-clock time of each
//! step, written as JSON once startup is complete, so a benchmark that
//! notes when it spawned the app can compute spawn-to-step times.
//!
//! Steps: `main` (the shell's first line), `page_script` (the page's first
//! script), `window_shown`, `engine_ready` (the hello result) and, with
//! `--open`, `first_frame` (the video's first presented frame). Startup is complete at the
//! first frame with `--open`, else once the window is shown and the engine
//! is ready. Every step is logged at info either way.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

pub struct Startup {
    report: Option<PathBuf>,
    wait_for_frame: bool,
    marks: Mutex<BTreeMap<String, f64>>,
}

fn wall_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |since| since.as_secs_f64() * 1000.0)
}

impl Startup {
    /// Records `main` now.
    pub fn new(report: Option<PathBuf>, wait_for_frame: bool) -> Self {
        let startup = Startup {
            report,
            wait_for_frame,
            marks: Mutex::new(BTreeMap::new()),
        };
        startup.mark("main");
        startup
    }

    /// Records a step the first time it happens. Returns true when this
    /// step completed startup and the report was written, so the caller
    /// can close the app.
    pub fn mark(&self, name: &str) -> bool {
        let Ok(mut marks) = self.marks.lock() else {
            return false;
        };
        if marks.contains_key(name) {
            return false;
        }
        let now = wall_ms();
        let since_main = marks.get("main").map_or(0.0, |main| now - main);
        marks.insert(name.to_owned(), now);
        log::info!("{name} at {since_main:.0} ms");
        let needed: &[&str] = if self.wait_for_frame {
            &["first_frame", "engine_ready"]
        } else {
            &["window_shown", "engine_ready"]
        };
        let complete = needed.iter().all(|step| marks.contains_key(*step));
        let Some(report) = self.report.as_ref().filter(|_| complete) else {
            return false;
        };
        let body = json!({ "marks": *marks });
        match std::fs::write(report, format!("{body:#}\n")) {
            Ok(()) => log::info!("startup report written to {}", report.display()),
            Err(error) => log::error!(
                "cannot write the startup report {}: {error}",
                report.display()
            ),
        }
        true
    }
}

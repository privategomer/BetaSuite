//! Changes the page holds but has not applied (the what-to-hide screen's
//! pending choices). While the page says it has some, closing the window
//! does not close it: the shell emits `close-requested` and the page asks
//! the user to apply, discard or stay. The page then calls `close_window`
//! to close for real.

use std::sync::atomic::{AtomicBool, Ordering};

/// Whether the page holds unapplied changes.
pub struct Unsaved(AtomicBool);

impl Unsaved {
    pub fn new() -> Self {
        Unsaved(AtomicBool::new(false))
    }

    pub fn set(&self, value: bool) {
        self.0.store(value, Ordering::SeqCst);
    }

    pub fn held(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// What to do when the window is asked to close.
#[derive(Debug, PartialEq, Eq)]
pub enum OnClose {
    /// Close now.
    Close,
    /// Keep the window and let the page ask first.
    AskPage,
}

/// The page is asked only while it holds changes and only once: a close it
/// asked for itself (`close_window`) goes through.
pub fn on_close(unsaved: bool, page_confirmed: bool) -> OnClose {
    if unsaved && !page_confirmed {
        OnClose::AskPage
    } else {
        OnClose::Close
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asks_only_with_unapplied_changes() {
        assert_eq!(on_close(false, false), OnClose::Close);
        assert_eq!(on_close(true, false), OnClose::AskPage);
        assert_eq!(on_close(true, true), OnClose::Close);
        assert_eq!(on_close(false, true), OnClose::Close);
    }

    #[test]
    fn remembers_what_the_page_said() {
        let unsaved = Unsaved::new();
        assert!(!unsaved.held());
        unsaved.set(true);
        assert!(unsaved.held());
        unsaved.set(false);
        assert!(!unsaved.held());
    }
}

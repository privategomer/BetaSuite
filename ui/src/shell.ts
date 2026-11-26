// The page's only contact with the native shell: typed wrappers over the
// commands registered in src-tauri/src/lib.rs.
import { invoke } from '@tauri-apps/api/core';

export type LogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error';

export function uiLog( level: LogLevel, message: string ): Promise<void> {
  return invoke( 'ui_log', { level, message } );
}

// The window starts hidden; this shows it. title is the window title, or
// null to keep the current one.
export function uiReady( title: string | null ): Promise<void> {
  return invoke( 'ui_ready', { title } );
}

export interface StartupInfo {
  version: string;
  locale: string | null;
  // light or dark from --theme; null follows the system.
  theme: 'light' | 'dark' | null;
  // The view from --view; null opens on the library (the player with --open).
  view: 'library' | 'player' | 'what-to-hide' | 'settings' | 'all-settings' | null;
  // All Settings' starting search text, from --search.
  search: string | null;
  // A help section to open at start, from --help-topic.
  help_topic: string | null;
  // The --open video, already allowed, or why it could not be opened.
  open: PlayableVideo | null;
  open_error: { code: string; params: Readonly<Record<string, unknown>> } | null;
}

export function startupInfo(): Promise<StartupInfo> {
  return invoke( 'startup_info' );
}

// A video the shell has allowed the page to load. file_url is set where
// video plays from file:// (Linux); elsewhere the page uses the asset
// protocol on path.
export interface PlayableVideo {
  path: string;
  name: string;
  file_url: string | null;
}

// Opens the native file dialog; null when the user cancels.
export function pickVideo( title: string, filterName: string, extensions: readonly string[] ): Promise<PlayableVideo | null> {
  return invoke( 'pick_video', { title, filterName, extensions } );
}

// A named moment for --measure-startup.
export function uiMark( name: string ): Promise<void> {
  return invoke( 'ui_mark', { name } );
}

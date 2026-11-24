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
}

export function startupInfo(): Promise<StartupInfo> {
  return invoke( 'startup_info' );
}

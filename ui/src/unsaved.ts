// Unapplied changes a screen holds (the what-to-hide screen's pending
// choices). While one is registered, leaving the screen or closing the
// window first asks: apply, discard, or stay. The shell is told, so it
// holds a window close and asks the page (src-tauri/src/unsaved.rs).
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { useSyncExternalStore } from 'react';
import { describeError, log } from './log';
import type { Check } from './settings/SettingRow';

export interface Guard {
  // How many changes, in the screen's own counting.
  readonly count: number;
  // What applying costs (the screen's config.set dry run).
  readonly check: Check;
  // Resolves true once applied, false when the engine refused.
  readonly apply: () => Promise<boolean>;
  readonly discard: () => void;
}

let current: Guard | null = null;
const listeners = new Set<() => void>();

export function setGuard( guard: Guard | null ): void {
  const held = guard !== null;
  const changed = held !== ( current !== null );
  current = guard;
  if ( changed ) {
    invoke( 'set_unsaved', { held } ).catch( ( error: unknown ) => {
      log.warn( `set_unsaved failed: ${describeError( error )}` );
    } );
  }
  listeners.forEach( ( listener ) => { listener(); } );
}

export function getGuard(): Guard | null {
  return current;
}

function subscribe( listener: () => void ): () => void {
  listeners.add( listener );
  return () => { listeners.delete( listener ); };
}

export function useGuard(): Guard | null {
  return useSyncExternalStore( subscribe, getGuard );
}

export function closeWindow(): Promise<void> {
  return invoke( 'close_window' );
}

export function onCloseRequested( handler: () => void ): Promise<UnlistenFn> {
  return listen( 'close-requested', () => { handler(); } );
}

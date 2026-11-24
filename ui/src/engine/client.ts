// The engine as the page sees it, through the shell (src-tauri/src/engine.rs).
// The shell owns the engine process; the page sends requests, reads the
// status, and hears events. Errors are codes with parameters, worded by
// i18n/errors.ts.
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export interface EngineError {
  code: string;
  params: Readonly<Record<string, unknown>>;
}

export type EngineStatus =
  | { state: 'idle' }
  | { state: 'starting' }
  | { state: 'ready'; version: string; protocol: number; pid: number }
  | { state: 'stopping' }
  | { state: 'stopped'; code: number | null; signal: number | null }
  | { state: 'exited'; code: number | null; signal: number | null }
  | { state: 'failed'; error: EngineError };

export interface EngineEvent {
  name: string;
  data: Readonly<Record<string, unknown>>;
}

export function isEngineError( value: unknown ): value is EngineError {
  return typeof value === 'object' && value !== null && 'code' in value && typeof value.code === 'string';
}

// Resolves with the result's data; rejects with an EngineError.
export function engineRequest( name: string, data: Readonly<Record<string, unknown>> = {} ): Promise<unknown> {
  return invoke( 'engine_request', { name, data } );
}

export function engineStatus(): Promise<EngineStatus> {
  return invoke( 'engine_status' );
}

export function restartEngine(): Promise<void> {
  return invoke( 'engine_restart' );
}

export function onEngineStatus( handler: ( status: EngineStatus ) => void ): Promise<UnlistenFn> {
  return listen<EngineStatus>( 'engine-status', ( event ) => { handler( event.payload ); } );
}

export function onEngineEvent( handler: ( event: EngineEvent ) => void ): Promise<UnlistenFn> {
  return listen<EngineEvent>( 'engine-event', ( event ) => { handler( event.payload ); } );
}

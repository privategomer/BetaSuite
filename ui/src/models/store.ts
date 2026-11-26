// The engine's models.state and models.progress events as React state: one
// store for the page, so the Models view and the status strip see the same
// downloads. useModelEvents() re-renders on each change.
import { useSyncExternalStore } from 'react';
import { isEngineError, onEngineEvent, type EngineError } from '../engine/client';
import { describeError, log } from '../log';
import { switchToModel, type UseResult } from './api';

export interface Progress {
  readonly bytes: number;
  readonly total: number;
}

export interface ModelEvents {
  // Downloads queued or running, with their latest progress (null until one arrives).
  readonly active: ReadonlyMap<string, Progress | null>;
  // The last failure of each model, until it is installed or tried again.
  readonly failures: ReadonlyMap<string, EngineError>;
  // Bumped on every state change, so a view knows to list again.
  readonly version: number;
}

let current: ModelEvents = { active: new Map(), failures: new Map(), version: 0 };
const listeners = new Set<() => void>();
let started = false;

function publish( next: ModelEvents ): void {
  current = next;
  listeners.forEach( ( listener ) => { listener(); } );
}

type UseDone = ( result: UseResult | null, error: EngineError | null ) => void;

// Recommended models being installed to be used: switched to when their
// download finishes, here rather than in the view, so leaving the page
// does not lose it.
const useAfterInstall = new Map<string, { gpu: boolean; done: UseDone }>();

export function installThenUse( id: string, gpu: boolean, done: UseDone ): void {
  useAfterInstall.set( id, { gpu, done } );
}

function followUp( id: string, state: string ): void {
  const pending = useAfterInstall.get( id );
  if ( pending === undefined || state === 'queued' || state === 'downloading' ) {
    return;
  }
  useAfterInstall.delete( id );
  if ( state !== 'installed' ) {
    return;
  }
  switchToModel( id, pending.gpu ).then(
    ( result ) => { pending.done( result, null ); },
    ( error: unknown ) => {
      log.error( `models.use after install failed: ${describeError( error )}` );
      pending.done( null, isEngineError( error ) ? error : { code: 'unknown', params: {} } );
    },
  );
}

function onState( id: string, state: string, data: Readonly<Record<string, unknown>> ): void {
  followUp( id, state );
  const active = new Map( current.active );
  const failures = new Map( current.failures );
  if ( state === 'queued' || state === 'downloading' ) {
    active.set( id, active.get( id ) ?? null );
    failures.delete( id );
  } else {
    active.delete( id );
  }
  if ( state === 'failed' ) {
    const error = { code: data[ 'code' ], params: data[ 'params' ] ?? {} };
    if ( isEngineError( error ) ) {
      failures.set( id, error );
    }
  } else if ( state === 'installed' || state === 'removed' ) {
    failures.delete( id );
  }
  publish( { active, failures, version: current.version + 1 } );
}

function onProgress( id: string, bytes: number, total: number ): void {
  if ( !current.active.has( id ) ) {
    return;
  }
  const active = new Map( current.active );
  active.set( id, { bytes, total } );
  publish( { ...current, active } );
}

function start(): void {
  if ( started ) {
    return;
  }
  started = true;
  onEngineEvent( ( event ) => {
    const id = event.data[ 'id' ];
    if ( typeof id !== 'string' ) {
      return;
    }
    if ( event.name === 'models.state' && typeof event.data[ 'state' ] === 'string' ) {
      log.debug( `model ${id}: ${event.data[ 'state' ]}` );
      onState( id, event.data[ 'state' ], event.data );
    } else if ( event.name === 'models.progress' ) {
      const bytes = event.data[ 'bytes' ];
      const total = event.data[ 'total' ];
      if ( typeof bytes === 'number' && typeof total === 'number' ) {
        onProgress( id, bytes, total );
      }
    }
  } ).catch( ( error: unknown ) => {
    log.error( `cannot follow model downloads: ${describeError( error )}` );
  } );
}

function subscribe( listener: () => void ): () => void {
  start();
  listeners.add( listener );
  return () => { listeners.delete( listener ); };
}

export function useModelEvents(): ModelEvents {
  return useSyncExternalStore( subscribe, () => current );
}

// For a download the page started: it shows as active before its first event.
export function markQueued( id: string ): void {
  start();
  onState( id, 'queued', {} );
}

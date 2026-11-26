// The governor, resource samples, Optimize progress, stage speeds and
// waiting settings changes as React state: one store, so the status strip
// and the Performance and Storage pages see the same thing. Samples flow
// only while something watches them (useSamples), so an idle app costs
// nothing.
import { useEffect, useSyncExternalStore } from 'react';
import { onEngineEvent, onEngineStatus, type EngineError } from '../engine/client';
import { describeError, log } from '../log';
import {
  governorState, pendingChanges, watchResources,
  type GovernorState, type PendingChange, type ProposalRow, type Sample,
} from './api';

const SAMPLES_KEPT = 120;
const EVENTS_KEPT = 100;

export interface OptimizeProgress {
  readonly state: 'running' | 'finished' | 'failed' | 'cancelled';
  readonly stage: string | null;
  readonly done: number;
  readonly total: number;
  readonly proposal: readonly ProposalRow[] | null;
  readonly error: EngineError | null;
}

export interface StageSpeed {
  readonly stage: string;
  readonly rate: number;
  readonly done: number;
  readonly total: number;
}

export interface Resources {
  readonly governor: GovernorState | null;
  // This session's governor changes, newest first.
  readonly changes: readonly GovernorState[];
  readonly samples: readonly Sample[];
  readonly optimize: OptimizeProgress | null;
  readonly stages: ReadonlyMap<string, StageSpeed>;
  readonly pending: readonly PendingChange[];
  // The last settings change made while jobs ran: written, or not (with why).
  readonly lastChange: { readonly applied: boolean; readonly error: EngineError | null } | null;
}

let current: Resources = {
  governor: null, changes: [], samples: [], optimize: null, stages: new Map(), pending: [], lastChange: null,
};
const listeners = new Set<() => void>();
let started = false;
let watchers = 0;

function publish( next: Partial<Resources> ): void {
  current = { ...current, ...next };
  listeners.forEach( ( listener ) => { listener(); } );
}

function asGovernor( data: Readonly<Record<string, unknown>> ): GovernorState | null {
  return Array.isArray( data[ 'reasons' ] ) && typeof data[ 'throttled' ] === 'boolean'
    ? data as unknown as GovernorState : null;
}

export function refreshGovernor(): void {
  governorState().then(
    ( state ) => {
      const history = [ ...( state.history ?? [] ) ].reverse();
      publish( { governor: state, changes: history.slice( 0, EVENTS_KEPT ) } );
    },
    ( error: unknown ) => { log.warn( `governor.get failed: ${describeError( error )}` ); },
  );
}

export function refreshPending(): void {
  pendingChanges().then(
    ( answer ) => { publish( { pending: answer.pending } ); },
    ( error: unknown ) => { log.warn( `config.pending failed: ${describeError( error )}` ); },
  );
}

function onEvent( name: string, data: Readonly<Record<string, unknown>> ): void {
  switch ( name ) {
    case 'governor.change': {
      const state = asGovernor( data );
      if ( state !== null ) {
        publish( { governor: state, changes: [ state, ...current.changes ].slice( 0, EVENTS_KEPT ) } );
      }
      return;
    }
    case 'resource.sample':
      publish( { samples: [ ...current.samples, data as unknown as Sample ].slice( -SAMPLES_KEPT ) } );
      return;
    case 'perf.optimize_progress': {
      const state = data[ 'state' ];
      if ( state !== 'running' && state !== 'finished' && state !== 'failed' && state !== 'cancelled' ) {
        return;
      }
      const code = data[ 'code' ];
      publish( { optimize: {
        state,
        stage: typeof data[ 'stage' ] === 'string' ? data[ 'stage' ] : null,
        done: typeof data[ 'done' ] === 'number' ? data[ 'done' ] : 0,
        total: typeof data[ 'total' ] === 'number' ? data[ 'total' ] : 0,
        proposal: Array.isArray( data[ 'proposal' ] ) ? data[ 'proposal' ] as readonly ProposalRow[] : null,
        error: typeof code === 'string' ? { code, params: ( data[ 'params' ] ?? {} ) as Record<string, unknown> } : null,
      } } );
      return;
    }
    case 'job.progress': {
      const stage = data[ 'stage' ];
      const rate = data[ 'rate' ];
      if ( typeof stage === 'string' && typeof rate === 'number' ) {
        const stages = new Map( current.stages );
        stages.set( stage, { stage, rate,
          done: typeof data[ 'done' ] === 'number' ? data[ 'done' ] : 0,
          total: typeof data[ 'total' ] === 'number' ? data[ 'total' ] : 0 } );
        publish( { stages } );
      }
      return;
    }
    case 'config.applied':
      publish( { lastChange: { applied: true, error: null } } );
      refreshPending();
      return;
    case 'config.pending_failed': {
      const error = data[ 'error' ] as EngineError | undefined;
      publish( { lastChange: { applied: false, error: error ?? null } } );
      refreshPending();
      return;
    }
    default:
  }
}

function start(): void {
  if ( started ) {
    return;
  }
  started = true;
  onEngineEvent( ( event ) => { onEvent( event.name, event.data ); } ).catch( ( error: unknown ) => {
    log.error( `cannot follow resource events: ${describeError( error )}` );
  } );
  onEngineStatus( ( status ) => {
    if ( status.state === 'ready' ) {
      refreshGovernor();
      refreshPending();
      if ( watchers > 0 ) {
        void watchResources( true ).catch( () => undefined );
      }
    }
  } ).catch( ( error: unknown ) => { log.error( `cannot follow the engine status: ${describeError( error )}` ); } );
  refreshGovernor();
  refreshPending();
}

function subscribe( listener: () => void ): () => void {
  start();
  listeners.add( listener );
  return () => { listeners.delete( listener ); };
}

export function useResources(): Resources {
  return useSyncExternalStore( subscribe, () => current );
}

// Samples arrive while at least one mounted view uses this.
export function useSamples(): readonly Sample[] {
  const resources = useResources();
  useEffect( () => {
    watchers += 1;
    if ( watchers === 1 ) {
      watchResources( true ).then(
        ( answer ) => { if ( answer.sample !== null ) { publish( { samples: [ answer.sample ] } ); } },
        ( error: unknown ) => { log.warn( `resources.watch failed: ${describeError( error )}` ); },
      );
    }
    return () => {
      watchers -= 1;
      if ( watchers === 0 ) {
        void watchResources( false ).catch( () => undefined );
      }
    };
  }, [] );
  return resources.samples;
}

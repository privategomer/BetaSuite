// The engine's job queue as the page sees it: rebuilt from job.list when an
// engine is ready (after a restart too, when the queue it kept comes back),
// and kept current by job events. Progress per job comes from job.progress,
// the only event that may be dropped, so it is a picture, never a count.
import { useSyncExternalStore } from 'react';
import { onEngineReady, onEvent } from '../engine/events';
import { describeError, log } from '../log';
import { listJobs, type JobRow, type JobState } from './api';

export interface Progress {
  readonly file: number;
  readonly stage: string;
  readonly done: number;
  readonly total: number | null;
  readonly rate: number;
}

export interface Queue {
  readonly rows: readonly JobRow[];
  readonly progress: Readonly<Record<string, Progress>>;
  // A job is queued, running or paused and holding the queue.
  readonly busy: boolean;
  readonly loaded: boolean;
}

const ACTIVE: readonly JobState[] = [ 'queued', 'running', 'paused' ];

let current: Queue = { rows: [], progress: {}, busy: false, loaded: false };
const listeners = new Set<() => void>();
const drainedHandlers = new Set<( rows: readonly JobRow[] ) => void>();
let started = false;
let pending: Promise<void> | null = null;
let again = false;

function publish( next: Queue ): void {
  const wasBusy = current.busy;
  current = next;
  listeners.forEach( ( listener ) => { listener(); } );
  if ( wasBusy && !next.busy ) {
    drainedHandlers.forEach( ( handler ) => { handler( next.rows ); } );
  }
}

// Whether a row still holds the queue: restored jobs are held, not busy.
export function holdsQueue( row: JobRow ): boolean {
  return ACTIVE.includes( row.state ) && row.restored !== true;
}

// Reads job.list; a refresh asked for meanwhile runs once more after it.
export function refreshJobs(): Promise<void> {
  if ( pending !== null ) {
    again = true;
    return pending;
  }
  pending = listJobs().then( ( rows ) => {
    const live = new Set( rows.filter( ( row ) => ACTIVE.includes( row.state ) ).map( ( row ) => row.job ) );
    const progress = Object.fromEntries( Object.entries( current.progress ).filter( ( [ job ] ) => live.has( job ) ) );
    publish( { rows, progress, busy: rows.some( holdsQueue ), loaded: true } );
  }, ( error: unknown ) => {
    log.warn( `job.list failed: ${describeError( error )}` );
  } ).finally( () => {
    pending = null;
    if ( again ) {
      again = false;
      void refreshJobs();
    }
  } );
  return pending;
}

function start(): void {
  if ( started ) {
    return;
  }
  started = true;
  onEngineReady( () => { void refreshJobs(); } );
  onEvent( 'job.state', () => { void refreshJobs(); } );
  onEvent( 'job.file', () => { void refreshJobs(); } );
  onEvent( 'job.progress', ( data ) => {
    const job = data[ 'job' ];
    if ( typeof job !== 'string' ) {
      return;
    }
    const progress: Progress = {
      file: Number( data[ 'file' ] ?? 0 ),
      stage: typeof data[ 'stage' ] === 'string' ? data[ 'stage' ] : '',
      done: Number( data[ 'done' ] ?? 0 ),
      total: typeof data[ 'total' ] === 'number' ? data[ 'total' ] : null,
      rate: Number( data[ 'rate' ] ?? 0 ),
    };
    current = { ...current, progress: { ...current.progress, [ job ]: progress } };
    listeners.forEach( ( listener ) => { listener(); } );
  } );
  void refreshJobs();
}

function subscribe( listener: () => void ): () => void {
  start();
  listeners.add( listener );
  return () => { listeners.delete( listener ); };
}

export function useQueue(): Queue {
  return useSyncExternalStore( subscribe, () => current );
}

export function queueNow(): Queue {
  start();
  return current;
}

// Called when the queue goes from busy to idle (the last job ended).
export function onQueueDrained( handler: ( rows: readonly JobRow[] ) => void ): () => void {
  start();
  drainedHandlers.add( handler );
  return () => { drainedHandlers.delete( handler ); };
}

// Seconds left for a job's current stage, at its last reported rate, as the
// command line's progress line works it out; null when it cannot be said.
export function stageSecondsLeft( progress: Progress | undefined ): number | null {
  if ( progress?.total === undefined || progress.total === null || progress.rate <= 0 ) {
    return null;
  }
  return Math.max( 0, ( progress.total - progress.done ) / progress.rate );
}

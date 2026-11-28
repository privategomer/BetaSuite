// The engine status as React state: one store for the page, fed by the
// shell's engine-status events. useEngineStatus() re-renders on each change.
import { useSyncExternalStore } from 'react';
import { describeError, log } from '../log';
import { engineStatus, onEngineStatus, type EngineStatus } from './client';

let current: EngineStatus = { state: 'starting' };
const listeners = new Set<() => void>();
let started = false;

function publish( status: EngineStatus ): void {
  current = status;
  listeners.forEach( ( listener ) => { listener(); } );
}

// Subscribes first, then reads the current status, so no change between
// the two is lost; a change that arrives first wins over the read.
function start(): void {
  if ( started ) {
    return;
  }
  started = true;
  let heard = false;
  onEngineStatus( ( status ) => {
    heard = true;
    log.debug( `engine status ${status.state}` );
    publish( status );
  } ).then(
    () => engineStatus().then( ( status ) => {
      if ( !heard ) {
        publish( status );
      }
    } ),
  ).catch( ( error: unknown ) => {
    log.error( `cannot follow the engine status: ${describeError( error )}` );
  } );
}

function subscribe( listener: () => void ): () => void {
  start();
  listeners.add( listener );
  return () => { listeners.delete( listener ); };
}

export function useEngineStatus(): EngineStatus {
  return useSyncExternalStore( subscribe, () => current );
}

// One subscription to the engine's events for the whole page, handed out
// by event name, so each screen listens to what it needs without opening
// a Tauri listener of its own.
import { describeError, log } from '../log';
import { onEngineEvent, onEngineStatus, type EngineEvent, type EngineStatus } from './client';

type Handler = ( data: Readonly<Record<string, unknown>> ) => void;

const handlers = new Map<string, Set<Handler>>();
const readyHandlers = new Set<() => void>();
let started = false;

function start(): void {
  if ( started ) {
    return;
  }
  started = true;
  onEngineEvent( ( event: EngineEvent ) => {
    handlers.get( event.name )?.forEach( ( handler ) => {
      try {
        handler( event.data );
      } catch ( error ) {
        log.error( `handling ${event.name} failed: ${describeError( error )}` );
      }
    } );
  } ).catch( ( error: unknown ) => { log.error( `cannot follow engine events: ${describeError( error )}` ); } );
  onEngineStatus( ( status: EngineStatus ) => {
    if ( status.state === 'ready' ) {
      readyHandlers.forEach( ( handler ) => { handler(); } );
    }
  } ).catch( ( error: unknown ) => { log.error( `cannot follow the engine status: ${describeError( error )}` ); } );
}

// Calls handler with each event of that name; returns the unsubscribe.
export function onEvent( name: string, handler: Handler ): () => void {
  start();
  let set = handlers.get( name );
  if ( set === undefined ) {
    set = new Set();
    handlers.set( name, set );
  }
  set.add( handler );
  return () => { set.delete( handler ); };
}

// Calls handler each time an engine (a new one after a restart too) is ready.
export function onEngineReady( handler: () => void ): () => void {
  start();
  readyHandlers.add( handler );
  return () => { readyHandlers.delete( handler ); };
}

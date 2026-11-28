// Page logging. Every record goes to the shell's log file, at the shell's
// level filter, with the target "page". Log text stays English: it is for
// whoever reads the log, never shown in the UI.
import { uiLog, type LogLevel } from './shell';

function send( level: LogLevel, message: string ): void {
  uiLog( level, message ).catch( () => {
    // The shell is gone, so there is nowhere left to log to.
  } );
}

export const log = {
  trace: ( message: string ): void => { send( 'trace', message ); },
  debug: ( message: string ): void => { send( 'debug', message ); },
  info: ( message: string ): void => { send( 'info', message ); },
  warn: ( message: string ): void => { send( 'warn', message ); },
  error: ( message: string ): void => { send( 'error', message ); },
};

export function describeError( reason: unknown ): string {
  if ( reason instanceof Error ) {
    return `${reason.name}: ${reason.message}`;
  }
  try {
    return JSON.stringify( reason );
  } catch {
    return typeof reason;
  }
}

export function logUncaughtErrors(): void {
  window.addEventListener( 'error', ( event ) => {
    log.error( `uncaught error: ${event.message} (${event.filename}:${event.lineno}:${event.colno})` );
  } );
  window.addEventListener( 'unhandledrejection', ( event ) => {
    log.error( `unhandled rejection: ${describeError( event.reason )}` );
  } );
}

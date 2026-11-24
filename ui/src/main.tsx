import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { initCatalog } from './i18n/format';
import { describeError, log, logUncaughtErrors } from './log';
import { startupInfo } from './shell';
import './styles.css';

logUncaughtErrors();
log.debug( 'page script started' );

const root = document.getElementById( 'root' );
if ( !root ) {
  throw new Error( 'index.html has no #root element' );
}
const element = root;

function start( requested: readonly string[] ): void {
  const locale = initCatalog( requested );
  log.info( `UI locale ${locale.tag}` );
  createRoot( element ).render( <StrictMode><App /></StrictMode> );
}

startupInfo().then(
  ( info ) => { start( info.locale === null ? navigator.languages : [ info.locale ] ); },
  ( error: unknown ) => {
    log.error( `startup_info failed: ${describeError( error )}` );
    start( navigator.languages );
  },
);

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { initCatalog } from './i18n/format';
import { describeError, log, logUncaughtErrors } from './log';
import { startupInfo, uiMark, type StartupInfo } from './shell';
import { applyTheme } from './theme/theme';
import './styles.css';

logUncaughtErrors();
log.debug( 'page script started' );

const root = document.getElementById( 'root' );
if ( !root ) {
  throw new Error( 'index.html has no #root element' );
}
const element = root;

function start( info: StartupInfo | null ): void {
  const locale = initCatalog( info?.locale != null ? [ info.locale ] : navigator.languages );
  applyTheme( info?.theme ?? null );
  log.info( `UI locale ${locale.tag}, theme ${info?.theme ?? 'system'}` );
  void uiMark( 'page_script' );
  createRoot( element ).render( <StrictMode><App startup={info} /></StrictMode> );
}

startupInfo().then(
  ( info ) => { start( info ); },
  ( error: unknown ) => {
    log.error( `startup_info failed: ${describeError( error )}` );
    start( null );
  },
);

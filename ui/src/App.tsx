import { useEffect } from 'react';
import { Player } from './components/Player';
import { StatusStrip } from './components/StatusStrip';
import { t } from './i18n/format';
import { describeError, log } from './log';
import { uiReady, type StartupInfo } from './shell';

interface AppProps {
  startup: StartupInfo | null;
}

export function App( { startup }: AppProps ) {
  useEffect( () => {
    // Show the window once the first render is in the DOM. WebKitGTK stalls
    // about a second when a window is shown before its page has content.
    // Not tied to requestAnimationFrame: a hidden window may never get one.
    uiReady( t( 'app.window_title' ) ).catch( ( error: unknown ) => {
      log.error( `ui_ready failed: ${describeError( error )}` );
    } );
  }, [] );

  return (
    <div className="app">
      <header className="app-header">
        <h1>{t( 'app.name' )}</h1>
      </header>
      <main className="app-main">
        <Player openAtStart={startup?.open ?? null} openError={startup?.open_error ?? null} />
      </main>
      <StatusStrip />
    </div>
  );
}

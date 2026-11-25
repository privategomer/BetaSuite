import { useEffect, useState } from 'react';
import { Player } from './components/Player';
import { StatusStrip } from './components/StatusStrip';
import { t } from './i18n/format';
import { describeError, log } from './log';
import { AllSettings } from './settings/AllSettings';
import { uiReady, type StartupInfo } from './shell';

interface AppProps {
  startup: StartupInfo | null;
}

type View = 'player' | 'all_settings';

export function App( { startup }: AppProps ) {
  const [ view, setView ] = useState<View>( startup?.view === 'all-settings' ? 'all_settings' : 'player' );

  useEffect( () => {
    // Show the window once the first render is in the DOM. WebKitGTK stalls
    // about a second when a window is shown before its page has content.
    // Not tied to requestAnimationFrame: a hidden window may never get one.
    uiReady( t( 'app.window_title' ) ).catch( ( error: unknown ) => {
      log.error( `ui_ready failed: ${describeError( error )}` );
    } );
  }, [] );

  const tab = ( id: View, label: string ) => (
    <button type="button" className="nav-tab" aria-current={view === id ? 'page' : undefined} onClick={() => { setView( id ); }}>
      {label}
    </button>
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>{t( 'app.name' )}</h1>
        <nav className="app-nav" aria-label={t( 'nav.label' )}>
          {tab( 'player', t( 'nav.player' ) )}
          {tab( 'all_settings', t( 'nav.all_settings' ) )}
        </nav>
      </header>
      <main className="app-main">
        {/* The player stays mounted, so a video keeps its place across views. */}
        <div className="view" hidden={view !== 'player'}>
          <Player openAtStart={startup?.open ?? null} openError={startup?.open_error ?? null} />
        </div>
        {view === 'all_settings' && (
          <AllSettings onHelp={( anchor ) => { log.info( `help requested: ${anchor}` ); }} />
        )}
      </main>
      <StatusStrip />
    </div>
  );
}

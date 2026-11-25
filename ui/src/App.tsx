import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import { Player } from './components/Player';
import { StatusStrip } from './components/StatusStrip';
import { t } from './i18n/format';
import { describeError, log } from './log';
import { AllSettings } from './settings/AllSettings';
import { SettingsView } from './settings/SettingsView';
import { uiReady, type StartupInfo } from './shell';

interface AppProps {
  startup: StartupInfo | null;
}

// The help panel and the reference it shows load on first use, keeping
// them out of the startup bundle.
const HelpPanel = lazy( () => import( './help/HelpPanel' ).then( ( module ) => ( { default: module.HelpPanel } ) ) );

type View = 'player' | 'settings' | 'all_settings';

function startView( view: StartupInfo[ 'view' ] | undefined ): View {
  switch ( view ) {
    case 'settings': return 'settings';
    case 'all-settings': return 'all_settings';
    default: return 'player';
  }
}

export function App( { startup }: AppProps ) {
  const [ view, setView ] = useState<View>( startView( startup?.view ) );
  const [ help, setHelp ] = useState<string | null>( startup?.help_topic ?? null );
  const onHelp = useCallback( ( anchor: string ) => { setHelp( anchor ); }, [] );
  const closeHelp = useCallback( () => { setHelp( null ); }, [] );

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
          {tab( 'settings', t( 'nav.settings' ) )}
          {tab( 'all_settings', t( 'nav.all_settings' ) )}
        </nav>
      </header>
      <div className="app-body">
      <main className="app-main">
        {/* The player stays mounted, so a video keeps its place across views. */}
        <div className="view" hidden={view !== 'player'}>
          <Player openAtStart={startup?.open ?? null} openError={startup?.open_error ?? null} />
        </div>
        {view === 'settings' && <SettingsView onHelp={onHelp} />}
        {view === 'all_settings' && <AllSettings initialQuery={startup?.search ?? ''} onHelp={onHelp} />}
      </main>
      {help !== null && (
        <Suspense fallback={null}>
          <HelpPanel anchor={help} onClose={closeHelp} />
        </Suspense>
      )}
      </div>
      <StatusStrip />
    </div>
  );
}

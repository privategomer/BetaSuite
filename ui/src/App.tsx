import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import { UnsavedDialog } from './components/UnsavedDialog';
import { Player } from './components/Player';
import { StatusStrip } from './components/StatusStrip';
import { t } from './i18n/format';
import { describeError, log } from './log';
import { StyleEditor } from './hide/StyleEditor';
import { WhatToHide } from './hide/WhatToHide';
import { useAppIcon } from './appIcon';
import { useBackground } from './library/background';
import { queueNow } from './library/jobs';
import { LibraryView } from './library/LibraryView';
import { RunCloseDialog } from './library/RunCloseDialog';
import { ModelsView } from './models/ModelsView';
import { PerformanceView } from './resources/PerformanceView';
import { StorageView } from './resources/StorageView';
import { AllSettings } from './settings/AllSettings';
import { SettingsView } from './settings/SettingsView';
import { registerStyleEditor } from './settings/structured';
import { uiReady, type StartupInfo } from './shell';
import { closeWindow, getGuard, onCloseRequested, useGuard } from './unsaved';

interface AppProps {
  startup: StartupInfo | null;
}

// The help panel and the reference it shows load on first use, keeping
// them out of the startup bundle.
const AboutView = lazy( () => import( './about/AboutView' ).then( ( module ) => ( { default: module.AboutView } ) ) );
const HelpPanel = lazy( () => import( './help/HelpPanel' ).then( ( module ) => ( { default: module.HelpPanel } ) ) );

registerStyleEditor( 'censor_style', StyleEditor );

type View = 'library' | 'player' | 'what_to_hide' | 'models' | 'performance' | 'storage' | 'settings' | 'all_settings' | 'about';

function startView( view: StartupInfo[ 'view' ] | undefined, opening: boolean ): View {
  switch ( view ) {
    case 'library': return 'library';
    case 'player': return 'player';
    case 'what-to-hide': return 'what_to_hide';
    case 'models': return 'models';
    case 'performance': return 'performance';
    case 'storage': return 'storage';
    case 'settings': return 'settings';
    case 'all-settings': return 'all_settings';
    case 'about': return 'about';
    default: return opening ? 'player' : 'library';
  }
}

export function App( { startup }: AppProps ) {
  const [ view, setView ] = useState<View>( startView( startup?.view, startup?.open !== null && startup?.open !== undefined ) );
  // A library video to play: the path, and a count so the same one plays again.
  const [ playing, setPlaying ] = useState<{ path: string; nonce: number } | null>( null );
  const [ closingRun, setClosingRun ] = useState( false );
  useBackground();
  useAppIcon();
  const play = useCallback( ( path: string ) => {
    setPlaying( ( current ) => ( { path, nonce: ( current?.nonce ?? 0 ) + 1 } ) );
    setView( 'player' );
  }, [] );
  const [ help, setHelp ] = useState<string | null>( startup?.help_topic ?? null );
  const onHelp = useCallback( ( anchor: string ) => { setHelp( anchor ); }, [] );
  const closeHelp = useCallback( () => { setHelp( null ); }, [] );
  // Where to go once the user has answered about unapplied changes: a view,
  // or 'close' for the window.
  const [ asking, setAsking ] = useState<View | 'close' | null>( null );
  const guard = useGuard();
  const go = ( target: View ) => {
    if ( target !== view && getGuard() !== null ) {
      setAsking( target );
    } else {
      setView( target );
    }
  };
  const answered = () => {
    const target = asking;
    setAsking( null );
    if ( target === 'close' ) {
      closeWindow().catch( ( error: unknown ) => { log.error( `close_window failed: ${describeError( error )}` ); } );
    } else if ( target !== null ) {
      setView( target );
    }
  };

  useEffect( () => {
    // The shell holds a window close while the page has unapplied changes.
    const unlisten = onCloseRequested( () => {
      if ( getGuard() !== null ) {
        setAsking( 'close' );
      } else if ( queueNow().busy ) {
        setClosingRun( true );
      } else {
        closeWindow().catch( ( error: unknown ) => { log.error( `close_window failed: ${describeError( error )}` ); } );
      }
    } );
    return () => { unlisten.then( ( stop ) => { stop(); }, () => undefined ); };
  }, [] );

  useEffect( () => {
    // Show the window once the first render is in the DOM. WebKitGTK stalls
    // about a second when a window is shown before its page has content.
    // Not tied to requestAnimationFrame: a hidden window may never get one.
    uiReady( t( 'app.window_title' ) ).catch( ( error: unknown ) => {
      log.error( `ui_ready failed: ${describeError( error )}` );
    } );
  }, [] );

  const tab = ( id: View, label: string ) => (
    <button type="button" className="nav-tab" aria-current={view === id ? 'page' : undefined} onClick={() => { go( id ); }}>
      {label}
    </button>
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>{t( 'app.name' )}</h1>
        <nav className="app-nav" aria-label={t( 'nav.label' )}>
          {tab( 'library', t( 'nav.library' ) )}
          {tab( 'player', t( 'nav.player' ) )}
          {tab( 'what_to_hide', t( 'nav.what_to_hide' ) )}
          {tab( 'models', t( 'nav.models' ) )}
          {tab( 'performance', t( 'nav.performance' ) )}
          {tab( 'storage', t( 'nav.storage' ) )}
          {tab( 'settings', t( 'nav.settings' ) )}
          {tab( 'all_settings', t( 'nav.all_settings' ) )}
          {tab( 'about', t( 'nav.about' ) )}
        </nav>
      </header>
      <div className="app-body">
      <main className="app-main">
        {/* The player stays mounted, so a video keeps its place across views. */}
        <div className="view" hidden={view !== 'player'}>
          <Player openAtStart={startup?.open ?? null} openError={startup?.open_error ?? null} request={playing} measureScrub={startup?.measure_scrub ?? false} />
        </div>
        {view === 'library' && <LibraryView onPlay={play} onHelp={onHelp} />}
        {view === 'what_to_hide' && <WhatToHide onHelp={onHelp} />}
        {view === 'models' && <ModelsView />}
        {view === 'performance' && <PerformanceView />}
        {view === 'storage' && <StorageView />}
        {view === 'settings' && <SettingsView onHelp={onHelp} />}
        {view === 'all_settings' && <AllSettings initialQuery={startup?.search ?? ''} onHelp={onHelp} />}
        {view === 'about' && <Suspense fallback={null}><AboutView /></Suspense>}
      </main>
      {help !== null && (
        <Suspense fallback={null}>
          <HelpPanel anchor={help} onClose={closeHelp} />
        </Suspense>
      )}
      </div>
      {asking !== null && guard !== null && (
        <UnsavedDialog guard={guard} closing={asking === 'close'} onDone={answered} onStay={() => { setAsking( null ); }} />
      )}
      {closingRun && (
        <RunCloseDialog
          onClose={() => {
            setClosingRun( false );
            closeWindow().catch( ( error: unknown ) => { log.error( `close_window failed: ${describeError( error )}` ); } );
          }}
          onStay={() => { setClosingRun( false ); }}
        />
      )}
      <StatusStrip />
    </div>
  );
}

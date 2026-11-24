import { useEffect } from 'react';
import { t } from './i18n/format';
import { describeError, log } from './log';
import { uiReady } from './shell';

export function App() {
  useEffect( () => {
    // Show the window once the first render is in the DOM. WebKitGTK stalls
    // about a second when a window is shown before its page has content.
    // Not tied to requestAnimationFrame: a hidden window may never get one.
    uiReady( t( 'app.window_title' ) ).catch( ( error: unknown ) => {
      log.error( `ui_ready failed: ${describeError( error )}` );
    } );
  }, [] );

  return <div className="app">Engine ready</div>;
}

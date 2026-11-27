// The app's own icon and tray: the shell is told the app_icon and
// tray_icon_shown settings once the engine is ready, and again whenever
// the settings are loaded after a change (useSettings), so a new icon
// shows without a restart (src-tauri/src/app_icon.rs, tray.rs).
import { invoke } from '@tauri-apps/api/core';
import { useEffect } from 'react';
import { useEngineStatus } from './engine/status';
import { describeError, log } from './log';
import { getValues } from './settings/api';

export const APP_ICON_PATH = 'app_icon';
export const TRAY_SHOWN_PATH = 'tray_icon_shown';

let appliedIcon: string | null = null;
let appliedShown: boolean | null = null;

// Sends the values the shell does not have yet; unchanged ones are skipped.
export function applyAppIcon( valueOf: ( path: string ) => unknown ): void {
  const icon = valueOf( APP_ICON_PATH );
  if ( typeof icon === 'string' && icon !== appliedIcon ) {
    appliedIcon = icon;
    invoke( 'set_app_icon', { id: icon } ).catch( ( error: unknown ) => {
      appliedIcon = null;
      log.warn( `set_app_icon ${icon} failed: ${describeError( error )}` );
    } );
  }
  const shown = valueOf( TRAY_SHOWN_PATH );
  if ( typeof shown === 'boolean' && shown !== appliedShown ) {
    appliedShown = shown;
    invoke( 'tray_shown', { shown } ).catch( ( error: unknown ) => {
      appliedShown = null;
      log.warn( `tray_shown failed: ${describeError( error )}` );
    } );
  }
}

async function readAndApply(): Promise<void> {
  const values = await getValues( { paths: [ APP_ICON_PATH, TRAY_SHOWN_PATH ] } );
  const byPath = new Map( values.settings.map( ( row ) => [ row.path, row.value ] ) );
  applyAppIcon( ( path ) => byPath.get( path ) );
}

// Mounted once by the app: applies the settings each time an engine
// process becomes ready.
export function useAppIcon(): void {
  const engine = useEngineStatus();
  const pid = engine.state === 'ready' ? engine.pid : null;
  useEffect( () => {
    if ( pid === null ) {
      return;
    }
    readAndApply().catch( ( error: unknown ) => { log.warn( `cannot read the app icon settings: ${describeError( error )}` ); } );
  }, [ pid ] );
}

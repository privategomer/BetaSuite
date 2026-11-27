// The app_icon picker: each icon the engine offers (the folders under
// assets/icons), and a preview of the chosen one as a tray shows it, at
// real size on a light and a dark panel, before the change is applied.
// The PNGs come from the shell (app_icon_image), the same files the window
// and tray use. Arrow keys move between icons, as in any radio group.
import { invoke } from '@tauri-apps/api/core';
import { useEffect, useState } from 'react';
import { hasMessage, msgKey, t } from '../i18n/format';
import { describeError, log } from '../log';
import type { EditorProps } from './editors';

// The sizes every icon is drawn at (tools/art/icons.py SIZES).
const SIZES = [ 16, 22, 24, 32, 48, 64, 128, 256, 512 ] as const;
// Tray sizes at a scale of 1: Windows (16) and KDE Plasma or GNOME (22).
const LARGEST = 512;
const TRAY_SIZES = [ 16, 22 ] as const;
const CHOICE_SIZE = 32;

// The smallest drawn size at least `want` pixels, as the shell picks.
export function drawnSize( want: number ): number {
  return SIZES.find( ( size ) => size >= Math.round( want ) ) ?? LARGEST;
}

const urls = new Map<string, string>();

function base64( bytes: Uint8Array ): string {
  let text = '';
  for ( const byte of bytes ) {
    text += String.fromCharCode( byte );
  }
  return btoa( text );
}

async function iconUrl( id: string, size: number ): Promise<string> {
  const key = `${id}@${String( size )}`;
  const known = urls.get( key );
  if ( known !== undefined ) {
    return known;
  }
  const answer = await invoke<ArrayBuffer | number[]>( 'app_icon_image', { id, size } );
  const bytes = answer instanceof ArrayBuffer ? new Uint8Array( answer ) : Uint8Array.from( answer );
  const url = `data:image/png;base64,${base64( bytes )}`;
  urls.set( key, url );
  return url;
}

// One icon at one size in CSS pixels, drawn from the PNG that fills it
// exactly on this screen.
function IconImage( { id, size }: { readonly id: string; readonly size: number } ) {
  const [ url, setUrl ] = useState<string | null>( null );
  const drawn = drawnSize( size * ( window.devicePixelRatio || 1 ) );
  useEffect( () => {
    let live = true;
    iconUrl( id, drawn ).then(
      ( loaded ) => { if ( live ) { setUrl( loaded ); } },
      ( error: unknown ) => { log.warn( `app icon ${id} at ${String( drawn )} px: ${describeError( error )}` ); },
    );
    return () => { live = false; };
  }, [ id, drawn ] );
  const style = { inlineSize: size, blockSize: size };
  return url === null ? <span className="icon-image" style={style} /> : <img className="icon-image" src={url} alt="" style={style} />;
}

export function iconName( id: string ): string {
  const key = msgKey( 'app_icon.', id );
  return hasMessage( key ) ? t( key ) : id;
}

export function IconPicker( { entry, value, onChange, inputId, disabled }: EditorProps ) {
  const choices = ( entry.choices ?? [] ).filter( ( choice ): choice is string => typeof choice === 'string' );
  const shown = typeof value === 'string' ? value : choices[ 0 ];
  return (
    <div className="icon-picker">
      <div id={inputId} className="icon-choices" role="radiogroup" aria-labelledby={`${inputId}-label`}>
        {choices.map( ( choice ) => (
          <label key={choice} className={`icon-choice${choice === shown ? ' selected' : ''}`}>
            <input
              type="radio"
              name={`${inputId}-icon`}
              value={choice}
              checked={choice === shown}
              disabled={disabled}
              onChange={() => { onChange( choice ); }}
            />
            <IconImage id={choice} size={CHOICE_SIZE} />
            <span>{iconName( choice )}</span>
          </label>
        ) )}
      </div>
      {shown !== undefined && (
        <figure className="tray-preview">
          <figcaption>{t( 'settings.icon.tray_preview', { name: iconName( shown ) } )}</figcaption>
          {( [ [ 'light', t( 'settings.icon.panel_light' ) ], [ 'dark', t( 'settings.icon.panel_dark' ) ] ] as const ).map( ( [ tone, label ] ) => (
            <div key={tone} className={`tray-panel ${tone}`}>
              <span className="tray-panel-label">{label}</span>
              {TRAY_SIZES.map( ( size ) => (
                <span key={size} className="tray-slot">
                  <IconImage id={shown} size={size} />
                  <span className="tray-size">{t( 'settings.icon.size', { size } )}</span>
                </span>
              ) )}
            </div>
          ) )}
        </figure>
      )}
    </div>
  );
}

// The player's scrub strip (betautils_catalog_strip): small frames at known
// times in one image. useStrip asks the engine for it and follows its two
// passes: the keyframes pass first, then the even pass when the engine
// announces it (catalog.strip_ready).
import { useEffect, useState } from 'react';
import { onEvent } from '../engine/events';
import { describeError, log } from '../log';
import { strip as askStrip, stripImage } from '../library/api';

export interface LoadedStrip {
  readonly image: HTMLImageElement;
  readonly frames: readonly number[];
  readonly tileWidth: number;
  readonly tileHeight: number;
  readonly columns: number;
  readonly density: string;
}

// The index of the last frame at or before seconds (the first frame for an
// earlier time); -1 when there are no frames.
export function frameAt( frames: readonly number[], seconds: number ): number {
  if ( frames.length === 0 ) {
    return -1;
  }
  let low = 0;
  let high = frames.length - 1;
  while ( low < high ) {
    const middle = Math.ceil( ( low + high ) / 2 );
    if ( ( frames[ middle ] ?? 0 ) <= seconds ) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  return low;
}

// Where frame index sits in the image: its source rectangle.
export function tileRect( strip: Pick<LoadedStrip, 'tileWidth' | 'tileHeight' | 'columns'>, index: number ) {
  const row = Math.floor( index / strip.columns );
  const column = index % strip.columns;
  return { x: column * strip.tileWidth, y: row * strip.tileHeight, width: strip.tileWidth, height: strip.tileHeight };
}

function loadImage( url: string ): Promise<HTMLImageElement> {
  return new Promise( ( resolve, reject ) => {
    const image = new Image();
    image.onload = () => { resolve( image ); };
    image.onerror = () => { reject( new Error( 'strip image did not load' ) ); };
    image.src = url;
  } );
}

// The best strip for path so far, or null; a newer pass replaces it.
export function useStrip( path: string | null ): LoadedStrip | null {
  const [ loaded, setLoaded ] = useState<{ path: string; strip: LoadedStrip } | null>( null );
  const [ generation, setGeneration ] = useState( 0 );

  useEffect( () => onEvent( 'catalog.strip_ready', ( data ) => {
    if ( data[ 'path' ] === path && data[ 'ok' ] === true ) {
      setGeneration( ( value ) => value + 1 );
    }
  } ), [ path ] );

  useEffect( () => {
    if ( path === null ) {
      return;
    }
    const state = { live: true };
    const started = performance.now();
    const load = async (): Promise<LoadedStrip | null> => {
      const answer = await askStrip( path );
      if ( !answer.ready || answer.image === undefined || answer.frames === undefined || answer.tile === undefined ) {
        return null;
      }
      const image = await loadImage( await stripImage( answer.image ) );
      return {
        image, frames: answer.frames, tileWidth: answer.tile.width, tileHeight: answer.tile.height,
        columns: answer.columns ?? 1, density: answer.density ?? 'keyframes',
      };
    };
    load().then( ( result ) => {
      if ( state.live && result !== null ) {
        log.info( `scrub strip: ${result.density}, ${String( result.frames.length )} frame(s), ready ${String( Math.round( performance.now() - started ) )} ms after asking` );
        setLoaded( { path, strip: result } );
      }
    }, ( error: unknown ) => { log.warn( `scrub strip: ${describeError( error )}` ); } );
    return () => { state.live = false; };
  }, [ path, generation ] );

  return loaded !== null && loaded.path === path ? loaded.strip : null;
}

// Every preview the screen shows, drawn in one preview.look request and
// redrawn when the requests or the pending changes (sent as per-run
// overrides) change. A piece of art the style lacks (a label a newer model
// added) is left out and reported as missing, so the screen can show a
// placeholder instead of nothing.
import { useEffect, useState } from 'react';
import { isEngineError, type EngineError } from '../engine/client';
import { describeError, log } from '../log';
import { imageUrl, previewLook, type PreviewRequest, type PreviewResult } from './api';

export interface Drawn {
  readonly url: string;
  readonly result: PreviewResult;
}

export interface Previews {
  readonly drawn: ReadonlyMap<string, Drawn>;
  readonly missing: ReadonlySet<string>;
  // An engine refusal, 'unknown' for anything else, null when drawn.
  readonly error: EngineError | 'unknown' | null;
  readonly busy: boolean;
}

const DELAY_MS = 120;
const MAX_MISSING_RETRIES = 32;

function itemKey( request: PreviewRequest ): string {
  return `${request.art.style}/${request.art.item}`;
}

async function drawAll( requests: ReadonlyMap<string, PreviewRequest>, overrides: Readonly<Record<string, unknown>> ) {
  const missing = new Set<string>();
  for ( let attempt = 0; attempt <= MAX_MISSING_RETRIES; attempt++ ) {
    const ids = [ ...requests.keys() ].filter( ( id ) => {
      const request = requests.get( id );
      return request !== undefined && !missing.has( itemKey( request ) );
    } );
    if ( ids.length === 0 ) {
      return { drawn: new Map<string, Drawn>(), missing };
    }
    try {
      const results = await previewLook( ids.map( ( id ) => requests.get( id ) as PreviewRequest ), overrides );
      const drawn = new Map<string, Drawn>();
      await Promise.all( results.map( async ( result, index ) => {
        const id = ids[ index ];
        if ( id !== undefined ) {
          drawn.set( id, { url: await imageUrl( result ), result } );
        }
      } ) );
      return { drawn, missing };
    } catch ( error ) {
      const style = isEngineError( error ) ? error.params[ 'style' ] : undefined;
      const item = isEngineError( error ) ? error.params[ 'item' ] : undefined;
      if ( isEngineError( error ) && error.code === 'preview.unknown_art' && typeof style === 'string' && typeof item === 'string' ) {
        log.warn( `no art for ${style}/${item}; showing a placeholder` );
        missing.add( `${style}/${item}` );
        continue;
      }
      throw error;
    }
  }
  return { drawn: new Map<string, Drawn>(), missing };
}

type Answer = Omit<Previews, 'busy'> & { readonly signature: string | null };

export function usePreviews( requests: ReadonlyMap<string, PreviewRequest>, overrides: Readonly<Record<string, unknown>>, enabled: boolean ): Previews {
  const [ state, setState ] = useState<Answer>( { drawn: new Map(), missing: new Set(), error: null, signature: null } );
  // Requests and overrides are rebuilt on every render; compare their content.
  const signature = JSON.stringify( [ [ ...requests.entries() ], overrides ] );

  useEffect( () => {
    if ( !enabled ) {
      return undefined;
    }
    let live = true;
    const timer = window.setTimeout( () => {
      const started = performance.now();
      drawAll( requests, overrides ).then(
        ( { drawn, missing } ) => {
          if ( !live ) {
            return;
          }
          log.debug( `previews drawn: ${String( drawn.size )} in ${String( Math.round( performance.now() - started ) )} ms, ${String( missing.size )} missing` );
          // Missing items map to their request ids for the screen.
          const missingIds = new Set( [ ...requests.entries() ].filter( ( [ , request ] ) => missing.has( itemKey( request ) ) ).map( ( [ id ] ) => id ) );
          setState( { drawn, missing: missingIds, error: null, signature } );
        },
        ( error: unknown ) => {
          if ( !live ) {
            return;
          }
          log.warn( `previews failed: ${describeError( error )}` );
          // Keep the last pictures: a busy engine or a refused override
          // should not blank the screen.
          setState( ( previous ) => ( { ...previous, error: isEngineError( error ) ? error : 'unknown', signature } ) );
        },
      );
    }, DELAY_MS );
    return () => { live = false; window.clearTimeout( timer ); };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- signature stands for requests and overrides
  }, [ signature, enabled ] );

  // Busy while the pictures shown were drawn for other requests.
  return { drawn: state.drawn, missing: state.missing, error: state.error, busy: enabled && state.signature !== signature };
}

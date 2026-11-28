// The settings as React state: the registry, every value with its layers,
// and each label as a run resolves it. Loaded once the engine is ready and
// again after any change, since some choices follow other settings
// (encode_preset's follow the codec).
import { useCallback, useEffect, useState } from 'react';
import { isEngineError, type EngineError } from '../engine/client';
import { applyAppIcon } from '../appIcon';
import { useEngineStatus } from '../engine/status';
import { describeError, log } from '../log';
import { getSchema, getValues, type LabelView } from './api';
import { buildRows, valuesByPath, type Row, type Schema, type ValueRow } from './model';

export interface Loaded {
  readonly schema: Schema;
  readonly rows: readonly Row[];
  readonly values: ReadonlyMap<string, ValueRow>;
  readonly labels: ReadonlyMap<string, LabelView>;
}

export type SettingsState =
  | { readonly state: 'waiting' }
  | { readonly state: 'loading'; readonly last: Loaded | null }
  | { readonly state: 'ready'; readonly data: Loaded }
  | { readonly state: 'failed'; readonly error: EngineError | null };

async function load(): Promise<Loaded> {
  const [ schema, values ] = await Promise.all( [ getSchema(), getValues( { labels: true } ) ] );
  return {
    schema,
    rows: buildRows( schema ),
    values: valuesByPath( values.settings ),
    labels: new Map( ( values.labels ?? [] ).map( ( view ) => [ view.label, view ] ) ),
  };
}

// Which engine process and which request a result answers.
type Result =
  | { readonly key: string; readonly data: Loaded }
  | { readonly key: string; readonly error: EngineError | null };

export function useSettings(): { settings: SettingsState; reload: () => void } {
  const engine = useEngineStatus();
  const [ generation, setGeneration ] = useState( 0 );
  const [ result, setResult ] = useState<Result | null>( null );
  const [ last, setLast ] = useState<Loaded | null>( null );
  const ready = engine.state === 'ready';
  const key = ready ? `${String( engine.pid )}:${String( generation )}` : null;

  // Results arrive in callbacks; whether one is current is worked out
  // below, from the generation it answers.
  useEffect( () => {
    if ( key === null ) {
      return undefined;
    }
    let live = true;
    load().then(
      ( data ) => {
        if ( live ) {
          log.debug( `settings loaded: ${String( data.rows.length )} rows, ${String( data.values.size )} values` );
          setResult( { key, data } );
          applyAppIcon( ( path ) => data.values.get( path )?.value );
          setLast( data );
        }
      },
      ( error: unknown ) => {
        // A load the engine refused because it is stopping or restarting
        // (an apply that closes the window reloads just as the engine
        // shuts down), or one already superseded, is not a failure: the
        // status bar shows the engine, and the next ready engine loads again.
        if ( !live || ( isEngineError( error ) && error.code === 'ui.engine.not_ready' ) ) {
          log.debug( `settings not loaded: ${describeError( error )}` );
        } else {
          log.error( `settings could not load: ${describeError( error )}` );
        }
        if ( live ) {
          setResult( { key, error: isEngineError( error ) ? error : null } );
        }
      },
    );
    return () => { live = false; };
  }, [ key ] );

  const reload = useCallback( () => { setGeneration( ( value ) => value + 1 ); }, [] );
  let settings: SettingsState;
  if ( key === null ) {
    settings = { state: 'waiting' };
  } else if ( result === null || result.key !== key ) {
    settings = { state: 'loading', last };
  } else if ( 'data' in result ) {
    settings = { state: 'ready', data: result.data };
  } else {
    settings = { state: 'failed', error: result.error };
  }
  return { settings, reload };
}

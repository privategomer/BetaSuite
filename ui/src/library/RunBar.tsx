// Run and Preview for the chosen videos (spec section 4): the full run is
// the main action, a 20-second preview sits beside it and is suggested the
// first time a look is used. Both show their time estimate first. Each
// video is its own job, so the queue can reorder them one by one.
import { useEffect, useState } from 'react';
import { isEngineError, type EngineError } from '../engine/client';
import { onEvent } from '../engine/events';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { describeError, log } from '../log';
import { estimate, getItem, PREVIEW_SECONDS, submitJob, type Estimate } from './api';
import { refreshJobs } from './jobs';
import { formatSeconds } from './names';

interface RunBarProps {
  readonly items: readonly string[];
  readonly onSubmitted: () => void;
}

// The chosen items' files where they are now (the first present location).
async function presentPaths( items: readonly string[] ): Promise<string[]> {
  const details = await Promise.all( items.map( ( item ) => getItem( item ) ) );
  return details.flatMap( ( detail ) => {
    const found = detail.locations.find( ( location ) => location.availability === 'present' );
    return found === undefined ? [] : [ found.path ];
  } );
}

function estimateText( result: Estimate | null, failed: boolean ): string {
  if ( failed ) {
    return t( 'run.estimate_failed' );
  }
  if ( result === null ) {
    return t( 'run.estimating' );
  }
  if ( result.seconds !== null ) {
    return result.basis === 'probe'
      ? t( 'run.estimate_probe', { time: formatSeconds( result.seconds ) } )
      : t( 'run.estimate', { time: formatSeconds( result.seconds ) } );
  }
  return result.probing ? t( 'run.estimate_measuring' ) : t( 'run.estimate_none' );
}

interface Measured {
  readonly key: string;
  readonly paths: string[];
  readonly full: Estimate | null;
  readonly preview: Estimate | null;
  readonly failed: boolean;
}

export function RunBar( { items, onSubmitted }: RunBarProps ) {
  const [ measured, setMeasured ] = useState<Measured | null>( null );
  const [ generation, setGeneration ] = useState( 0 );
  const [ problem, setProblem ] = useState<EngineError | null>( null );
  const [ submitting, setSubmitting ] = useState( false );
  const key = `${items.join( ',' )}#${String( generation )}`;

  // A probe that ends makes the estimate worth asking again.
  useEffect( () => onEvent( 'job.probed', () => { setGeneration( ( value ) => value + 1 ); } ), [] );

  useEffect( () => {
    const state = { live: true };
    const chosen = key.split( '#' )[ 0 ] ?? '';
    const load = async (): Promise<Measured> => {
      const found = await presentPaths( chosen === '' ? [] : chosen.split( ',' ) );
      if ( found.length === 0 ) {
        return { key, paths: found, full: null, preview: null, failed: false };
      }
      const [ wholeRun, slice ] = await Promise.all( [ estimate( found, false ), estimate( found, true ) ] );
      return { key, paths: found, full: wholeRun, preview: slice, failed: false };
    };
    load().then( ( result ) => {
      if ( state.live ) {
        setMeasured( result );
      }
    }, ( error: unknown ) => {
      log.warn( `estimate failed: ${describeError( error )}` );
      if ( state.live ) {
        setMeasured( { key, paths: [], full: null, preview: null, failed: true } );
      }
    } );
    return () => { state.live = false; };
  }, [ key ] );

  // What was measured for the videos chosen now; null while it is worked out.
  const current = measured?.key === key ? measured : null;
  const paths = current?.paths ?? null;
  const full = current?.full ?? null;
  const preview = current?.preview ?? null;
  const failed = current?.failed ?? false;

  const run = ( asPreview: boolean ) => {
    if ( paths === null || paths.length === 0 ) {
      return;
    }
    setSubmitting( true );
    setProblem( null );
    const submitAll = async () => {
      for ( const path of paths ) {
        await submitJob( [ path ], asPreview );
      }
    };
    submitAll().then( () => {
      log.info( `${String( paths.length )} ${asPreview ? 'preview' : 'full'} job(s) submitted` );
      onSubmitted();
    }, ( error: unknown ) => {
      log.warn( `job.submit failed: ${describeError( error )}` );
      setProblem( isEngineError( error ) ? error : { code: 'unknown', params: {} } );
    } ).finally( () => {
      setSubmitting( false );
      void refreshJobs();
    } );
  };

  const nudge = full !== null && !full.look_seen;
  const none = paths !== null && paths.length === 0;
  return (
    <div className="run-bar" role="region" aria-label={t( 'run.region' )}>
      <span className="run-count">{t( 'run.chosen', { count: items.length } )}</span>
      <div className="run-action">
        <button type="button" className="primary" disabled={submitting || none} onClick={() => { run( false ); }}>
          {t( 'run.run' )}
        </button>
        <span className="run-estimate">{estimateText( full, failed )}</span>
      </div>
      <div className="run-action">
        <button type="button" className={nudge ? 'nudged' : undefined} disabled={submitting || none} onClick={() => { run( true ); }}>
          {t( 'run.preview', { seconds: PREVIEW_SECONDS } )}
        </button>
        <span className="run-estimate">{estimateText( preview, failed )}</span>
      </div>
      {nudge && <p className="run-nudge">{t( 'run.nudge', { seconds: PREVIEW_SECONDS } )}</p>}
      {none && <p className="run-problem">{t( 'run.none_present' )}</p>}
      {problem !== null && <p className="run-problem" role="alert">{errorText( problem )}</p>}
    </div>
  );
}

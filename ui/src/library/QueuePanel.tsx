// The queue (spec section 4): one job at a time, reorder, pause, resume and
// cancel. A running job shows its stage's progress and the time left at its
// current rate, the way the command line's progress line does. Jobs kept
// from the last session come back paused, marked as such.
import { useState } from 'react';
import { isEngineError, type EngineError } from '../engine/client';
import { errorText } from '../i18n/errors';
import { msgKey, t } from '../i18n/format';
import { describeError, log } from '../log';
import { jobCommand, moveJob, type JobRow } from './api';
import { refreshJobs, stageSecondsLeft, useQueue, type Progress } from './jobs';
import { formatSeconds } from './names';

function baseName( path: string ): string {
  const parts = path.split( /[\\/]/ );
  return parts[ parts.length - 1 ] ?? path;
}

function jobName( row: JobRow ): string {
  const first = baseName( row.files[ 0 ] ?? '' );
  return row.files.length > 1 ? t( 'queue.many_files', { name: first, more: row.files.length - 1 } ) : first;
}

function isPreview( row: JobRow ): boolean {
  return row.overrides[ 'preview_mode_enabled' ] === true;
}

function ProgressLine( { row, progress }: { row: JobRow; progress: Progress | undefined } ) {
  const done = row.counts.processed + row.counts.skipped + row.counts.failed;
  const left = stageSecondsLeft( progress );
  const fraction = progress?.total ? Math.min( 1, progress.done / progress.total ) : null;
  return (
    <div className="queue-progress">
      {fraction !== null && <progress max={1} value={fraction} aria-label={t( 'queue.progress_label' )} />}
      <span>
        {progress === undefined
          ? t( 'queue.files_done', { done, total: row.counts.total } )
          : t( 'queue.stage', {
            stage: t( msgKey( 'queue.stage.', progress.stage === '' ? 'other' : progress.stage ) ),
            file: progress.file + 1, total: row.counts.total,
          } )}
      </span>
      {left !== null && <span>{t( 'queue.left', { time: formatSeconds( left ) } )}</span>}
    </div>
  );
}

export function QueuePanel( { onPlay }: { onPlay: ( path: string ) => void } ) {
  const queue = useQueue();
  const [ problem, setProblem ] = useState<EngineError | null>( null );
  const live = queue.rows.filter( ( row ) => row.position !== null );
  const ended = queue.rows.filter( ( row ) => row.position === null ).slice( 0, 5 );
  const waiting = live.filter( ( row ) => row.state === 'queued' );

  const act = ( promise: Promise<void>, what: string ) => {
    setProblem( null );
    promise.catch( ( error: unknown ) => {
      log.warn( `${what} failed: ${describeError( error )}` );
      if ( isEngineError( error ) ) {
        setProblem( error );
      }
    } ).finally( () => { void refreshJobs(); } );
  };

  const move = ( row: JobRow, by: number ) => {
    if ( row.position !== null ) {
      act( moveJob( row.job, Math.max( 0, row.position + by ) ), 'job.move' );
    }
  };

  return (
    <section className="queue" aria-labelledby="queue-title">
      <h3 id="queue-title">{t( 'queue.title' )}</h3>
      {live.length === 0 && <p className="queue-empty">{t( 'queue.empty' )}</p>}
      <ol className="queue-list">
        {live.map( ( row ) => (
          <li key={row.job} className={`queue-row queue-${row.state}`}>
            <div className="queue-head">
              <span className="queue-name">{jobName( row )}</span>
              <span className="queue-tags">
                <span>{t( msgKey( 'queue.state.', row.state ) )}</span>
                {isPreview( row ) && <span>{t( 'queue.preview' )}</span>}
                {row.restored === true && <span>{t( 'queue.restored' )}</span>}
              </span>
            </div>
            {( row.state === 'running' || row.state === 'paused' ) && (
              <ProgressLine row={row} progress={queue.progress[ row.job ]} />
            )}
            <div className="queue-actions">
              {row.state === 'running' && (
                <button type="button" onClick={() => { act( jobCommand( 'job.pause', row.job ), 'job.pause' ); }}>{t( 'queue.pause' )}</button>
              )}
              {row.state === 'paused' && (
                <button type="button" className="primary" onClick={() => { act( jobCommand( 'job.resume', row.job ), 'job.resume' ); }}>{t( 'queue.resume' )}</button>
              )}
              {row.state === 'queued' && waiting.indexOf( row ) > 0 && (
                <button type="button" aria-label={t( 'queue.up_label', { name: jobName( row ) } )} onClick={() => { move( row, -1 ); }}>{t( 'queue.up' )}</button>
              )}
              {row.state === 'queued' && waiting.indexOf( row ) < waiting.length - 1 && (
                <button type="button" aria-label={t( 'queue.down_label', { name: jobName( row ) } )} onClick={() => { move( row, 1 ); }}>{t( 'queue.down' )}</button>
              )}
              <button type="button" onClick={() => { act( jobCommand( 'job.cancel', row.job ), 'job.cancel' ); }}>{t( 'queue.cancel' )}</button>
            </div>
          </li>
        ) )}
      </ol>
      {problem !== null && <p className="run-problem" role="alert">{errorText( problem )}</p>}
      {ended.length > 0 && (
        <details className="queue-ended">
          <summary>{t( 'queue.ended', { count: ended.length } )}</summary>
          <ul>
            {ended.map( ( row ) => (
              <li key={row.job} className="queue-tags">
                <span>{jobName( row )}</span>
                <span>{t( msgKey( 'queue.state.', row.state ) )}</span>
                {row.code !== undefined && <span>{errorText( { code: row.code, params: row.params ?? {} } )}</span>}
                {( queue.outputs[ row.job ] ?? [] ).map( ( output ) => (
                  <button key={output} type="button" aria-label={t( 'queue.play_label', { name: baseName( output ) } )}
                    onClick={() => { onPlay( output ); }}>{isPreview( row ) ? t( 'queue.play_preview' ) : t( 'queue.play' )}</button>
                ) )}
              </li>
            ) )}
          </ul>
        </details>
      )}
    </section>
  );
}

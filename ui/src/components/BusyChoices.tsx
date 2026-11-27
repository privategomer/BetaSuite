// What to do with a settings change while jobs run, instead of losing it:
// keep it for when the jobs are done, pause the job to apply it and carry
// on, or cancel the jobs and start them again with it. Shown wherever a
// change meets config.busy (betautils_serve_changes in the engine).
import { useState } from 'react';
import { isEngineError } from '../engine/client';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { describeError, log } from '../log';
import { changeAfterJobs, restartWithChange, type SettingsChange } from '../resources/api';
import { refreshPending } from '../resources/store';

export type BusyOutcome = 'after_jobs' | 'resume' | 'requeue';

interface BusyChoicesProps {
  readonly change: SettingsChange;
  // The change is taken care of: the caller can forget its pending edit.
  readonly onDone: ( outcome: BusyOutcome ) => void;
}

export function BusyChoices( { change, onDone }: BusyChoicesProps ) {
  const [ working, setWorking ] = useState( false );
  const [ confirmRequeue, setConfirmRequeue ] = useState( false );
  const [ problem, setProblem ] = useState<string | null>( null );

  const go = ( outcome: BusyOutcome ) => {
    setWorking( true );
    setProblem( null );
    const request = outcome === 'after_jobs' ? changeAfterJobs( change ) : restartWithChange( change, outcome );
    log.info( `settings change while jobs run: ${outcome}` );
    request.then(
      () => { refreshPending(); onDone( outcome ); },
      ( error: unknown ) => {
        log.error( `settings change (${outcome}) failed: ${describeError( error )}` );
        setProblem( isEngineError( error ) ? errorText( error ) : t( 'busy.failed' ) );
      },
    ).finally( () => { setWorking( false ); } );
  };

  return (
    <div className="busy-choices" role="group" aria-label={t( 'busy.title' )}>
      <p className="busy-choices-title">{t( 'busy.title' )}</p>
      <div className="busy-choices-actions">
        <button type="button" className="primary" disabled={working} onClick={() => { go( 'after_jobs' ); }}>
          {t( 'busy.after_jobs' )}
        </button>
        <button type="button" disabled={working} onClick={() => { go( 'resume' ); }}>{t( 'busy.resume' )}</button>
        {!confirmRequeue && (
          <button type="button" disabled={working} onClick={() => { setConfirmRequeue( true ); }}>{t( 'busy.requeue' )}</button>
        )}
      </div>
      <p className="busy-choices-note">{t( 'busy.note' )}</p>
      {confirmRequeue && (
        <div className="busy-choices-confirm" role="alert">
          <p>{t( 'busy.requeue_warning' )}</p>
          <button type="button" className="danger" disabled={working} onClick={() => { go( 'requeue' ); }}>
            {t( 'busy.requeue_confirm' )}
          </button>
          <button type="button" disabled={working} onClick={() => { setConfirmRequeue( false ); }}>{t( 'busy.requeue_back' )}</button>
        </div>
      )}
      {problem !== null && <p className="busy-choices-problem" role="alert">{problem}</p>}
    </div>
  );
}

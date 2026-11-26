// Closing the window while a job runs asks once: keep it running in the
// background (the window hides to the tray), or pause it and close. A
// paused queue is kept: the next start shows its jobs paused, ready to
// resume from where they stopped. Without a tray only pausing is offered.
import { useEffect, useRef, useState } from 'react';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { isEngineError } from '../engine/client';
import { describeError, log } from '../log';
import { hasTray, hideToTray } from './background';

interface RunCloseDialogProps {
  readonly onClose: () => void;
  readonly onStay: () => void;
}

export function RunCloseDialog( { onClose, onStay }: RunCloseDialogProps ) {
  const dialog = useRef<HTMLDialogElement>( null );
  const [ problem, setProblem ] = useState<string | null>( null );
  const tray = hasTray();

  useEffect( () => {
    const element = dialog.current;
    if ( element !== null && !element.open ) {
      element.showModal();
    }
    return () => { element?.close(); };
  }, [] );

  const background = () => {
    hideToTray().then( onStay, ( error: unknown ) => {
      log.warn( `hide_to_tray failed: ${describeError( error )}` );
      setProblem( isEngineError( error ) ? errorText( error ) : t( 'run_close.hide_failed' ) );
    } );
  };

  return (
    <dialog
      ref={dialog}
      className="unsaved-dialog"
      aria-labelledby="run-close-title"
      aria-describedby="run-close-body"
      onCancel={( event ) => { event.preventDefault(); onStay(); }}
    >
      <h2 id="run-close-title">{t( 'run_close.title' )}</h2>
      <p id="run-close-body">{tray ? t( 'run_close.body' ) : t( 'run_close.body_no_tray' )}</p>
      {problem !== null && <p className="unsaved-failed" role="alert">{problem}</p>}
      <div className="unsaved-actions">
        {tray && <button type="button" className="primary" onClick={background}>{t( 'run_close.background' )}</button>}
        <button type="button" className={tray ? undefined : 'primary'} onClick={onClose}>{t( 'run_close.pause' )}</button>
        <button type="button" onClick={onStay}>{t( 'run_close.stay' )}</button>
      </div>
    </dialog>
  );
}

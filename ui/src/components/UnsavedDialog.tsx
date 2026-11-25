// Asks what to do with unapplied changes before leaving their screen or
// closing the window: apply them, discard them, or stay. A modal dialog, so
// focus stays inside it; Escape stays.
import { useEffect, useRef, useState } from 'react';
import { t } from '../i18n/format';
import { CostNotice } from '../settings/SettingRow';
import type { Guard } from '../unsaved';

interface UnsavedDialogProps {
  readonly guard: Guard;
  readonly closing: boolean;
  readonly onDone: () => void;
  readonly onStay: () => void;
}

export function UnsavedDialog( { guard, closing, onDone, onStay }: UnsavedDialogProps ) {
  const dialog = useRef<HTMLDialogElement>( null );
  const [ applying, setApplying ] = useState( false );
  const [ failed, setFailed ] = useState( false );

  useEffect( () => {
    const element = dialog.current;
    if ( element !== null && !element.open ) {
      element.showModal();
    }
    return () => { element?.close(); };
  }, [] );

  const apply = () => {
    setApplying( true );
    guard.apply().then( ( ok ) => {
      if ( ok ) {
        onDone();
      } else {
        setFailed( true );
      }
    }, () => { setFailed( true ); } ).finally( () => { setApplying( false ); } );
  };

  return (
    <dialog
      ref={dialog}
      className="unsaved-dialog"
      aria-labelledby="unsaved-title"
      aria-describedby="unsaved-body"
      onCancel={( event ) => { event.preventDefault(); onStay(); }}
    >
      <h2 id="unsaved-title">{t( 'unsaved.title' )}</h2>
      <p id="unsaved-body">{closing ? t( 'unsaved.body_close', { count: guard.count } ) : t( 'unsaved.body_leave', { count: guard.count } )}</p>
      <CostNotice check={guard.check} />
      {failed && <p className="unsaved-failed" role="alert">{t( 'unsaved.apply_failed' )}</p>}
      <div className="unsaved-actions">
        <button type="button" className="primary" disabled={applying || guard.check.state !== 'ok'} onClick={apply}>{t( 'unsaved.apply' )}</button>
        <button type="button" disabled={applying} onClick={() => { guard.discard(); onDone(); }}>{t( 'unsaved.discard' )}</button>
        <button type="button" disabled={applying} onClick={onStay}>{t( 'unsaved.stay' )}</button>
      </div>
    </dialog>
  );
}

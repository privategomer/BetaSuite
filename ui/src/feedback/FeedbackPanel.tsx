// "Missed something" and "censored too much" under the player (spec
// section 4, feedback v1: capture only). Shown only when the engine says
// the file playing is a result. A press pauses the video, keeps the time
// it was pressed at, and asks what the mark is about: a label, or "not
// sure". The marks already made on this result are listed under it;
// choosing one seeks there, and each can be removed. Marks stay in the
// library catalog on this computer. The player keys it by path, so a new
// file starts it afresh.
import { useEffect, useRef, useState } from 'react';
import { isEngineError } from '../engine/client';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { labelName } from '../i18n/labels';
import { describeError, log } from '../log';
import { clock } from '../components/clock';
import { addMark, listMarks, removeMark, type Mark, type Marks, type Verdict } from './api';

interface FeedbackPanelProps {
  // The file playing.
  readonly path: string;
  // The exact time in the video now.
  readonly now: () => number;
  readonly pause: () => void;
  readonly seek: ( seconds: number ) => void;
}

interface Asking {
  readonly verdict: Verdict;
  readonly time: number;
}

function failure( reason: unknown ): string {
  return isEngineError( reason ) ? errorText( reason ) : t( 'feedback.error' );
}

function markText( mark: Mark ): string {
  return t( 'feedback.mark', {
    time: clock( mark.time ),
    verdict: mark.verdict,
    label: mark.label === null ? t( 'feedback.not_sure' ) : labelName( mark.label ),
  } );
}

export function FeedbackPanel( { path, now, pause, seek }: FeedbackPanelProps ) {
  const [ marks, setMarks ] = useState<Marks | null>( null );
  const [ asking, setAsking ] = useState<Asking | null>( null );
  const [ busy, setBusy ] = useState( false );
  const [ error, setError ] = useState<string | null>( null );
  const [ saved, setSaved ] = useState<string | null>( null );
  const firstChoice = useRef<HTMLButtonElement>( null );
  const opener = useRef<HTMLButtonElement | null>( null );

  useEffect( () => {
    let live = true;
    listMarks( path ).then( ( found ) => {
      if ( live ) {
        setMarks( found );
      }
    }, ( reason: unknown ) => {
      // No buttons then: feedback is never in the way of playing.
      log.warn( `feedback.list failed: ${describeError( reason )}` );
    } );
    return () => { live = false; };
  }, [ path ] );

  // While the picker is open: focus its first choice; Escape closes it.
  const isAsking = asking !== null;
  useEffect( () => {
    if ( !isAsking ) {
      return;
    }
    firstChoice.current?.focus();
    const onKey = ( event: KeyboardEvent ) => {
      if ( event.key === 'Escape' ) {
        event.preventDefault();
        setAsking( null );
        opener.current?.focus();
      }
    };
    window.addEventListener( 'keydown', onKey );
    return () => { window.removeEventListener( 'keydown', onKey ); };
  }, [ isAsking ] );

  if ( marks === null || marks.result === null ) {
    return null;
  }

  const ask = ( verdict: Verdict, button: HTMLButtonElement ) => {
    const time = now();
    pause();
    opener.current = button;
    setError( null );
    setSaved( null );
    setAsking( { verdict, time } );
  };

  const close = () => {
    setAsking( null );
    opener.current?.focus();
  };

  const choose = ( label: string | null ) => {
    if ( asking === null ) {
      return;
    }
    setBusy( true );
    addMark( path, asking.time, asking.verdict, label ).then( ( mark ) => {
      log.info( `feedback: ${mark.verdict} at ${mark.time.toFixed( 3 )} s (${mark.label ?? 'no label'})` );
      setMarks( { ...marks, records: [ ...marks.records, mark ].sort( ( a, b ) => a.time - b.time || a.id - b.id ) } );
      setSaved( t( 'feedback.saved', { time: clock( mark.time ) } ) );
      close();
    }, ( reason: unknown ) => {
      log.warn( `feedback.add failed: ${describeError( reason )}` );
      setError( failure( reason ) );
    } ).finally( () => { setBusy( false ); } );
  };

  const remove = ( mark: Mark ) => {
    setBusy( true );
    setSaved( null );
    removeMark( mark.id ).then( () => {
      setMarks( { ...marks, records: marks.records.filter( ( other ) => other.id !== mark.id ) } );
    }, ( reason: unknown ) => {
      log.warn( `feedback.remove failed: ${describeError( reason )}` );
      setError( failure( reason ) );
    } ).finally( () => { setBusy( false ); } );
  };

  return (
    <div className="feedback" role="group" aria-label={t( 'feedback.group' )}>
      <div className="feedback-actions">
        <button type="button" disabled={busy || asking !== null}
          onClick={( event ) => { ask( 'missed', event.currentTarget ); }}>{t( 'feedback.missed' )}</button>
        <button type="button" disabled={busy || asking !== null}
          onClick={( event ) => { ask( 'too_much', event.currentTarget ); }}>{t( 'feedback.too_much' )}</button>
        <span className="feedback-note">{t( 'feedback.note' )}</span>
        <span className="feedback-saved" role="status">{saved}</span>
      </div>
      {error !== null && <p className="feedback-error" role="alert">{error}</p>}
      {asking !== null && (
        <fieldset className="feedback-ask">
          <legend>{t( 'feedback.ask', { verdict: asking.verdict, time: clock( asking.time ) } )}</legend>
          <div className="feedback-labels">
            {marks.labels.map( ( label, index ) => (
              <button key={label} type="button" disabled={busy} ref={index === 0 ? firstChoice : undefined}
                onClick={() => { choose( label ); }}>{labelName( label )}</button>
            ) )}
            <button type="button" disabled={busy} onClick={() => { choose( null ); }}>{t( 'feedback.not_sure' )}</button>
            <button type="button" className="feedback-cancel" disabled={busy} onClick={close}>{t( 'feedback.cancel' )}</button>
          </div>
        </fieldset>
      )}
      {marks.records.length > 0 && (
        <ul className="feedback-marks" aria-label={t( 'feedback.marks' )}>
          {marks.records.map( ( mark ) => (
            <li key={mark.id}>
              <button type="button" className="feedback-mark" onClick={() => { seek( mark.time ); }}
                title={t( 'feedback.go_to', { time: clock( mark.time ) } )}>{markText( mark )}</button>
              <button type="button" className="feedback-remove" disabled={busy}
                aria-label={t( 'feedback.remove', { mark: markText( mark ) } )}
                onClick={() => { remove( mark ); }}>{t( 'feedback.remove_symbol' )}</button>
            </li>
          ) )}
        </ul>
      )}
    </div>
  );
}

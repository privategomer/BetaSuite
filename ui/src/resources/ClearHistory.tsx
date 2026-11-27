// "Clear history": like a browser's "clear browsing data". A time range,
// a box per kind of trace with its size and what clearing it means (the
// cheap traces ticked by default, work to redo unticked), and "Delete
// permanently", which starts as the storage_delete_permanently setting.
// Without it, files go to the trash, and the dialog says so plainly:
// they can still be opened there until it is emptied.
import { useEffect, useRef, useState } from 'react';
import { engineRequest, isEngineError } from '../engine/client';
import { errorText } from '../i18n/errors';
import { msgKey, t } from '../i18n/format';
import { describeError, log } from '../log';
import { sizeText } from './text';

export interface HistoryKind {
  readonly id: string;
  readonly default: boolean;
  readonly confirm: boolean;
  readonly ranged: boolean;
  readonly bytes: number;
  readonly files: number;
  readonly rows: number;
}

interface HistoryAnswer {
  readonly kinds: readonly HistoryKind[];
  readonly busy: boolean;
  readonly permanent: boolean;
}

export interface Cleared {
  readonly bytes: number;
  readonly trashed: boolean;
}

const RANGES = [ 'hour', 'day', 'week', 'all' ] as const;
type Range = ( typeof RANGES )[ number ];

function amount( kind: HistoryKind ): string {
  if ( kind.rows > 0 && kind.bytes === 0 ) {
    return t( 'history.records', { count: kind.rows } );
  }
  if ( kind.rows > 0 ) {
    return t( 'history.size_and_records', { size: sizeText( kind.bytes ), count: kind.rows } );
  }
  return sizeText( kind.bytes );
}

export function ClearHistory( { onClose }: { readonly onClose: ( cleared: Cleared | null ) => void } ) {
  const dialog = useRef<HTMLDialogElement>( null );
  const [ range, setRange ] = useState<Range>( 'all' );
  const [ answer, setAnswer ] = useState<HistoryAnswer | null>( null );
  const [ chosen, setChosen ] = useState<ReadonlySet<string> | null>( null );
  const [ permanent, setPermanent ] = useState<boolean | null>( null );
  const [ working, setWorking ] = useState( false );
  const [ problem, setProblem ] = useState<string | null>( null );

  useEffect( () => {
    const element = dialog.current;
    if ( element !== null && !element.open ) {
      element.showModal();
    }
    return () => { element?.close(); };
  }, [] );

  useEffect( () => {
    let live = true;
    engineRequest( 'storage.history', { range } ).then(
      ( found ) => {
        if ( !live ) {
          return;
        }
        const history = found as HistoryAnswer;
        setAnswer( history );
        setChosen( ( was ) => was ?? new Set( history.kinds.filter( ( kind ) => kind.default ).map( ( kind ) => kind.id ) ) );
        setPermanent( ( was ) => was ?? history.permanent );
      },
      ( error: unknown ) => {
        log.error( `storage.history failed: ${describeError( error )}` );
        if ( live ) {
          setProblem( isEngineError( error ) ? errorText( error ) : t( 'storage.request_failed' ) );
        }
      },
    );
    return () => { live = false; };
  }, [ range ] );

  const picked = answer?.kinds.filter( ( kind ) => chosen?.has( kind.id ) === true ) ?? [];
  const total = picked.reduce( ( sum, kind ) => sum + kind.bytes, 0 );
  const costly = picked.filter( ( kind ) => kind.confirm );
  const removesRows = picked.some( ( kind ) => kind.rows > 0 );

  const toggle = ( id: string, on: boolean ) => {
    setChosen( ( was ) => {
      const next = new Set( was ?? [] );
      if ( on ) {
        next.add( id );
      } else {
        next.delete( id );
      }
      return next;
    } );
  };

  const clear = () => {
    setWorking( true );
    setProblem( null );
    const kinds = picked.map( ( kind ) => kind.id );
    log.info( `storage: clear history ${range}: ${kinds.join( ', ' )}${permanent === true ? ' (permanently)' : ''}` );
    engineRequest( 'storage.clear_history', { kinds, range, confirm: costly.length > 0, permanent: permanent === true } ).then(
      ( done ) => {
        const result = done as { cleared: readonly { bytes: number }[]; trashed: boolean };
        onClose( { bytes: result.cleared.reduce( ( sum, row ) => sum + row.bytes, 0 ), trashed: result.trashed } );
      },
      ( error: unknown ) => {
        log.error( `storage.clear_history failed: ${describeError( error )}` );
        setProblem( isEngineError( error ) ? errorText( error ) : t( 'storage.request_failed' ) );
        setWorking( false );
      },
    );
  };

  return (
    <dialog
      ref={dialog}
      className="unsaved-dialog history-dialog"
      aria-labelledby="history-title"
      onCancel={( event ) => { event.preventDefault(); if ( !working ) { onClose( null ); } }}
    >
      <h2 id="history-title">{t( 'history.title' )}</h2>
      <label className="history-range">
        {t( 'history.when' )}
        <select value={range} disabled={working} onChange={( event ) => { setRange( event.target.value as Range ); }}>
          {RANGES.map( ( name ) => <option key={name} value={name}>{t( msgKey( 'history.range.', name ) )}</option> )}
        </select>
      </label>
      {answer === null && problem === null && <p role="status">{t( 'storage.loading' )}</p>}
      {answer !== null && (
        <ul className="history-kinds">
          {answer.kinds.map( ( kind ) => (
            <li key={kind.id}>
              <label>
                <input
                  type="checkbox"
                  checked={chosen?.has( kind.id ) === true}
                  disabled={working}
                  onChange={( event ) => { toggle( kind.id, event.target.checked ); }}
                />
                <span className="history-name">{t( msgKey( 'history.kind.', `${kind.id}.name` ) )}</span>
                {' '}
                <span className="history-amount">{t( 'history.amount', { amount: amount( kind ) } )}</span>
              </label>
              <p className="history-what">
                {t( msgKey( 'history.kind.', `${kind.id}.what` ) )}
                {!kind.ranged && range !== 'all' && <> {t( 'history.all_time_only' )}</>}
              </p>
            </li>
          ) )}
        </ul>
      )}
      {answer !== null && (
        <div className="history-permanent">
          <label>
            <input type="checkbox" checked={permanent === true} disabled={working}
              onChange={( event ) => { setPermanent( event.target.checked ); }} />
            {t( 'history.permanent' )}
          </label>
          <p className="history-what">{permanent === true ? t( 'history.permanent_on' ) : t( 'history.permanent_off' )}</p>
          {removesRows && <p className="history-what">{t( 'history.records_gone' )}</p>}
        </div>
      )}
      {costly.length > 0 && <p className="history-costly" role="note">{t( 'history.costly', { count: costly.length } )}</p>}
      {answer?.busy === true && <p className="unsaved-failed" role="alert">{t( 'history.busy' )}</p>}
      {problem !== null && <p className="unsaved-failed" role="alert">{problem}</p>}
      <div className="unsaved-actions">
        <button type="button" className="primary"
          disabled={working || answer === null || answer.busy || picked.length === 0}
          onClick={clear}
        >
          {permanent === true ? t( 'history.delete', { size: sizeText( total ) } ) : t( 'history.trash', { size: sizeText( total ) } )}
        </button>
        <button type="button" disabled={working} onClick={() => { onClose( null ); }}>{t( 'history.cancel' )}</button>
      </div>
    </dialog>
  );
}

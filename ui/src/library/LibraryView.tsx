// The library (spec section 3): every video the catalog points at, with its
// blurred thumbnail, status, video type and why, censors applied, results,
// size and when it was added. Sorted and filtered by the engine, a page at
// a time, kept live by catalog.changed. Videos come in by the add buttons,
// a watched folder, or a drop on the window.
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { isEngineError, type EngineError } from '../engine/client';
import { onEvent } from '../engine/events';
import { useEngineStatus } from '../engine/status';
import { errorText } from '../i18n/errors';
import { msgKey, t } from '../i18n/format';
import { labelName } from '../i18n/labels';
import { describeError, log } from '../log';
import {
  addFiles, addFolder, listFolders, listItems, pickFiles, pickFolder, removeFolder, rescan, videoTypes,
  type Filters, type Folder, type Item, type OutputState, type Sort, type Status,
} from './api';
import { ItemDrawer } from './ItemDrawer';
import { formatBytes, lookName, statusName, typeName, typeWhy, VIDEO_EXTENSIONS } from './names';
import { QueuePanel } from './QueuePanel';
import { RunBar } from './RunBar';
import { Thumb } from './Thumb';

const PAGE = 100;
const STATUSES: readonly Status[] = [ 'new', 'queued', 'processing', 'done', 'failed' ];
const OUTPUTS: readonly OutputState[] = [ 'present', 'missing', 'none' ];
const SORTS: readonly Sort[] = [ 'name', 'added_at', 'size', 'status' ];

interface LibraryViewProps {
  readonly onPlay: ( path: string ) => void;
}

function dateText( seconds: number ): string {
  return new Intl.DateTimeFormat( document.documentElement.lang || undefined, { dateStyle: 'medium' } )
    .format( new Date( seconds * 1000 ) );
}

function FoldersPanel( { onProblem }: { onProblem: ( error: EngineError ) => void } ) {
  const [ folders, setFolders ] = useState<readonly Folder[]>( [] );
  const [ byFolder, setByFolder ] = useState( false );
  const [ generation, setGeneration ] = useState( 0 );

  useEffect( () => onEvent( 'catalog.changed', () => { setGeneration( ( value ) => value + 1 ); } ), [] );
  useEffect( () => {
    listFolders().then( setFolders, ( error: unknown ) => { log.warn( `catalog.folders: ${describeError( error )}` ); } );
  }, [ generation ] );

  const fail = ( error: unknown ) => {
    log.warn( `folder change failed: ${describeError( error )}` );
    if ( isEngineError( error ) ) {
      onProblem( error );
    }
  };
  const add = () => {
    pickFolder( t( 'library.folder_dialog' ) ).then( async ( path ) => {
      if ( path !== null ) {
        await addFolder( path, byFolder );
        setGeneration( ( value ) => value + 1 );
      }
    } ).catch( fail );
  };

  return (
    <section className="folders" aria-labelledby="folders-title">
      <h3 id="folders-title">{t( 'library.folders' )}</h3>
      <ul>
        {folders.map( ( folder ) => (
          <li key={folder.folder}>
            <span className="item-path">{folder.path}</span>
            <span className="item-why">
              {folder.available ? t( 'library.folder_items', { count: folder.items } ) : t( 'library.folder_offline' )}
              {folder.type_from_folder && <span>{t( 'library.folder_types' )}</span>}
            </span>
            <button type="button" onClick={() => {
              removeFolder( folder.folder ).then( () => { setGeneration( ( value ) => value + 1 ); }, fail );
            }}>{t( 'library.folder_remove' )}</button>
          </li>
        ) )}
      </ul>
      <label className="folders-option">
        <input type="checkbox" checked={byFolder} onChange={( event ) => { setByFolder( event.target.checked ); }} />
        <span>{t( 'library.folder_by_name' )}</span>
      </label>
      <div className="folders-actions">
        <button type="button" onClick={add}>{t( 'library.add_folder' )}</button>
        <button type="button" onClick={() => { rescan().catch( fail ); }}>{t( 'library.rescan' )}</button>
      </div>
    </section>
  );
}

export function LibraryView( { onPlay }: LibraryViewProps ) {
  const engine = useEngineStatus();
  const [ items, setItems ] = useState<readonly Item[]>( [] );
  const [ total, setTotal ] = useState( 0 );
  const [ pending, setPending ] = useState( 0 );
  const [ shown, setShown ] = useState( PAGE );
  const [ sort, setSort ] = useState<Sort>( 'added_at' );
  const [ descending, setDescending ] = useState( true );
  const [ status, setStatus ] = useState<Status | ''>( '' );
  const [ type, setType ] = useState( '' );
  const [ output, setOutput ] = useState<OutputState | ''>( '' );
  const [ label, setLabel ] = useState( '' );
  const [ look, setLook ] = useState( '' );
  const [ types, setTypes ] = useState<readonly string[]>( [] );
  const [ chosen, setChosen ] = useState<ReadonlySet<string>>( new Set() );
  const [ open, setOpen ] = useState<string | null>( null );
  const [ folders, setFolders ] = useState( false );
  const [ problem, setProblem ] = useState<EngineError | null>( null );
  const [ notice, setNotice ] = useState<string | null>( null );
  const [ generation, setGeneration ] = useState( 0 );
  const [ versions, setVersions ] = useState<Readonly<Record<string, number>>>( {} );
  const ready = engine.state === 'ready';

  const filter = useMemo( (): Filters => {
    const out: Filters = {};
    if ( status !== '' ) { out.status = [ status ]; }
    if ( type !== '' ) { out.type = [ type ]; }
    if ( output !== '' ) { out.output = [ output ]; }
    if ( label !== '' || look !== '' ) {
      out.censored = {};
      if ( label !== '' ) { out.censored.labels = [ label ]; }
      if ( look !== '' ) { out.censored.looks = [ look ]; }
    }
    return out;
  }, [ status, type, output, label, look ] );

  // The catalog changed: list again soon, a few times a second at most.
  useEffect( () => {
    let timer: number | undefined;
    const off = onEvent( 'catalog.changed', ( data ) => {
      const changed = data[ 'items' ];
      if ( Array.isArray( changed ) ) {
        setVersions( ( current ) => {
          const next = { ...current };
          for ( const id of changed ) {
            if ( typeof id === 'string' ) {
              next[ id ] = ( next[ id ] ?? 0 ) + 1;
            }
          }
          return next;
        } );
      }
      window.clearTimeout( timer );
      timer = window.setTimeout( () => { setGeneration( ( value ) => value + 1 ); }, 300 );
    } );
    return () => { off(); window.clearTimeout( timer ); };
  }, [] );

  useEffect( () => {
    if ( !ready ) {
      return;
    }
    videoTypes().then( setTypes, ( error: unknown ) => { log.warn( `video types: ${describeError( error )}` ); } );
  }, [ ready ] );

  useEffect( () => {
    if ( !ready ) {
      return;
    }
    let live = true;
    listItems( filter, sort, descending, 0, shown ).then( ( page ) => {
      if ( live ) {
        setItems( page.items );
        setTotal( page.total );
        setPending( page.pending );
      }
    }, ( error: unknown ) => {
      log.warn( `catalog.list failed: ${describeError( error )}` );
      if ( live && isEngineError( error ) ) {
        setProblem( error );
      }
    } );
    return () => { live = false; };
  }, [ ready, filter, sort, descending, shown, generation ] );

  const add = useCallback( ( paths: readonly string[] ) => {
    if ( paths.length === 0 ) {
      return;
    }
    setProblem( null );
    addFiles( paths ).then( async ( result ) => {
      // A dropped folder is not a file: watch it instead.
      const folderPaths = result.refused.filter( ( refusal ) => refusal.reason === 'not_a_file' )
        .map( ( refusal ) => paths[ refusal.index ] ).filter( ( path ): path is string => path !== undefined );
      for ( const path of folderPaths ) {
        await addFolder( path, false );
      }
      const refused = result.refused.length - folderPaths.length;
      setNotice( t( 'library.added', { files: result.accepted, folders: folderPaths.length, refused } ) );
    } ).catch( ( error: unknown ) => {
      log.warn( `adding failed: ${describeError( error )}` );
      if ( isEngineError( error ) ) {
        setProblem( error );
      }
    } );
  }, [] );

  // Files and folders dropped on the window (Tauri's own drag-drop events).
  useEffect( () => {
    const off = getCurrentWebview().onDragDropEvent( ( event ) => {
      if ( event.payload.type === 'drop' ) {
        log.info( `${String( event.payload.paths.length )} path(s) dropped` );
        add( event.payload.paths );
      }
    } );
    return () => { off.then( ( stop ) => { stop(); }, () => undefined ); };
  }, [ add ] );

  const chooseFiles = () => {
    pickFiles( t( 'library.files_dialog' ), t( 'library.files_filter' ), VIDEO_EXTENSIONS ).then( add, ( error: unknown ) => {
      log.warn( `pick_files failed: ${describeError( error )}` );
    } );
  };

  const toggle = ( item: string ) => {
    setChosen( ( current ) => {
      const next = new Set( current );
      if ( next.has( item ) ) {
        next.delete( item );
      } else {
        next.add( item );
      }
      return next;
    } );
  };
  const allChosen = items.length > 0 && items.every( ( item ) => chosen.has( item.item ) );
  const labels = useMemo( () => [ ...new Set( [ ...items.flatMap( ( item ) => item.censors.map( ( censor ) => censor.label ) ),
    ...( label === '' ? [] : [ label ] ) ] ) ].sort(), [ items, label ] );
  // Kinds of look are data too: those the listed videos' results used.
  const looks = useMemo( () => [ ...new Set( [ ...items.flatMap( ( item ) => item.censors.flatMap( ( censor ) => censor.looks ) ),
    ...( look === '' ? [] : [ look ] ) ] ) ].sort(), [ items, look ] );

  return (
    <section className="view library" aria-labelledby="library-title">
      <div className="library-head">
        <h2 id="library-title">{t( 'library.title' )}</h2>
        <div className="library-actions">
          <button type="button" className="primary" onClick={chooseFiles}>{t( 'library.add_files' )}</button>
          <button type="button" aria-expanded={folders} onClick={() => { setFolders( !folders ); }}>{t( 'library.folders' )}</button>
        </div>
      </div>
      <p className="library-hint">{t( 'library.drop_hint' )}</p>
      {notice !== null && <p role="status">{notice}</p>}
      {problem !== null && <p className="run-problem" role="alert">{errorText( problem )}</p>}
      {folders && <FoldersPanel onProblem={setProblem} />}

      <div className="library-filters" role="group" aria-label={t( 'library.filters' )}>
        <label>
          <span>{t( 'library.col.status' )}</span>
          <select value={status} onChange={( event ) => { setStatus( event.target.value as Status | '' ); }}>
            <option value="">{t( 'library.any' )}</option>
            {STATUSES.map( ( value ) => <option key={value} value={value}>{statusName( value )}</option> )}
          </select>
        </label>
        <label>
          <span>{t( 'library.col.type' )}</span>
          <select value={type} onChange={( event ) => { setType( event.target.value ); }}>
            <option value="">{t( 'library.any' )}</option>
            <option value="auto">{t( 'type.auto' )}</option>{/* catalog-ignore: the engine's filter token for no pinned type */}
            {types.map( ( value ) => <option key={value} value={value}>{typeName( value )}</option> )}
          </select>
        </label>
        <label>
          <span>{t( 'library.col.results' )}</span>
          <select value={output} onChange={( event ) => { setOutput( event.target.value as OutputState | '' ); }}>
            <option value="">{t( 'library.any' )}</option>
            {OUTPUTS.map( ( value ) => <option key={value} value={value}>{t( msgKey( 'library.output.', value ) )}</option> )}
          </select>
        </label>
        <label>
          <span>{t( 'library.filter_label' )}</span>
          <select value={label} onChange={( event ) => { setLabel( event.target.value ); }}>
            <option value="">{t( 'library.any' )}</option>
            {labels.map( ( value ) => <option key={value} value={value}>{labelName( value )}</option> )}
          </select>
        </label>
        <label>
          <span>{t( 'library.filter_look' )}</span>
          <select value={look} onChange={( event ) => { setLook( event.target.value ); }}>
            <option value="">{t( 'library.any' )}</option>
            {looks.map( ( value ) => <option key={value} value={value}>{lookName( value )}</option> )}
          </select>
        </label>
        <label>
          <span>{t( 'library.sort' )}</span>
          <select value={sort} onChange={( event ) => { setSort( event.target.value as Sort ); }}>
            {SORTS.map( ( value ) => <option key={value} value={value}>{t( msgKey( 'library.sort.', value ) )}</option> )}
          </select>
        </label>
        <button type="button" aria-pressed={descending} onClick={() => { setDescending( !descending ); }}>
          {descending ? t( 'library.descending' ) : t( 'library.ascending' )}
        </button>
      </div>

      <div className="library-body">
        <div className="library-table-wrap">
          <table className="library-table">
            <caption className="visually-hidden">
              {t( 'library.caption', { shown: items.length, total } )}
            </caption>
            <thead>
              <tr>
                <th scope="col">
                  <input
                    type="checkbox"
                    aria-label={t( 'library.choose_all' )}
                    checked={allChosen}
                    onChange={() => { setChosen( allChosen ? new Set() : new Set( items.map( ( item ) => item.item ) ) ); }}
                  />
                </th>
                <th scope="col"><span className="visually-hidden">{t( 'library.col.thumb' )}</span></th>
                <th scope="col">{t( 'library.col.name' )}</th>
                <th scope="col">{t( 'library.col.status' )}</th>
                <th scope="col">{t( 'library.col.type' )}</th>
                <th scope="col">{t( 'library.col.censors' )}</th>
                <th scope="col">{t( 'library.col.results' )}</th>
                <th scope="col">{t( 'library.col.size' )}</th>
                <th scope="col">{t( 'library.col.added' )}</th>
              </tr>
            </thead>
            <tbody>
              {items.map( ( item ) => (
                <tr key={item.item} aria-selected={chosen.has( item.item )} className={open === item.item ? 'is-open' : undefined}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={t( 'library.choose', { name: item.name } )}
                      checked={chosen.has( item.item )}
                      onChange={() => { toggle( item.item ); }}
                    />
                  </td>
                  <td><Thumb item={item.item} name={item.name} version={versions[ item.item ] ?? 0} /></td>
                  <td>
                    <button type="button" className="link" onClick={() => { setOpen( item.item ); }}>{item.name}</button>
                    {item.availability !== 'present' && (
                      <span className="item-why">{t( msgKey( 'availability.', item.availability ) )}</span>
                    )}
                  </td>
                  <td>{statusName( item.status )}</td>
                  <td>
                    <span>{typeName( item.type.profile )}</span>
                    <span className="item-why">{typeWhy( item.type )}</span>
                  </td>
                  <td>
                    {item.censors.map( ( censor ) => (
                      <span key={censor.label} className="chip" title={censor.looks.map( lookName ).join( ', ' )}>
                        {labelName( censor.label )}
                      </span>
                    ) )}
                  </td>
                  <td>
                    {item.outputs > 0 && t( 'library.results_count', { count: item.outputs } )}
                    {item.outputs_missing > 0 && <span className="item-why">{t( 'library.results_missing', { count: item.outputs_missing } )}</span>}
                  </td>
                  <td>{formatBytes( item.size )}</td>
                  <td>{dateText( item.added_at )}</td>
                </tr>
              ) )}
            </tbody>
          </table>
          {items.length === 0 && ready && <p className="queue-empty">{total === 0 && pending === 0 ? t( 'library.empty' ) : t( 'library.none_match' )}</p>}
          {pending > 0 && <p role="status">{t( 'library.pending', { count: pending } )}</p>}
          {items.length < total && (
            <button type="button" onClick={() => { setShown( shown + PAGE ); }}>{t( 'library.more', { count: total - items.length } )}</button>
          )}
        </div>
        <div className="library-side">
          {chosen.size > 0 && <RunBar items={[ ...chosen ]} onSubmitted={() => { setChosen( new Set() ); }} />}
          <QueuePanel />
        </div>
      </div>
      {open !== null && <ItemDrawer item={open} onPlay={onPlay} onClose={() => { setOpen( null ); }} />}
    </section>
  );
}

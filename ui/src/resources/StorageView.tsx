// The Storage view: what the app's files use on disk, what clearing each
// kind costs, safe clean-up, free space, and where the cache folder is.
// Detections, shot cuts, benchmarks and unfinished renders ask first,
// since clearing them means redoing work; outputs, models, the catalog,
// logs and stats are listed only.
import { useCallback, useEffect, useState } from 'react';
import { engineRequest, isEngineError } from '../engine/client';
import { useEngineStatus } from '../engine/status';
import { errorText } from '../i18n/errors';
import { msgKey, t } from '../i18n/format';
import { describeError, log } from '../log';
import { cleanStorage, storageUsage, type Category, type StorageUsage } from './api';
import { categoryName, kindName, policyText, sizeText } from './text';

type Notice = { readonly tone: 'ok' | 'problem'; readonly text: string } | null;
const CLEANABLE = new Set( [ 'safe', 'costly', 'resume' ] );
const ASKS = new Set( [ 'costly', 'resume' ] );

function Row( { category, busy, onClean }: { category: Category; busy: boolean; onClean: ( category: Category ) => void } ) {
  const [ asking, setAsking ] = useState( false );
  const cleanable = CLEANABLE.has( category.policy ) && category.bytes > 0;
  return (
    <li className={`storage-row policy-${category.policy}`}>
      <div className="storage-head">
        <span className="storage-name">{categoryName( category )}</span>
        <span className="storage-size">{sizeText( category.bytes )}</span>
        <span className="model-note">{t( 'storage.files', { count: category.files } )}</span>
      </div>
      <p className="model-note">{policyText( category )}</p>
      {cleanable && !asking && (
        <button type="button" disabled={busy} onClick={() => { if ( ASKS.has( category.policy ) ) { setAsking( true ); } else { onClean( category ); } }}>
          {t( 'storage.clean' )}
        </button>
      )}
      {asking && (
        <div className="busy-choices-confirm" role="alert">
          <p>{t( 'storage.confirm', { name: categoryName( category ), size: sizeText( category.bytes ) } )}</p>
          <button type="button" className="danger" disabled={busy} onClick={() => { setAsking( false ); onClean( category ); }}>
            {t( 'storage.confirm_clean' )}
          </button>
          <button type="button" disabled={busy} onClick={() => { setAsking( false ); }}>{t( 'storage.confirm_back' )}</button>
        </div>
      )}
    </li>
  );
}

function CacheFolder( { usage, onChanged }: { usage: StorageUsage; onChanged: () => void } ) {
  const [ folder, setFolder ] = useState( '' );
  const [ notice, setNotice ] = useState<Notice>( null );
  const save = ( value: string | null ) => {
    const request = value === null
      ? engineRequest( 'config.reset', { paths: [ 'cache_dir' ] } )
      : engineRequest( 'config.set', { values: { cache_dir: value } } );
    request.then(
      () => { setNotice( { tone: 'ok', text: t( 'storage.cache.saved' ) } ); onChanged(); },
      ( error: unknown ) => {
        log.error( `cache_dir change failed: ${describeError( error )}` );
        setNotice( { tone: 'problem', text: isEngineError( error ) ? errorText( error ) : t( 'storage.request_failed' ) } );
      },
    );
  };
  return (
    <section className="perf-section" aria-labelledby="storage-cache">
      <h3 id="storage-cache">{t( 'storage.cache.title' )}</h3>
      <p className="view-intro">{t( msgKey( 'storage.cache.source_', usage.cache_source ) )}</p>
      {usage.cache_source !== 'variable' && (
        <div className="storage-cache">
          <label>
            {t( 'storage.cache.label' )}
            <input type="text" value={folder} onChange={( event ) => { setFolder( event.target.value ); }} />
          </label>
          <button type="button" className="primary" disabled={folder.trim() === ''} onClick={() => { save( folder.trim() ); }}>
            {t( 'storage.cache.move' )}
          </button>
          {usage.cache_source === 'setting' && (
            <button type="button" onClick={() => { save( null ); }}>{t( 'storage.cache.default' )}</button>
          )}
        </div>
      )}
      <p className="model-note">{t( 'storage.cache.restart' )}</p>
      {notice !== null && <p className={`models-result ${notice.tone}`} role={notice.tone === 'problem' ? 'alert' : 'status'}>{notice.text}</p>}
    </section>
  );
}

export function StorageView() {
  const engine = useEngineStatus();
  const [ usage, setUsage ] = useState<StorageUsage | null>( null );
  const [ working, setWorking ] = useState( false );
  const [ notice, setNotice ] = useState<Notice>( null );

  const refresh = useCallback( () => {
    storageUsage().then( setUsage, ( error: unknown ) => {
      log.error( `storage.usage failed: ${describeError( error )}` );
      setNotice( { tone: 'problem', text: isEngineError( error ) ? errorText( error ) : t( 'storage.request_failed' ) } );
    } );
  }, [] );
  useEffect( () => { if ( engine.state === 'ready' ) { refresh(); } }, [ engine.state, refresh ] );

  const clean = ( category: Category ) => {
    setWorking( true );
    setNotice( null );
    log.info( `storage: cleaning ${category.id}` );
    cleanStorage( [ category.id ], ASKS.has( category.policy ) ).then(
      ( answer ) => {
        const freed = answer.freed.reduce( ( sum, row ) => sum + row.bytes, 0 );
        setNotice( { tone: 'ok', text: answer.trashed ? t( 'storage.trashed', { size: sizeText( freed ) } )
          : t( 'storage.freed', { size: sizeText( freed ) } ) } );
      },
      ( error: unknown ) => {
        log.error( `storage.clean failed: ${describeError( error )}` );
        setNotice( { tone: 'problem', text: isEngineError( error ) ? errorText( error ) : t( 'storage.request_failed' ) } );
      },
    ).finally( () => { setWorking( false ); refresh(); } );
  };

  if ( usage === null ) {
    return (
      <section className="view storage-view" aria-labelledby="storage-title">
        <h2 id="storage-title" className="models-title">{t( 'storage.title' )}</h2>
        <p className="settings-wait" role="status">{notice?.text ?? t( 'storage.loading' )}</p>
      </section>
    );
  }
  const volumes = new Map<number, { free: number; total: number; kinds: string[] }>();
  for ( const disk of usage.disks ) {
    const entry = volumes.get( disk.volume ) ?? { free: disk.free_bytes, total: disk.total_bytes, kinds: [] };
    entry.kinds.push( kindName( disk.kind ) );
    volumes.set( disk.volume, entry );
  }
  return (
    <section className="view storage-view" aria-labelledby="storage-title">
      <h2 id="storage-title" className="models-title">{t( 'storage.title' )}</h2>
      <p className="view-intro">{t( 'storage.intro' )}</p>
      {usage.busy && <p className="models-notice">{t( 'storage.busy' )}</p>}
      {notice !== null && <p className={`models-result ${notice.tone}`} role={notice.tone === 'problem' ? 'alert' : 'status'}>{notice.text}</p>}
      <section className="perf-section" aria-labelledby="storage-disks">
        <h3 id="storage-disks">{t( 'storage.disks' )}</h3>
        <ul className="storage-disks">
          {[ ...volumes.entries() ].map( ( [ volume, disk ] ) => (
            <li key={volume}>
              <meter min={0} max={disk.total} value={disk.total - disk.free} aria-label={t( 'storage.disk_used' )} />
              {' '}
              {t( 'storage.disk', { free: sizeText( disk.free ), total: sizeText( disk.total ),
                kinds: new Intl.ListFormat( document.documentElement.lang || undefined ).format( disk.kinds ) } )}
            </li>
          ) )}
        </ul>
      </section>
      <ul className="storage-list">
        {usage.categories.map( ( category ) => (
          <Row key={category.id} category={category} busy={working} onClean={clean} />
        ) )}
      </ul>
      <CacheFolder usage={usage} onChanged={refresh} />
    </section>
  );
}

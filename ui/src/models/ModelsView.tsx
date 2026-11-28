// The Models page: every model the app can install, grouped by what it
// finds, with install (resumable), use, remove and "I already have the
// file". The detector recommended for this PC is marked and installs and
// switches in one click, turning the GPU on when the PC has one it can use,
// so the recommendation never runs on the slow path.
import { useCallback, useEffect, useState } from 'react';
import { engineRequest, isEngineError, type EngineError } from '../engine/client';
import { useEngineStatus } from '../engine/status';
import { errorText } from '../i18n/errors';
import { msgKey, t } from '../i18n/format';
import { describeError, log } from '../log';
import {
  cancelModel, importModel, installModel, listModels, pickModelFile, removeModel, switchToModel,
  type ModelKind, type ModelRow, type ModelsList,
} from './api';
import { installThenUse, markQueued, useModelEvents, type Progress } from './store';

type Notice = { tone: 'ok' | 'problem'; text: string } | null;

// Catalog keys are dotted lower case; model ids use hyphens.
export function modelKeyPart( id: string ): string {
  return id.replace( /-/g, '_' );
}

export function modelName( id: string ): string {
  return t( msgKey( 'models.name.', modelKeyPart( id ) ) );
}

function megabytes( bytes: number ): string {
  return t( 'models.size', { mb: Math.max( 1, Math.round( bytes / 1e6 ) ) } );
}

function percent( done: number, total: number ): number {
  return total > 0 ? Math.min( 100, Math.floor( ( done * 100 ) / total ) ) : 0;
}

function asError( error: unknown ): EngineError | null {
  return isEngineError( error ) ? error : null;
}

interface RowProps {
  row: ModelRow;
  inUse: boolean;
  recommended: boolean;
  progress: Progress | null | undefined;
  failure: EngineError | undefined;
  busy: boolean;
  onInstall: () => void;
  onCancel: () => void;
  onUse: () => void;
  onRemove: () => void;
}

function stateText( row: ModelRow, progress: Progress | null | undefined ): string {
  if ( progress !== undefined ) {
    return progress === null
      ? t( 'models.state.queued' )
      : t( 'models.state.downloading', { percent: percent( progress.bytes, progress.total ) } );
  }
  switch ( row.state ) {
    case 'installed': return t( 'models.state.installed' );
    case 'partial': return t( 'models.state.partial', { percent: percent( row.bytes, row.size ) } );
    case 'modified': return t( 'models.state.modified' );
    case 'queued': return t( 'models.state.queued' );
    case 'downloading': return t( 'models.state.downloading', { percent: percent( row.bytes, row.size ) } );
    case 'absent': return t( 'models.state.absent' );
  }
}

function ModelItem( props: RowProps ) {
  const { row, inUse, recommended, progress, failure, busy } = props;
  const downloading = progress !== undefined || row.state === 'queued' || row.state === 'downloading';
  const installed = row.state === 'installed' && !downloading;
  const titleId = `model-${row.id}`;
  return (
    <li className={`model-row${inUse ? ' in-use' : ''}`} aria-labelledby={titleId}>
      <div className="model-head">
        <h4 id={titleId} className="model-name">{modelName( row.id )}</h4>
        {inUse && ( installed
          ? <span className="badge in-use">{t( 'models.badge.in_use' )}</span>
          : <span className="badge missing">{t( 'models.badge.selected_missing' )}</span> )}
        {recommended && <span className="badge recommended">{t( 'models.badge.recommended' )}</span>}
      </div>
      <p className="model-desc">{t( msgKey( 'models.desc.', modelKeyPart( row.id ) ) )}</p>
      <p className="model-facts">
        {megabytes( row.size )}
        {' · '}
        {t( 'models.license', { license: row.license.spdx, project: row.license.project } )}
      </p>
      {/* The names the settings, SETUP.md and the tools use, since the page's
          small, medium and large do not always match them (RF-DETR's are
          nano, small and medium). */}
      <p className="model-facts">
        {row.variant === null
          ? t( 'models.setting_backend', { backend: row.backend } )
          : t( 'models.setting_variant', { backend: row.backend, variant: row.variant } )}
      </p>
      <div className="model-status">
        <span className={`model-state state-${downloading ? 'downloading' : row.state}`}>{stateText( row, progress )}</span>
        {progress !== undefined && progress !== null && (
          <progress className="model-progress" max={progress.total} value={progress.bytes}
            aria-label={t( 'models.progress_label', { name: modelName( row.id ) } )} />
        )}
      </div>
      {failure !== undefined && <p className="model-failure" role="alert">{errorText( failure )}</p>}
      <div className="model-actions">
        {downloading && (
          <button type="button" onClick={props.onCancel} disabled={busy}>
            {progress === null || row.state === 'queued' ? t( 'models.action.cancel' ) : t( 'models.action.pause' )}
          </button>
        )}
        {!downloading && !installed && (
          <button type="button" className="primary" onClick={props.onInstall} disabled={busy}>
            {row.state === 'partial' ? t( 'models.action.resume' )
              : row.state === 'modified' ? t( 'models.action.replace' ) : t( 'models.action.install' )}
          </button>
        )}
        {installed && !inUse && (
          <>
            <button type="button" className="primary" onClick={props.onUse} disabled={busy}>{t( 'models.action.use' )}</button>
            <button type="button" onClick={props.onRemove} disabled={busy}>{t( 'models.action.remove' )}</button>
          </>
        )}
        {installed && inUse && <span className="model-note">{t( 'models.in_use_note' )}</span>}
      </div>
    </li>
  );
}

interface GpuNoticeProps {
  list: ModelsList;
  busy: boolean;
  onUseGpu: () => void;
}

function GpuNotice( { list, busy, onUseGpu }: GpuNoticeProps ) {
  const { gpu, in_use: inUse } = list;
  if ( gpu.usable && !gpu.enabled ) {
    return (
      <div className="models-notice" role="status">
        <p>{gpu.name === null ? t( 'models.gpu.off' ) : t( 'models.gpu.off_named', { name: gpu.name } )}</p>
        <button type="button" className="primary" onClick={onUseGpu} disabled={busy}>{t( 'models.gpu.turn_on' )}</button>
      </div>
    );
  }
  if ( !gpu.usable && inUse.detector !== null && inUse.detector !== list.recommended.detector ) {
    return (
      <div className="models-notice" role="status">
        <p>{t( 'models.gpu.slow', { name: modelName( inUse.detector ), recommended: modelName( list.recommended.detector ) } )}</p>
      </div>
    );
  }
  return null;
}

interface RecommendProps {
  list: ModelsList;
  active: boolean;
  busy: boolean;
  onGo: () => void;
}

function Recommendation( { list, active, busy, onGo }: RecommendProps ) {
  const id = list.recommended.detector;
  const row = list.models.find( ( model ) => model.id === id );
  const gpuReady = !list.gpu.usable || list.gpu.enabled;
  if ( row === undefined || ( list.in_use.detector === id && gpuReady ) ) {
    return null;
  }
  const reason = list.recommended.reason === 'gpu'
    ? ( list.gpu.name === null ? t( 'models.recommend.gpu' ) : t( 'models.recommend.gpu_named', { name: list.gpu.name } ) )
    : t( 'models.recommend.cpu' );
  return (
    <section className="models-recommend" aria-labelledby="models-recommend-title">
      <h3 id="models-recommend-title">{t( 'models.recommend.title', { name: modelName( id ) } )}</h3>
      <p>{reason}</p>
      <button type="button" className="primary" onClick={onGo} disabled={busy || active}>
        {active ? t( 'models.recommend.working' )
          : row.state === 'installed' ? t( 'models.recommend.use' ) : t( 'models.recommend.install_and_use' )}
      </button>
    </section>
  );
}

function groups(): readonly { kind: ModelKind; title: string; intro: string }[] {
  return [
    { kind: 'detector', title: t( 'models.group.detector' ), intro: t( 'models.group.detector_intro' ) },
    { kind: 'person', title: t( 'models.group.person' ), intro: t( 'models.group.person_intro' ) },
  ];
}

export function ModelsView() {
  const engine = useEngineStatus();
  const events = useModelEvents();
  const [ list, setList ] = useState<ModelsList | null>( null );
  const [ loadError, setLoadError ] = useState<EngineError | null>( null );
  const [ busy, setBusy ] = useState( false );
  const [ notice, setNotice ] = useState<Notice>( null );

  const refresh = useCallback( () => {
    listModels().then(
      ( answer ) => { setList( answer ); setLoadError( null ); },
      ( error: unknown ) => {
        log.error( `models.list failed: ${describeError( error )}` );
        setLoadError( asError( error ) ?? { code: 'unknown', params: {} } );
      },
    );
  }, [] );

  // List again when the engine is ready and after every install state change.
  useEffect( () => {
    if ( engine.state === 'ready' ) {
      refresh();
    }
  }, [ engine.state, events.version, refresh ] );

  const run = async ( what: string, action: () => Promise<Notice> ) => {
    setBusy( true );
    setNotice( null );
    try {
      setNotice( await action() );
    } catch ( error: unknown ) {
      log.error( `${what} failed: ${describeError( error )}` );
      const engineError = asError( error );
      setNotice( { tone: 'problem', text: engineError === null ? t( 'models.request_failed' ) : errorText( engineError ) } );
    } finally {
      setBusy( false );
      refresh();
    }
  };

  const usedText = ( id: string, moves: readonly string[] ) => moves.includes( 'detection' )
    ? t( 'models.notice.using_redetect', { name: modelName( id ) } )
    : t( 'models.notice.using', { name: modelName( id ) } );

  const install = ( id: string ) => run( 'models.install', async () => {
    markQueued( id );
    await installModel( id );
    return null;
  } );
  const cancel = ( id: string ) => run( 'models.cancel', async () => { await cancelModel( id ); return null; } );
  const switchTo = ( id: string ) => run( 'models.use', async () => {
    const result = await switchToModel( id, false );
    return { tone: 'ok', text: usedText( id, result.moves ) };
  } );
  const remove = ( id: string ) => run( 'models.remove', async () => {
    await removeModel( id );
    return { tone: 'ok', text: t( 'models.notice.removed', { name: modelName( id ) } ) };
  } );
  const turnOnGpu = () => run( 'config.set', async () => {
    await engineRequest( 'config.set', { values: { gpu_enabled: 1 } } );
    return { tone: 'ok', text: t( 'models.notice.gpu_on' ) };
  } );
  const importFile = () => run( 'models.import', async () => {
    const path = await pickModelFile( t( 'models.import.dialog_title' ), t( 'models.import.filter' ) );
    if ( path === null ) {
      return null;
    }
    try {
      const id = await importModel( path );
      return { tone: 'ok', text: t( 'models.notice.imported', { name: modelName( id ) } ) };
    } catch ( error: unknown ) {
      // The engine answers a path with no file behind it as a bad
      // parameter; for the person who picked it, the file is not there.
      if ( isEngineError( error ) && error.code === 'command.bad_params' && error.params[ 'field' ] === 'path' ) {
        return { tone: 'problem', text: t( 'models.import.not_found' ) };
      }
      throw error;
    }
  } );
  const recommended = ( current: ModelsList ) => {
    const id = current.recommended.detector;
    const gpu = current.gpu.usable && !current.gpu.enabled;
    const row = current.models.find( ( model ) => model.id === id );
    if ( row?.state === 'installed' ) {
      return run( 'models.use', async () => {
        const result = await switchToModel( id, gpu );
        return { tone: 'ok', text: usedText( id, result.moves ) };
      } );
    }
    return run( 'models.install', async () => {
      installThenUse( id, gpu, ( result, error ) => {
        setNotice( error === null
          ? { tone: 'ok', text: usedText( id, result?.moves ?? [] ) }
          : { tone: 'problem', text: errorText( error ) } );
        refresh();
      } );
      markQueued( id );
      await installModel( id );
      return null;
    } );
  };

  if ( list === null ) {
    return (
      <section className="view models-view" aria-labelledby="models-title">
        <h2 id="models-title" className="models-title">{t( 'models.title' )}</h2>
        <p className="settings-wait" role="status">
          {loadError === null ? t( 'models.loading' ) : errorText( loadError )}
        </p>
      </section>
    );
  }

  const recommendedId = list.recommended.detector;
  return (
    <section className="view models-view" aria-labelledby="models-title">
      <h2 id="models-title" className="models-title">{t( 'models.title' )}</h2>
      <p className="view-intro">{t( 'models.intro' )}</p>
      <GpuNotice list={list} busy={busy} onUseGpu={() => { void turnOnGpu(); }} />
      <Recommendation list={list} busy={busy} active={events.active.has( recommendedId )}
        onGo={() => { void recommended( list ); }} />
      {notice !== null && (
        <p className={`models-result ${notice.tone}`} role={notice.tone === 'problem' ? 'alert' : 'status'}>{notice.text}</p>
      )}
      <div className="models-import">
        <button type="button" onClick={() => { void importFile(); }} disabled={busy}>{t( 'models.import.button' )}</button>
        <span className="model-note">{t( 'models.import.hint' )}</span>
      </div>
      {groups().map( ( group ) => (
        <section key={group.kind} className="models-group" aria-labelledby={`models-${group.kind}`}>
          <h3 id={`models-${group.kind}`}>{group.title}</h3>
          <p className="view-intro">{group.intro}</p>
          {list.custom[ group.kind ] !== undefined && (
            <p className="models-notice" role="status">{t( 'models.custom', { path: list.custom[ group.kind ] ?? '' } )}</p>
          )}
          <ul className="models-list">
            {list.models.filter( ( row ) => row.kind === group.kind ).map( ( row ) => (
              <ModelItem key={row.id} row={row} busy={busy}
                inUse={list.in_use[ group.kind ] === row.id}
                recommended={row.id === recommendedId}
                progress={events.active.get( row.id )}
                failure={events.failures.get( row.id )}
                onInstall={() => { void install( row.id ); }}
                onCancel={() => { void cancel( row.id ); }}
                onUse={() => { void switchTo( row.id ); }}
                onRemove={() => { void remove( row.id ); }} />
            ) )}
          </ul>
        </section>
      ) )}
    </section>
  );
}

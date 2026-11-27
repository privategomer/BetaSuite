// The status strip along the bottom of the window: one item per subsystem,
// each with an icon whose shape (not only its color) says how it is, a
// plain-language line, and an action when there is something to do: the
// engine, and model downloads while there are any.
import { useState } from 'react';
import { restartEngine, type EngineStatus } from '../engine/client';
import { useEngineStatus } from '../engine/status';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { describeError, log } from '../log';
import { modelName } from '../models/ModelsView';
import { useModelEvents } from '../models/store';

type Tone = 'ok' | 'busy' | 'problem';

function StatusIcon( { tone }: { tone: Tone } ) {
  return (
    <svg className={`status-icon ${tone}`} viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      {tone === 'ok' && (
        <>
          <circle cx="8" cy="8" r="7" fill="currentColor" />
          <path d="M4.5 8.2l2.3 2.3 4.7-4.9" fill="none" stroke="var(--color-surface)" strokeWidth="1.8" />
        </>
      )}
      {tone === 'busy' && (
        <path d="M8 1.5a6.5 6.5 0 1 1-6.5 6.5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      )}
      {tone === 'problem' && (
        <>
          <path d="M8 1l7.5 13.5h-15z" fill="currentColor" />
          <path d="M8 5.5v4.5M8 11.6v1.2" stroke="var(--color-surface)" strokeWidth="1.8" />
        </>
      )}
    </svg>
  );
}

interface View {
  tone: Tone;
  text: string;
  detail: string | null;
  canRestart: boolean;
}

function exitDetail( code: number | null, signal: number | null ): string | null {
  if ( signal !== null ) {
    return t( 'status.engine.exit_signal', { signal } );
  }
  if ( code !== null ) {
    return t( 'status.engine.exit_code', { code } );
  }
  return null;
}

function view( status: EngineStatus ): View {
  switch ( status.state ) {
    case 'idle':
    case 'starting':
      return { tone: 'busy', text: t( 'status.engine.starting' ), detail: null, canRestart: false };
    case 'ready':
      return {
        tone: 'ok',
        text: t( 'status.engine.ready' ),
        detail: t( 'status.engine.version', { version: status.version } ),
        canRestart: false,
      };
    case 'stopping':
      return { tone: 'busy', text: t( 'status.engine.stopping' ), detail: null, canRestart: false };
    case 'stopped':
      return { tone: 'problem', text: t( 'status.engine.stopped' ), detail: null, canRestart: true };
    case 'exited':
      return {
        tone: 'problem',
        text: t( 'status.engine.exited' ),
        detail: exitDetail( status.code, status.signal ),
        canRestart: true,
      };
    case 'failed':
      return { tone: 'problem', text: t( 'status.engine.failed' ), detail: errorText( status.error ), canRestart: true };
  }
}

function EngineItem() {
  const status = useEngineStatus();
  const [ restarting, setRestarting ] = useState( false );
  const { tone, text, detail, canRestart } = view( status );

  const restart = () => {
    setRestarting( true );
    log.info( 'restart requested from the status strip' );
    restartEngine()
      .catch( ( error: unknown ) => { log.error( `engine_restart failed: ${describeError( error )}` ); } )
      .finally( () => { setRestarting( false ); } );
  };

  return (
    <div className="status-item" data-state={status.state}>
      <StatusIcon tone={tone} />
      <span>{text}</span>
      {detail !== null && <span className="status-detail">{detail}</span>}
      {canRestart && (
        <button type="button" className="primary" onClick={restart} disabled={restarting}>
          {t( 'status.engine.restart' )}
        </button>
      )}
    </div>
  );
}

function ModelsItem() {
  const { active } = useModelEvents();
  if ( active.size === 0 ) {
    return null;
  }
  let text = t( 'status.models.waiting', { count: active.size } );
  for ( const [ id, progress ] of active ) {
    if ( active.size === 1 && progress !== null ) {
      text = t( 'status.models.downloading', {
        name: modelName( id ),
        percent: progress.total > 0 ? Math.min( 100, Math.floor( ( progress.bytes * 100 ) / progress.total ) ) : 0,
      } );
    }
  }
  return (
    <div className="status-item" data-state="downloading">
      <StatusIcon tone="busy" />
      <span>{text}</span>
    </div>
  );
}

export function StatusStrip() {
  return (
    <footer className="status-strip" role="status" aria-live="polite" aria-label={t( 'status.strip_label' )}>
      <EngineItem />
      <ModelsItem />
    </footer>
  );
}

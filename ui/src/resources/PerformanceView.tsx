// The Performance page: what the governor is doing and why, with its
// overrides and its on/off switch; the machine live (processor, memory,
// graphics card, disk); speed now; this PC and the settings its hardware
// settles; what past runs measured, with nudges; and Optimize for this PC.
import { useCallback, useEffect, useState } from 'react';
import { engineRequest, isEngineError, type EngineError } from '../engine/client';
import { useEngineStatus } from '../engine/status';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { describeError, log } from '../log';
import { setValues } from '../settings/api';
import { CostNotice, refusal, type Check } from '../settings/SettingRow';
import {
  applyOptimize, cancelOptimize, clearFullSpeed, detectMachine, enableGovernor, evidence as loadEvidence,
  fullSpeed, optimize, type Evidence, type GovernorState, type Machine, type Nudge, type PlanStep, type Sample,
} from './api';
import { refreshGovernor, useResources, useSamples } from './store';
import { bandName, decisionText, leversText, nudgeText, planText, reasonText, sizeText, stageName } from './text';

type Notice = { readonly tone: 'ok' | 'problem'; readonly text: string } | null;
const COLUMN = { scope: 'col' } as const;
const ROW = { scope: 'row' } as const;

function failure( error: unknown ): string {
  return isEngineError( error ) ? errorText( error ) : t( 'perf.request_failed' );
}

// A small line chart of the last samples; the latest value is in the text beside it.
function Sparkline( { values, max, label }: { values: readonly ( number | null )[]; max: number; label: string } ) {
  const width = 160;
  const height = 32;
  const known = values.map( ( value, index ) => ( value === null ? null : { index, value } ) )
    .filter( ( point ): point is { index: number; value: number } => point !== null );
  const step = values.length > 1 ? width / ( values.length - 1 ) : width;
  const points = known.map( ( point ) => {
    const y = height - ( Math.min( point.value, max ) / ( max || 1 ) ) * ( height - 2 ) - 1;
    return `${( point.index * step ).toFixed( 1 )},${y.toFixed( 1 )}`;
  } ).join( ' ' );
  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
      <rect x="0" y="0" width={width} height={height} className="sparkline-frame" />
      {known.length > 1 && <polyline points={points} className="sparkline-line" />}
    </svg>
  );
}

function latest( samples: readonly Sample[], pick: ( sample: Sample ) => number | null ): number | null {
  for ( let index = samples.length - 1; index >= 0; index -= 1 ) {
    const sample = samples[ index ];
    if ( sample !== undefined ) {
      const value = pick( sample );
      if ( value !== null ) {
        return value;
      }
    }
  }
  return null;
}

function Live( { samples }: { samples: readonly Sample[] } ) {
  const rows: { key: string; label: string; values: ( number | null )[]; max: number; text: string }[] = [];
  const own = latest( samples, ( s ) => s.own_cpu_percent );
  const other = latest( samples, ( s ) => s.other_cpu_percent );
  rows.push( { key: 'cpu-own', label: t( 'perf.live.cpu_run' ), values: samples.map( ( s ) => s.own_cpu_percent ), max: 100,
               text: own === null ? t( 'perf.live.unknown' ) : t( 'perf.live.percent', { value: Math.round( own ) } ) } );
  rows.push( { key: 'cpu-other', label: t( 'perf.live.cpu_other' ), values: samples.map( ( s ) => s.other_cpu_percent ), max: 100,
               text: other === null ? t( 'perf.live.unknown' ) : t( 'perf.live.percent', { value: Math.round( other ) } ) } );
  const total = latest( samples, ( s ) => s.memory_total_mb );
  const available = latest( samples, ( s ) => s.memory_available_mb );
  rows.push( { key: 'memory', label: t( 'perf.live.memory' ),
               values: samples.map( ( s ) => ( s.memory_total_mb !== null && s.memory_available_mb !== null
                 ? s.memory_total_mb - s.memory_available_mb : null ) ),
               max: total ?? 1,
               text: available === null ? t( 'perf.live.unknown' ) : t( 'perf.live.memory_free', { free: sizeText( available * 2 ** 20 ) } ) } );
  const gpu = latest( samples, ( s ) => s.gpu_percent );
  if ( gpu !== null ) {
    rows.push( { key: 'gpu', label: t( 'perf.live.gpu' ), values: samples.map( ( s ) => s.gpu_percent ), max: 100,
                 text: t( 'perf.live.percent', { value: gpu } ) } );
    const vramTotal = latest( samples, ( s ) => s.vram_total_mb );
    const vramUsed = latest( samples, ( s ) => s.vram_used_mb );
    rows.push( { key: 'vram', label: t( 'perf.live.vram' ), values: samples.map( ( s ) => s.vram_used_mb ), max: vramTotal ?? 1,
                 text: vramUsed === null ? t( 'perf.live.unknown' )
                   : t( 'perf.live.vram_used', { used: sizeText( vramUsed * 2 ** 20 ), total: sizeText( ( vramTotal ?? 0 ) * 2 ** 20 ) } ) } );
  }
  const pressure = latest( samples, ( s ) => s.memory_pressure );
  const cacheFree = latest( samples, ( s ) => s.cache_free_bytes );
  const battery = samples[ samples.length - 1 ];
  return (
    <section className="perf-section" aria-labelledby="perf-live">
      <h3 id="perf-live">{t( 'perf.live.title' )}</h3>
      {samples.length === 0 && <p className="settings-wait" role="status">{t( 'perf.live.waiting' )}</p>}
      <ul className="perf-live">
        {rows.map( ( row ) => (
          <li key={row.key}>
            <span className="perf-live-label">{row.label}</span>
            <Sparkline values={row.values} max={row.max} label={row.label} />
            <span className="perf-live-value">{row.text}</span>
          </li>
        ) )}
      </ul>
      <p className="model-note">
        {pressure !== null && <>{t( 'perf.live.pressure', { value: Math.round( pressure ) } )} </>}
        {cacheFree !== null && <>{t( 'perf.live.cache_free', { free: sizeText( cacheFree ) } )} </>}
        {battery?.on_battery === true && t( 'perf.live.on_battery', { percent: battery.battery_percent ?? 0 } )}
      </p>
    </section>
  );
}

function Governor( { governor }: { governor: GovernorState } ) {
  const [ working, setWorking ] = useState( false );
  const [ notice, setNotice ] = useState<Notice>( null );
  const [ guardWarning, setGuardWarning ] = useState( false );
  const act = ( what: string, action: () => Promise<Notice> ) => {
    setWorking( true );
    setNotice( null );
    action().then( ( next ) => { setNotice( next ); refreshGovernor(); }, ( error: unknown ) => {
      log.error( `${what} failed: ${describeError( error )}` );
      setNotice( { tone: 'problem', text: failure( error ) } );
    } ).finally( () => { setWorking( false ); } );
  };
  const toggle = ( on: boolean ) => { act( 'governor.enable', async () => {
    const answer = await enableGovernor( on );
    return answer.saved === 'after_jobs' ? { tone: 'ok', text: t( 'perf.governor.saved_after_jobs' ) } : null;
  } ); };
  const full = ( scope: 'run' | 'session', memoryGuard: boolean ) => { act( 'governor.override', async () => {
    await fullSpeed( scope, memoryGuard );
    setGuardWarning( false );
    return null;
  } ); };
  const levers = leversText( governor );
  return (
    <section className="perf-section" aria-labelledby="perf-governor">
      <h3 id="perf-governor">{t( 'perf.governor.title' )}</h3>
      <p className="view-intro">{t( 'perf.governor.intro' )}</p>
      <label className="perf-switch">
        <input type="checkbox" checked={governor.enabled} disabled={working}
          onChange={( event ) => { toggle( event.target.checked ); }} />
        {t( 'perf.governor.enabled' )}
      </label>
      <p className={`perf-state ${governor.throttled ? 'slowed' : 'full'}`} role="status">{decisionText( governor )}</p>
      {governor.reasons.length > 0 && governor.throttled && (
        <ul className="perf-reasons">
          {governor.reasons.filter( ( r ) => r.code !== 'governor.override' ).map( ( reason ) => (
            <li key={reason.code}>{reasonText( reason )}</li>
          ) )}
        </ul>
      )}
      {levers.length > 0 && <ul className="perf-levers">{levers.map( ( line ) => <li key={line}>{line}</li> )}</ul>}
      {governor.guard && <p className="perf-guard">{t( 'perf.governor.guard_explained' )}</p>}
      <div className="perf-actions">
        {governor.override === null && governor.enabled && (
          <>
            <button type="button" className={governor.throttled ? 'primary' : undefined} disabled={working}
              onClick={() => { full( 'run', true ); }}>
              {t( 'perf.governor.full_run' )}
            </button>
            <button type="button" disabled={working} onClick={() => { full( 'session', true ); }}>
              {t( 'perf.governor.full_session' )}
            </button>
          </>
        )}
        {governor.override !== null && (
          <button type="button" disabled={working} onClick={() => { act( 'governor.override_clear', async () => { await clearFullSpeed(); return null; } ); }}>
            {t( 'perf.governor.let_slow' )}
          </button>
        )}
        {governor.guard && !guardWarning && (
          <button type="button" disabled={working} onClick={() => { setGuardWarning( true ); }}>
            {t( 'perf.governor.ignore_memory' )}
          </button>
        )}
      </div>
      {guardWarning && (
        <div className="busy-choices-confirm" role="alert">
          <p>{t( 'perf.governor.ignore_memory_warning' )}</p>
          <button type="button" className="danger" disabled={working} onClick={() => { full( governor.override?.scope ?? 'run', false ); }}>
            {t( 'perf.governor.ignore_memory_confirm' )}
          </button>
          <button type="button" disabled={working} onClick={() => { setGuardWarning( false ); }}>{t( 'perf.governor.ignore_memory_back' )}</button>
        </div>
      )}
      {notice !== null && <p className={`models-result ${notice.tone}`} role={notice.tone === 'problem' ? 'alert' : 'status'}>{notice.text}</p>}
    </section>
  );
}

function History( { changes }: { changes: readonly GovernorState[] } ) {
  return (
    <section className="perf-section" aria-labelledby="perf-history">
      <h3 id="perf-history">{t( 'perf.history.title' )}</h3>
      {changes.length === 0
        ? <p className="model-note">{t( 'perf.history.none' )}</p>
        : (
          <ol className="perf-history">
            {changes.map( ( change, index ) => (
              <li key={`${change.time}-${index}`}>
                <time dateTime={new Date( change.time * 1000 ).toISOString()}>
                  {t( 'perf.history.time', { time: new Date( change.time * 1000 ) } )}
                </time>
                {' '}
                {decisionText( change )}
                {change.throttled && change.reasons[ 0 ] !== undefined && change.reasons[ 0 ].code !== 'governor.override' && (
                  <> {t( 'perf.history.because', { reason: reasonText( change.reasons[ 0 ] ) } )}</>
                )}
              </li>
            ) )}
          </ol>
        )}
    </section>
  );
}

function SpeedNow() {
  const { stages, governor } = useResources();
  const rows = [ ...stages.values() ];
  return (
    <section className="perf-section" aria-labelledby="perf-speed">
      <h3 id="perf-speed">{t( 'perf.speed.title' )}</h3>
      {rows.length === 0 && <p className="model-note">{t( 'perf.speed.none' )}</p>}
      {rows.length > 0 && (
        <ul className="perf-stages">
          {rows.map( ( row ) => (
            <li key={row.stage}>{t( 'perf.speed.stage', { stage: row.stage, rate: Math.round( row.rate * 10 ) / 10, done: row.done, total: row.total } )}</li>
          ) )}
        </ul>
      )}
      {governor?.planned != null && (
        <p className="model-note">{t( 'perf.speed.workers', { running: governor.running ?? 0, planned: governor.planned,
          cap: governor.worker_cap ?? governor.planned } )}</p>
      )}
    </section>
  );
}

function ThisPc( { machine, onChanged }: { machine: Machine; onChanged: () => void } ) {
  const [ notice, setNotice ] = useState<Notice>( null );
  const apply = ( setting: string, value: unknown ) => {
    engineRequest( 'config.set', { values: { [ setting ]: value } } ).then(
      () => { setNotice( { tone: 'ok', text: t( 'perf.pc.applied' ) } ); onChanged(); },
      ( error: unknown ) => { setNotice( { tone: 'problem', text: failure( error ) } ); },
    );
  };
  return (
    <section className="perf-section" aria-labelledby="perf-pc">
      <h3 id="perf-pc">{t( 'perf.pc.title' )}</h3>
      <dl className="perf-facts">
        <dt>{t( 'perf.pc.cpu' )}</dt>
        <dd>{t( 'perf.pc.cpu_value', { model: machine.cpu.model ?? t( 'perf.live.unknown' ), threads: machine.cpu.threads ?? 0 } )}</dd>
        <dt>{t( 'perf.pc.memory' )}</dt>
        <dd>{machine.memory.total_mb === null ? t( 'perf.live.unknown' ) : sizeText( machine.memory.total_mb * 2 ** 20 )}</dd>
        <dt>{t( 'perf.pc.gpu' )}</dt>
        <dd>{machine.gpu === null ? t( 'perf.pc.no_gpu' )
          : t( 'perf.pc.gpu_value', { name: machine.gpu.name ?? '', vram: sizeText( ( machine.gpu.vram_mb ?? 0 ) * 2 ** 20 ) } )}</dd>
      </dl>
      {!machine.cuda_usable && <p className="perf-guard">{t( 'perf.pc.cpu_only' )}</p>}
      {machine.safe_settings.map( ( pick ) => (
        <p key={pick.setting} className="models-notice">
          {pick.code === 'hw.gpu_usable' ? t( 'perf.pc.safe.gpu_usable' ) : t( 'perf.pc.safe.gpu_not_usable' )}
          {' '}
          <button type="button" className="primary" onClick={() => { apply( pick.setting, pick.value ); }}>{t( 'perf.pc.safe.apply' )}</button>
        </p>
      ) )}
      {notice !== null && <p className={`models-result ${notice.tone}`} role={notice.tone === 'problem' ? 'alert' : 'status'}>{notice.text}</p>}
    </section>
  );
}

function PastRuns( { found, onNudge, hidden }: { found: Evidence; onNudge: ( nudge: Nudge ) => void; hidden: ReadonlySet<string> } ) {
  return (
    <section className="perf-section" aria-labelledby="perf-runs">
      <h3 id="perf-runs">{t( 'perf.runs.title' )}</h3>
      <p className="view-intro">{t( 'perf.runs.intro', { runs: found.runs, measured: found.measured } )}</p>
      {found.bands.length > 0 && (
        <table className="perf-bands">
          <thead>
            <tr>
              <th {...COLUMN}>{t( 'perf.runs.band' )}</th>
              <th {...COLUMN}>{t( 'perf.runs.count' )}</th>
              <th {...COLUMN}>{t( 'perf.runs.detect' )}</th>
              <th {...COLUMN}>{t( 'perf.runs.render' )}</th>
              <th {...COLUMN}>{t( 'perf.runs.total' )}</th>
            </tr>
          </thead>
          <tbody>
            {found.bands.map( ( band ) => (
              <tr key={band.band}>
                <th {...ROW}>{bandName( band.band )}</th>
                <td>{band.runs}</td>
                <td>{band.detect_speed === null ? t( 'perf.live.unknown' ) : t( 'perf.runs.speed', { value: band.detect_speed } )}</td>
                <td>{band.render_fps === null ? t( 'perf.live.unknown' ) : t( 'perf.runs.fps', { value: Math.round( band.render_fps ) } )}</td>
                <td>{band.total_speed === null ? t( 'perf.live.unknown' ) : t( 'perf.runs.speed', { value: band.total_speed } )}</td>
              </tr>
            ) )}
          </tbody>
        </table>
      )}
      {found.memory_capped.of > 0 && (
        <p className="model-note">{t( 'perf.runs.memory_capped', { runs: found.memory_capped.runs, of: found.memory_capped.of } )}</p>
      )}
      {found.nudges.filter( ( nudge ) => !hidden.has( nudge.code ) ).map( ( nudge ) => (
        <p key={nudge.code} className="models-notice">
          {nudgeText( nudge )}
          {' '}
          <button type="button" className="primary" onClick={() => { onNudge( nudge ); }}>
            {nudge.action.kind === 'set' ? t( 'perf.nudge.do_set' ) : t( 'perf.nudge.do_optimize' )}
          </button>
        </p>
      ) )}
    </section>
  );
}

function Optimize( { plan, onStart }: { plan: readonly PlanStep[] | null; onStart: ( full: boolean, force: boolean ) => void } ) {
  const { optimize: progress } = useResources();
  const [ full, setFull ] = useState( false );
  const [ notice, setNotice ] = useState<Notice>( null );
  const running = progress?.state === 'running';
  // What applying the proposal costs (re-detect, re-render), checked as a
  // dry run the way every settings change is, before Apply is offered.
  // The answer is kept with the proposal it was for, so a new proposal
  // reads as checking until its own answer arrives.
  const proposal = progress?.state === 'finished' ? progress.proposal : null;
  const [ answer, setAnswer ] = useState<{ readonly for: readonly unknown[]; readonly check: Check } | null>( null );
  const check: Check = proposal === null || proposal.length === 0
    ? { state: 'idle' }
    : answer !== null && answer.for === proposal ? answer.check : { state: 'checking' };
  useEffect( () => {
    if ( proposal === null || proposal.length === 0 ) {
      return;
    }
    let live = true;
    const values = Object.fromEntries( proposal.map( ( row ) => [ row.setting, row.value ] ) );
    setValues( values, true ).then(
      ( change ) => { if ( live ) { setAnswer( { for: proposal, check: { state: 'ok', change } } ); } },
      ( error: unknown ) => { if ( live ) { setAnswer( { for: proposal, check: refusal( error ) } ); } },
    );
    return () => { live = false; };
  }, [ proposal ] );
  const apply = () => {
    applyOptimize().then(
      ( answer ) => { setNotice( { tone: 'ok', text: t( 'perf.optimize.applied', { count: answer.applied.length } ) } ); },
      ( error: unknown ) => { setNotice( { tone: 'problem', text: failure( error ) } ); },
    );
  };
  return (
    <section className="perf-section" aria-labelledby="perf-optimize">
      <h3 id="perf-optimize">{t( 'perf.optimize.title' )}</h3>
      <p className="view-intro">{t( 'perf.optimize.intro' )}</p>
      <label className="perf-switch">
        <input type="checkbox" checked={full} disabled={running} onChange={( event ) => { setFull( event.target.checked ); }} />
        {t( 'perf.optimize.full' )}
      </label>
      <div className="perf-actions">
        {!running && <button type="button" className="primary" onClick={() => { onStart( full, false ); }}>{t( 'perf.optimize.start' )}</button>}
        {running && <button type="button" onClick={() => { void cancelOptimize(); }}>{t( 'perf.optimize.cancel' )}</button>}
      </div>
      {plan !== null && (
        <ul className="perf-plan">
          {plan.map( ( step ) => (
            <li key={step.stage} className={step.run ? 'run' : 'skip'}>
              {!step.run ? t( 'perf.optimize.step_skip', { stage: stageName( step.stage ) } )
                : step.narrow ? t( 'perf.optimize.step_narrow', { stage: stageName( step.stage ) } )
                  : t( 'perf.optimize.step_full', { stage: stageName( step.stage ) } )}
              {' '}
              <span className="model-note">{planText( step.code )}</span>
            </li>
          ) )}
        </ul>
      )}
      {progress !== null && progress.state === 'running' && (
        <p role="status">
          <progress max={Math.max( 1, progress.total )} value={progress.done} />
          {' '}
          {t( 'perf.optimize.progress', { done: progress.done, total: progress.total } )}
        </p>
      )}
      {progress?.state === 'failed' && progress.error !== null && <p className="models-result problem" role="alert">{errorText( progress.error )}</p>}
      {progress?.state === 'cancelled' && <p className="model-note">{t( 'perf.optimize.cancelled' )}</p>}
      {progress?.state === 'finished' && progress.proposal !== null && (
        progress.proposal.length === 0
          ? <p className="models-result ok" role="status">{t( 'perf.optimize.nothing' )}</p>
          : (
            <>
              <table className="perf-bands">
                <thead>
                  <tr>
                    <th {...COLUMN}>{t( 'perf.optimize.setting' )}</th>
                    <th {...COLUMN}>{t( 'perf.optimize.now' )}</th>
                    <th {...COLUMN}>{t( 'perf.optimize.measured' )}</th>
                  </tr>
                </thead>
                <tbody>
                  {progress.proposal.map( ( row ) => (
                    <tr key={row.setting}>
                      <th {...ROW}><code>{row.setting}</code></th>
                      <td>{JSON.stringify( row.current )}</td>
                      <td>{JSON.stringify( row.value )}</td>
                    </tr>
                  ) )}
                </tbody>
              </table>
              <CostNotice check={check} />
              <button type="button" className="primary" disabled={check.state !== 'ok'} onClick={apply}>
                {t( 'perf.optimize.apply' )}
              </button>
            </>
          )
      )}
      {notice !== null && <p className={`models-result ${notice.tone}`} role={notice.tone === 'problem' ? 'alert' : 'status'}>{notice.text}</p>}
    </section>
  );
}

export function PerformanceView() {
  const engine = useEngineStatus();
  const { governor, changes } = useResources();
  const samples = useSamples();
  const [ machine, setMachine ] = useState<Machine | null>( null );
  const [ found, setFound ] = useState<Evidence | null>( null );
  const [ plan, setPlan ] = useState<readonly PlanStep[] | null>( null );
  const [ problem, setProblem ] = useState<EngineError | string | null>( null );
  const [ busyMachine, setBusyMachine ] = useState<{ full: boolean; stages?: readonly string[] } | null>( null );

  const refresh = useCallback( () => {
    detectMachine().then( setMachine, ( error: unknown ) => { log.warn( `hw.detect failed: ${describeError( error )}` ); } );
    loadEvidence().then( setFound, ( error: unknown ) => { log.warn( `perf.evidence failed: ${describeError( error )}` ); } );
  }, [] );
  useEffect( () => {
    if ( engine.state === 'ready' ) {
      refresh();
      refreshGovernor();
    }
  }, [ engine.state, refresh ] );

  const start = ( full: boolean, force: boolean, stages?: readonly string[] ) => {
    setProblem( null );
    setBusyMachine( null );
    optimize( stages === undefined ? { full, force } : { full, force, stages } ).then(
      ( answer ) => { setPlan( answer.plan ); },
      ( error: unknown ) => {
        if ( isEngineError( error ) && error.code === 'perf.machine_busy' ) {
          setBusyMachine( stages === undefined ? { full } : { full, stages } );
          return;
        }
        log.error( `perf.optimize failed: ${describeError( error )}` );
        setProblem( isEngineError( error ) ? error : t( 'perf.request_failed' ) );
      },
    );
  };
  const onNudge = ( nudge: Nudge ) => {
    if ( nudge.action.kind === 'set' ) {
      engineRequest( 'config.set', { values: nudge.action.values ?? {} } ).then( refresh, ( error: unknown ) => {
        setProblem( isEngineError( error ) ? error : t( 'perf.request_failed' ) );
      } );
    } else {
      start( false, false, nudge.action.stages ?? undefined );
    }
  };

  return (
    <section className="view perf-view" aria-labelledby="perf-title">
      <h2 id="perf-title" className="models-title">{t( 'perf.title' )}</h2>
      {governor === null
        ? <p className="settings-wait" role="status">{t( 'perf.loading' )}</p>
        : <Governor governor={governor} />}
      <Live samples={samples} />
      <SpeedNow />
      <History changes={changes} />
      {machine !== null && <ThisPc machine={machine} onChanged={refresh} />}
      {found !== null && (
        // The GPU fix is already offered under This computer; not twice.
        <PastRuns found={found} onNudge={onNudge}
          hidden={new Set( machine?.safe_settings.some( ( pick ) => pick.setting === 'gpu_enabled' ) ? [ 'perf.gpu_off' ] : [] )} />
      )}
      <Optimize plan={plan} onStart={( full, force ) => { start( full, force ); }} />
      {busyMachine !== null && (
        <div className="busy-choices-confirm" role="alert">
          <p>{errorText( { code: 'perf.machine_busy', params: {} } )}</p>
          <button type="button" onClick={() => { start( busyMachine.full, true, busyMachine.stages ); }}>{t( 'perf.optimize.force' )}</button>
        </div>
      )}
      {problem !== null && (
        <p className="models-result problem" role="alert">{typeof problem === 'string' ? problem : errorText( problem )}</p>
      )}
    </section>
  );
}

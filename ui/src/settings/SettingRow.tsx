// One setting: what it is, where its value comes from, and a way to change
// or reset it. No change is written until the engine has said what it
// costs (a dry run) and the user has applied it.
import { useEffect, useId, useMemo, useState } from 'react';
import { BusyChoices, type BusyOutcome } from '../components/BusyChoices';
import { isEngineError } from '../engine/client';
import type { SettingsChange } from '../resources/api';
import { t } from '../i18n/format';
import { instanceName, unavailableText } from '../i18n/labels';
import { describeError, log } from '../log';
import { resetValues, setValues, type Change } from './api';
import { EDITORS } from './editors';
import { labelFacts, labelStatusText } from './labelStatus';
import { instances, unmet, worstCost, type Row, type ValueRow } from './model';
import { costLines, failureLines, requirementText, rowCostTag, settingLabel, settingTip, sourceName } from './text';
import type { Loaded } from './useSettings';

const DRY_RUN_DELAY_MS = 300;

export type Check =
  | { readonly state: 'idle' }
  | { readonly state: 'checking' }
  | { readonly state: 'ok'; readonly change: Change }
  | { readonly state: 'refused'; readonly lines: readonly string[]; readonly busy: boolean };

type Pending = { readonly action: 'set'; readonly value: unknown } | { readonly action: 'reset' };

interface SettingRowProps {
  readonly row: Row;
  readonly data: Loaded;
  readonly onHelp: ( anchor: string ) => void;
  readonly onChanged: () => void;
}

function same( a: unknown, b: unknown ): boolean {
  return JSON.stringify( a ) === JSON.stringify( b );
}

export function refusal( error: unknown ): Check {
  if ( isEngineError( error ) ) {
    return { state: 'refused', lines: failureLines( error ), busy: error.code === 'config.busy' };
  }
  log.error( `settings request failed: ${describeError( error )}` );
  return { state: 'refused', lines: [ t( 'settings.request_failed' ) ], busy: false };
}

interface CostNoticeProps {
  readonly check: Check;
  // The change being checked, so a refusal because jobs run can offer to
  // keep it (BusyChoices) rather than lose it.
  readonly change?: SettingsChange;
  readonly onBusyDone?: ( outcome: BusyOutcome ) => void;
}

export function CostNotice( { check, change, onBusyDone }: CostNoticeProps ) {
  switch ( check.state ) {
    case 'idle':
      return null;
    case 'checking':
      return <p className="cost-notice checking" role="status">{t( 'settings.cost.checking' )}</p>;
    case 'ok':
      return (
        <div className="cost-notice" role="status">
          <span className="cost-notice-title">{t( 'settings.cost.title' )}</span>
          <ul>{costLines( check.change ).map( ( line ) => <li key={line}>{line}</li> )}</ul>
        </div>
      );
    case 'refused':
      return (
        <div className="cost-notice refused" role="alert">
          <span className="cost-notice-title">{check.busy ? t( 'settings.busy_title' ) : t( 'settings.refused_title' )}</span>
          <ul>{check.lines.map( ( line ) => <li key={line}>{line}</li> )}</ul>
          {check.busy && change !== undefined && onBusyDone !== undefined && (
            <BusyChoices change={change} onDone={onBusyDone} />
          )}
        </div>
      );
  }
}

export function SettingRow( { row, data, onHelp, onChanged }: SettingRowProps ) {
  const { entry } = row;
  const inputId = useId();
  const choices = useMemo( () => ( row.pattern ? instances( data.schema, [ ...data.values.keys() ], entry.path ) : [ entry.path ] ),
    [ row.pattern, data, entry.path ] );
  const [ chosen, setChosen ] = useState( choices[ 0 ] ?? entry.path );
  const path = choices.includes( chosen ) ? chosen : ( choices[ 0 ] ?? entry.path );
  const held: ValueRow | undefined = data.values.get( path );
  const [ pending, setPending ] = useState<Pending | null>( null );
  const [ answer, setAnswer ] = useState<{ readonly to: Pending; readonly check: Check } | null>( null );
  const [ applying, setApplying ] = useState( false );

  // Every edit, and a reset asked for, goes past the engine as a dry run
  // first: the notice shows what applying it would cost. An answer counts
  // only for the edit it was asked about.
  useEffect( () => {
    if ( pending === null ) {
      return undefined;
    }
    let live = true;
    const timer = window.setTimeout( () => {
      const request = pending.action === 'set' ? setValues( { [ path ]: pending.value }, true ) : resetValues( [ path ], true );
      request.then(
        ( change ) => { if ( live ) { setAnswer( { to: pending, check: { state: 'ok', change } } ); } },
        ( error: unknown ) => { if ( live ) { setAnswer( { to: pending, check: refusal( error ) } ); } },
      );
    }, DRY_RUN_DELAY_MS );
    return () => { live = false; window.clearTimeout( timer ); };
  }, [ pending, path ] );
  const check: Check = pending === null ? { state: 'idle' } : answer?.to === pending ? answer.check : { state: 'checking' };

  const apply = () => {
    if ( pending === null ) {
      return;
    }
    setApplying( true );
    const request = pending.action === 'set' ? setValues( { [ path ]: pending.value }, false ) : resetValues( [ path ], false );
    log.info( `settings: ${pending.action} ${path}` );
    request.then(
      () => { setPending( null ); onChanged(); },
      ( error: unknown ) => { setAnswer( { to: pending, check: refusal( error ) } ); },
    ).finally( () => { setApplying( false ); } );
  };

  const facts = row.pattern ? labelFacts( entry.path, path, data.labels ) : null;
  // A per-label value not set at this tier starts from what it resolves to.
  const current = held !== undefined ? held.value : facts?.value ?? null;
  const shown = pending?.action === 'set' ? pending.value : current;
  const Editor = row.editor === null ? null : EDITORS[ row.editor ];
  const valueOf = ( other: string ) => data.values.get( other )?.value;
  const context = { paths: [ ...data.values.keys() ], valueOf };
  const missing = unmet( entry, valueOf );
  const inactive = missing.length > 0 || facts?.offBy != null || facts?.unused === true;
  const parentLabel = ( parentPath: string ) => {
    const parent = data.rows.find( ( other ) => other.entry.path === parentPath );
    return parent === undefined ? parentPath : settingLabel( parent.entry );
  };

  return (
    <section className={`setting-row${inactive ? ' inactive' : ''}`} aria-labelledby={`${inputId}-label`} data-path={entry.path}>
      <div className="setting-head">
        <label id={`${inputId}-label`} htmlFor={inputId} className="setting-label">{settingLabel( entry )}</label>
        {held !== undefined && <span className={`badge source-${held.source}`}>{sourceName( held.source )}</span>}
        {held?.changed === true && <span className="badge changed">{t( 'settings.changed' )}</span>}
        <span className={`badge cost-${worstCost( entry.moves ) ?? ( entry.may_move.length > 0 ? 'some' : 'none' )}`}>{rowCostTag( entry )}</span>
      </div>
      <p className="setting-tip">
        {settingTip( entry )}
        {' '}
        <button type="button" className="link" onClick={() => { onHelp( entry.help ); }}>{t( 'settings.learn_more' )}</button>
      </p>
      {missing.map( ( requirement ) => (
        <p key={requirement.path} className="setting-note">{requirementText( requirement, parentLabel( requirement.path ) )}</p>
      ) )}
      {row.pattern && choices.length > 0 && (
        <label className="setting-instance">
          {t( 'settings.instance' )}
          <select value={path} onChange={( event ) => { setPending( null ); setChosen( event.target.value ); }}>
            {choices.map( ( choice ) => <option key={choice} value={choice}>{instanceName( choice, data.schema.placeholders[ '<label>' ] ?? [] )}</option> )}
          </select>
        </label>
      )}
      {facts !== null && (
        <p className={`setting-note label-status ${facts.status.kind}`}>
          {labelStatusText( facts )}
          {facts.offBy !== null && <> {t( 'settings.label.off', { parent: facts.offBy } )}</>}
          {facts.unused && <> {t( 'settings.label.unused', { label: facts.label } )}</>}
        </p>
      )}
      {facts !== null && data.labels.get( facts.label )?.unavailable !== undefined && (
        <p className="setting-note label-status unavailable">{unavailableText( data.labels.get( facts.label )?.unavailable ?? { code: 'other', params: {} } )}</p>
      )}
      <div className="setting-value">
        {Editor !== null && (
          <Editor
            entry={entry}
            schema={data.schema}
            context={context}
            value={shown}
            path={path}
            inputId={inputId}
            disabled={applying}
            onChange={( value ) => { setPending( held !== undefined && same( value, held.value ) ? null : { action: 'set', value } ); }}
          />
        )}
      </div>
      {held !== undefined && held.layers.length > 1 && (
        <p className="setting-layers">
          {t( 'settings.default_was', { value: JSON.stringify( held.default ) } )}
        </p>
      )}
      <CostNotice check={check}
        change={pending === null ? undefined : pending.action === 'set' ? { values: { [ path ]: pending.value } } : { resets: [ path ] }}
        onBusyDone={() => { setPending( null ); onChanged(); }} />
      <div className="setting-actions">
        {pending !== null && (
          <>
            <button type="button" className="primary" disabled={applying || check.state !== 'ok'} onClick={apply}>
              {pending.action === 'set' ? t( 'settings.apply' ) : t( 'settings.reset_confirm' )}
            </button>
            <button type="button" disabled={applying} onClick={() => { setPending( null ); }}>{t( 'settings.cancel' )}</button>
          </>
        )}
        {pending === null && held?.source === 'local' && (
          <button type="button" onClick={() => { setPending( { action: 'reset' } ); }}>{t( 'settings.reset' )}</button>
        )}
      </div>
    </section>
  );
}

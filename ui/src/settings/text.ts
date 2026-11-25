// Words for the settings pages: a setting's label, tip and "None" meaning
// from its catalog keys, rule ids, screens, sources and costs. The engine
// sends keys and codes only; every sentence is the catalog's.
import type { EngineError } from '../engine/client';
import { errorText } from '../i18n/errors';
import { hasMessage, msgKey, t } from '../i18n/format';
import type { Change } from './api';
import { COST_ORDER, worstCost, type Cost, type FieldSchema, type Requirement, type Screen, type SettingSchema, type Source } from './model';

const SETTING_PREFIX = 'setting.';

// 'setting.encode_crf.label' -> the message for it.
function settingKey( key: string ): string {
  if ( !key.startsWith( SETTING_PREFIX ) ) {
    return key;
  }
  return t( msgKey( 'setting.', key.slice( SETTING_PREFIX.length ) ) );
}

export function settingLabel( entry: SettingSchema | FieldSchema ): string {
  return settingKey( entry.label );
}

export function settingTip( entry: SettingSchema | FieldSchema ): string {
  return settingKey( entry.tip );
}

export function noneMeaning( entry: SettingSchema ): string | null {
  return entry.none === undefined ? null : settingKey( entry.none );
}

export function screenName( screen: Screen ): string {
  switch ( screen ) {
    case 'run': return t( 'settings.screen.run' );
    case 'what_to_hide': return t( 'settings.screen.what_to_hide' );
    case 'models': return t( 'settings.screen.models' );
    case 'library': return t( 'settings.screen.library' );
    case 'reverse': return t( 'settings.screen.reverse' );
    case 'resources': return t( 'settings.screen.resources' );
    case 'other': return t( 'settings.screen.other' );
  }
}

export function sourceName( source: Source ): string {
  switch ( source ) {
    case 'default': return t( 'settings.source.default' );
    case 'local': return t( 'settings.source.local' );
    case 'run': return t( 'settings.source.run' );
  }
}

// The short tag a row carries before any edit: what changing it re-makes,
// or that it re-makes only some videos (its profile or preview slice).
export function rowCostTag( entry: SettingSchema ): string {
  const cost = worstCost( entry.moves );
  if ( cost === null && entry.may_move.length > 0 ) {
    return t( 'settings.cost.tag.some' );
  }
  return costTag( cost );
}

export function costTag( cost: Cost | null ): string {
  switch ( cost ) {
    case 'detection': return t( 'settings.cost.tag.detection' );
    case 'censor': return t( 'settings.cost.tag.censor' );
    case 'encode': return t( 'settings.cost.tag.encode' );
    case null: return t( 'settings.cost.tag.none' );
  }
}

function costLine( cost: Cost ): string {
  switch ( cost ) {
    case 'detection': return t( 'settings.cost.detection' );
    case 'censor': return t( 'settings.cost.censor' );
    case 'encode': return t( 'settings.cost.encode' );
  }
}

function someFilesLine( cost: Cost ): string {
  switch ( cost ) {
    case 'detection': return t( 'settings.cost.some.detection' );
    case 'censor': return t( 'settings.cost.some.censor' );
    case 'encode': return t( 'settings.cost.some.encode' );
  }
}

// What a change costs, worst first, from the keys the engine says it moves.
export function costLines( change: Change ): string[] {
  const lines: string[] = [];
  for ( const cost of COST_ORDER ) {
    if ( change.moves.includes( cost ) ) {
      lines.push( costLine( cost ) );
    }
  }
  for ( const cost of COST_ORDER ) {
    if ( change.may_move.includes( cost ) && !change.moves.includes( cost ) ) {
      lines.push( someFilesLine( cost ) );
    }
  }
  if ( lines.length === 0 ) {
    lines.push( t( 'settings.cost.none' ) );
  }
  return lines;
}

export function requirementText( requirement: Requirement, parentLabel: string ): string {
  switch ( requirement.test ) {
    case 'on': return t( 'settings.requires.on', { parent: parentLabel } );
    case 'off': return t( 'settings.requires.off', { parent: parentLabel } );
    case 'above_zero': return t( 'settings.requires.above_zero', { parent: parentLabel } );
  }
}

// One problem from config.invalid: a rule by its own wording, anything
// else through the error catalog.
export function problemText( problem: EngineError ): string {
  if ( problem.code === 'config.rule' ) {
    const rule = problem.params[ 'rule' ];
    if ( typeof rule === 'string' && hasMessage( msgKey( 'rule.', rule ) ) ) {
      return t( msgKey( 'rule.', rule ) );
    }
  }
  return errorText( problem );
}

// The problems a failed config.set or config.reset carries, one per line.
export function failureLines( error: EngineError ): string[] {
  if ( error.code === 'config.invalid' ) {
    const problems = error.params[ 'problems' ];
    if ( Array.isArray( problems ) && problems.length > 0 ) {
      return problems.filter( isError ).map( problemText );
    }
  }
  return [ errorText( error ) ];
}

function isError( value: unknown ): value is EngineError {
  return typeof value === 'object' && value !== null && 'code' in value && typeof value.code === 'string'
    && 'params' in value && typeof value.params === 'object' && value.params !== null;
}

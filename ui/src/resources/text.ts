// Wording for the engine's resource codes: governor reasons, nudges,
// Optimize plan codes, storage categories and policies. The engine sends
// codes and numbers only; the words are here, from the catalog.
import { msgKey, t } from '../i18n/format';
import type { Category, GovernorState, Nudge, Reason } from './api';

function tail( code: string ): string {
  const dot = code.indexOf( '.' );
  return dot < 0 ? code : code.slice( dot + 1 );
}

function num( value: unknown ): number {
  return typeof value === 'number' ? value : 0;
}

// One reason the governor gave, as a short phrase.
export function reasonText( reason: Reason ): string {
  const params = reason.params;
  return t( msgKey( 'resources.reason.', tail( reason.code ) ), {
    other: Math.round( num( params[ 'other_cpu' ] ) ),
    idle: Math.round( num( params[ 'idle_seconds' ] ) ),
    percent: Math.round( num( params[ 'percent' ] ) ),
    pressure: Math.round( num( params[ 'pressure' ] ) ),
    available: Math.round( num( params[ 'available_mb' ] ) ),
  } );
}

// The governor's whole decision in one line, for the status strip.
export function decisionText( state: GovernorState ): string {
  if ( !state.enabled ) {
    return t( 'resources.state.off' );
  }
  const main = state.reasons[ 0 ];
  const reason = main === undefined ? '' : reasonText( main );
  if ( !state.active ) {
    // No run: what one would meet now.
    if ( state.override !== null ) {
      return t( 'resources.state.idle_override' );
    }
    return state.throttled ? t( 'resources.state.would_slow', { reason } ) : t( 'resources.state.idle' );
  }
  if ( state.guard ) {
    return t( 'resources.state.guard' );
  }
  if ( state.override !== null ) {
    return t( 'resources.state.full_override' );
  }
  if ( !state.throttled ) {
    return t( 'resources.state.full' );
  }
  return t( 'resources.state.slowed', { reason } );
}

// What the governor is doing to the run, in plain words.
export function leversText( state: GovernorState ): string[] {
  const lines: string[] = [];
  if ( state.worker_cap !== null ) {
    lines.push( t( 'resources.levers.workers', { cap: state.worker_cap, planned: state.planned ?? state.worker_cap } ) );
  }
  if ( state.priority === 'low' ) {
    lines.push( t( 'resources.levers.priority' ) );
  }
  if ( state.pace > 0 ) {
    lines.push( t( 'resources.levers.pace', { percent: Math.round( state.pace * 100 ) } ) );
  }
  return lines;
}

export function nudgeText( nudge: Nudge ): string {
  return t( msgKey( 'perf.nudge.', tail( nudge.code ) ), {
    gpu: typeof nudge.params[ 'gpu' ] === 'string' ? nudge.params[ 'gpu' ] : '',
    runs: num( nudge.params[ 'runs' ] ),
    of: num( nudge.params[ 'of' ] ),
  } );
}

export function planText( code: string ): string {
  return t( msgKey( 'perf.plan.', tail( code ) ) );
}

export function stageName( stage: string ): string {
  return t( msgKey( 'perf.stage.', stage ) );
}

export function bandName( band: string ): string {
  return t( msgKey( 'perf.band.', band ) );
}

export function categoryName( category: Category ): string {
  return t( msgKey( 'storage.category.', category.id ) );
}

// What clearing a category costs. Slow-to-get-back kinds each say how
// they come back; the rest share their policy's sentence.
const OWN_COST = new Set( [ 'detections', 'shot_cuts', 'benchmarks' ] );

export function policyText( category: Category ): string {
  if ( category.policy === 'costly' && OWN_COST.has( category.id ) ) {
    return t( msgKey( 'storage.cost.', category.id ) );
  }
  return t( msgKey( 'storage.policy.', category.policy ) );
}

export function kindName( kind: string ): string {
  return t( msgKey( 'storage.kind.', kind ) );
}

const UNITS = [ 'b', 'kb', 'mb', 'gb', 'tb' ] as const;

// 1536 -> "1.5 KB": sizes as a file manager shows them (powers of 1024).
export function sizeText( bytes: number ): string {
  let value = bytes;
  let unit = 0;
  while ( value >= 1024 && unit < UNITS.length - 1 ) {
    value /= 1024;
    unit += 1;
  }
  return t( msgKey( 'storage.size.', UNITS[ unit ] ?? 'b' ), { value: unit === 0 ? value : Math.round( value * 10 ) / 10 } );
}

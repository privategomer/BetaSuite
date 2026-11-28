// Words for the library's values. Video types, the rules that choose them
// and kinds of look are data (profiles, rule codes, censor styles), so each
// has a catalog message when there is one and shows its own name otherwise.
import { hasMessage, msgKey, t } from '../i18n/format';
import type { Status, VideoType } from './api';

export function typeName( profile: string | null ): string {
  if ( profile === null ) {
    return t( 'type.auto' );
  }
  const key = msgKey( 'type.name.', profile );
  return hasMessage( key ) ? t( key ) : profile;
}

export function typeHint( profile: string ): string | null {
  const key = msgKey( 'type.hint.', profile );
  return hasMessage( key ) ? t( key ) : null;
}

// Why the item gets its type: the rule, and when it was decided.
export function typeWhy( type: VideoType ): string {
  if ( type.when === 'at_run' ) {
    return t( 'type.why.at_run' );
  }
  const folder = type.params[ 'folder' ];
  const key = msgKey( 'type.rule.', type.rule ?? 'unknown' );
  const rule = hasMessage( key ) ? t( key, { folder: typeof folder === 'string' ? folder : '' } ) : t( 'type.rule.unknown' );
  return type.when === 'last_run' ? t( 'type.why.last_run', { rule } ) : rule;
}

export function statusName( status: Status ): string {
  return t( msgKey( 'status.', status ) );
}

export function lookName( look: string ): string {
  const key = msgKey( 'look.kind.', look );
  return hasMessage( key ) ? t( key ) : look;
}

// The files the catalog takes as videos (betautils_catalog_index.VIDEO_EXTENSIONS).
export const VIDEO_EXTENSIONS: readonly string[] = [
  'mp4', 'mkv', 'webm', 'avi', 'mov', 'm4v', 'wmv', 'flv', 'ts', 'm2ts', 'mts', 'mpg', 'mpeg', '3gp', 'ogv',
];

export function formatBytes( bytes: number ): string {
  const units = [ 'byte', 'kilobyte', 'megabyte', 'gigabyte', 'terabyte' ] as const;
  let value = bytes;
  let unit = 0;
  while ( value >= 1000 && unit < units.length - 1 ) {
    value /= 1000;
    unit += 1;
  }
  return new Intl.NumberFormat( document.documentElement.lang || undefined, {
    style: 'unit', unit: units[ unit ], unitDisplay: 'short', maximumFractionDigits: unit === 0 ? 0 : 1,
  } ).format( value );
}

// A duration in seconds as the catalog words it.
export function formatSeconds( seconds: number ): string {
  const whole = Math.max( 0, Math.round( seconds ) );
  const hours = Math.floor( whole / 3600 );
  const minutes = Math.floor( ( whole % 3600 ) / 60 );
  const rest = whole % 60;
  if ( hours > 0 ) {
    return t( 'duration.hours', { hours, minutes } );
  }
  if ( minutes > 0 ) {
    return t( 'duration.minutes', { minutes, seconds: rest } );
  }
  return t( 'duration.seconds', { seconds: rest } );
}

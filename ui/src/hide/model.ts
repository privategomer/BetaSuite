// The what-to-hide screen's model. Everything about labels, models, presets
// and art styles comes from the engine as data (config.schema and
// config.get), so a newer model's labels, a new preset or a new art style
// shows up here with no code change. Pure functions only.

export type Sex = 'female' | 'male';
export type View = 'front' | 'back';
export type Dress = 'unclothed' | 'clothed';

export interface LabelInfo {
  readonly id: string;
  readonly part: string;
  readonly state: string;
  readonly sex: string;
  readonly view: string;
}

export interface Preset {
  readonly id: string;
  readonly version: number;
  readonly name_key?: string;
  readonly name?: string;
  readonly style: unknown;
}

export const ITEMS_PATH = 'items_to_censor';
export const ART_STYLE_PATH = 'art_style';
export const DEFAULT_ART_STYLE = 'flat';

export function stylePath( label: string ): string {
  return `item_overrides.${label}.censor_style`;
}

// The figure sex a label's art is drawn on: its own when the model ties it
// to one, else whichever figure is showing. One switch either way.
export function tileSex( label: LabelInfo, figure: Sex ): Sex {
  return label.sex === 'female' || label.sex === 'male' ? label.sex : figure;
}

export function tileItem( label: LabelInfo, figure: Sex ): string {
  return `tile.${label.id}.${tileSex( label, figure )}`;
}

export function figureItem( sex: Sex, view: View, dress: Dress ): string {
  return `figure.${sex}.${view}.${dress}`;
}

// The figure that shows a label: covered labels on the clothed figure.
export function dressFor( label: LabelInfo, current: Dress ): Dress {
  if ( label.state === 'covered' ) {
    return 'clothed';
  }
  if ( label.state === 'exposed' ) {
    return 'unclothed';
  }
  return current;
}

// Labels grouped by body part, in the engine's display order.
export function byPart( labels: readonly LabelInfo[] ): { part: string; labels: LabelInfo[] }[] {
  const groups: { part: string; labels: LabelInfo[] }[] = [];
  for ( const label of labels ) {
    const last = groups.at( -1 );
    if ( last?.part === label.part ) {
      last.labels.push( label );
    } else {
      groups.push( { part: label.part, labels: [ label ] } );
    }
  }
  return groups;
}

// JSON with object keys in a fixed order, so two equal values compare
// equal whatever order their keys came in.
export function canonical( value: unknown ): string {
  if ( Array.isArray( value ) ) {
    return `[${value.map( canonical ).join( ',' )}]`;
  }
  if ( typeof value === 'object' && value !== null ) {
    const entries = Object.entries( value as Record<string, unknown> ).sort( ( [ a ], [ b ] ) => ( a < b ? -1 : a > b ? 1 : 0 ) );
    return `{${entries.map( ( [ key, item ] ) => `${JSON.stringify( key )}:${canonical( item )}` ).join( ',' )}}`;
  }
  return value === undefined ? 'null' : JSON.stringify( value );
}

export function matchingPreset( style: unknown, presets: readonly Preset[] ): Preset | null {
  const wanted = canonical( style );
  return presets.find( ( preset ) => canonical( preset.style ) === wanted ) ?? null;
}

export function withLabel( items: readonly string[], label: string, on: boolean ): string[] {
  const rest = items.filter( ( item ) => item !== label );
  return on ? [ ...rest, label ] : rest;
}

// A box as percentages of its image, for an overlay that scales with it.
export function boxPercent( box: { x: number; y: number; w: number; h: number }, width: number, height: number ) {
  return {
    insetInlineStart: `${String( ( box.x / width ) * 100 )}%`,
    insetBlockStart: `${String( ( box.y / height ) * 100 )}%`,
    inlineSize: `${String( ( box.w / width ) * 100 )}%`,
    blockSize: `${String( ( box.h / height ) * 100 )}%`,
  };
}

// The tile nearest in a direction, by where tiles sit on screen: the grid
// wraps and is split by part headings, so rows are not a fixed width.
export interface Rect { readonly left: number; readonly top: number; readonly width: number; readonly height: number }

export function neighbour( rects: readonly Rect[], from: number, direction: 'up' | 'down' ): number {
  const here = rects[ from ];
  if ( here === undefined ) {
    return from;
  }
  const centre = here.left + here.width / 2;
  let best = from;
  let bestScore = Infinity;
  rects.forEach( ( rect, index ) => {
    const dy = direction === 'down' ? rect.top - here.top : here.top - rect.top;
    if ( index === from || dy < here.height / 2 ) {
      return;
    }
    const score = dy * 1000 + Math.abs( rect.left + rect.width / 2 - centre );
    if ( score < bestScore ) {
      bestScore = score;
      best = index;
    }
  } );
  return best;
}

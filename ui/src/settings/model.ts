// The settings pages' model: rows built from config.schema and the values
// config.get reports. Pure functions with no imports, so
// scripts/check-settings-reach.mjs runs this same file against the
// registry fixture and proves every setting is reachable.

export type Cost = 'detection' | 'censor' | 'encode';
export type RequireTest = 'on' | 'off' | 'above_zero';
export type Screen = 'run' | 'what_to_hide' | 'models' | 'library' | 'reverse' | 'resources' | 'other';
export type Level = 'simple' | 'advanced';
export type Source = 'default' | 'local' | 'run';

export interface Requirement {
  readonly path: string;
  readonly test: RequireTest;
}

export interface SettingSchema {
  readonly path: string;
  readonly kind: string;
  readonly scope: string;
  readonly level: Level;
  readonly moves: readonly Cost[];
  readonly may_move: readonly Cost[];
  readonly when: readonly string[];
  readonly per_run: boolean;
  readonly tip: string;
  readonly label: string;
  readonly help: string;
  readonly screen: Screen;
  readonly requires: readonly Requirement[];
  readonly minimum?: number;
  readonly maximum?: number;
  readonly min_open?: true;
  readonly max_open?: true;
  readonly nullable?: true;
  readonly odd?: true;
  readonly choices?: readonly unknown[];
  readonly none?: string;
}

export interface FieldSchema {
  readonly kind: string;
  readonly tip: string;
  readonly label: string;
  readonly minimum?: number;
  readonly maximum?: number;
  readonly min_open?: true;
  readonly max_open?: true;
  readonly choices?: readonly unknown[];
}

export interface Schema {
  readonly settings: readonly SettingSchema[];
  readonly styles: {
    readonly common: Readonly<Record<string, FieldSchema>>;
    readonly types: Readonly<Record<string, Readonly<Record<string, FieldSchema>>>>;
    readonly frame_extra: Readonly<Record<string, FieldSchema>>;
  };
  readonly rules: Readonly<Record<string, Readonly<Record<string, FieldSchema>>>>;
  readonly placeholders: Readonly<Record<string, readonly string[]>>;
  readonly conditions: readonly string[];
  readonly screens: readonly Screen[];
  readonly geometry_actions: readonly string[];
  readonly catalog: string;
  readonly help: string;
}

export interface Layer {
  readonly layer: Source;
  readonly value: unknown;
}

export interface ValueRow {
  readonly path: string;
  readonly value: unknown;
  readonly default?: unknown;
  readonly source: Source;
  readonly changed: boolean;
  readonly layers: readonly Layer[];
}

// The tabs, in the order the app shows them. 'other' holds the tuning no
// screen's task needs; every setting is also on All Settings.
export const SCREENS: readonly Screen[] = [ 'run', 'what_to_hide', 'models', 'library', 'reverse', 'resources', 'other' ];

// Which editor shows each kind of value. The page's editor table is keyed
// by EditorId, so the type checker requires a component for each one.
export const EDITOR_FOR_KIND = {
  bool: 'toggle',
  int: 'number',
  number: 'number',
  choice: 'choice',
  string: 'text',
  path: 'path',
  paths: 'paths',
  labels: 'labels',
  color: 'color',
  sizes: 'sizes',
  censor_style: 'style_slot',
  window_style: 'style_slot',
  frame_style: 'style_slot',
  geometry_action: 'geometry_action',
  suppression_rules: 'suppression_rules',
  promotion_rules: 'promotion_rules',
  refused_pairs: 'refused_pairs',
  profile_map: 'profile_map',
} as const;

export type Kind = keyof typeof EDITOR_FOR_KIND;
export type EditorId = ( typeof EDITOR_FOR_KIND )[ Kind ];

export function editorFor( kind: string ): EditorId | null {
  return Object.hasOwn( EDITOR_FOR_KIND, kind ) ? EDITOR_FOR_KIND[ kind as Kind ] : null;
}

// Worst first: re-detecting costs more than re-rendering, which costs
// more than re-encoding.
export const COST_ORDER: readonly Cost[] = [ 'detection', 'censor', 'encode' ];

export function worstCost( moves: readonly Cost[] ): Cost | null {
  return COST_ORDER.find( ( cost ) => moves.includes( cost ) ) ?? null;
}

// --- paths and placeholders ------------------------------------------------

export function segments( path: string ): string[] {
  return path.split( '.' );
}

export function isPlaceholder( segment: string ): boolean {
  return segment.startsWith( '<' ) && segment.endsWith( '>' );
}

export function hasPlaceholders( path: string ): boolean {
  return segments( path ).some( isPlaceholder );
}

export function matches( pattern: string, path: string ): boolean {
  const want = segments( pattern );
  const got = segments( path );
  return want.length === got.length && want.every( ( part, index ) => isPlaceholder( part ) || part === got[ index ] );
}

// The values a placeholder can take at one position of a pattern: the
// schema's list for the ones it names (backends, labels, variants), else
// the keys the config holds there (a structure profile, a folder).
export function placeholderValues( schema: Schema, held: readonly string[], prefix: readonly string[], placeholder: string ): string[] {
  const known = schema.placeholders[ placeholder ];
  if ( known !== undefined ) {
    return [ ...known ];
  }
  const found = new Set<string>();
  for ( const path of held ) {
    const parts = segments( path );
    if ( parts.length > prefix.length && prefix.every( ( part, index ) => part === parts[ index ] ) ) {
      const value = parts[ prefix.length ];
      if ( value !== undefined ) {
        found.add( value );
      }
    }
  }
  return [ ...found ].sort();
}

// Every concrete path an entry stands for, with each placeholder filled.
export function instances( schema: Schema, held: readonly string[], pattern: string ): string[] {
  let prefixes: string[][] = [ [] ];
  for ( const part of segments( pattern ) ) {
    if ( !isPlaceholder( part ) ) {
      prefixes = prefixes.map( ( prefix ) => [ ...prefix, part ] );
      continue;
    }
    prefixes = prefixes.flatMap( ( prefix ) =>
      placeholderValues( schema, held, prefix, part ).map( ( value ) => [ ...prefix, value ] ) );
  }
  return prefixes.map( ( prefix ) => prefix.join( '.' ) );
}

// --- requirements ----------------------------------------------------------

export function testPasses( test: RequireTest, value: unknown ): boolean {
  switch ( test ) {
    case 'on':
      return Boolean( value );
    case 'off':
      return !value;
    case 'above_zero':
      return typeof value === 'number' && value > 0;
  }
}

// The requirements an entry does not meet with these values: an empty list
// means the setting is in effect as far as its parents go.
export function unmet( entry: SettingSchema, valueOf: ( path: string ) => unknown ): Requirement[] {
  return entry.requires.filter( ( requirement ) => !testPasses( requirement.test, valueOf( requirement.path ) ) );
}

// --- rows ------------------------------------------------------------------

export interface Row {
  readonly entry: SettingSchema;
  readonly editor: EditorId | null;
  readonly pattern: boolean;
}

export function buildRows( schema: Schema ): Row[] {
  return schema.settings.map( ( entry ) => ( {
    entry,
    editor: editorFor( entry.kind ),
    pattern: hasPlaceholders( entry.path ),
  } ) );
}

export function valuesByPath( values: readonly ValueRow[] ): Map<string, ValueRow> {
  return new Map( values.map( ( row ) => [ row.path, row ] ) );
}

// Whether any value the row stands for differs from betaconfig.py's.
export function rowChanged( row: Row, values: ReadonlyMap<string, ValueRow> ): boolean {
  if ( !row.pattern ) {
    return values.get( row.entry.path )?.changed ?? false;
  }
  for ( const [ path, value ] of values ) {
    if ( value.changed && matches( row.entry.path, path ) ) {
      return true;
    }
  }
  return false;
}

// --- search ----------------------------------------------------------------

export interface Filters {
  readonly query: string;
  readonly changedOnly: boolean;
  readonly screen: Screen | null;
  readonly cost: Cost | null;
}

export const NO_FILTERS: Filters = { query: '', changedOnly: false, screen: null, cost: null };

function fold( text: string ): string {
  return text.normalize( 'NFKD' ).replace( /\p{M}/gu, '' ).toLocaleLowerCase();
}

// The words a row is found by: its label and tip as shown, its path (with
// dots and underscores as spaces too) and its placeholders' values.
export function searchText( row: Row, label: string, tip: string, schema: Schema ): string {
  const path = row.entry.path;
  const parts = [ label, tip, path, path.replace( /[._<>]/g, ' ' ) ];
  for ( const part of segments( path ).filter( isPlaceholder ) ) {
    parts.push( ...( schema.placeholders[ part ] ?? [] ) );
  }
  return fold( parts.join( ' ' ) );
}

export function filterRows(
  rows: readonly Row[],
  filters: Filters,
  textOf: ( row: Row ) => string,
  values: ReadonlyMap<string, ValueRow>,
): Row[] {
  const words = fold( filters.query ).split( /\s+/ ).filter( ( word ) => word.length > 0 );
  return rows.filter( ( row ) => {
    if ( filters.screen !== null && row.entry.screen !== filters.screen ) {
      return false;
    }
    if ( filters.cost !== null && worstCost( row.entry.moves ) !== filters.cost ) {
      return false;
    }
    if ( filters.changedOnly && !rowChanged( row, values ) ) {
      return false;
    }
    if ( words.length === 0 ) {
      return true;
    }
    const text = textOf( row );
    return words.every( ( word ) => text.includes( word ) );
  } );
}

// Rows grouped by screen in tab order, each group simple settings first.
export function byScreen( rows: readonly Row[] ): Map<Screen, Row[]> {
  const groups = new Map<Screen, Row[]>( SCREENS.map( ( screen ) => [ screen, [] ] ) );
  for ( const row of rows ) {
    groups.get( row.entry.screen )?.push( row );
  }
  for ( const group of groups.values() ) {
    group.sort( ( a, b ) => Number( a.entry.level === 'advanced' ) - Number( b.entry.level === 'advanced' ) );
  }
  return groups;
}

// Rows with each one placed under the parent it requires, when that parent
// is in the same list: [{ row, depth }], parents first, the list's own
// order kept otherwise. The nearest parent is the last requirement listed
// (person windows' settings sit under person windows, not reverse mode).
export interface Placed {
  readonly row: Row;
  readonly depth: number;
}

export function nestByParent( rows: readonly Row[] ): Placed[] {
  const present = new Set( rows.map( ( row ) => row.entry.path ) );
  const children = new Map<string, Row[]>();
  const roots: Row[] = [];
  for ( const row of rows ) {
    const parent = [ ...row.entry.requires ].reverse().find( ( requirement ) => present.has( requirement.path ) );
    if ( parent === undefined ) {
      roots.push( row );
    } else {
      children.set( parent.path, [ ...( children.get( parent.path ) ?? [] ), row ] );
    }
  }
  const placed: Placed[] = [];
  const visit = ( row: Row, depth: number ) => {
    placed.push( { row, depth } );
    for ( const child of children.get( row.entry.path ) ?? [] ) {
      visit( child, depth + 1 );
    }
  };
  roots.forEach( ( row ) => { visit( row, 0 ); } );
  return placed;
}

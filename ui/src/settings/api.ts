// The config.* commands, typed. The shapes are the engine's
// (protocol/v1/commands/config.*.schema.json); errors arrive as
// EngineError codes and are worded by i18n/errors.ts and rules.ts.
import { engineRequest } from '../engine/client';
import type { Cost, Schema, ValueRow } from './model';

export interface ChainLayer {
  readonly source: string;
  readonly value?: unknown;
  readonly wins?: true;
  readonly profile?: string;
  readonly inactive?: 'backend_not_selected' | 'variant_not_selected' | 'not_backend_tunable'
    | 'look_key_in_window' | 'reverse_off' | 'not_a_window' | 'is_a_window' | 'per_profile';
}

export interface LabelField {
  readonly value: unknown;
  readonly source: string;
  readonly chain: readonly ChainLayer[];
  readonly off?: string;
}

export interface LabelView {
  readonly label: string;
  readonly rendered: boolean;
  readonly window: boolean;
  // Whether the selected detector backend reports this label at all, and
  // why not as a code with parameters.
  readonly available?: boolean;
  readonly unavailable?: { readonly code: string; readonly params: Readonly<Record<string, unknown>> };
  readonly fields: Readonly<Record<string, LabelField>>;
}

export interface Values {
  readonly settings: readonly ValueRow[];
  readonly labels?: readonly LabelView[];
}

// What a change would cost, or did: the cache keys it moves for every
// file, and those it moves for some files only.
export interface Change {
  readonly moves: readonly Cost[];
  readonly may_move: readonly Cost[];
  readonly written: boolean;
  readonly dry_run: boolean;
}

export async function getSchema(): Promise<Schema> {
  return await engineRequest( 'config.schema' ) as Schema;
}

export async function getValues( options: { paths?: readonly string[]; overrides?: Readonly<Record<string, unknown>>; labels?: boolean } = {} ): Promise<Values> {
  return await engineRequest( 'config.get', { ...options } ) as Values;
}

export async function setValues( values: Readonly<Record<string, unknown>>, dryRun: boolean ): Promise<Change> {
  return await engineRequest( 'config.set', { values, dry_run: dryRun } ) as Change;
}

export async function resetValues( paths: readonly string[], dryRun: boolean ): Promise<Change> {
  return await engineRequest( 'config.reset', { paths, dry_run: dryRun } ) as Change;
}

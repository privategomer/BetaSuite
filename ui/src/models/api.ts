// The models.* engine commands, typed.
// Shapes: protocol/v1/commands/models.*.schema.json and events/models.*.
import { invoke } from '@tauri-apps/api/core';
import { engineRequest } from '../engine/client';

export type ModelKind = 'detector' | 'person';
export type ModelState = 'installed' | 'modified' | 'partial' | 'absent' | 'queued' | 'downloading';

export interface ModelRow {
  readonly id: string;
  readonly kind: ModelKind;
  readonly backend: string;
  readonly variant: string | null;
  readonly size: number;
  readonly state: ModelState;
  readonly bytes: number;
  readonly license: { readonly spdx: string; readonly project: string; readonly home: string };
}

export interface ModelsList {
  readonly models: readonly ModelRow[];
  readonly in_use: { readonly detector: string | null; readonly person: string | null };
  readonly custom: Readonly<Partial<Record<ModelKind, string>>>;
  readonly recommended: { readonly detector: string; readonly reason: 'gpu' | 'cpu' };
  readonly gpu: { readonly usable: boolean; readonly enabled: boolean; readonly name: string | null };
}

export interface UseResult {
  readonly moves: readonly string[];
  readonly may_move: readonly string[];
  readonly written: boolean;
  readonly dry_run: boolean;
}

export async function listModels(): Promise<ModelsList> {
  return await engineRequest( 'models.list' ) as ModelsList;
}

export async function installModel( id: string ): Promise<void> {
  await engineRequest( 'models.install', { id } );
}

export async function cancelModel( id: string ): Promise<void> {
  await engineRequest( 'models.cancel', { id } );
}

export async function importModel( path: string ): Promise<string> {
  const answer = await engineRequest( 'models.import', { path } ) as { id: string };
  return answer.id;
}

export async function removeModel( id: string ): Promise<void> {
  await engineRequest( 'models.remove', { id } );
}

export async function switchToModel( id: string, gpu: boolean, dryRun = false ): Promise<UseResult> {
  return await engineRequest( 'models.use', { id, gpu, dry_run: dryRun } ) as UseResult;
}

// The shell's native file dialog for one model file; null when cancelled.
export function pickModelFile( title: string, filterName: string ): Promise<string | null> {
  return invoke( 'pick_model', { title, filterName } );
}

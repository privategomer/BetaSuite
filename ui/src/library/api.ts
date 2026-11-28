// The catalog.* and job.* commands the library and the player use, typed,
// and the shell commands that go with them. Shapes are the engine's
// (protocol/v1/commands/*.schema.json); errors arrive as EngineError codes.
import { invoke } from '@tauri-apps/api/core';
import { engineRequest } from '../engine/client';
import type { PlayableVideo } from '../shell';

export type Status = 'new' | 'queued' | 'processing' | 'done' | 'failed';
export type Availability = 'present' | 'missing' | 'offline';
export type OutputState = 'present' | 'missing' | 'none';
export type Sort = 'name' | 'added_at' | 'size' | 'status';

export interface VideoType {
  readonly profile: string | null;
  readonly rule: string | null;
  readonly params: Readonly<Record<string, unknown>>;
  readonly when: 'declared' | 'last_run' | 'at_run';
}

export interface Censor {
  readonly label: string;
  readonly looks: readonly string[];
}

export interface Item {
  readonly item: string;
  readonly name: string;
  readonly kind: string;
  readonly size: number;
  readonly status: Status;
  readonly availability: Availability;
  readonly copies: number;
  readonly pin: string | null;
  readonly type: VideoType;
  readonly outputs: number;
  readonly outputs_missing: number;
  readonly censors: readonly Censor[];
  readonly added_at: number;
}

export interface OutputFile {
  readonly path: string;
  readonly detection_key: string;
  readonly censor_key: string;
  readonly encode_key: string;
  readonly present: boolean;
}

export interface Run {
  readonly program: string | null;
  readonly started_at: number;
  readonly ended_at: number;
  readonly outcome: string;
  readonly reason: string | null;
  readonly output: string | null;
  readonly profile: string | null;
  readonly preview: boolean;
}

export interface ItemDetail extends Item {
  readonly locations: readonly { readonly path: string; readonly folder: number | null; readonly availability: Availability }[];
  readonly output_files: readonly OutputFile[];
  readonly history: readonly Run[];
}

export interface Filters {
  status?: Status[];
  availability?: Availability[];
  type?: string[];
  output?: OutputState[];
  censored?: { labels?: string[]; looks?: string[] };
  folder?: number;
  pinned?: boolean;
  duplicates?: boolean;
}

export interface ListPage {
  readonly items: readonly Item[];
  readonly total: number;
  readonly pending: number;
  readonly reclaimable_bytes: number;
}

export interface Folder {
  readonly folder: number;
  readonly path: string;
  readonly type_from_folder: boolean;
  readonly available: boolean;
  readonly items: number;
}

export async function listItems( filter: Filters, sort: Sort, descending: boolean, offset: number, limit: number ): Promise<ListPage> {
  return await engineRequest( 'catalog.list', { filter, sort, descending, offset, limit } ) as ListPage;
}

export async function getItem( item: string ): Promise<ItemDetail> {
  return await engineRequest( 'catalog.get', { item } ) as ItemDetail;
}

export async function addFiles( paths: readonly string[] ): Promise<{ accepted: number; refused: readonly { index: number; reason: string }[] }> {
  return await engineRequest( 'catalog.add_files', { paths } ) as { accepted: number; refused: { index: number; reason: string }[] };
}

export async function listFolders(): Promise<readonly Folder[]> {
  return ( await engineRequest( 'catalog.folders' ) as { folders: Folder[] } ).folders;
}

export async function addFolder( path: string, typeFromFolder: boolean ): Promise<void> {
  await engineRequest( 'catalog.add_folder', { path, type_from_folder: typeFromFolder } );
}

export async function removeFolder( folder: number ): Promise<void> {
  await engineRequest( 'catalog.remove_folder', { folder } );
}

export async function rescan(): Promise<void> {
  await engineRequest( 'catalog.rescan' );
}

// The video types (structure profiles) the selected detector backend has,
// read from the settings, so a profile added in the config shows up here.
export async function videoTypes(): Promise<string[]> {
  const rows = ( await engineRequest( 'config.get', { paths: [ 'detector_backend' ] } ) as {
    settings: { path: string; value?: unknown }[];
  } ).settings;
  const selected = rows.find( ( row ) => row.path === 'detector_backend.selected' )?.value;
  if ( typeof selected !== 'string' ) {
    return [];
  }
  const prefix = `detector_backend.${selected}.profiles.variants.`;
  const names = new Set<string>();
  for ( const row of rows ) {
    if ( row.path.startsWith( prefix ) ) {
      const name = row.path.slice( prefix.length ).split( '.' )[ 0 ];
      if ( name !== undefined && name !== '' ) {
        names.add( name );
      }
    }
  }
  return [ ...names ].sort();
}

export async function pinType( item: string, type: string | null ): Promise<void> {
  await engineRequest( 'catalog.pin_type', { item, type } );
}

export interface Thumbnail {
  readonly mime: string;
  readonly data: string;
  readonly width: number;
  readonly height: number;
}

// A source's blurred thumbnail, its clear frame with reveal (made in
// memory, written nowhere), or a censored output's thumbnail.
export async function thumbnail( item: string, reveal: boolean, of: 'source' | 'output' = 'source' ): Promise<Thumbnail> {
  return await engineRequest( 'catalog.thumbnail', { item, reveal, of } ) as Thumbnail;
}

export function dataUrl( thumb: Thumbnail ): string {
  return `data:${thumb.mime};base64,${thumb.data}`;
}

export interface Strip {
  readonly path: string;
  readonly blurred: boolean;
  readonly building: boolean;
  readonly ready: boolean;
  readonly density?: 'keyframes' | 'even';
  readonly frames?: readonly number[];
  readonly tile?: { readonly width: number; readonly height: number };
  readonly columns?: number;
  readonly duration?: number;
  readonly image?: string;
}

export async function strip( path: string ): Promise<Strip> {
  return await engineRequest( 'catalog.strip', { path } ) as Strip;
}

// The strip's JPEG as a data: URL, read by the shell from the engine's cache.
export async function stripImage( path: string ): Promise<string> {
  const bytes = new Uint8Array( await invoke<ArrayBuffer>( 'strip_image', { path } ) );
  let binary = '';
  for ( let index = 0; index < bytes.length; index += 0x8000 ) {
    binary += String.fromCharCode( ...bytes.subarray( index, index + 0x8000 ) );
  }
  return `data:image/jpeg;base64,${btoa( binary )}`;
}

// --- jobs ------------------------------------------------------------------

export type JobState = 'queued' | 'running' | 'paused' | 'completed' | 'cancelled' | 'stopped';

export interface Counts {
  readonly total: number;
  readonly processed: number;
  readonly skipped: number;
  readonly failed: number;
}

export interface JobRow {
  readonly job: string;
  readonly state: JobState;
  readonly position: number | null;
  readonly files: readonly string[];
  readonly counts: Counts;
  readonly overrides: Readonly<Record<string, unknown>>;
  readonly restored?: boolean;
  readonly code?: string;
  readonly params?: Readonly<Record<string, unknown>>;
}

// The preview's length, the spec's one-click 20 seconds.
export const PREVIEW_SECONDS = 20;

// A run's overrides. Both set the preview switch, so neither depends on
// what the person's config holds.
export function runOverrides( preview: boolean ): Record<string, unknown> {
  return preview
    ? { preview_mode_enabled: true, preview_max_seconds: PREVIEW_SECONDS }
    : { preview_mode_enabled: false };
}

export async function submitJob( files: readonly string[], preview: boolean ): Promise<{ job: string; position: number }> {
  return await engineRequest( 'job.submit', { files, overrides: runOverrides( preview ) } ) as { job: string; position: number };
}

export async function listJobs(): Promise<readonly JobRow[]> {
  return ( await engineRequest( 'job.list' ) as { jobs: JobRow[] } ).jobs;
}

export async function jobCommand( name: 'job.pause' | 'job.resume' | 'job.cancel', job: string ): Promise<void> {
  await engineRequest( name, { job } );
}

export async function moveJob( job: string, position: number ): Promise<void> {
  await engineRequest( 'job.move', { job, position } );
}

export interface Estimate {
  readonly files: readonly { readonly seconds: number | null; readonly detect_cache: string | null }[];
  readonly seconds: number | null;
  readonly basis: 'history' | 'probe' | 'none';
  readonly look_seen: boolean;
  readonly probing: boolean;
  readonly missing: readonly string[];
}

export async function estimate( files: readonly string[], preview: boolean ): Promise<Estimate> {
  return await engineRequest( 'job.estimate', { files, overrides: runOverrides( preview ) } ) as Estimate;
}

// --- shell ---------------------------------------------------------------

export function pickFiles( title: string, filterName: string, extensions: readonly string[] ): Promise<string[]> {
  return invoke( 'pick_files', { title, filterName, extensions } );
}

export function pickFolder( title: string ): Promise<string | null> {
  return invoke( 'pick_folder', { title } );
}

export function saveAs( source: string, title: string, defaultName: string ): Promise<string | null> {
  return invoke( 'save_as', { source, title, defaultName } );
}

export function openVideo( path: string ): Promise<PlayableVideo> {
  return invoke( 'open_video', { path } );
}

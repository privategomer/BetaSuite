// preview.look, typed, and the shell's look_image: the engine draws a look
// with its own censor code into a PNG in its cache folder; the page asks
// the shell for the bytes and shows them from a blob: URL.
// Shapes: protocol/v1/commands/preview.look.schema.json.
import { invoke } from '@tauri-apps/api/core';
import { engineRequest } from '../engine/client';

export interface Box {
  readonly label: string;
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

export interface PreviewRequest {
  readonly art: { readonly style: string; readonly item: string };
  readonly labels?: readonly string[];
  readonly seed?: number;
}

export interface PreviewResult {
  readonly path: string;
  readonly key: string;
  readonly cached: boolean;
  readonly width: number;
  readonly height: number;
  readonly boxes: readonly Box[];
}

export async function previewLook( previews: readonly PreviewRequest[], overrides: Readonly<Record<string, unknown>> ): Promise<PreviewResult[]> {
  const data: Record<string, unknown> = { previews };
  if ( Object.keys( overrides ).length > 0 ) {
    data[ 'overrides' ] = overrides;
  }
  const answer = await engineRequest( 'preview.look', data ) as { previews: PreviewResult[] };
  return answer.previews;
}

// A preview's key names its bytes exactly, so one data: URL per key serves
// every later request for it. Kept to a bounded number, oldest dropped.
// data: rather than blob: URLs: WebKitGTK does not load blob: URLs made
// under the app's own page scheme, and img-src already allows data:.
const MAX_URLS = 400;
const urls = new Map<string, string>();

function base64( bytes: Uint8Array ): string {
  let text = '';
  const chunk = 0x8000;
  for ( let start = 0; start < bytes.length; start += chunk ) {
    text += String.fromCharCode( ...bytes.subarray( start, start + chunk ) );
  }
  return btoa( text );
}

export async function imageUrl( result: PreviewResult ): Promise<string> {
  const known = urls.get( result.key );
  if ( known !== undefined ) {
    urls.delete( result.key );
    urls.set( result.key, known );
    return known;
  }
  const answer = await invoke<ArrayBuffer | number[]>( 'look_image', { path: result.path } );
  const bytes = answer instanceof ArrayBuffer ? new Uint8Array( answer ) : Uint8Array.from( answer );
  const url = `data:image/png;base64,${base64( bytes )}`;
  urls.set( result.key, url );
  while ( urls.size > MAX_URLS ) {
    const oldest = urls.keys().next().value;
    if ( oldest === undefined ) {
      break;
    }
    urls.delete( oldest );
  }
  return url;
}

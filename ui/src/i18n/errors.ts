// Words an error code from the engine or the shell: error.<code> in the
// catalog, with the error's parameters as the message's values. A code the
// catalog does not know (a newer engine, say) falls back to error.unknown.
import type { EngineError } from '../engine/client';
import { hasMessage, msgKey, t, type MessageValues } from './format';

function values( params: EngineError[ 'params' ] ): MessageValues {
  const out: Record<string, string | number> = {};
  for ( const [ name, value ] of Object.entries( params ) ) {
    out[ name ] = typeof value === 'number' || typeof value === 'string' ? value : JSON.stringify( value );
  }
  return out;
}

export function errorText( error: EngineError ): string {
  const key = msgKey( 'error.', error.code );
  if ( !hasMessage( key ) ) {
    return t( 'error.unknown', { code: error.code } );
  }
  return t( key, values( error.params ) );
}

// t() turns a catalog key and its values into text for the active locale.
// Plurals, numbers and dates are the message format's job, never string
// building: a message is one ICU MessageFormat pattern.
import { IntlMessageFormat } from 'intl-messageformat';
import { describeError, log } from '../log';
import { BASE_LOCALE, catalogs, type CatalogKey } from './catalog';
import { resolveLocale, type ActiveLocale } from './locale';
import { pseudo } from './pseudo';

export type MessageValues = Readonly<Record<string, string | number | Date>>;

declare const dynamic: unique symbol;

// A key assembled at run time, such as error.<code> for an engine error.
// Only msgKey() makes one, so the catalog check can find every prefix.
export type DynamicKey = string & { readonly [ dynamic ]: true };

export function msgKey( prefix: string, part: string ): DynamicKey {
  return `${prefix}${part}` as DynamicKey;
}

let locale: ActiveLocale = resolveLocale( [] );
let messages: Readonly<Record<string, string>> = {};
const base: Readonly<Record<string, string>> = catalogs.get( BASE_LOCALE ) ?? {};
const compiled = new Map<string, IntlMessageFormat>();

// Picks the locale from the tags asked for, and marks the document with
// its language and direction. Call before the first render.
export function initCatalog( requested: readonly string[] ): ActiveLocale {
  locale = resolveLocale( requested );
  messages = catalogs.get( locale.messages ) ?? base;
  compiled.clear();
  document.documentElement.lang = locale.tag;
  document.documentElement.dir = locale.dir;
  return locale;
}

// The catalog locale in use, for files kept per locale (help pages).
export function activeLocale(): string {
  return locale.messages;
}

export function hasMessage( key: string ): boolean {
  return key in messages || key in base;
}

export function t( key: CatalogKey | DynamicKey, values?: MessageValues ): string {
  let format = compiled.get( key );
  if ( format === undefined ) {
    const source = messages[ key ] ?? base[ key ];
    if ( source === undefined ) {
      log.error( `no message for ${key}` );
      return key;
    }
    format = new IntlMessageFormat( source, locale.format );
    compiled.set( key, format );
  }
  try {
    const result = format.format( values );
    const text = Array.isArray( result ) ? result.join( '' ) : String( result );
    return locale.pseudo ? pseudo( text ) : text;
  } catch ( error ) {
    log.error( `cannot format ${key}: ${describeError( error )}` );
    return key;
  }
}

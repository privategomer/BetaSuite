// Picks the UI locale: the first requested tag that has a catalog, by exact
// tag or else by language, falling back to the base locale.
import { BASE_LOCALE, catalogs, PSEUDO_LOCALE } from './catalog';

export interface ActiveLocale {
  // The tag in effect, set as the document's lang.
  tag: string;
  // Whose catalog supplies the messages.
  messages: string;
  // Whose rules format numbers, dates and plurals.
  format: string;
  pseudo: boolean;
  dir: 'ltr' | 'rtl';
}

const RTL_LANGUAGES = new Set( [ 'ar', 'dv', 'fa', 'he', 'ps', 'ur', 'yi' ] );

function language( tag: string ): string {
  return tag.split( '-' )[ 0 ]?.toLowerCase() ?? '';
}

function canonical( tag: string ): string | null {
  try {
    return Intl.getCanonicalLocales( tag )[ 0 ] ?? null;
  } catch {
    return null;
  }
}

function active( tag: string ): ActiveLocale {
  return {
    tag,
    messages: tag,
    format: tag,
    pseudo: false,
    dir: RTL_LANGUAGES.has( language( tag ) ) ? 'rtl' : 'ltr',
  };
}

export function resolveLocale( requested: readonly string[] ): ActiveLocale {
  const available = [ ...catalogs.keys() ];
  for ( const asked of requested ) {
    if ( asked.toLowerCase() === PSEUDO_LOCALE ) {
      return { ...active( BASE_LOCALE ), tag: PSEUDO_LOCALE, pseudo: true };
    }
    const tag = canonical( asked );
    if ( tag === null ) {
      continue;
    }
    const exact = available.find( ( locale ) => locale.toLowerCase() === tag.toLowerCase() );
    if ( exact !== undefined ) {
      return active( exact );
    }
    const sameLanguage = available.find( ( locale ) => language( locale ) === language( tag ) );
    if ( sameLanguage !== undefined ) {
      return active( sameLanguage );
    }
  }
  return active( BASE_LOCALE );
}

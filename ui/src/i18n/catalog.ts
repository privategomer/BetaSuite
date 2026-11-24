// The message catalogs: every i18n/<locale>/<namespace>.json in the
// repository, bundled into the page. Keys are unique across a locale's
// files (scripts/check-catalog.mjs enforces it), so each locale becomes one
// flat map. Messages are ICU MessageFormat.
import type errorsCatalog from '../../../i18n/en-US/errors.json';
import type uiCatalog from '../../../i18n/en-US/ui.json';

// Keys the type checker can verify. Keys built at run time (an engine
// error code, a setting's tip) go through msgKey() in format.ts instead.
export type CatalogKey = keyof typeof uiCatalog | keyof typeof errorsCatalog;

export const BASE_LOCALE = 'en-US';

// Accents and pads every string (pseudo.ts), to show untranslated text and
// layouts that cannot take longer words.
export const PSEUDO_LOCALE = 'qps-ploc';

const files = import.meta.glob<Record<string, string>>( '../../../i18n/*/*.json', {
  eager: true,
  import: 'default',
} );

function byLocale(): Map<string, Readonly<Record<string, string>>> {
  const locales = new Map<string, Readonly<Record<string, string>>>();
  for ( const [ path, messages ] of Object.entries( files ) ) {
    const locale = path.split( '/' ).at( -2 );
    if ( locale !== undefined ) {
      locales.set( locale, { ...locales.get( locale ), ...messages } );
    }
  }
  return locales;
}

export const catalogs: ReadonlyMap<string, Readonly<Record<string, string>>> = byLocale();

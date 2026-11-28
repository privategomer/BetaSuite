// Checks the theme colors in src/theme/tokens.css against WCAG 2.1 AA, in
// both themes: 4.5:1 for text, 3:1 for icons and the focus ring. Also checks
// that the system-dark block and the data-theme="dark" block agree.
//
//   node scripts/check-contrast.mjs [--log-level L] [--console-level L] [--log-file PATH]
//
// No input needed. Exits 0 when every pair passes, 1 otherwise.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createLogger, parseLogArgs } from './lib/log.mjs';

// [foreground, background, minimum ratio, what it is]
const PAIRS = [
  [ 'text', 'bg', 4.5, 'body text' ],
  [ 'text', 'surface', 4.5, 'text on panels' ],
  [ 'text-muted', 'bg', 4.5, 'secondary text' ],
  [ 'text-muted', 'surface', 4.5, 'secondary text on panels' ],
  [ 'accent-text', 'accent', 4.5, 'primary button label' ],
  [ 'status-ok', 'surface', 3, 'ready icon' ],
  [ 'status-busy', 'surface', 3, 'working icon' ],
  [ 'status-problem', 'surface', 3, 'problem icon' ],
  [ 'status-problem', 'surface', 4.5, 'why a part cannot be hidden (tile note)' ],
  [ 'accent', 'surface', 3, 'selected tile border' ],
  [ 'focus', 'bg', 3, 'focus ring' ],
  [ 'focus', 'surface', 3, 'focus ring on panels' ],
];

function variables( block ) {
  return Object.fromEntries( [ ...block.matchAll( /--color-([a-z-]+):\s*(#[0-9a-fA-F]{6})/g ) ].map( ( m ) => [ m[ 1 ], m[ 2 ] ] ) );
}

function blockAfter( css, selector ) {
  const start = css.indexOf( selector );
  if ( start < 0 ) {
    throw new Error( `no ${selector} block in tokens.css` );
  }
  const open = css.indexOf( '{', start );
  let depth = 0;
  for ( let i = open; i < css.length; i += 1 ) {
    depth += css[ i ] === '{' ? 1 : css[ i ] === '}' ? -1 : 0;
    if ( depth === 0 ) {
      return css.slice( open + 1, i );
    }
  }
  throw new Error( `unclosed ${selector} block` );
}

function luminance( hex ) {
  const channel = ( i ) => {
    const c = parseInt( hex.slice( i, i + 2 ), 16 ) / 255;
    return c <= 0.03928 ? c / 12.92 : ( ( c + 0.055 ) / 1.055 ) ** 2.4;
  };
  return 0.2126 * channel( 1 ) + 0.7152 * channel( 3 ) + 0.0722 * channel( 5 );
}

export function ratio( a, b ) {
  const [ hi, lo ] = [ luminance( a ), luminance( b ) ].sort( ( x, y ) => y - x );
  return ( hi + 0.05 ) / ( lo + 0.05 );
}

function main() {
  const ui = path.dirname( path.dirname( fileURLToPath( import.meta.url ) ) );
  const options = parseLogArgs( process.argv.slice( 2 ) );
  const log = createLogger( 'check-contrast', {
    logFile: options.logFile ?? path.join( ui, 'scripts', 'logs', 'check-contrast.log' ),
    logLevel: options.logLevel,
    consoleLevel: options.consoleLevel,
  } );
  const css = fs.readFileSync( path.join( ui, 'src', 'theme', 'tokens.css' ), 'utf8' );
  const themes = {
    light: variables( blockAfter( css, ':root,' ) ),
    dark: variables( blockAfter( css, ':root[data-theme="dark"]' ) ),
  };
  const systemDark = variables( blockAfter( blockAfter( css, '@media (prefers-color-scheme: dark)' ), ':root' ) );
  let failures = 0;
  if ( JSON.stringify( systemDark ) !== JSON.stringify( themes.dark ) ) {
    failures += 1;
    log.error( 'the prefers-color-scheme dark block and the data-theme="dark" block differ' );
  }
  for ( const [ name, colors ] of Object.entries( themes ) ) {
    for ( const [ fg, bg, minimum, what ] of PAIRS ) {
      if ( !( fg in colors ) || !( bg in colors ) ) {
        failures += 1;
        log.error( `${name}: --color-${fg} or --color-${bg} is missing` );
        continue;
      }
      const value = ratio( colors[ fg ], colors[ bg ] );
      const line = `${name}: ${what}, ${fg} ${colors[ fg ]} on ${bg} ${colors[ bg ]}: ${value.toFixed( 2 )}:1 (needs ${minimum}:1)`;
      if ( value < minimum ) {
        failures += 1;
        log.error( line );
      } else {
        log.info( line );
      }
    }
  }
  log.info( `${failures} failure(s); log: ${log.file}` );
  log.close();
  process.exitCode = failures === 0 ? 0 : 1;
}

main();

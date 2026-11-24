// Proves each rule of check-catalog.mjs fires, and that a clean project
// passes. Each case starts from a small valid project, changes a few files,
// and names exactly the rules it must trigger.
//
//   node scripts/check-catalog.selftest.mjs [--log-level L] [--console-level L] [--log-file PATH]
//
// No input needed. Exits 0 when every case matches, 1 otherwise.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { check } from './check-catalog.mjs';
import { createLogger, parseLogArgs } from './lib/log.mjs';

const SCHEMA = JSON.stringify( {
  $defs: {
    'protocol.bad_json': { properties: { code: { const: 'protocol.bad_json' }, params: { properties: {} } } },
    'command.unknown': { properties: { code: { const: 'command.unknown' }, params: { properties: { name: {} } } } },
  },
} );

const BASE = {
  'i18n/en-US/ui.json': JSON.stringify( {
    'app.title': 'BetaSuite',
    'status.files': '{count, plural, one {# file} other {# files}}',
    'action.open': 'Open',
  } ),
  'i18n/en-US/errors.json': JSON.stringify( {
    'error.protocol.bad_json': 'Unreadable message.',
    'error.command.unknown': 'Unknown request {name}.',
    'error.ui.engine.exited': 'The engine stopped.',
    'error.unknown': 'Something went wrong ({code}).',
  } ),
  'errors.schema.json': SCHEMA,
  'rust/engine.rs': 'const EXITED: &str = "ui.engine.exited";\n',
  'index.html': '<!doctype html>\n<html><head><title></title></head><body><div id="root"></div></body></html>\n',
  'src/styles.css': '.strip { padding-inline-start: 4px; inset-inline-end: 0; }\n',
  'src/App.tsx': [
    "import { Msg } from './Msg';",
    "import { t, msgKey } from './format';",
    "const KEYS = { open: 'action.open' } as const;",
    "export function App( { count, code }: { count: number; code: string } ) {",
    "  log.info( 'App rendered with some English' );",
    "  if ( code === 'job.state' ) { throw new Error( 'Bad state here' ); }",
    "  return (",
    "    <main className=\"app main\" aria-hidden=\"false\" data-state=\"ready now\">",
    "      <h1>{t( 'app.title' )}</h1>",
    "      <p><Msg id=\"status.files\" values={{ count }} /></p>",
    "      <p>{t( msgKey( 'error.', code ) )}</p>",
    "      <button type=\"button\">{t( KEYS.open as 'action.open' )}{' '}</button>",
    "    </main>",
    "  );",
    "}",
    "t( 'action.open' );",
  ].join( '\n' ),
};

const CASES = {
  'clean project': { files: {}, expect: [] },
  'JSX text': { files: { 'src/A.tsx': 'export const A = () => <p>Hello there</p>;' }, expect: [ 'source.literal' ] },
  'one-word JSX text': { files: { 'src/A.tsx': 'export const A = () => <b>restart</b>;' }, expect: [ 'source.literal' ] },
  'string in a JSX expression': { files: { 'src/A.tsx': "export const A = () => <b>{'restart'}</b>;" }, expect: [ 'source.literal' ] },
  'conditional in JSX': { files: { 'src/A.tsx': "export const A = ( p: { on: boolean } ) => <b>{p.on ? 'on' : t( 'action.open' )}</b>;" }, expect: [ 'source.literal' ] },
  'text attribute': { files: { 'src/A.tsx': 'export const A = () => <img src="x.png" alt="A photo" />;' }, expect: [ 'source.literal' ] },
  'aria-label attribute': { files: { 'src/A.tsx': 'export const A = () => <button aria-label="close" />;' }, expect: [ 'source.literal' ] },
  'title in an expression': { files: { 'src/A.tsx': "export const A = () => <span title={'Engine'} />;" }, expect: [ 'source.literal' ] },
  'prose in a variable': { files: { 'src/a.ts': "export const label = 'Engine ready';" }, expect: [ 'source.literal' ] },
  'capitalized word in a variable': { files: { 'src/a.ts': "export const label = 'Restart';" }, expect: [ 'source.literal' ] },
  'prose in a template': { files: { 'src/a.ts': 'export const label = ( n: number ) => `${n} files left`;' }, expect: [ 'source.literal' ] },
  'class names built in an expression pass': { files: { 'src/A.tsx': 'export const A = ( p: { tone: string } ) => <b className={`status-icon ${p.tone}`} />;' }, expect: [] },
  'text passed to a setter named like an error': { files: { 'src/a.ts': "setError( 'Engine broke down' );" }, expect: [ 'source.literal' ] },
  'a key used inside a setter counts as used': {
    files: { 'src/a.ts': "setError( t( 'action.spare' ) );", 'i18n/en-US/ui.json': JSON.stringify( { 'app.title': 'BetaSuite', 'status.files': '{count, plural, one {# file} other {# files}}', 'action.open': 'Open', 'action.spare': 'Spare' } ) },
    expect: [],
  },
  'an Error constructor is exempt': { files: { 'src/a.ts': "throw new TypeError( 'Bad input here' );" }, expect: [] },
  'DOM token passes': { files: { 'src/a.ts': "export const isEscape = ( k: string ) => k === 'Escape';" }, expect: [] },
  'exemption with a reason': {
    files: { 'src/a.ts': "// catalog-ignore: a fixture value, never shown\nexport const label = 'Engine ready';" },
    expect: [],
  },
  'exemption without a reason': {
    files: { 'src/a.ts': "// catalog-ignore:\nexport const label = 'Engine ready';" },
    expect: [ 'ignore.reason', 'source.literal' ],
  },
  'unknown key': { files: { 'src/a.ts': "t( 'nope.missing' );" }, expect: [ 'source.key_unknown' ] },
  'unknown Msg id': { files: { 'src/A.tsx': 'export const A = () => <Msg id="nope.missing" />;' }, expect: [ 'source.key_unknown' ] },
  'missing values': { files: { 'src/a.ts': "t( 'status.files' );" }, expect: [ 'source.args' ] },
  'wrong values': { files: { 'src/a.ts': "t( 'status.files', { total: 2 } );" }, expect: [ 'source.args' ] },
  'extra values on Msg': { files: { 'src/A.tsx': 'export const A = () => <Msg id="action.open" values={{ n: 1 }} />;' }, expect: [ 'source.args' ] },
  'msgKey without a literal prefix': { files: { 'src/a.ts': 'export const k = ( p: string ) => msgKey( p, "x" );' }, expect: [ 'source.key_dynamic' ] },
  'msgKey with an unknown prefix': { files: { 'src/a.ts': "export const k = ( c: string ) => msgKey( 'oops.', c );" }, expect: [ 'source.key_dynamic' ] },
  'unused key': { files: { 'i18n/en-US/ui.json': JSON.stringify( { 'app.title': 'BetaSuite', 'status.files': '{count, plural, one {# file} other {# files}}', 'action.open': 'Open', 'action.spare': 'Spare' } ) }, expect: [ 'catalog.unused' ] },
  'not a flat map': { files: { 'i18n/en-US/extra.json': JSON.stringify( { 'a.b': { nested: 'x' } } ) }, expect: [ 'catalog.shape' ] },
  'not JSON': { files: { 'i18n/en-US/extra.json': '{ "a.b": ' }, expect: [ 'catalog.shape' ] },
  'bad key': { files: { 'i18n/en-US/extra.json': JSON.stringify( { 'Bad Key': 'x' } ) }, expect: [ 'catalog.key' ] },
  'duplicate key': { files: { 'i18n/en-US/extra.json': JSON.stringify( { 'app.title': 'Again' } ) }, expect: [ 'catalog.duplicate' ] },
  'bad ICU': { files: { 'i18n/en-US/extra.json': JSON.stringify( { 'x.broken': 'Hello {name' } ) }, expect: [ 'catalog.icu' ] },
  'plural without other': { files: { 'i18n/en-US/extra.json': JSON.stringify( { 'x.files': '{n, plural, one {# file}}' } ) }, expect: [ 'catalog.plural_other' ] },
  'em dash': { files: { 'i18n/en-US/extra.json': JSON.stringify( { 'x.dash': 'Ready \u2014 go' } ) }, expect: [ 'catalog.em_dash' ] },
  'translation with an extra key': { files: { 'i18n/fr-FR/ui.json': JSON.stringify( { 'app.title': 'BetaSuite', 'x.only_here': 'Seul' } ) }, expect: [ 'catalog.locale_extra' ] },
  'engine code without a message': {
    files: { 'errors.schema.json': SCHEMA.replace( '"command.unknown":', '"engine.internal":{"properties":{"code":{"const":"engine.internal"},"params":{"properties":{}}}},"command.unknown":' ) },
    expect: [ 'errors.missing' ],
  },
  'shell code without a message': { files: { 'rust/more.rs': 'const X: &str = "ui.engine.hello_timeout";\n' }, expect: [ 'errors.missing' ] },
  'error message with a value its code lacks': {
    files: { 'i18n/en-US/errors.json': JSON.stringify( { 'error.protocol.bad_json': 'Unreadable line {line}.', 'error.command.unknown': 'Unknown request {name}.', 'error.ui.engine.exited': 'The engine stopped.', 'error.unknown': 'Something went wrong ({code}).' } ) },
    expect: [ 'errors.args' ],
  },
  'error message without a code': {
    files: { 'i18n/en-US/errors.json': JSON.stringify( { 'error.protocol.bad_json': 'Unreadable message.', 'error.command.unknown': 'Unknown request {name}.', 'error.ui.engine.exited': 'The engine stopped.', 'error.unknown': 'Something went wrong ({code}).', 'error.gone.away': 'Old.' } ) },
    expect: [ 'errors.orphan' ],
  },
  'no fallback error message': {
    files: { 'i18n/en-US/errors.json': JSON.stringify( { 'error.protocol.bad_json': 'Unreadable message.', 'error.command.unknown': 'Unknown request {name}.', 'error.ui.engine.exited': 'The engine stopped.' } ) },
    expect: [ 'errors.missing' ],
  },
  'physical CSS property': { files: { 'src/more.css': '.a { margin-left: 4px; }\n' }, expect: [ 'css.physical' ] },
  'physical CSS value': { files: { 'src/more.css': '.a { text-align: right; }\n' }, expect: [ 'css.physical' ] },
  'CSS content text': { files: { 'src/more.css': '.a::after { content: "Loading"; }\n' }, expect: [ 'css.content' ] },
  'HTML title': { files: { 'index.html': '<html><head><title>BetaSuite</title></head><body></body></html>' }, expect: [ 'html.text' ] },
  'HTML alt': { files: { 'index.html': '<html><head><title></title></head><body><img alt="logo"></body></html>' }, expect: [ 'html.text' ] },
};

function project( files ) {
  const dir = fs.mkdtempSync( path.join( os.tmpdir(), 'catalog-selftest-' ) );
  for ( const [ name, content ] of Object.entries( { ...BASE, ...files } ) ) {
    fs.mkdirSync( path.dirname( path.join( dir, name ) ), { recursive: true } );
    fs.writeFileSync( path.join( dir, name ), content );
  }
  return dir;
}

function main() {
  const ui = path.dirname( path.dirname( fileURLToPath( import.meta.url ) ) );
  const options = parseLogArgs( process.argv.slice( 2 ) );
  const log = createLogger( 'catalog-selftest', {
    logFile: options.logFile ?? path.join( ui, 'scripts', 'logs', 'check-catalog-selftest.log' ),
    logLevel: options.logLevel,
    consoleLevel: options.consoleLevel,
  } );
  let failed = 0;
  for ( const [ name, { files, expect } ] of Object.entries( CASES ) ) {
    const dir = project( files );
    try {
      const { findings } = check( {
        i18nDir: path.join( dir, 'i18n' ),
        srcDir: path.join( dir, 'src' ),
        htmlFiles: [ path.join( dir, 'index.html' ) ],
        errorsSchema: path.join( dir, 'errors.schema.json' ),
        rustDir: path.join( dir, 'rust' ),
      } );
      const got = [ ...new Set( findings.map( ( f ) => f.rule ) ) ].sort();
      const want = [ ...expect ].sort();
      if ( JSON.stringify( got ) === JSON.stringify( want ) ) {
        log.info( `ok    ${name}: ${want.join( ', ' ) || 'clean'}` );
      } else {
        failed += 1;
        log.error( `FAIL  ${name}: expected ${want.join( ', ' ) || 'clean'}, got ${got.join( ', ' ) || 'clean'}` );
        for ( const f of findings ) {
          log.debug( `      ${path.relative( dir, f.file )}:${f.line}: ${f.rule}: ${f.message}` );
        }
      }
    } finally {
      fs.rmSync( dir, { recursive: true, force: true } );
    }
  }
  const total = Object.keys( CASES ).length;
  log.info( `${total - failed} of ${total} cases match; log: ${log.file}` );
  log.close();
  process.exitCode = failed === 0 ? 0 : 1;
}

main();

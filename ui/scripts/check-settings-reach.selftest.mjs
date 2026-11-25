// Proves each rule of check-settings-reach.mjs fires, and that the real
// fixture passes. Each case changes a copy of the fixture, the catalog or
// the model, and names exactly the rules it must trigger.
//
//   node scripts/check-settings-reach.selftest.mjs [--log-level L] [--console-level L] [--log-file PATH]
//
// No input needed. Exits 0 when every case matches, 1 otherwise.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { check, loadModel } from './check-settings-reach.mjs';
import { createLogger, parseLogArgs } from './lib/log.mjs';

const UI = path.dirname( path.dirname( fileURLToPath( import.meta.url ) ) );
const REPO = path.dirname( UI );

function edited( fixture, path, change ) {
  const copy = structuredClone( fixture );
  const entry = copy.schema.settings.find( ( item ) => item.path === path );
  change( entry, copy );
  return copy;
}

const CASES = {
  'the real fixture': {},
  'a kind with no editor': { fixture: ( f ) => edited( f, 'encode_crf', ( e ) => { e.kind = 'mystery'; } ), expect: [ 'reach.editor' ] },
  'a screen that is no tab': { fixture: ( f ) => edited( f, 'encode_crf', ( e ) => { e.screen = 'nowhere'; } ), expect: [ 'reach.screen' ] },
  'a setting listed twice': {
    fixture: ( f ) => edited( f, 'encode_crf', ( e, copy ) => { copy.schema.settings.push( structuredClone( e ) ); } ),
    expect: [ 'reach.row' ],
  },
  'a user-named key the config never holds': {
    fixture: ( f ) => {
      const copy = structuredClone( f );
      copy.paths = copy.paths.filter( ( p ) => !p.startsWith( 'profile_by_source_dir.' ) );
      return copy;
    },
    expect: [ 'reach.instances' ],
  },
  'a label missing from the catalog': {
    catalog: ( c ) => { const copy = { ...c }; delete copy[ 'setting.encode_crf.label' ]; return copy; },
    expect: [ 'reach.catalog' ],
  },
  'a parent with no row': {
    fixture: ( f ) => edited( f, 'preview_max_seconds', ( e ) => { e.requires = [ { path: 'no_such_setting', test: 'on' } ]; } ),
    expect: [ 'reach.requires' ],
  },
  'a parent with placeholders': {
    fixture: ( f ) => edited( f, 'preview_max_seconds', ( e ) => { e.requires = [ { path: 'item_overrides.<label>.censor_shape', test: 'on' } ]; } ),
    expect: [ 'reach.requires' ],
  },
  'two settings that require each other': {
    fixture: ( f ) => edited( edited( f, 'encode_crf', ( e ) => { e.requires = [ { path: 'encode_preset', test: 'on' } ]; } ),
      'encode_preset', ( e ) => { e.requires = [ { path: 'encode_crf', test: 'on' } ]; } ),
    expect: [ 'reach.nesting' ],
  },
  'search that finds nothing': { model: ( m ) => ( { ...m, searchText: () => '' } ), expect: [ 'reach.search' ] },
  'search that ignores the tip and path': {
    model: ( m ) => ( { ...m, searchText: ( row, label, _tip, schema ) => m.searchText( { ...row, entry: { ...row.entry, path: '' } }, label, '', schema ) } ),
    expect: [ 'reach.search' ],
  },
};

async function main() {
  const options = parseLogArgs( process.argv.slice( 2 ) );
  const log = createLogger( 'settings-reach-selftest', {
    logFile: options.logFile ?? path.join( UI, 'scripts', 'logs', 'check-settings-reach-selftest.log' ),
    logLevel: options.logLevel,
    consoleLevel: options.consoleLevel,
  } );
  const model = await loadModel();
  const fixture = JSON.parse( fs.readFileSync( path.join( UI, 'scripts', 'fixtures', 'settings.json' ), 'utf8' ) );
  const catalog = JSON.parse( fs.readFileSync( path.join( REPO, 'i18n', 'en-US', 'settings.json' ), 'utf8' ) );
  let failed = 0;
  for ( const [ name, change ] of Object.entries( CASES ) ) {
    const { findings } = check( ( change.model ?? ( ( m ) => m ) )( model ),
      ( change.fixture ?? ( ( f ) => f ) )( fixture ), ( change.catalog ?? ( ( c ) => c ) )( catalog ) );
    const got = [ ...new Set( findings.map( ( finding ) => finding.rule ) ) ].sort();
    const want = [ ...( change.expect ?? [] ) ].sort();
    if ( JSON.stringify( got ) === JSON.stringify( want ) ) {
      log.info( `ok    ${name}: ${got.join( ', ' ) || 'clean'}` );
    } else {
      failed += 1;
      log.error( `FAIL  ${name}: expected [${want.join( ', ' )}], got [${got.join( ', ' )}]` );
      for ( const finding of findings.slice( 0, 5 ) ) {
        log.debug( `      ${finding.setting}: ${finding.rule}: ${finding.message}` );
      }
    }
  }
  const total = Object.keys( CASES ).length;
  log.info( `${total - failed} of ${total} cases match; log: ${log.file}` );
  log.close();
  process.exitCode = failed === 0 ? 0 : 1;
}

await main();

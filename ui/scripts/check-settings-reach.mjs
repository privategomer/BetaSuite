// The reach check: every setting in the registry can be found, read and
// changed in All Settings.
//
//   node scripts/check-settings-reach.mjs [--fixture PATH] [--log-level L] [--console-level L] [--log-file PATH]
//
// No input needed. Reads the registry as the engine sends it
// (scripts/fixtures/settings.json, kept current by the engine's
// tests/test_settings_fixture.py) and runs the page's own model
// (src/settings/model.ts) over it. Exits 0 when every setting is reachable,
// 1 with one line per problem. Rules:
//
//   reach.row         a setting has no row, or more than one
//   reach.editor      a setting's kind has no editor
//   reach.screen      a setting's screen is not a tab
//   reach.instances   a setting with placeholders stands for no concrete setting
//   reach.search      searching for a setting's label does not find it
//   reach.catalog     a setting's label or tip is not in the catalog
//   reach.requires    a setting requires a parent All Settings has no row for
//   reach.nesting     placing settings under their parents loses one (a cycle)
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import ts from 'typescript';
import { createLogger, parseLogArgs } from './lib/log.mjs';

const UI = path.dirname( path.dirname( fileURLToPath( import.meta.url ) ) );
const REPO = path.dirname( UI );

// The page's model, compiled to plain JavaScript the way the build does.
export async function loadModel( source = path.join( UI, 'src', 'settings', 'model.ts' ) ) {
  const output = ts.transpileModule( fs.readFileSync( source, 'utf8' ), {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
    fileName: source,
  } );
  const dir = fs.mkdtempSync( path.join( os.tmpdir(), 'settings-model-' ) );
  const file = path.join( dir, 'model.mjs' );
  fs.writeFileSync( file, output.outputText );
  try {
    return await import( pathToFileURL( file ).href );
  } finally {
    fs.rmSync( dir, { recursive: true, force: true } );
  }
}

export function check( model, fixture, catalog ) {
  const findings = [];
  const report = ( rule, setting, message ) => findings.push( { rule, setting, message } );
  const { schema, paths } = fixture;
  const rows = model.buildRows( schema );
  const byPath = new Map();
  for ( const row of rows ) {
    byPath.set( row.entry.path, [ ...( byPath.get( row.entry.path ) ?? [] ), row ] );
  }
  const noValues = new Map();
  const textOf = ( row ) => model.searchText( row, catalog[ row.entry.label ] ?? '', catalog[ row.entry.tip ] ?? '', schema );
  let instanceCount = 0;
  for ( const entry of schema.settings ) {
    const found = byPath.get( entry.path ) ?? [];
    if ( found.length !== 1 ) {
      report( 'reach.row', entry.path, `${found.length} rows` );
      continue;
    }
    const row = found[ 0 ];
    if ( row.editor === null ) {
      report( 'reach.editor', entry.path, `no editor for kind ${entry.kind}` );
    }
    if ( !model.SCREENS.includes( entry.screen ) ) {
      report( 'reach.screen', entry.path, `screen ${entry.screen} is not a tab` );
    }
    const concrete = model.instances( schema, paths, entry.path );
    instanceCount += concrete.length;
    if ( concrete.length === 0 ) {
      report( 'reach.instances', entry.path, 'stands for no concrete setting' );
    }
    for ( const key of [ entry.label, entry.tip ] ) {
      if ( typeof catalog[ key ] !== 'string' ) {
        report( 'reach.catalog', entry.path, `${key} is not in the catalog` );
      }
    }
    const label = catalog[ entry.label ];
    if ( typeof label === 'string' ) {
      for ( const query of [ label, entry.path ] ) {
        const hits = model.filterRows( rows, { ...model.NO_FILTERS, query }, textOf, noValues );
        if ( !hits.includes( row ) ) {
          report( 'reach.search', entry.path, `searching for "${query}" does not find it` );
        }
      }
    }
    for ( const requirement of entry.requires ) {
      const parent = byPath.get( requirement.path ) ?? [];
      if ( parent.length !== 1 || parent[ 0 ].pattern ) {
        report( 'reach.requires', entry.path, `requires ${requirement.path}, which has no single row` );
      }
    }
  }
  // Every tab and fold lists its rows nested under their parents; none may
  // drop out on the way.
  const groups = [ rows, ...model.byScreen( rows ).values(),
    ...[ ...model.byScreen( rows ).values() ].map( ( group ) => group.filter( ( row ) => row.entry.level === 'advanced' ) ) ];
  for ( const group of groups ) {
    const placed = new Set( model.nestByParent( group ).map( ( item ) => item.row ) );
    for ( const row of group ) {
      if ( !placed.has( row ) ) {
        report( 'reach.nesting', row.entry.path, 'lost when settings are placed under their parents' );
      }
    }
  }
  return { findings, rows: rows.length, instances: instanceCount };
}

async function main() {
  const options = parseLogArgs( process.argv.slice( 2 ) );
  const log = createLogger( 'check-settings-reach', {
    logFile: options.logFile ?? path.join( UI, 'scripts', 'logs', 'check-settings-reach.log' ),
    logLevel: options.logLevel,
    consoleLevel: options.consoleLevel,
  } );
  let fixturePath = path.join( UI, 'scripts', 'fixtures', 'settings.json' );
  for ( let i = 0; i < options.rest.length; i += 1 ) {
    const [ flag, inline ] = options.rest[ i ].split( /=(.*)/s, 2 );
    if ( flag === '--fixture' ) {
      fixturePath = inline ?? options.rest[ ( i += 1 ) ];
    } else {
      log.error( `unknown option ${options.rest[ i ]}` );
      log.close();
      process.exitCode = 2;
      return;
    }
  }
  const catalogPath = path.join( REPO, 'i18n', 'en-US', 'settings.json' );
  log.debug( `fixture: ${fixturePath}; catalog: ${catalogPath}` );
  const model = await loadModel();
  const fixture = JSON.parse( fs.readFileSync( fixturePath, 'utf8' ) );
  const catalog = JSON.parse( fs.readFileSync( catalogPath, 'utf8' ) );
  const { findings, rows, instances } = check( model, fixture, catalog );
  for ( const { rule, setting, message } of findings ) {
    log.error( `${setting}: ${rule}: ${message}` );
  }
  log.info( `${fixture.schema.settings.length} settings, ${rows} rows, ${instances} concrete settings; `
    + `${findings.length} problem(s); log: ${log.file}` );
  log.close();
  process.exitCode = findings.length === 0 ? 0 : 1;
}

if ( process.argv[ 1 ] && path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url ) ) {
  await main();
}
